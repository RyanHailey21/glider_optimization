import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca
import numpy as onp
from config import *

def optimize_trajectory(best):
    print(f"\n  [Phase 2] Solving Trajectory for {best['wing_af_name'].upper()}...")
    opti = asb.Opti()
    
    # ── Trajectory Variables ──
    N_dive  = 20
    N_glide = 60
    N       = N_dive + N_glide
    T_guess = 12.0
    
    # ── Geometry from Phase 1 ──
    mac = best["chord"] * (2 / 3) * (1 + best["taper"] + best["taper"] ** 2) / (1 + best["taper"])
    cg_x = 0.25 * mac
    wing_af_obj = asb.Airfoil(best["wing_af_name"])
    
    # Calculate sweep offset for a straight leading edge (optional, but typical for gliders)
    # The leading edge x-coordinate at the tip will be 0
    
    airplane = asb.Airplane(
        name="Glider", xyz_ref=[cg_x, 0, 0],
        wings=[
            asb.Wing(
                name="Main Wing", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=best["chord"], twist=0.0, airfoil=wing_af_obj),
                    asb.WingXSec(
                        xyz_le=[0, best["span"]/2, best["span"]/2 * 0.04], 
                        chord=best["chord"] * best["taper"], 
                        twist=best["twist"], 
                        airfoil=wing_af_obj
                    ),
                ]
            ),
            asb.Wing(
                name="Horizontal Stabilizer", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=best["tail_chord"], twist=best["i_tail"], airfoil=TAIL_AF),
                    asb.WingXSec(xyz_le=[0, best["tail_span"], best["tail_span"] * 0.02], chord=best["tail_chord"], twist=best["i_tail"], airfoil=TAIL_AF),
                ]
            ).translate([cg_x + best["tail_arm"] - 0.25 * best["tail_chord"], 0, 0]),
        ]
    )

    T_final = opti.variable(init_guess=T_guess, lower_bound=1.0, upper_bound=120.0)
    tau = onp.linspace(0.0, 1.0, N)
    time = T_final * tau

    h_dive_g= min(0.35 * DROP_HEIGHT, 6.0)
    alt_g   = np.concatenate([
        np.linspace(DROP_HEIGHT, DROP_HEIGHT - h_dive_g, N_dive),
        np.linspace(DROP_HEIGHT - h_dive_g, 0.05, N_glide),
    ])
    x_g     = np.concatenate([
        np.linspace(0, 0.5, N_dive),
        np.linspace(0.5, 130.0, N_glide),
    ])
    V_g     = np.concatenate([
        np.linspace(5.0, 15.0, N_dive),
        np.full(N_glide, 15.0),
    ])
    gam_g   = np.concatenate([
        np.linspace(np.radians(-80), np.radians(-12), N_dive),
        np.linspace(np.radians(-12), np.radians(-3),  N_glide),
    ])
    
    # Model alpha as a low-order piecewise-linear control schedule.
    alpha_stall_deg = float(best.get("alpha_stall_deg", 12.0))
    alpha_stall_margin_deg = float(best.get("alpha_stall_margin_deg", 2.0))
    alpha_upper_bound = min(10.0, alpha_stall_deg - alpha_stall_margin_deg)
    alpha_lower_bound = -10.0
    n_alpha_ctrl = int(max(2, TRAJ_ALPHA_CTRL_POINTS))
    alpha_ctrl = opti.variable(
        init_guess=np.full(n_alpha_ctrl, 5.0),
        lower_bound=alpha_lower_bound,
        upper_bound=alpha_upper_bound,
    )

    tau_ctrl = onp.linspace(0.0, 1.0, n_alpha_ctrl)
    W = onp.zeros((N, n_alpha_ctrl))
    for i, t in enumerate(tau):
        j = onp.searchsorted(tau_ctrl, t, side="right") - 1
        j = int(onp.clip(j, 0, n_alpha_ctrl - 2))
        dt_local = tau_ctrl[j + 1] - tau_ctrl[j]
        w_hi = (t - tau_ctrl[j]) / dt_local
        w_lo = 1.0 - w_hi
        W[i, j] = w_lo
        W[i, j + 1] = w_hi

    alpha_profile = ca.mtimes(ca.DM(W), alpha_ctrl)

    dyn = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=best["mass"]),
        x_e   = opti.variable(init_guess=x_g),
        z_e   = opti.variable(init_guess=-alt_g),
        speed = opti.variable(init_guess=V_g, lower_bound=3.0, upper_bound=35.0),
        gamma = opti.variable(init_guess=gam_g, lower_bound=np.radians(-89), upper_bound=np.radians(45)),
        alpha = alpha_profile,
    )

    aero = asb.AeroBuildup(airplane=airplane, op_point=dyn.op_point).run()

    dyn.add_gravity_force(g=g)
    dyn.add_force(*aero["F_w"], axes="wind")
    dyn.constrain_derivatives(opti, time)

    opti.subject_to([
        dyn.x_e[0]   == 0.0,
        dyn.z_e[0]   == -DROP_HEIGHT,
        dyn.speed[0] <= 5.0,
        dyn.gamma[0] <= np.radians(-40),
        dyn.z_e[-1]  == 0.0,
        # natural pull-ups might violate arbitrary gamma constraints, so we remove them
        dyn.z_e      <= 0.0,
    ])

    dt = np.diff(time)
    alpha_dot = np.diff(dyn.alpha) / dt
    opti.subject_to(alpha_dot ** 2 <= TRAJ_ALPHA_DOT_MAX_DEG_S ** 2)

    # Stall protection is enforced consistently via alpha bounds derived from
    # Phase 1 wing-stall estimates, instead of mixing whole-aircraft CL with wing CL limits.
    opti.subject_to([
        dyn.alpha <= alpha_upper_bound,
        dyn.alpha >= alpha_lower_bound,
    ])

    gamma_smoothness = np.sum(np.diff(dyn.gamma) ** 2)
    alpha_smoothness = np.sum(np.diff(dyn.alpha) ** 2)
    
    # Penalize pitch moment softly; normalized by horizon length to keep
    # flight-time maximization as the dominant objective.
    cm_penalty = ca.sum1(aero["Cm"] ** 2) / N
    
    # The objective: maximize flight time, smoothly pull out, and stay in natural pitch trim
    opti.minimize(
        -T_final
        + 1e-3 * gamma_smoothness
        + TRAJ_ALPHA_SMOOTH_WEIGHT * alpha_smoothness
        + TRAJ_CM_PENALTY_WEIGHT * cm_penalty
    )

    try:
        sol = opti.solve(verbose=False, max_iter=3000)
        return {
            "trajectory_feasible": True,
            "T_opt" : float(sol(T_final)),
            "t_sol" : sol(time).flatten(),
            "x_sol" : sol(dyn.x_e).flatten(),
            "z_sol" : sol(dyn.z_e).flatten(),
            "V_sol" : sol(dyn.speed).flatten(),
            "g_sol" : np.degrees(sol(dyn.gamma).flatten()),
            "a_sol" : sol(dyn.alpha).flatten(),
            "CL_sol": sol(aero["CL"]).flatten(),
            "CD_sol": sol(aero["CD"]).flatten(),
            "Cm_sol": sol(aero["Cm"]).flatten()
        }
    except Exception as exc:
        print(f"  ✗ [Trajectory] NLP Soft-Converged or Failed: {exc}")
        
        try:
            # When CasADi is extremely close (e.g., 1e-9 violation) it might still throw an Exception.
            # We can still extract the visually correct physics path via debug.value()
            return {
                "trajectory_feasible": False,
                "T_opt" : float(opti.debug.value(T_final)),
                "t_sol" : opti.debug.value(time).flatten(),
                "x_sol" : opti.debug.value(dyn.x_e).flatten(),
                "z_sol" : opti.debug.value(dyn.z_e).flatten(),
                "V_sol" : opti.debug.value(dyn.speed).flatten(),
                "g_sol" : np.degrees(opti.debug.value(dyn.gamma).flatten()),
                "a_sol" : opti.debug.value(dyn.alpha).flatten(),
                "CL_sol": opti.debug.value(aero["CL"]).flatten(),
                "CD_sol": opti.debug.value(aero["CD"]).flatten(),
                "Cm_sol": opti.debug.value(aero["Cm"]).flatten()
            }
        except:
            # Fallback empty trajectory dict for plotting if the dynamic sim fails
            # but we still want to see the glider build guide
            return {
                "trajectory_feasible": False,
                "T_opt" : 0.0, "t_sol" : np.zeros(N),
                "x_sol" : np.zeros(N), "z_sol" : np.zeros(N),
                "V_sol" : np.zeros(N), "g_sol" : np.zeros(N), "a_sol" : np.zeros(N),
                "CL_sol": np.zeros(N), "CD_sol": np.zeros(N), "Cm_sol": np.zeros(N)
            }
