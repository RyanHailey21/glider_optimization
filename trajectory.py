import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca
from config import g, DROP_HEIGHT, WING_AF, TAIL_AF

def optimize_trajectory(bc):
    SPAN       = bc["span"]
    CHRD       = bc["chord"]
    MASS       = bc["mass"]
    TAIL_ARM   = bc["tail_arm"]
    TAIL_CHORD = bc["tail_chord"]
    TAIL_SPAN  = bc["tail_span"]
    TAIL_INC   = bc["i_tail"]
    CG_X       = 0.25 * CHRD

    V_glide = bc["V"]
    sink    = bc["sink"]
    N_dive  = 20
    N_glide = 60
    N       = N_dive + N_glide

    t_dive  = V_glide / g
    h_dive  = min(0.5 * g * t_dive**2, DROP_HEIGHT * 0.35)
    h_glide = DROP_HEIGHT - h_dive
    T_guess = t_dive + h_glide / sink

    alt_g   = np.concatenate([
        np.linspace(DROP_HEIGHT, DROP_HEIGHT - h_dive, N_dive),
        np.linspace(DROP_HEIGHT - h_dive, 0.05, N_glide),
    ])
    x_g     = np.concatenate([
        np.linspace(0, 0.3, N_dive),
        np.linspace(0.3, V_glide * h_glide / sink * 0.95, N_glide),
    ])
    V_g     = np.concatenate([
        np.linspace(1.0, V_glide, N_dive),
        np.full(N_glide, V_glide),
    ])
    gam_g   = np.concatenate([
        np.linspace(np.radians(-80), np.radians(-12), N_dive),
        np.linspace(np.radians(-12), np.radians(-3),  N_glide),
    ])
    alpha_g = np.concatenate([
        np.linspace(0.0, bc["alpha"] * 0.5, N_dive),
        np.full(N_glide, bc["alpha"]),
    ])

    print(f"  Geometry: span={SPAN*100:.0f}cm  chord={CHRD*100:.0f}cm  tail_arm={TAIL_ARM*100:.0f}cm  tail_inc={TAIL_INC:.1f}°")
    print(f"  Warm-start: T_guess={T_guess:.1f}s  N={N}")

    opti    = asb.Opti()
    T_final = opti.variable(init_guess=T_guess, lower_bound=1.0, upper_bound=120.0)
    time    = T_final * ca.linspace(0, 1, N)

    dyn = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=MASS),
        x_e   = opti.variable(init_guess=x_g),
        z_e   = opti.variable(init_guess=-alt_g),
        speed = opti.variable(init_guess=V_g, lower_bound=0.5, upper_bound=25.0),
        gamma = opti.variable(init_guess=gam_g, lower_bound=np.radians(-89), upper_bound=np.radians(45)),
        alpha = opti.variable(init_guess=alpha_g, lower_bound=-12.0, upper_bound=14.0),
    )

    airplane = asb.Airplane(
        name="Glider", xyz_ref=[CG_X, 0, 0],
        wings=[
            asb.Wing(
                name="Main Wing", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=CHRD, twist=0.0, airfoil=WING_AF),
                    asb.WingXSec(xyz_le=[0, SPAN/2, SPAN/2 * 0.04], chord=CHRD, twist=-2.0, airfoil=WING_AF),
                ]
            ),
            asb.Wing(
                name="Horizontal Stabilizer", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=TAIL_CHORD, twist=TAIL_INC, airfoil=TAIL_AF),
                    asb.WingXSec(xyz_le=[0, TAIL_SPAN, TAIL_SPAN * 0.02], chord=TAIL_CHORD, twist=TAIL_INC, airfoil=TAIL_AF),
                ]
            ).translate([TAIL_ARM, 0, 0]),
        ]
    )

    aero = asb.AeroBuildup(airplane=airplane, op_point=dyn.op_point).run()

    dyn.add_gravity_force(g=g)
    dyn.add_force(*aero["F_w"], axes="wind")
    dyn.constrain_derivatives(opti, time)

    opti.subject_to([
        dyn.x_e[0]   == 0.0,
        dyn.z_e[0]   == -DROP_HEIGHT,
        dyn.speed[0] <= 1.5,
        dyn.gamma[0] == np.radians(-80),
        dyn.z_e[-1]   == 0.0,
        dyn.gamma[-1] >= np.radians(-30),
        dyn.gamma[-1] <= np.radians(0),
        dyn.speed[-1] >= bc["V"] * 0.7,
        dyn.z_e      <= 0.0,
        dyn.gamma    <= np.radians(30),
    ])

    alpha_smoothness = np.sum(np.diff(dyn.alpha) ** 2)
    opti.minimize(-T_final + 1e-3 * alpha_smoothness)

    try:
        sol = opti.solve(verbose=True, max_iter=3000)
        print("  ✓ NLP converged.")
        return {
            "T_opt" : float(sol(T_final)),
            "t_sol" : sol(time).flatten(),
            "x_sol" : sol(dyn.x_e).flatten(),
            "z_sol" : sol(dyn.z_e).flatten(),
            "V_sol" : sol(dyn.speed).flatten(),
            "g_sol" : np.degrees(sol(dyn.gamma).flatten()),
            "a_sol" : sol(dyn.alpha).flatten(),
            "CL_sol": sol(aero["CL"]).flatten(),
            "CD_sol": sol(aero["CD"]).flatten(),
            "Cm_sol": sol(aero["Cm"]).flatten(),
            "h_dive": h_dive
        }
    except Exception as exc:
        print(f"  ✗ NLP failed: {exc}")
        return {
            "T_opt" : T_guess,
            "t_sol" : np.linspace(0, T_guess, N),
            "x_sol" : x_g,
            "z_sol" : -alt_g,
            "V_sol" : V_g,
            "g_sol" : np.degrees(gam_g),
            "a_sol" : alpha_g,
            "CL_sol": np.full(N, bc["CL"]),
            "CD_sol": np.full(N, bc["CD"]),
            "Cm_sol": np.zeros(N),
            "h_dive": h_dive
        }