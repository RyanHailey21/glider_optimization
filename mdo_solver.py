import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca
import numpy as onp
from config import *
from mass_model import estimate_mass

def optimize_glider_mdo(wing_af_name):
    print(f"\n  [MDO] Building aerodynamic surrogate model for {wing_af_name.upper()}...")
    try:
        from neuralfoil import get_aero_from_airfoil
        
        # Grid definition
        alphas = np.linspace(-10, 15, 26)
        Res    = np.array([20e3, 40e3, 60e3, 80e3, 100e3, 150e3])
        
        af_wing = asb.Airfoil(wing_af_name)
        A, R = np.meshgrid(alphas, Res, indexing='ij')
        
        CL_w = np.zeros_like(A); CD_w = np.zeros_like(A); Cm_w = np.zeros_like(A)
        CL_t = np.zeros_like(A); CD_t = np.zeros_like(A); Cm_t = np.zeros_like(A)
        
        for i, a in enumerate(alphas):
            for j, re in enumerate(Res):
                rw = get_aero_from_airfoil(af_wing, alpha=a, Re=re)
                rt = get_aero_from_airfoil(TAIL_AF, alpha=a, Re=re)
                
                CL_w[i, j] = float(np.atleast_1d(rw["CL"])[0])
                CD_w[i, j] = float(np.atleast_1d(rw["CD"])[0])
                Cm_w[i, j] = float(np.atleast_1d(rw["CM"])[0])
                
                CL_t[i, j] = float(np.atleast_1d(rt["CL"])[0])
                CD_t[i, j] = float(np.atleast_1d(rt["CD"])[0])
                Cm_t[i, j] = float(np.atleast_1d(rt["CM"])[0])
                
        # Conservative static stall references from the sampled 2D table.
        cl_w_max_2d = float(np.max(CL_w))
        cl_w_min_2d = float(np.min(CL_w))
        alpha_stall_deg = float(np.min(alphas[np.argmax(CL_w, axis=0)]))

        spline_cl_w = ca.interpolant("clw", "bspline", [alphas, Res], CL_w.ravel(order='F'))
        spline_cd_w = ca.interpolant("cdw", "bspline", [alphas, Res], CD_w.ravel(order='F'))
        spline_cm_w = ca.interpolant("cmw", "bspline", [alphas, Res], Cm_w.ravel(order='F'))
        
        spline_cl_t = ca.interpolant("clt", "bspline", [alphas, Res], CL_t.ravel(order='F'))
        spline_cd_t = ca.interpolant("cdt", "bspline", [alphas, Res], CD_t.ravel(order='F'))
        spline_cm_t = ca.interpolant("cmt", "bspline", [alphas, Res], Cm_t.ravel(order='F'))
        
    except ImportError:
        print("NeuralFoil missing. Skipping surrogate generation.")
        return None

    print(f"  [MDO] Solving static NLP for minimum sink rate (flight time)...")
    
    # NLP Setup
    opti = asb.Opti()
    
    # === 1. Geometric & Payload Variables ===
    span       = opti.variable(init_guess=0.40, lower_bound=MIN_SPAN,  upper_bound=MAX_SPAN)
    root_chord = opti.variable(init_guess=0.08, lower_bound=MIN_CHORD, upper_bound=MAX_CHORD)
    taper      = opti.variable(init_guess=0.6,  lower_bound=0.3,       upper_bound=1.0)
    twist      = opti.variable(init_guess=-2.0, lower_bound=-6.0,      upper_bound=2.0)
    
    tail_arm   = opti.variable(init_guess=0.40, lower_bound=MIN_TAIL_ARM,   upper_bound=MAX_TAIL_ARM)
    tail_chord = opti.variable(init_guess=0.04, lower_bound=MIN_TAIL_CHORD, upper_bound=MAX_TAIL_CHORD)
    tail_span  = opti.variable(init_guess=0.08, lower_bound=MIN_TAIL_SPAN,  upper_bound=MAX_TAIL_SPAN)
    i_tail     = opti.variable(init_guess=-5.0, lower_bound=-15.0, upper_bound=5.0)
    
    batt_x     = opti.variable(init_guess=-0.10,          lower_bound=MIN_BATTERY_X, upper_bound=MAX_BATTERY_X)
    motor_x    = opti.variable(init_guess=0.20,           lower_bound=MIN_MOTOR_X,   upper_bound=MAX_MOTOR_X)
    
    # MAC calculation for swept/tapered wings
    # For a straight leading-edge wing, mac = root_chord * 2/3 * (1 + taper + taper**2) / (1 + taper)
    mac = root_chord * (2/3) * (1 + taper + taper**2) / (1 + taper)
    S    = span * root_chord * (1 + taper) / 2
    cg_x = 0.25 * mac
    
    opti.subject_to([
        motor_x <= tail_arm + cg_x,          
        (span / mac) >= 4.0,               
        (span / mac) <= 15.0
    ])
    
    # === 2. Mass & Balance Physics ===
    fc_x    = 0.0
    other_x = (batt_x + cg_x + tail_arm) / 2
    ballast = opti.variable(init_guess=0.01, lower_bound=0.0, upper_bound=0.300)

    # Note: Using root_chord as a proxy for structural estimate
    structural_mass, structural_moment = estimate_mass(
        span=span,
        root_chord=root_chord,
        taper=taper,
        tail_span=tail_span,
        tail_chord=tail_chord,
        tail_arm=tail_arm,
        target_cg_x=cg_x,
    )
    total_mass = structural_mass + PAYLOAD_TOTAL + ballast
    W = total_mass * g
    
    total_moment = structural_moment + ((PAYLOAD_BATTERY + ballast) * batt_x) + (PAYLOAD_MOTOR * motor_x) + (PAYLOAD_FC * fc_x) + (PAYLOAD_OTHER * other_x)
    opti.subject_to(total_moment / total_mass == cg_x)

    # === 3. State Variables for Trim ===
    alpha = opti.variable(init_guess=5.0, lower_bound=-5.0, upper_bound=12.0)
    V     = opti.variable(init_guess=10.0, lower_bound=3.0, upper_bound=25.0)

    # === 4. Surrogate Aerodynamics & 3D Corrections ===
    # For a twisted wing, calculate aero at MAC 
    effective_alpha_MAC = alpha + twist * (0.5 * (1 + taper) * 0.5) # Roughly semi-span integrated twist
    wing_re = ca.fmax(ca.fmin(rho * V * mac / mu, 150e3), 20e3)
    
    op_pts_w = ca.horzcat(effective_alpha_MAC, wing_re).T
    cl_w_2d  = spline_cl_w(op_pts_w).T
    cd_w_2d  = spline_cd_w(op_pts_w).T
    cm_w_2d  = spline_cm_w(op_pts_w).T
    
    AR_w = span**2 / S
    e_w  = 0.95
    CL_wing  = cl_w_2d * (AR_w / (AR_w + 2))
    CDi_wing = (CL_wing ** 2) / (np.pi * AR_w * e_w)
    CD_wing  = cd_w_2d + CDi_wing
    
    eps = 2 * CL_wing / (np.pi * AR_w)
    tail_alpha = alpha + i_tail - eps * (180/np.pi)
    
    tail_re = ca.fmax(ca.fmin(rho * V * tail_chord / mu, 150e3), 20e3)
    op_pts_t = ca.horzcat(tail_alpha, tail_re).T
    cl_t_2d  = spline_cl_t(op_pts_t).T
    cd_t_2d  = spline_cd_t(op_pts_t).T
    cm_t_2d  = spline_cm_t(op_pts_t).T
    
    AR_t = tail_span * 2 / tail_chord
    e_t  = 0.95
    CL_tail  = cl_t_2d * (AR_t / (AR_t + 2))
    CDi_tail = (CL_tail ** 2) / (np.pi * AR_t * e_t)
    CD_tail  = cd_t_2d + CDi_tail
    
    CD0_fuselage = 0.005
    S_tail = tail_span * 2 * tail_chord
    q = 0.5 * rho * V ** 2
    
    Total_Lift = q * (CL_wing * S + CL_tail * S_tail)
    Total_Drag = q * (CD_wing * S + CD_tail * S_tail + CD0_fuselage * S)
    Moment_Y = q * (cm_w_2d * S * mac + cm_t_2d * S_tail * tail_chord) - (q * CL_tail * S_tail * tail_arm)

    # === 5. Trim Constraints ===
    opti.subject_to([
        Total_Lift == W,
        # Tolerance on pitch moment to help continuous convergence
        (Moment_Y / (q * S * mac)) ** 2 <= 1e-6
    ])

    # Static trim must stay away from stall in both alpha and lift coefficient.
    stall_margin_alpha_deg = 2.0
    stall_margin_cl = 0.90
    cl_w_max_3d_est = cl_w_max_2d * (AR_w / (AR_w + 2))
    cl_w_min_3d_est = cl_w_min_2d * (AR_w / (AR_w + 2))
    opti.subject_to(alpha <= alpha_stall_deg - stall_margin_alpha_deg)
    opti.subject_to(CL_wing <= stall_margin_cl * cl_w_max_3d_est)
    opti.subject_to(CL_wing >= stall_margin_cl * cl_w_min_3d_est)

    # === 6. Objective ===
    # Maximize flight time (minimize sink rate)
    opti.minimize(V * Total_Drag / Total_Lift)

    # === 7. Solve ===
    # Multi-start improves robustness on this nonconvex NLP.
    rng = onp.random.default_rng(7)
    n_starts = 6
    best_sol = None
    best_sink = None

    for i in range(n_starts):
        if i > 0:
            opti.set_initial(span, float(rng.uniform(MIN_SPAN, MAX_SPAN)))
            opti.set_initial(root_chord, float(rng.uniform(MIN_CHORD, MAX_CHORD)))
            opti.set_initial(taper, float(rng.uniform(0.3, 1.0)))
            opti.set_initial(twist, float(rng.uniform(-6.0, 2.0)))
            opti.set_initial(tail_arm, float(rng.uniform(MIN_TAIL_ARM, MAX_TAIL_ARM)))
            opti.set_initial(tail_chord, float(rng.uniform(MIN_TAIL_CHORD, MAX_TAIL_CHORD)))
            opti.set_initial(tail_span, float(rng.uniform(MIN_TAIL_SPAN, MAX_TAIL_SPAN)))
            opti.set_initial(i_tail, float(rng.uniform(-15.0, 5.0)))
            opti.set_initial(batt_x, float(rng.uniform(MIN_BATTERY_X, MAX_BATTERY_X)))
            opti.set_initial(motor_x, float(rng.uniform(MIN_MOTOR_X, 0.9)))
            opti.set_initial(ballast, float(rng.uniform(0.0, 0.08)))
            opti.set_initial(alpha, float(rng.uniform(-2.0, 8.0)))
            opti.set_initial(V, float(rng.uniform(4.0, 20.0)))

        try:
            sol_i = opti.solve(verbose=False, max_iter=2000)
            sink_i = float(sol_i(V * Total_Drag / Total_Lift))

            if best_sink is None or sink_i < best_sink:
                best_sink = sink_i
                best_sol = sol_i
        except Exception:
            continue

    if best_sol is None:
        print(f"  ✗ [{wing_af_name.upper()}] Static NLP Failed in all multi-start attempts.")
        return None

    sol = best_sol
    print(f"  ✓ [{wing_af_name.upper()}] NLP Converged! Sink = {float(sol(V * Total_Drag / Total_Lift)):.2f} m/s")

    return {
        "wing_af_name": wing_af_name,
        # Geometry
        "span": float(sol(span)), "chord": float(sol(root_chord)),
        "taper": float(sol(taper)), "twist": float(sol(twist)),
        "tail_arm": float(sol(tail_arm)), "tail_chord": float(sol(tail_chord)), "tail_span": float(sol(tail_span)),
        "i_tail": float(sol(i_tail)), "AR": float(sol(AR_w)), "S": float(sol(S)),
        # Mass & Balance
        "mass": float(sol(total_mass)), "structural_mass": float(sol(structural_mass)), "ballast": float(sol(ballast)),
        "batt_x": float(sol(batt_x)), "motor_x": float(sol(motor_x)),
        # Stall estimates passed to the trajectory phase
        "alpha_stall_deg": alpha_stall_deg,
        "alpha_stall_margin_deg": stall_margin_alpha_deg,
        "CL_max_est": float(sol(cl_w_max_3d_est)),
        "CL_min_est": float(sol(cl_w_min_3d_est)),
        # Extracted Trim KPIs
        "V": float(sol(V)), "LD": float(sol(Total_Lift / Total_Drag)),
        "alpha": float(sol(alpha)),
        "sink": float(sol(V * Total_Drag / Total_Lift)),
        "Re": rho * float(sol(V)) * float(sol(mac)) / mu
    }

if __name__ == "__main__":
    optimize_glider_mdo("sd7037")
