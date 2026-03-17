import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca
from config import *
from mass_model import estimate_mass

def optimize_glider_mdo(wing_af_name):
    """
    Multi-Disciplinary Optimization (MDO) formulation.
    Finds the exact combination of Span, Chord, Tail dimensions, 
    Payload placement, and Flight trajectory to maximize Range from a fixed drop height.
    """
    print(f"\n  [MDO] Solving for {wing_af_name.upper()}...")
    
    # NLP Setup & Guesses
    opti = asb.Opti()
    
    # === 1. Geometric & Payload Variables ===
    span       = opti.variable(init_guess=0.40, lower_bound=MIN_SPAN,  upper_bound=MAX_SPAN)
    chord      = opti.variable(init_guess=0.08, lower_bound=MIN_CHORD, upper_bound=MAX_CHORD)
    tail_arm   = opti.variable(init_guess=0.40, lower_bound=MIN_TAIL_ARM,   upper_bound=MAX_TAIL_ARM)
    tail_chord = opti.variable(init_guess=0.04, lower_bound=MIN_TAIL_CHORD, upper_bound=MAX_TAIL_CHORD)
    tail_span  = opti.variable(init_guess=0.08, lower_bound=MIN_TAIL_SPAN,  upper_bound=MAX_TAIL_SPAN)
    i_tail     = opti.variable(init_guess=-5.0, lower_bound=-15.0, upper_bound=5.0)
    
    batt_x     = opti.variable(init_guess=-0.10,          lower_bound=MIN_BATTERY_X, upper_bound=MAX_BATTERY_X)
    motor_x    = opti.variable(init_guess=0.20,           lower_bound=MIN_MOTOR_X,   upper_bound=MAX_MOTOR_X)
    
    S    = span * chord
    cg_x = 0.25 * chord
    
    # Internal constraints
    opti.subject_to([
        motor_x <= tail_arm + cg_x,          # motor must fit on boom
        (span / chord) >= 4.0,               # AR bounds
        (span / chord) <= 15.0
    ])
    
    # === 2. Mass & Balance Physics ===
    fc_x    = 0.0
    other_x = (batt_x + cg_x + tail_arm) / 2
    ballast = opti.variable(init_guess=0.01, lower_bound=0.0, upper_bound=0.300)

    structural_mass, structural_moment = estimate_mass(span, chord, tail_span, tail_chord, tail_arm)
    total_mass = structural_mass + PAYLOAD_TOTAL + ballast
    W = total_mass * g
    
    total_moment = structural_moment + ((PAYLOAD_BATTERY + ballast) * batt_x) + (PAYLOAD_MOTOR * motor_x) + (PAYLOAD_FC * fc_x) + (PAYLOAD_OTHER * other_x)
    opti.subject_to(total_moment / total_mass == cg_x)

    # === 3. Trajectory Variables ===
    N_dive  = 20
    N_glide = 60
    N       = N_dive + N_glide
    T_guess = 12.0

    T_final = opti.variable(init_guess=T_guess, lower_bound=1.0, upper_bound=120.0)
    time    = T_final * ca.linspace(0, 1, N)

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
        np.linspace(1.0, 15.0, N_dive),
        np.full(N_glide, 15.0),
    ])
    gam_g   = np.concatenate([
        np.linspace(np.radians(-80), np.radians(-12), N_dive),
        np.linspace(np.radians(-12), np.radians(-3),  N_glide),
    ])
    alpha_g = np.full(N, 5.0)

    dyn = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=total_mass),
        x_e   = opti.variable(init_guess=x_g),
        z_e   = opti.variable(init_guess=-alt_g),
        speed = opti.variable(init_guess=V_g, lower_bound=0.5, upper_bound=35.0),
        gamma = opti.variable(init_guess=gam_g, lower_bound=np.radians(-89), upper_bound=np.radians(45)),
        alpha = opti.variable(init_guess=alpha_g, lower_bound=-14.0, upper_bound=14.0),
    )

    # === 4. Airplane Geometry & Aerodynamics ===
    wing_af_obj = asb.Airfoil(wing_af_name)
    
    airplane = asb.Airplane(
        name="Glider", xyz_ref=[cg_x, 0, 0],
        wings=[
            asb.Wing(
                name="Main Wing", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=chord, twist=0.0, airfoil=wing_af_obj),
                    asb.WingXSec(xyz_le=[0, span/2, span/2 * 0.04], chord=chord, twist=-2.0, airfoil=wing_af_obj),
                ]
            ),
            asb.Wing(
                name="Horizontal Stabilizer", symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=tail_chord, twist=i_tail, airfoil=TAIL_AF),
                    asb.WingXSec(xyz_le=[0, tail_span, tail_span * 0.02], chord=tail_chord, twist=i_tail, airfoil=TAIL_AF),
                ]
            ).translate([tail_arm, 0, 0]),
        ]
    )

    aero = asb.AeroBuildup(airplane=airplane, op_point=dyn.op_point).run()

    # Apply forces and link dynamics
    dyn.add_gravity_force(g=g)
    dyn.add_force(*aero["F_w"], axes="wind")
    dyn.constrain_derivatives(opti, time)

    # === 5. Trajectory Boundary Constraints ===
    opti.subject_to([
        aero["Cm"] == 0,                          # Glider must be pitch-trimmed at all points in the trajectory automatically!
        dyn.x_e[0]   == 0.0,                      # Start at origin
        dyn.z_e[0]   == -DROP_HEIGHT,             # Start at drop height
        dyn.speed[0] <= 1.5,                      # Drop speed near 0
        dyn.gamma[0] <= np.radians(-80),          # Nose-down drop
        dyn.z_e[-1]  == 0.0,                      # Reaches ground
        dyn.gamma[-1] >= np.radians(-30),         # Don't lawn-dart
        dyn.gamma[-1] <= np.radians(0),
        dyn.z_e      <= 0.0,                      # Don't go above ceiling
        dyn.gamma    <= np.radians(30),
    ])

    # === 6. Objective ===
    alpha_smoothness = np.sum(np.diff(dyn.alpha) ** 2)
    # Give primary weight to Range (x_e[-1]) and a secondary weight to time to prevent stalling behaviors
    opti.minimize(-dyn.x_e[-1] - 0.5 * T_final + 1e-3 * alpha_smoothness)

    # === 7. Solve ===
    try:
        sol = opti.solve(verbose=False, max_iter=3000)
        print(f"  ✓ [{wing_af_name.upper()}] NLP Converged! Range = {float(sol(dyn.x_e[-1])):.1f} m")
        
        # Pull level flight trim states from the middle of the glide (index N_dive + 20)
        idx_trim = N_dive + 20
        V_trim  = float(sol(dyn.speed[idx_trim]))
        CL_trim = float(sol(aero["CL"][idx_trim]))
        CD_trim = float(sol(aero["CD"][idx_trim]))
        
        return {
            "wing_af_name": wing_af_name,
            # Geometry
            "span": float(sol(span)), "chord": float(sol(chord)), 
            "tail_arm": float(sol(tail_arm)), "tail_chord": float(sol(tail_chord)), "tail_span": float(sol(tail_span)),
            "i_tail": float(sol(i_tail)), "AR": float(sol(span/chord)), "S": float(sol(S)),
            # Mass & Balance
            "mass": float(sol(total_mass)), "structural_mass": float(sol(structural_mass)), "ballast": float(sol(ballast)),
            "batt_x": float(sol(batt_x)), "motor_x": float(sol(motor_x)),
            # Trajectory Series
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
            # Extracted Trim KPIs
            "V": V_trim, "LD": CL_trim/CD_trim, 
            "alpha": float(sol(dyn.alpha[idx_trim])),
            "sink": V_trim * CD_trim / CL_trim,
            "Re": rho * V_trim * float(sol(chord)) / mu
        }
    except Exception as exc:
        print(f"  ✗ [{wing_af_name.upper()}] NLP Failed: {exc}")
        return None
