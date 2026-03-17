import aerosandbox as asb
from config import *
from mass_model import estimate_mass

def trimmed_min_sink(span, chord, wing_af_name):
    """
    Solve for minimum-sink-rate trim with Cm=0.
    Free vars: alpha (wing AoA), tail geometry, payload positions, trim speed.
    Constraints: sum of moments about aerodynamic center = 0  and  L=W.
    Objective: sink rate = V * CD / CL.
    """
    S = span * chord
    cg_x = 0.25 * chord

    opti = asb.Opti()
    
    # Aerodynamic states
    alpha_v = opti.variable(init_guess=5.0,  lower_bound=-5.0,  upper_bound=14.0)
    it_v    = opti.variable(init_guess=-3.0, lower_bound=-12.0, upper_bound=3.0)
    
    # Geometric variables
    tail_arm   = opti.variable(init_guess=0.50, lower_bound=MIN_TAIL_ARM,   upper_bound=MAX_TAIL_ARM)
    tail_chord = opti.variable(init_guess=0.06, lower_bound=MIN_TAIL_CHORD, upper_bound=MAX_TAIL_CHORD)
    tail_span  = opti.variable(init_guess=0.08, lower_bound=MIN_TAIL_SPAN,  upper_bound=MAX_TAIL_SPAN)
    
    # Payload Point Masses
    batt_x  = opti.variable(init_guess=-0.10,        lower_bound=MIN_BATTERY_X, upper_bound=MAX_BATTERY_X)
    motor_x = opti.variable(init_guess=0.40,         lower_bound=MIN_MOTOR_X,   upper_bound=MAX_MOTOR_X)
    
    # Ensure motor_x fits on the fuselage boom
    opti.subject_to(motor_x <= tail_arm + cg_x)
    
    fc_x    = 0.0          # FC near CG / wing LE
    other_x = (batt_x + cg_x + tail_arm) / 2 # center of fuselage

    # Ballast to satisfy CG if moving payload isn't enough
    ballast = opti.variable(init_guess=0.01, lower_bound=0.0, upper_bound=0.300)

    # Dynamic Mass Calculation
    structural_mass, structural_moment = estimate_mass(span, chord, tail_span, tail_chord, tail_arm)
    total_mass = structural_mass + PAYLOAD_TOTAL + ballast
    W = total_mass * g
    
    # Balance Constraint: Center of Mass = 25% MAC (cg_x)
    total_moment = structural_moment + ((PAYLOAD_BATTERY + ballast) * batt_x) + (PAYLOAD_MOTOR * motor_x) + (PAYLOAD_FC * fc_x) + (PAYLOAD_OTHER * other_x)
    opti.subject_to(total_moment / total_mass == cg_x)

    # Initial guess for velocity based on an approximate 160g total mass
    approx_W = 0.160 * g
    V0_guess = (2 * approx_W / (rho * S * 0.85)) ** 0.5
    V_v = opti.variable(init_guess=V0_guess, lower_bound=1.5, upper_bound=20.0)

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
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=tail_chord, twist=it_v, airfoil=TAIL_AF),
                    asb.WingXSec(xyz_le=[0, tail_span, tail_span * 0.02], chord=tail_chord, twist=it_v, airfoil=TAIL_AF),
                ]
            ).translate([tail_arm, 0, 0]),
        ]
    )

    op   = asb.OperatingPoint(velocity=V_v, alpha=alpha_v, atmosphere=asb.Atmosphere(altitude=0))
    aero = asb.AeroBuildup(airplane=airplane, op_point=op).run()

    opti.subject_to([
        aero["Cm"] == 0,                              # pitch trim
        aero["CL"] * 0.5 * rho * V_v**2 * S == W,     # level flight
    ])
    opti.minimize(V_v * aero["CD"] / aero["CL"])      # minimise sink rate

    try:
        sol = opti.solve(verbose=False, max_iter=500)
        CL  = float(sol(aero["CL"]))
        CD  = float(sol(aero["CD"]))
        V   = float(sol(V_v))
        if V > 19.5:          # trim hit upper bound → unphysical
            return None
        return dict(
            span=span, chord=chord, wing_af_name=wing_af_name, mass=float(sol(total_mass)), 
            structural_mass=float(sol(structural_mass)), ballast=float(sol(ballast)),
            AR=span/chord, S=S, WL=float(sol(W))/S,
            Re=rho * V * chord / mu,
            alpha=float(sol(alpha_v)),
            i_tail=float(sol(it_v)),
            V=V, CL=CL, CD=CD, LD=CL/CD,
            sink=V * CD / CL,
            tail_arm=float(sol(tail_arm)), tail_chord=float(sol(tail_chord)), tail_span=float(sol(tail_span)),
            batt_x=float(sol(batt_x)), motor_x=float(sol(motor_x))
        )
    except Exception as e:
        # print(f"Solver failed for Span={span*100:.0f}cm Chord={chord*100:.0f}cm: {e}")
        return None