import aerosandbox as asb
from config import g, rho, mu, TAIL_AF
from mass_model import estimate_mass

def trimmed_min_sink(span, chord, wing_af_name,
                     tail_arm_frac   = 8.0,   # tail_arm   = frac * chord
                     tail_chord_frac = 0.55,  # tail_chord = frac * wing_chord
                     tail_span_frac  = 0.40): # tail_sspan = frac * (span/2)
    """
    Solve for minimum-sink-rate trim with Cm=0.
    Free vars: alpha (wing AoA), tail_incidence, trim_speed  — all scalars.
    Constraints: Cm=0  and  L=W  (2 equalities, 3 unknowns → 1 DOF minimised).
    Objective: sink rate = V * CD / CL.
    """
    S          = span * chord
    tail_arm   = tail_arm_frac   * chord
    tail_chord = tail_chord_frac * chord
    tail_span  = tail_span_frac  * (span / 2)
    
    mass, ballast, structural_mass = estimate_mass(span, chord, tail_span, tail_chord, tail_arm)
    W          = mass * g
    
    cg_x       = 0.25 * chord
    V0         = (2 * W / (rho * S * 0.85)) ** 0.5   # level-flight speed guess

    opti    = asb.Opti()
    alpha_v = opti.variable(init_guess=5.0,  lower_bound=-5.0,  upper_bound=14.0)
    it_v    = opti.variable(init_guess=-3.0, lower_bound=-12.0, upper_bound=3.0)
    V_v     = opti.variable(init_guess=V0,   lower_bound=1.5,   upper_bound=20.0)

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
            span=span, chord=chord, wing_af_name=wing_af_name, mass=mass, 
            ballast=ballast, structural_mass=structural_mass,
            AR=span/chord, S=S, WL=W/S,
            Re=rho * V * chord / mu,
            alpha=float(sol(alpha_v)),
            i_tail=float(sol(it_v)),
            V=V, CL=CL, CD=CD, LD=CL/CD,
            sink=V * CD / CL,
            tail_arm=tail_arm, tail_chord=tail_chord, tail_span=tail_span,
        )
    except Exception:
        return None