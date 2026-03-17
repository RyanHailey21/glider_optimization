from config import PRINT_AREA_DENSITY, BOOM_LINEAR_DENSITY, PAYLOAD_TOTAL

def estimate_mass(span, chord, tail_span, tail_chord, tail_arm):
    """
    Estimates the total mass of the glider and the required nose ballast to achieve a 25% MAC CG.
    
    Returns:
        total_mass: Total flying weight of the glider (kg)
        ballast: Additional weight required in the nose (kg)
    """
    # Structural geometry
    wing_area = span * chord
    tail_area = (tail_span * 2) * tail_chord  # tail_span is semi-span
    
    # Target CG
    target_cg = 0.25 * chord
    
    # 1. Component Masses
    # We assume 3D printed wing and tail mass scale with surface area.
    wing_mass = wing_area * PRINT_AREA_DENSITY
    
    # Tail mass scales with surface area.
    tail_mass = tail_area * PRINT_AREA_DENSITY
    
    # Boom mass scales with tail arm length.
    boom_mass = tail_arm * BOOM_LINEAR_DENSITY
    
    # 2. Moment Arms (relative to wing leading edge (LE), where x=0)
    # Assume wing mass is centered near its own quarter-chord.
    wing_arm_x = target_cg
    
    # Tail is located at tail_arm behind the wing quarter-chord.
    tail_arm_x = target_cg + tail_arm
    
    # Boom connects from some point (say target_cg) back to the tail_arm_x.
    # We will approximate its center of gravity at the midpoint between wing CG and tail.
    boom_arm_x = target_cg + (tail_arm / 2)
    
    # Payload is placed in the nose.
    # We constrain the payload to be at most 0.10 m (10 cm) ahead of the wing LE.
    nose_arm_x = -0.10
    
    # 3. Calculate Moments
    # Moment = mass * distance_from_LE
    structural_moment = (wing_mass * wing_arm_x) + (tail_mass * tail_arm_x) + (boom_mass * boom_arm_x)
    structural_mass = wing_mass + tail_mass + boom_mass
    
    # 4. Ballast Calculation
    # We need: (structural_moment + payload_moment + ballast_moment) / total_mass = target_cg
    # We assume the entire payload can be shifted to `nose_arm_x` to balance.
    # First, calculate what moment the payload provides if placed entirely in the nose.
    
    # We want to solve for `m_nose` where `m_nose` includes both the Payload + Ballast
    # m_nose * nose_arm_x + structural_moment = target_cg * (structural_mass + m_nose)
    # m_nose * nose_arm_x - target_cg * m_nose = target_cg * structural_mass - structural_moment
    # m_nose * (nose_arm_x - target_cg) = target_cg * structural_mass - structural_moment
    
    required_nose_mass = (target_cg * structural_mass - structural_moment) / (nose_arm_x - target_cg)
    ballast = max(0, required_nose_mass - PAYLOAD_TOTAL)
    
    total_mass = structural_mass + PAYLOAD_TOTAL + ballast
    
    return total_mass, ballast, structural_mass
