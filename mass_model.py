from config import PRINT_AREA_DENSITY, BOOM_LINEAR_DENSITY

def estimate_mass(span, chord, tail_span, tail_chord, tail_arm):
    """
    Estimates the mass and moment of the glider's physical printed parts and carbon boom.
    Returns:
        structural_mass: Mass of the wing, tail, and boom (kg)
        structural_moment: Pitching moment (mass * arm) of the structure about LE (kg*m)
    """
    wing_area = span * chord
    tail_area = (tail_span * 2) * tail_chord
    
    target_cg = 0.25 * chord
    
    wing_mass = wing_area * PRINT_AREA_DENSITY
    tail_mass = tail_area * PRINT_AREA_DENSITY
    boom_mass = tail_arm * BOOM_LINEAR_DENSITY
    
    wing_arm_x = target_cg
    tail_arm_x = target_cg + tail_arm
    boom_arm_x = target_cg + (tail_arm / 2)
    
    structural_moment = (wing_mass * wing_arm_x) + (tail_mass * tail_arm_x) + (boom_mass * boom_arm_x)
    structural_mass = wing_mass + tail_mass + boom_mass
    
    return structural_mass, structural_moment
