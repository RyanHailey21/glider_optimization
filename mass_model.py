from config import PRINT_AREA_DENSITY, BOOM_LINEAR_DENSITY

def estimate_mass(span, root_chord, taper, tail_span, tail_chord, tail_arm, target_cg_x=None):
    """
    Estimates the mass and moment of the glider's physical printed parts and carbon boom.
    Returns:
        structural_mass: Mass of the wing, tail, and boom (kg)
        structural_moment: Pitching moment (mass * arm) of the structure about LE (kg*m)
    """
    # Keep geometry consistent with the optimizer's tapered-wing planform model.
    wing_area = span * root_chord * (1 + taper) / 2
    tail_area = (tail_span * 2) * tail_chord

    if target_cg_x is None:
        mac = root_chord * (2 / 3) * (1 + taper + taper ** 2) / (1 + taper)
        target_cg_x = 0.25 * mac
    
    wing_mass = wing_area * PRINT_AREA_DENSITY
    tail_mass = tail_area * PRINT_AREA_DENSITY
    boom_mass = tail_arm * BOOM_LINEAR_DENSITY
    
    wing_arm_x = target_cg_x
    tail_arm_x = target_cg_x + tail_arm
    boom_arm_x = target_cg_x + (tail_arm / 2)
    
    structural_moment = (wing_mass * wing_arm_x) + (tail_mass * tail_arm_x) + (boom_mass * boom_arm_x)
    structural_mass = wing_mass + tail_mass + boom_mass
    
    return structural_mass, structural_moment
