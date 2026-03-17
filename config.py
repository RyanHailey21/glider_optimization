import aerosandbox as asb

# ── constants & constraints ───────────────────────────────────────────────────
g   = 9.81
rho = 1.225
mu  = 1.789e-5
DROP_HEIGHT = 60 * 0.3048        # 18.288 m

MAX_SPAN  = 0.40
MIN_SPAN  = 0.15   # printer bed = 400 mm
MAX_CHORD = 0.12
MIN_CHORD = 0.04   # <4 cm → Re < 20k → drag-dominated

# Mass estimation constants
PRINT_AREA_DENSITY  = 1.2  # kg/m^2 (approximate weight of 3D printed wing/tail shells per area)
BOOM_LINEAR_DENSITY = 0.015 # kg/m (e.g., 4mm carbon tube)

PAYLOAD_BATTERY = 0.027  # 27g
PAYLOAD_FC      = 0.018  # 18g
PAYLOAD_MOTOR   = 0.020  # 20g
PAYLOAD_OTHER   = 0.055  # 55g (wires, props, etc.)
PAYLOAD_TOTAL   = PAYLOAD_BATTERY + PAYLOAD_FC + PAYLOAD_MOTOR + PAYLOAD_OTHER # 120g


WING_AF = asb.Airfoil("sd7037")    # NeuralFoil: Re-accurate, undercambered
TAIL_AF = asb.Airfoil("naca0009")  # Symmetric, thin stabiliser