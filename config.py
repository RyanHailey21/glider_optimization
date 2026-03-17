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

MIN_TAIL_ARM = 0.30
MAX_TAIL_ARM = 1.00
MIN_TAIL_CHORD = 0.03
MAX_TAIL_CHORD = 0.08
MIN_TAIL_SPAN = 0.04
MAX_TAIL_SPAN = 0.15

MIN_BATTERY_X = -0.20
MAX_BATTERY_X = 0.00
MIN_MOTOR_X   = 0.10
MAX_MOTOR_X   = 1.00  # Will be bounded dynamically by tail_arm

# Mass estimation constants
PRINT_AREA_DENSITY  = 1.2  # kg/m^2 (approximate weight of 3D printed wing/tail shells per area)
BOOM_LINEAR_DENSITY = 0.015 # kg/m (e.g., 4mm carbon tube)

PAYLOAD_BATTERY = 0.027  # 27g
PAYLOAD_FC      = 0.018  # 18g
PAYLOAD_MOTOR   = 0.020  # 20g
PAYLOAD_OTHER   = 0.055  # 55g (wires, props, etc.)
PAYLOAD_TOTAL   = PAYLOAD_BATTERY + PAYLOAD_FC + PAYLOAD_MOTOR + PAYLOAD_OTHER # 120g

CANDIDATE_AIRFOILS = ["sd7037", "ag12", "clarky", "e205", "s1223", "e423", "fx63137"]
TAIL_AF = asb.Airfoil("naca0009")  # Symmetric, thin stabiliser