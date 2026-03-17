import aerosandbox.numpy as np
import warnings

from config import *
from mdo_solver import optimize_glider_mdo
from visualization import generate_plots

warnings.filterwarnings("ignore")

def print_build_guide(bc, traj):
    CG_X = 0.25 * bc["chord"]
    tail_vol = (bc["tail_chord"] * bc["tail_span"] * 2 * bc["tail_arm"]) / (bc["S"] * bc["chord"])
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          COMPETITION GLIDER — BUILD GUIDE                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  WING AIRFOIL: {bc['wing_af_name'].upper()}                                            ║
    ║    Re(trim):        {bc['Re']:.0f}                                              ║
    ║    Trim AoA:        {bc['alpha']:+.1f}°                                              ║
    ║                                                                  ║
    ║  WING PLANFORM                                                   ║
    ║    Span:            {bc['span']*100:.0f} cm                                             ║
    ║    Chord:           {bc['chord']*100:.0f} cm                                             ║
    ║    Aspect ratio:    {bc['AR']:.1f}                                               ║
    ║    Washout:         -2° at tip                                         ║
    ║    Dihedral:        4%                                                 ║
    ║    Spar:            2 mm carbon rod at 25% chord                       ║
    ║                                                                  ║
    ║  HORIZONTAL TAIL: NACA0009 (symmetric)                           ║
    ║    Arm from wing LE:{bc['tail_arm']*100:.0f} cm                               ║
    ║    Chord:           {bc['tail_chord']*100:.0f} cm                               ║
    ║    Semi-span:       {bc['tail_span']*100:.0f} cm                               ║
    ║    Set incidence:   {bc['i_tail']:+.1f}° relative to wing                       ║
    ║    Tail vol coeff:  {tail_vol:.3f}                                               ║
    ║                                                                  ║
    ║  MASS & BALANCE                                                  ║
    ║    CG Position:     {CG_X*100:.1f} cm from wing leading edge (25% MAC)           ║
    ║    Total Mass:      {bc['mass']*1000:.1f} g                                        ║
    ║    Structure:       {bc['structural_mass']*1000:.1f} g                                        ║
    ║    Payload:         120.0 g (Batt={PAYLOAD_BATTERY*1000:.0f}g, Mot={PAYLOAD_MOTOR*1000:.0f}g, FC={PAYLOAD_FC*1000:.0f}g)  ║
    ║                                                                  ║
    ║  PAYLOAD PLACEMENT (from wing LE)                                ║
    ║    Battery:         {bc['batt_x']*100:+.1f} cm                                     ║
    ║    Motor(s):        {bc['motor_x']*100:+.1f} cm                                     ║
    ║                                                                  ║
    ║  COMPETITION STRATEGY                                            ║
    ║    Release method:  Hold vertical, nose down                           ║
    ║    Dive recovery:   Starts at ~{traj['h_dive']:.1f} m                             ║
    ║    Est. flight time:{traj['T_opt']:.1f} s                                               ║
    ║    Est. range:      {float(traj['x_sol'][-1]):.1f} m                                              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print("\n" + "="*58)
    print("  MDO SOLVER — Monolithic Optimization")
    print("="*58)

    best = None
    results = []
    
    for af in CANDIDATE_AIRFOILS:
        r = optimize_glider_mdo(af)
        if r is None:
            continue
            
        results.append(r)
        
        # Primary objective logic (maximize range)
        range_val = float(r["x_sol"][-1])
        if best is None or range_val > float(best["x_sol"][-1]):
            best = r
            
    if not results:
        print("\n  ✗ No valid MDO configurations found.")
        import sys; sys.exit(1)
        
    print(f"\n  {len(results)} valid configurations completely generated.")
    
    # Generate Visualization
    best_traj = {
        "T_opt" : best["T_opt"],
        "t_sol" : best["t_sol"],
        "x_sol" : best["x_sol"],
        "z_sol" : best["z_sol"],
        "V_sol" : best["V_sol"],
        "g_sol" : best["g_sol"],
        "a_sol" : best["a_sol"],
        "CL_sol": best["CL_sol"],
        "CD_sol": best["CD_sol"],
        "Cm_sol": best["Cm_sol"]
    }
    
    print("\n  Generating Optimal Performance Visuals...")
    generate_plots(best, best_traj)

    # Output Build Guide for the overall winner
    print("\n" + "="*58)
    print(f"  WINNER: {best['wing_af_name'].upper()}")
    print("="*58)
    print_build_guide(best, best_traj)