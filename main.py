import aerosandbox.numpy as np
from itertools import product as iterproduct
import warnings

from config import *
from trim_solver import trimmed_min_sink
from trajectory import optimize_trajectory
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
    ║    Re(trim) = {bc['Re']:.0f}                                              ║
    ║    Undercambered: generates lift at 0° AoA                       ║
    ║    Trim AoA = {bc['alpha']:+.1f}°  (peak L/D region, not near stall)       ║
    ║                                                                  ║
    ║  WING PLANFORM                                                   ║
    ║    Span  {bc['span']*100:.0f} cm  ×  Chord  {bc['chord']*100:.0f} cm   (AR = {bc['AR']:.1f})              ║
    ║    -2° washout at tip  (prevents tip stall during pull-out)      ║
    ║    4% dihedral  (V-shape for roll stability)                     ║
    ║    2 mm carbon rod spar at 25% chord                             ║
    ║                                                                  ║
    ║  HORIZONTAL TAIL: NACA0009 (symmetric)                           ║
    ║    Arm from wing LE: {bc['tail_arm']*100:.0f} cm                               ║
    ║    Chord:            {bc['tail_chord']*100:.0f} cm                               ║
    ║    Semi-span:        {bc['tail_span']*100:.0f} cm                               ║
    ║    Set incidence:   {bc['i_tail']:+.1f}° relative to wing (trim value)     ║
    ║    Tail vol coeff:   {tail_vol:.3f}  (>0.3 = adequate stability)      ║
    ║                                                                  ║
    ║  MASS & BALANCE                                                  ║
    ║    CG Position: {CG_X*100:.1f} cm from wing leading edge  (25% MAC)        ║
    ║    Total Mass:  {bc['mass']*1000:.1f} g                                        ║
    ║    Structure:   {bc['structural_mass']*1000:.1f} g                                        ║
    ║    Payload:     120.0 g                                          ║
    ║    Added Nose Ballast: {bc['ballast']*1000:.1f} g                                 ║
    ║                                                                  ║
    ║  COMPETITION STRATEGY                                            ║
    ║    Hold vertical, nose down. DO NOT throw — just release.        ║
    ║    Glider dives ~{traj['h_dive']:.1f} m, accelerates, then auto-trims level.    ║
    ║    Expected: {traj['T_opt']:.0f} s flight,  {float(traj['x_sol'][-1]):.0f} m range                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print("\n" + "="*58)
    print("  PHASE 1 — Trimmed parametric sweep  (SD7037 / NACA0009)")
    print("="*58)

    spans  = np.linspace(MIN_SPAN,  MAX_SPAN,  8)
    chords = np.linspace(MIN_CHORD, MAX_CHORD, 7)

    best    = None
    results = []
    count   = 0
    total   = len(spans) * len(chords) * len(CANDIDATE_AIRFOILS)

    for b, c, af in iterproduct(spans, chords, CANDIDATE_AIRFOILS):
        AR = b / c
        if AR < 4 or AR > 15:
            continue
        count += 1
        r = trimmed_min_sink(b, c, af)
        if r is None:
            continue
        results.append(r)
        if best is None or r["sink"] < best["sink"]:
            best = r
        if count % 10 == 0:
            print(f"  [{count:3d}/{total}]  best sink={best['sink']:.3f} m/s  "
                  f"(span={best['span']*100:.0f}cm  chord={best['chord']*100:.0f}cm  "
                  f"mass={best['mass']*1000:.0f}g)")

    print(f"\n  {len(results)} valid configs evaluated\n")
    print(f"  ╔══ BEST TRIMMED DESIGN ══════════════════════════╗")
    print(f"  ║  WING  — {best['wing_af_name'].upper()}                                  ║")
    print(f"  ║    Span:        {best['span']*100:5.1f} cm                       ║")
    print(f"  ║    Chord:       {best['chord']*100:5.1f} cm                       ║")
    print(f"  ║    Aspect ratio:{best['AR']:5.1f}                           ║")
    print(f"  ║    Re (trim):   {best['Re']:7.0f}                        ║")
    print(f"  ╠══ TAIL — NACA0009 ═══════════════════════════════╣")
    print(f"  ║    Arm:         {best['tail_arm']*100:5.1f} cm                       ║")
    print(f"  ║    Chord:       {best['tail_chord']*100:5.1f} cm                       ║")
    print(f"  ║    Semi-span:   {best['tail_span']*100:5.1f} cm                       ║")
    print(f"  ║    Incidence:   {best['i_tail']:+5.1f}°                        ║")
    print(f"  ╠══ PERFORMANCE ═══════════════════════════════════╣")
    print(f"  ║    Total Mass:  {best['mass']*1000:5.1f} g                        ║")
    print(f"  ║    Structure:   {best['structural_mass']*1000:5.1f} g                        ║")
    print(f"  ║    Ballast:     {best['ballast']*1000:5.1f} g                        ║")
    print(f"  ║    Trim alpha:  {best['alpha']:+5.1f}°                        ║")
    print(f"  ║    L/D:         {best['LD']:5.2f}                           ║")
    print(f"  ║    Trim speed:  {best['V']:5.2f} m/s                     ║")
    print(f"  ║    Sink rate:   {best['sink']:5.3f} m/s                     ║")
    print(f"  ╚═════════════════════════════════════════════════╝")

    print("\n" + "="*58)
    print("  PHASE 2 — Trajectory NLP  (AeroBuildup, Cm-trim tail)")
    print("="*58)

    traj_data = optimize_trajectory(best)

    print(f"\n  Flight time  = {traj_data['T_opt']:.2f} s")
    print(f"  Range        = {float(traj_data['x_sol'][-1]):.1f} m")
    print(f"  Peak speed   = {float(np.max(traj_data['V_sol'])):.2f} m/s")
    print(f"  Alpha range  = {float(np.min(traj_data['a_sol'])):.1f}° → {float(np.max(traj_data['a_sol'])):.1f}°")
    print(f"  Cm range     = {float(np.min(traj_data['Cm_sol'])):.3f} → {float(np.max(traj_data['Cm_sol'])):.3f}")

    # Generate Visualization
    generate_plots(best, traj_data)

    # Print Build Guide
    print_build_guide(best, traj_data)