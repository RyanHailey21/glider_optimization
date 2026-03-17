import aerosandbox.numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import DROP_HEIGHT

DARK  = "#0d1117";  PANEL = "#161b22"
GOLD  = "#f0c040";  CYAN  = "#58d8f0"
GREEN = "#3fb950";  RED   = "#f85149"
GREY  = "#8b949e";  PURP  = "#d2a8ff"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.set_title(title, color=GOLD, fontsize=9, pad=6, fontweight="bold")
    ax.grid(color="#21262d", linestyle="--", linewidth=0.5, alpha=0.7)

def generate_plots(bc, traj, out_fig="glider_optimization_v2.png"):
    print("\n  Building performance plots...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ── A: Trajectory ─────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    style_ax(ax0, f"Optimal Trajectory  (T = {traj['T_opt']:.2f} s,  Range = {float(traj['x_sol'][-1]):.1f} m)")
    sc = ax0.scatter(traj["x_sol"], -traj["z_sol"], c=traj["V_sol"], cmap="plasma", s=14, zorder=3)
    ax0.plot(traj["x_sol"], -traj["z_sol"], color=CYAN, lw=1.5, alpha=0.5)
    ax0.axhline(DROP_HEIGHT, color=GREY, ls=":", lw=1, label=f"Release height {DROP_HEIGHT:.1f} m")
    ax0.axhline(0, color=GREEN, lw=1.5, label="Ground")
    cb = fig.colorbar(sc, ax=ax0, pad=0.01)
    cb.set_label("Speed (m/s)", color=GREY, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=GREY, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=GREY)
    ax0.set_xlabel("Horizontal distance (m)");  ax0.set_ylabel("Altitude (m)")
    ax0.legend(fontsize=8, facecolor=PANEL, edgecolor="#30363d", labelcolor=GREY)
    ax0.set_ylim(-1, DROP_HEIGHT + 2)

    # ── B: Angles + Cm vs time ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    style_ax(ax1, "Flight Angles & Pitch Moment vs Time")
    ax1b = ax1.twinx()
    ax1.plot(traj["t_sol"], traj["g_sol"],  color=CYAN,  lw=1.8, label="γ (flight path)")
    ax1.plot(traj["t_sol"], traj["a_sol"],  color=GOLD,  lw=1.8, label="α (wing AoA)")
    ax1b.plot(traj["t_sol"], traj["CL_sol"], color=GREEN, lw=1.4, ls="--", label="CL")
    ax1b.plot(traj["t_sol"], traj["Cm_sol"], color=PURP,  lw=1.0, ls=":",  label="Cm (pitch)")
    ax1.axhline(0, color=GREY, lw=0.5, ls="--")
    ax1b.axhline(0, color=GREY, lw=0.4, ls=":")
    ax1.set_xlabel("Time (s)");  ax1.set_ylabel("Angle (deg)")
    ax1b.set_ylabel("CL / Cm");  ax1b.tick_params(colors=GREY, labelsize=7)
    ax1b.yaxis.label.set_color(GREY)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=7.5, facecolor=PANEL, edgecolor="#30363d", labelcolor=GREY)

    # ── C: Summary card ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL);  ax3.set_xticks([]);  ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_edgecolor("#30363d")
    ax3.set_title("Optimal Design", color=GOLD, fontsize=10, pad=6, fontweight="bold")

    tail_vol = (bc["tail_chord"] * bc["tail_span"] * 2 * bc["tail_arm"]) / (bc["S"] * bc["chord"])
    card = [
        (f"─── WING  ({bc['wing_af_name'].upper()}) ───────────", GOLD),
        (f"  Span         {bc['span']*100:5.1f} cm",  CYAN),
        (f"  Chord        {bc['chord']*100:5.1f} cm",  CYAN),
        (f"  Aspect ratio {bc['AR']:5.1f}",            CYAN),
        (f"  Re (trim)    {bc['Re']:7.0f}",      CYAN),
        ("", GREY),
        ("─── TAIL  (NACA0009) ─────────", GOLD),
        (f"  Arm from LE  {bc['tail_arm']*100:5.1f} cm",   PURP),
        (f"  Chord        {bc['tail_chord']*100:5.1f} cm",  PURP),
        (f"  Semi-span    {bc['tail_span']*100:5.1f} cm",   PURP),
        (f"  Incidence   {bc['i_tail']:+5.1f}°",          PURP),
        (f"  Vol coeff    {tail_vol:5.3f}",            PURP),
        ("", GREY),
        ("─── MASS & BALANCE ───────────", GOLD),
        (f"  Total Mass   {bc['mass']*1000:5.1f} g",   GREEN),
        (f"  Structural   {bc['structural_mass']*1000:5.1f} g", GREEN),
        (f"  Ballast      {bc['ballast']*1000:5.1f} g", GREEN),
        (f"  Batt Pos    {bc['batt_x']*100:+5.1f} cm", GREEN),
        (f"  Motor Pos   {bc['motor_x']*100:+5.1f} cm", GREEN),
        (f"  Trim alpha  {bc['alpha']:+5.1f}°",  GREEN),
        (f"  L/D          {bc['LD']:5.2f}",      GREEN),
        (f"  Trim speed   {bc['V']:5.2f} m/s",   GREEN),
        (f"  Sink rate    {bc['sink']:5.3f} m/s", GREEN),
        ("", GREY),
        (f"  FLIGHT TIME  {traj['T_opt']:5.2f} s",       RED),
        (f"  Range        {float(traj['x_sol'][-1]):5.1f} m", GREEN),
    ]
    y = 0.97
    for txt, col in card:
        ax3.text(0.04, y, txt, transform=ax3.transAxes, color=col,
                 fontsize=8.2, fontfamily="monospace", verticalalignment="top")
        y -= 0.046

    fig.suptitle(
        f"Glider Optimizer v2  ·  {bc['wing_af_name'].upper()} wing + NACA0009 tail  ·  "
        "AeroBuildup  ·  Cm-trimmed  ·  60 ft nose-down drop",
        color="white", fontsize=11, fontweight="bold", y=0.99
    )

    plt.savefig(out_fig, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"\n  Figure → {out_fig}")