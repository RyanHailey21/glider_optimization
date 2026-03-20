import csv
import os

import aerosandbox.numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from config import DROP_HEIGHT


PALETTE = {
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "surface": "#f8fafc",
    "primary": "#0f766e",
    "accent": "#ea580c",
    "accent2": "#0284c7",
    "good": "#16a34a",
}


def _style_ax(ax, title):
    ax.set_facecolor(PALETTE["surface"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.set_title(title, color=PALETTE["ink"], fontsize=11, pad=8, fontweight="semibold")
    ax.grid(color=PALETTE["grid"], linestyle="-", linewidth=0.8, alpha=0.9)


def _write_trajectory_csv(file_path, traj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "x_m",
            "altitude_m",
            "speed_mps",
            "gamma_deg",
            "alpha_deg",
            "CL",
            "CD",
            "Cm",
        ])
        for row in zip(
            traj["t_sol"],
            traj["x_sol"],
            -traj["z_sol"],
            traj["V_sol"],
            traj["g_sol"],
            traj["a_sol"],
            traj["CL_sol"],
            traj["CD_sol"],
            traj["Cm_sol"],
        ):
            writer.writerow([float(v) for v in row])


def _write_summary_csv(file_path, bc, traj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tail_vol = (bc["tail_chord"] * bc["tail_span"] * 2 * bc["tail_arm"]) / (bc["S"] * bc["chord"])
    rows = [
        ("wing_airfoil", bc["wing_af_name"].upper(), "-"),
        ("trim_Re", float(bc["Re"]), "-"),
        ("trim_alpha_deg", float(bc["alpha"]), "deg"),
        ("wing_span_m", float(bc["span"]), "m"),
        ("wing_chord_m", float(bc["chord"]), "m"),
        ("wing_taper", float(bc["taper"]), "-"),
        ("wing_twist_deg", float(bc["twist"]), "deg"),
        ("wing_AR", float(bc["AR"]), "-"),
        ("tail_arm_m", float(bc["tail_arm"]), "m"),
        ("tail_chord_m", float(bc["tail_chord"]), "m"),
        ("tail_semispan_m", float(bc["tail_span"]), "m"),
        ("tail_incidence_deg", float(bc["i_tail"]), "deg"),
        ("tail_volume_coefficient", float(tail_vol), "-"),
        ("total_mass_kg", float(bc["mass"]), "kg"),
        ("structural_mass_kg", float(bc["structural_mass"]), "kg"),
        ("ballast_mass_kg", float(bc["ballast"]), "kg"),
        ("battery_x_m", float(bc["batt_x"]), "m"),
        ("motor_x_m", float(bc["motor_x"]), "m"),
        ("trim_speed_mps", float(bc["V"]), "m/s"),
        ("trim_L_over_D", float(bc["LD"]), "-"),
        ("sink_rate_mps", float(bc["sink"]), "m/s"),
        ("optimized_flight_time_s", float(traj["T_opt"]), "s"),
        ("optimized_range_m", float(traj["x_sol"][-1]), "m"),
    ]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "units"])
        for row in rows:
            writer.writerow(row)


def generate_plots(bc, traj, out_fig="glider_optimization_v2.png"):
    print("\n  Building performance plots...")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["muted"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "font.size": 10,
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.22)

    # A) Trajectory with speed colormap
    ax0 = fig.add_subplot(gs[0, :])
    _style_ax(
        ax0,
        f"Optimized Drop-Glide Trajectory | T = {traj['T_opt']:.2f} s, Range = {float(traj['x_sol'][-1]):.1f} m",
    )
    sc = ax0.scatter(
        traj["x_sol"],
        -traj["z_sol"],
        c=traj["V_sol"],
        cmap="viridis",
        s=24,
        edgecolors="none",
        zorder=3,
    )
    ax0.plot(traj["x_sol"], -traj["z_sol"], color=PALETTE["accent2"], lw=1.2, alpha=0.75)
    ax0.axhline(DROP_HEIGHT, color=PALETTE["muted"], ls="--", lw=1.0, label=f"Release height ({DROP_HEIGHT:.1f} m)")
    ax0.axhline(0, color=PALETTE["good"], lw=1.8, label="Ground")
    cbar = fig.colorbar(sc, ax=ax0, pad=0.015)
    cbar.set_label("Speed [m/s]", color=PALETTE["muted"], fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["muted"])
    ax0.set_xlabel("Horizontal Distance [m]")
    ax0.set_ylabel("Altitude [m]")
    ax0.set_ylim(-0.5, DROP_HEIGHT + 2)
    ax0.legend(loc="upper right", fontsize=9, frameon=False)

    # B) Angles and speed
    ax1 = fig.add_subplot(gs[1, 0])
    _style_ax(ax1, "Flight States vs Time")
    ax1b = ax1.twinx()
    ax1.plot(traj["t_sol"], traj["g_sol"], color=PALETTE["accent2"], lw=2.0, label="Flight path angle gamma [deg]")
    ax1.plot(traj["t_sol"], traj["a_sol"], color=PALETTE["accent"], lw=2.0, label="Angle of attack alpha [deg]")
    ax1b.plot(traj["t_sol"], traj["V_sol"], color=PALETTE["primary"], lw=2.0, ls="--", label="Speed [m/s]")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angle [deg]")
    ax1b.set_ylabel("Speed [m/s]", color=PALETTE["muted"])
    ax1b.tick_params(colors=PALETTE["muted"], labelsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8.5, frameon=False)

    # C) Aerodynamic coefficients
    ax2 = fig.add_subplot(gs[1, 1])
    _style_ax(ax2, "Aerodynamic Coefficients vs Time")
    ax2.plot(traj["t_sol"], traj["CL_sol"], color=PALETTE["good"], lw=2.0, label="CL")
    ax2.plot(traj["t_sol"], traj["CD_sol"], color=PALETTE["accent"], lw=2.0, label="CD")
    ax2.plot(traj["t_sol"], traj["Cm_sol"], color=PALETTE["primary"], lw=1.8, ls="--", label="Cm")
    ax2.axhline(0, color=PALETTE["muted"], lw=1.0, ls=":")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Coefficient [-]")
    ax2.legend(loc="best", fontsize=8.5, frameon=False)

    fig.suptitle(
        f"Passive Glider Optimization Results | Winner: {bc['wing_af_name'].upper()} | 60 ft Nose-Down Release",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )

    # Save figure products
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    pdf_path = os.path.splitext(out_fig)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save source data behind all plotted series
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    trajectory_csv = os.path.join(results_dir, "trajectory_timeseries.csv")
    summary_csv = os.path.join(results_dir, "design_summary.csv")
    _write_trajectory_csv(trajectory_csv, traj)
    _write_summary_csv(summary_csv, bc, traj)

    print(f"\n  Figure -> {out_fig}")
    print(f"  Figure -> {pdf_path}")
    print(f"  Data   -> {trajectory_csv}")
    print(f"  Data   -> {summary_csv}")
