import hashlib
import os

import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca
import numpy as onp

from config import *
from mass_model import estimate_mass


def _cache_path(airfoil_key, alphas, res):
    sig = hashlib.sha1()
    sig.update(onp.asarray(alphas, dtype=float).tobytes())
    sig.update(onp.asarray(res, dtype=float).tobytes())
    grid_key = sig.hexdigest()[:12]
    filename = f"{airfoil_key.lower()}_{grid_key}.npz"
    return os.path.join(SURROGATE_CACHE_DIR, filename)


def _load_or_build_grid(airfoil_key, airfoil_obj, alphas, res, get_aero_from_airfoil):
    alphas = onp.asarray(alphas, dtype=float)
    res = onp.asarray(res, dtype=float)

    cache_file = _cache_path(airfoil_key, alphas, res)
    if SURROGATE_USE_CACHE and os.path.exists(cache_file):
        data = onp.load(cache_file)
        return data["CL"], data["CD"], data["CM"]

    A, _ = onp.meshgrid(alphas, res, indexing="ij")
    CL = onp.zeros_like(A)
    CD = onp.zeros_like(A)
    CM = onp.zeros_like(A)

    for i, alpha in enumerate(alphas):
        for j, re in enumerate(res):
            aero = get_aero_from_airfoil(airfoil_obj, alpha=float(alpha), Re=float(re))
            CL[i, j] = float(onp.atleast_1d(aero["CL"])[0])
            CD[i, j] = float(onp.atleast_1d(aero["CD"])[0])
            CM[i, j] = float(onp.atleast_1d(aero["CM"])[0])

    if SURROGATE_USE_CACHE:
        os.makedirs(SURROGATE_CACHE_DIR, exist_ok=True)
        onp.savez(
            cache_file,
            alphas=alphas,
            res=res,
            CL=CL,
            CD=CD,
            CM=CM,
        )

    return CL, CD, CM


def _build_splines(prefix, alphas, res, CL, CD, CM):
    spline_cl = ca.interpolant(f"{prefix}_cl", "bspline", [alphas, res], CL.ravel(order="F"))
    spline_cd = ca.interpolant(f"{prefix}_cd", "bspline", [alphas, res], CD.ravel(order="F"))
    spline_cm = ca.interpolant(f"{prefix}_cm", "bspline", [alphas, res], CM.ravel(order="F"))
    return spline_cl, spline_cd, spline_cm


def _solve_static_nlp(wing_af_name, splines_w, splines_t, stall_refs):
    spline_cl_w, spline_cd_w, spline_cm_w = splines_w
    spline_cl_t, spline_cd_t, spline_cm_t = splines_t

    cl_w_max_2d = float(stall_refs["cl_w_max_2d"])
    cl_w_min_2d = float(stall_refs["cl_w_min_2d"])
    alpha_stall_deg = float(stall_refs["alpha_stall_deg"])

    print(f"  [MDO] Solving static NLP for minimum sink rate (flight time)...")

    opti = asb.Opti()

    # 1) Geometry and payload placement
    span = opti.variable(init_guess=0.40, lower_bound=MIN_SPAN, upper_bound=MAX_SPAN)
    root_chord = opti.variable(init_guess=0.08, lower_bound=MIN_CHORD, upper_bound=MAX_CHORD)
    taper = opti.variable(init_guess=0.6, lower_bound=0.3, upper_bound=1.0)
    twist = opti.variable(init_guess=-2.0, lower_bound=-6.0, upper_bound=2.0)

    tail_arm = opti.variable(init_guess=0.40, lower_bound=MIN_TAIL_ARM, upper_bound=MAX_TAIL_ARM)
    tail_chord = opti.variable(init_guess=0.04, lower_bound=MIN_TAIL_CHORD, upper_bound=MAX_TAIL_CHORD)
    tail_span = opti.variable(init_guess=0.08, lower_bound=MIN_TAIL_SPAN, upper_bound=MAX_TAIL_SPAN)
    i_tail = opti.variable(init_guess=-5.0, lower_bound=-15.0, upper_bound=5.0)

    batt_x = opti.variable(init_guess=-0.10, lower_bound=MIN_BATTERY_X, upper_bound=MAX_BATTERY_X)
    motor_x = opti.variable(init_guess=0.20, lower_bound=MIN_MOTOR_X, upper_bound=MAX_MOTOR_X)

    mac = root_chord * (2 / 3) * (1 + taper + taper ** 2) / (1 + taper)
    S = span * root_chord * (1 + taper) / 2
    cg_x = 0.25 * mac

    opti.subject_to([
        motor_x <= tail_arm + cg_x,
        (span / mac) >= 4.0,
        (span / mac) <= 15.0,
    ])

    # 2) Mass and balance
    fc_x = 0.0
    other_x = (batt_x + cg_x + tail_arm) / 2
    ballast = opti.variable(init_guess=0.01, lower_bound=0.0, upper_bound=0.300)

    structural_mass, structural_moment = estimate_mass(
        span=span,
        root_chord=root_chord,
        taper=taper,
        tail_span=tail_span,
        tail_chord=tail_chord,
        tail_arm=tail_arm,
        target_cg_x=cg_x,
    )

    total_mass = structural_mass + PAYLOAD_TOTAL + ballast
    W = total_mass * g

    total_moment = (
        structural_moment
        + ((PAYLOAD_BATTERY + ballast) * batt_x)
        + (PAYLOAD_MOTOR * motor_x)
        + (PAYLOAD_FC * fc_x)
        + (PAYLOAD_OTHER * other_x)
    )
    opti.subject_to(total_moment / total_mass == cg_x)

    # 3) Trim state variables
    alpha = opti.variable(init_guess=5.0, lower_bound=-5.0, upper_bound=12.0)
    V = opti.variable(init_guess=10.0, lower_bound=3.0, upper_bound=25.0)

    # 4) Surrogate aerodynamics and simple 3D corrections
    effective_alpha_mac = alpha + twist * (0.5 * (1 + taper) * 0.5)
    wing_re = ca.fmax(ca.fmin(rho * V * mac / mu, 150e3), 20e3)

    op_pts_w = ca.horzcat(effective_alpha_mac, wing_re).T
    cl_w_2d = spline_cl_w(op_pts_w).T
    cd_w_2d = spline_cd_w(op_pts_w).T
    cm_w_2d = spline_cm_w(op_pts_w).T

    AR_w = span ** 2 / S
    e_w = 0.95
    CL_wing = cl_w_2d * (AR_w / (AR_w + 2))
    CDi_wing = (CL_wing ** 2) / (np.pi * AR_w * e_w)
    CD_wing = cd_w_2d + CDi_wing

    eps = 2 * CL_wing / (np.pi * AR_w)
    tail_alpha = alpha + i_tail - eps * (180 / np.pi)

    tail_re = ca.fmax(ca.fmin(rho * V * tail_chord / mu, 150e3), 20e3)
    op_pts_t = ca.horzcat(tail_alpha, tail_re).T
    cl_t_2d = spline_cl_t(op_pts_t).T
    cd_t_2d = spline_cd_t(op_pts_t).T
    cm_t_2d = spline_cm_t(op_pts_t).T

    AR_t = tail_span * 2 / tail_chord
    e_t = 0.95
    CL_tail = cl_t_2d * (AR_t / (AR_t + 2))
    CDi_tail = (CL_tail ** 2) / (np.pi * AR_t * e_t)
    CD_tail = cd_t_2d + CDi_tail

    CD0_fuselage = 0.005
    S_tail = tail_span * 2 * tail_chord
    q = 0.5 * rho * V ** 2

    total_lift = q * (CL_wing * S + CL_tail * S_tail)
    total_drag = q * (CD_wing * S + CD_tail * S_tail + CD0_fuselage * S)
    moment_y = q * (cm_w_2d * S * mac + cm_t_2d * S_tail * tail_chord) - (q * CL_tail * S_tail * tail_arm)

    # 5) Trim and stall constraints
    opti.subject_to([
        total_lift == W,
        (moment_y / (q * S * mac)) ** 2 <= 1e-6,
    ])

    stall_margin_alpha_deg = 2.0
    stall_margin_cl = 0.90
    cl_w_max_3d_est = cl_w_max_2d * (AR_w / (AR_w + 2))
    cl_w_min_3d_est = cl_w_min_2d * (AR_w / (AR_w + 2))
    opti.subject_to(alpha <= alpha_stall_deg - stall_margin_alpha_deg)
    opti.subject_to(CL_wing <= stall_margin_cl * cl_w_max_3d_est)
    opti.subject_to(CL_wing >= stall_margin_cl * cl_w_min_3d_est)

    # 6) Objective
    sink_expr = V * total_drag / total_lift
    opti.minimize(sink_expr)

    # 7) Multi-start solve
    rng = onp.random.default_rng(7)
    best_sol = None
    best_sink = None

    for i in range(N_STARTS_MDO):
        if i > 0:
            opti.set_initial(span, float(rng.uniform(MIN_SPAN, MAX_SPAN)))
            opti.set_initial(root_chord, float(rng.uniform(MIN_CHORD, MAX_CHORD)))
            opti.set_initial(taper, float(rng.uniform(0.3, 1.0)))
            opti.set_initial(twist, float(rng.uniform(-6.0, 2.0)))
            opti.set_initial(tail_arm, float(rng.uniform(MIN_TAIL_ARM, MAX_TAIL_ARM)))
            opti.set_initial(tail_chord, float(rng.uniform(MIN_TAIL_CHORD, MAX_TAIL_CHORD)))
            opti.set_initial(tail_span, float(rng.uniform(MIN_TAIL_SPAN, MAX_TAIL_SPAN)))
            opti.set_initial(i_tail, float(rng.uniform(-15.0, 5.0)))
            opti.set_initial(batt_x, float(rng.uniform(MIN_BATTERY_X, MAX_BATTERY_X)))
            opti.set_initial(motor_x, float(rng.uniform(MIN_MOTOR_X, 0.9)))
            opti.set_initial(ballast, float(rng.uniform(0.0, 0.08)))
            opti.set_initial(alpha, float(rng.uniform(-2.0, 8.0)))
            opti.set_initial(V, float(rng.uniform(4.0, 20.0)))

        try:
            sol_i = opti.solve(verbose=False, max_iter=2000)
            sink_i = float(sol_i(sink_expr))
            if best_sink is None or sink_i < best_sink:
                best_sink = sink_i
                best_sol = sol_i
        except Exception:
            continue

    if best_sol is None:
        print(f"  ✗ [{wing_af_name.upper()}] Static NLP Failed in all multi-start attempts.")
        return None

    sol = best_sol
    print(f"  ✓ [{wing_af_name.upper()}] NLP Converged! Sink = {float(sol(sink_expr)):.2f} m/s")

    return {
        "wing_af_name": wing_af_name,
        # Geometry
        "span": float(sol(span)),
        "chord": float(sol(root_chord)),
        "taper": float(sol(taper)),
        "twist": float(sol(twist)),
        "tail_arm": float(sol(tail_arm)),
        "tail_chord": float(sol(tail_chord)),
        "tail_span": float(sol(tail_span)),
        "i_tail": float(sol(i_tail)),
        "AR": float(sol(AR_w)),
        "S": float(sol(S)),
        # Mass and balance
        "mass": float(sol(total_mass)),
        "structural_mass": float(sol(structural_mass)),
        "ballast": float(sol(ballast)),
        "batt_x": float(sol(batt_x)),
        "motor_x": float(sol(motor_x)),
        # Stall estimates passed to trajectory phase
        "alpha_stall_deg": alpha_stall_deg,
        "alpha_stall_margin_deg": stall_margin_alpha_deg,
        "CL_max_est": float(sol(cl_w_max_3d_est)),
        "CL_min_est": float(sol(cl_w_min_3d_est)),
        # Trim KPIs
        "V": float(sol(V)),
        "LD": float(sol(total_lift / total_drag)),
        "alpha": float(sol(alpha)),
        "sink": float(sol(sink_expr)),
        "Re": rho * float(sol(V)) * float(sol(mac)) / mu,
    }


def optimize_glider_mdo(wing_af_name):
    print(f"\n  [MDO] Building aerodynamic surrogate model for {wing_af_name.upper()}...")
    try:
        from neuralfoil import get_aero_from_airfoil
    except ImportError:
        print("NeuralFoil missing. Skipping surrogate generation.")
        return None

    af_wing = asb.Airfoil(wing_af_name)

    alphas_coarse = onp.linspace(-10.0, 15.0, 26)
    res_coarse = onp.array([20e3, 40e3, 60e3, 80e3, 100e3, 150e3], dtype=float)

    CL_w_c, CD_w_c, CM_w_c = _load_or_build_grid(
        airfoil_key=wing_af_name,
        airfoil_obj=af_wing,
        alphas=alphas_coarse,
        res=res_coarse,
        get_aero_from_airfoil=get_aero_from_airfoil,
    )
    CL_t_c, CD_t_c, CM_t_c = _load_or_build_grid(
        airfoil_key="naca0009_tail",
        airfoil_obj=TAIL_AF,
        alphas=alphas_coarse,
        res=res_coarse,
        get_aero_from_airfoil=get_aero_from_airfoil,
    )

    splines_w_c = _build_splines(f"{wing_af_name}_coarse_w", alphas_coarse, res_coarse, CL_w_c, CD_w_c, CM_w_c)
    splines_t_c = _build_splines(f"{wing_af_name}_coarse_t", alphas_coarse, res_coarse, CL_t_c, CD_t_c, CM_t_c)

    stall_refs_c = {
        "cl_w_max_2d": float(onp.max(CL_w_c)),
        "cl_w_min_2d": float(onp.min(CL_w_c)),
        "alpha_stall_deg": float(onp.min(alphas_coarse[onp.argmax(CL_w_c, axis=0)])),
    }

    coarse_result = _solve_static_nlp(wing_af_name, splines_w_c, splines_t_c, stall_refs_c)
    if coarse_result is None:
        return None

    if not SURROGATE_LOCAL_REFINEMENT:
        return coarse_result

    alpha_center = float(coarse_result["alpha"])
    re_center = float(coarse_result["Re"])

    alpha_refine = onp.linspace(
        max(-10.0, alpha_center - REFINE_ALPHA_WINDOW_DEG),
        min(15.0, alpha_center + REFINE_ALPHA_WINDOW_DEG),
        REFINE_ALPHA_POINTS,
    )
    re_refine = onp.linspace(
        max(20e3, REFINE_RE_MIN_SCALE * re_center),
        min(150e3, REFINE_RE_MAX_SCALE * re_center),
        REFINE_RE_POINTS,
    )

    alphas_refined = onp.unique(onp.concatenate([alphas_coarse, alpha_refine]))
    res_refined = onp.unique(onp.concatenate([res_coarse, re_refine]))

    print(f"  [MDO] Refining surrogate near alpha={alpha_center:+.2f} deg, Re={re_center:.0f}...")

    CL_w_r, CD_w_r, CM_w_r = _load_or_build_grid(
        airfoil_key=wing_af_name,
        airfoil_obj=af_wing,
        alphas=alphas_refined,
        res=res_refined,
        get_aero_from_airfoil=get_aero_from_airfoil,
    )
    CL_t_r, CD_t_r, CM_t_r = _load_or_build_grid(
        airfoil_key="naca0009_tail",
        airfoil_obj=TAIL_AF,
        alphas=alphas_refined,
        res=res_refined,
        get_aero_from_airfoil=get_aero_from_airfoil,
    )

    splines_w_r = _build_splines(f"{wing_af_name}_refined_w", alphas_refined, res_refined, CL_w_r, CD_w_r, CM_w_r)
    splines_t_r = _build_splines(f"{wing_af_name}_refined_t", alphas_refined, res_refined, CL_t_r, CD_t_r, CM_t_r)

    stall_refs_r = {
        "cl_w_max_2d": float(onp.max(CL_w_r)),
        "cl_w_min_2d": float(onp.min(CL_w_r)),
        "alpha_stall_deg": float(onp.min(alphas_refined[onp.argmax(CL_w_r, axis=0)])),
    }

    refined_result = _solve_static_nlp(wing_af_name, splines_w_r, splines_t_r, stall_refs_r)
    if refined_result is None:
        print("  [MDO] Refined pass failed; keeping coarse-pass result.")
        return coarse_result

    if float(refined_result["sink"]) <= float(coarse_result["sink"]):
        return refined_result

    print("  [MDO] Refined pass converged but did not improve sink; keeping coarse-pass result.")
    return coarse_result


if __name__ == "__main__":
    optimize_glider_mdo("sd7037")
