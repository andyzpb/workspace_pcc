from __future__ import annotations
import argparse
import math
from tempfile import mkstemp
from time import time as _time
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pccik_native.pcc_workspace.specs import (
    SegmentSpec,
    TranslationSpec,
    TouchPointSpec,
    IKOptions,
)
from IK_pipeline import PCCIKClosedFormFast, create_solver

from workspace_npz import prepare_ws_card_from_solver, reach_and_solve_many
import testcases_generated as tc


class Timer:
    clock = _time

    def __call__(self):
        return self.clock()


# --------------------------- Segment definitions ---------------------------
# Outer tube: fixed length 0.05 m (bendable), no passive (frame of interest is its tip)
outer = SegmentSpec(
    name="outer",
    # lock outer to the local tip frame: no length, no bend, no roll
    L_min=0.0,
    L_max=0.0,
    samples_length=1,
    passive_L_min=0.0,
    passive_L_max=0.0,
    samples_passive_length=1,
    theta_min=0.0,
    theta_max=0.0,
    phi_min=0.0,
    phi_max=0.0,
    samples_theta=1,
    samples_phi=1,
)

# Inner tube: active <= 0.006 m, rigid tip = 0.003 m, passive <= 0.006 m
inner = SegmentSpec(
    name="inner",
    L_min=0.00,
    L_max=0.0125,
    samples_length=60,
    passive_L_min=0.0,
    passive_L_max=0.006,
    samples_passive_length=61,
    theta_min=np.deg2rad(-90.2),
    theta_max=np.deg2rad(90.2),
    samples_theta=360,
    phi_min=np.deg2rad(-180.0),
    phi_max=np.deg2rad(180.0) - 1e-5,
    samples_phi=720,
    active_L_max=0.006,
    is_inner=True,
    bevel_angle_deg=45.0,
    roll_offset_deg=0.0,
)

# --------------------------- Solver builder ---------------------------


def build_solver(
    d_min: float, d_max: float, s2_fixed: float = 0.006
) -> PCCIKClosedFormFast:
    tr = TranslationSpec(d_min=d_min, d_max=d_max, samples=100)
    opts = IKOptions(
        pos_tol=5e-3,
        ang_tol_deg=1.0,
        topk=1,
        require_frontside=True,
        use_bevel_alignment=True,
        angle_target_deg=45,
        enforce_axis_band=True,
        active_first=True,
        active_first_tol=2e-2,
        nms_enable=True,
    )
    solver = create_solver(
        [inner, outer], tr, opts, inner_rigid_tip=0.003, s2_fixed=s2_fixed
    )
    solver.debug = False
    return solver


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)


def _R_cc(phi: float, th: float) -> np.ndarray:
    c = math.cos(phi)
    s = math.sin(phi)
    ct = math.cos(th)
    st = math.sin(th)
    return np.array(
        [
            [ct * c * c + s * s, (ct - 1.0) * c * s, st * c],
            [(ct - 1.0) * c * s, ct * s * s + c * c, st * s],
            [-st * c, -st * s, ct],
        ],
        float,
    )


def _A_B_scalar(t: float) -> Tuple[float, float]:
    at = abs(t)
    if at < 1e-6:
        A = 0.5 * t - (t**3) / 24.0
        B = 1.0 - (t**2) / 6.0 + (t**4) / 120.0
    else:
        A = (1.0 - math.cos(t)) / t
        B = math.sin(t) / t
    return A, B


def _cc_T(
    phi: float, kappa: float, theta: float, s_hint: float | None = None
) -> np.ndarray:
    T = np.eye(4, dtype=float)
    eps = 1e-12
    if abs(theta) < eps or abs(kappa) < eps:
        s = (
            (theta / kappa)
            if (s_hint is None and abs(kappa) > eps)
            else (s_hint or 0.0)
        )
        p0 = np.array([0.0, 0.0, float(s)], dtype=float)
        # axis = Rz(phi) * ey
        c, s_ = math.cos(phi), math.sin(phi)
        axis = np.array([0.0, 1.0, 0.0])
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]], float)
        ax = Rz @ axis
        # Rodrigues
        kx, ky, kz = ax.tolist()
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], float)
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        p = Rz @ p0
    else:
        r = 1.0 / kappa
        c = math.cos(theta)
        s = math.sin(theta)
        p0 = np.array([r * (1.0 - c), 0.0, r * s], dtype=float)
        # same axis as above
        cph, sph = math.cos(phi), math.sin(phi)
        axis = np.array([0.0, 1.0, 0.0])
        Rz = np.array([[cph, -sph, 0.0], [sph, cph, 0.0], [0.0, 0.0, 1.0]], float)
        ax = Rz @ axis
        kx, ky, kz = ax.tolist()
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], float)
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        p = Rz @ p0
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _cc_centerline_points(
    theta: float, phi: float, s_active: float, n: int = 60
) -> np.ndarray:
    pts = [np.zeros(3, float)]
    if abs(theta) < 1e-9 or s_active < 1e-12:
        zs = np.linspace(0.0, s_active, n)
        for z in zs[1:]:
            pts.append(np.array([0.0, 0.0, z], float))
        return np.vstack(pts)
    kappa = theta / s_active
    for i in range(1, n + 1):
        th_i = theta * (i / n)
        s_i = s_active * (i / n)
        T = _cc_T(phi, kappa, th_i, s_hint=s_i)
        pts.append(T[:3, 3].copy())
    return np.vstack(pts)


# --------------------------- Workspace card for a given normal ---------------------------


def inner_angles_from_normal(
    n_star: np.ndarray, bevel_deg: float
) -> Tuple[float, float, np.ndarray]:
    n = _normalize(n_star)
    alpha = math.radians(bevel_deg)
    sa, ca = math.sin(alpha), math.cos(alpha)
    phi2 = math.atan2(float(n[1]), float(n[0]) - sa)
    c, s = math.cos(phi2), math.sin(phi2)
    u = np.array([sa * c, -sa * s, ca], float)
    w = np.array([c * n[0] + s * n[1], -s * n[0] + c * n[1], n[2]], float)
    ux, uz = u[0], u[2]
    wx, wz = w[0], w[2]
    den = ux * ux + uz * uz if (ux * ux + uz * uz) > 1e-16 else 1e-16
    cos_th = (ux * wx + uz * wz) / den
    sin_th = (uz * wx - ux * wz) / den
    theta2 = math.atan2(sin_th, cos_th)
    return float((phi2 % (2.0 * math.pi))), float(theta2), _R_cc(phi2, theta2)


def build_workspace_card(solver: PCCIKClosedFormFast, n_star: np.ndarray) -> dict:
    bevel_deg = float(getattr(solver.seg2, "bevel_angle_deg", 45.0))
    phi2, theta2, R2 = inner_angles_from_normal(np.asarray(n_star, float), bevel_deg)
    # base geometry
    s2 = float(solver.s2_fixed)
    A2, B2 = _A_B_scalar(theta2)
    v2 = np.array([math.cos(phi2) * s2 * A2, math.sin(phi2) * s2 * A2, s2 * B2], float)
    u = R2 @ (float(solver._Lrig) * np.array([0.0, 0.0, 1.0]))
    base = v2 + u
    ws = {
        "phi2": phi2,
        "theta2": theta2,
        "R2": R2,
        "xy_base": base[:2].copy(),
        "z_base": float(base[2]),
        "Smin": float(solver._L2p_min + solver.translation.d_min),
        "Smax": float(solver._L2p_max + solver.translation.d_max),
        "bevel_deg": bevel_deg,
    }
    return ws


# --------------------------- Pretty plotting ---------------------------


def style_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, alpha=0.25)
    ax.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
    ax.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
    ax.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def draw_equal_xyz(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    r = max(xr, yr, zr)
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    cz = 0.5 * (zlim[0] + zlim[1])
    ax.set_xlim3d(cx - r / 2, cx + r / 2)
    ax.set_ylim3d(cy - r / 2, cy + r / 2)
    ax.set_zlim3d(cz - r / 2, cz + r / 2)


def draw_axes_triad(ax, origin=np.zeros(3), scale=0.02):
    o = np.asarray(origin, float)
    ax.quiver(o[0], o[1], o[2], scale, 0, 0, color="#ef4444", linewidth=2)
    ax.quiver(o[0], o[1], o[2], 0, scale, 0, color="#22c55e", linewidth=2)
    ax.quiver(o[0], o[1], o[2], 0, 0, scale, color="#3b82f6", linewidth=2)


def make_cloud_from_card(
    card: Dict[str, Any],
    pos_tol: float,
    N: int = 30000,
    radius_scale: float = 1.03,
    z_margin: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = np.asarray(card["xy_base"], float).reshape(2)
    z_base = float(card["z_base"])
    Smin, Smax = np.asarray(card["sum_interval"], float).tolist()
    r_max = float(pos_tol) * float(radius_scale)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    rr = r_max * np.sqrt(rng.uniform(0.0, 1.0, size=N))
    x = xy[0] + rr * np.cos(theta)
    y = xy[1] + rr * np.sin(theta)
    z = rng.uniform(z_base + Smin - z_margin, z_base + Smax + z_margin, size=N)
    return np.stack([x, y, z], axis=1)


def plot_elegant(
    ax,
    solver: "PCCIKClosedFormFast",
    touch: TouchPointSpec,
    sol,
    ws: dict,
    *,
    cloud_P: np.ndarray | None = None,  # shape (N,3)
    cloud_mask: np.ndarray | None = None,  # shape (N,)
    cloud_reasons: (
        np.ndarray | None
    ) = None,  # shape (N,)  {1:xy_miss,2:sum_interval,3:bevel_tol,4:other}
    cloud_size: float = 4.0,
    cloud_alpha: float = 0.35,
):

    P = np.array(touch.coordinates, float)
    n = np.array(touch.normal, float)

    ax.scatter(
        [P[0]],
        [P[1]],
        [P[2]],
        s=40,
        c="#111827",
        marker="o",
        depthshade=True,
        label="target",
    )
    ax.quiver(P[0], P[1], P[2], *(n * 0.02), color="#1f77b4", linewidth=2.0)

    s2 = float(solver.s2_fixed)
    L2p = float(sol.segments[1].L_passive)
    phi2 = float(sol.segments[1].phi)
    theta2 = float(sol.segments[1].theta)

    pts_pas = np.vstack([np.zeros(3), np.array([0.0, 0.0, L2p])])
    pts_act = _cc_centerline_points(theta2, phi2, s2, n=80)
    pts_act[:, 2] += L2p

    tip_end = pts_act[-1] + (
        ws["R2"] @ (float(solver._Lrig) * np.array([0.0, 0.0, 1.0]))
    )

    ax.plot(
        pts_pas[:, 0],
        pts_pas[:, 1],
        pts_pas[:, 2],
        color="#64748b",
        linewidth=2.5,
        label="inner passive",
    )
    ax.plot(
        pts_act[:, 0],
        pts_act[:, 1],
        pts_act[:, 2],
        color="#111827",
        linewidth=2.8,
        label="inner active",
    )
    ax.plot(
        [pts_act[-1, 0], tip_end[0]],
        [pts_act[-1, 1], tip_end[1]],
        [pts_act[-1, 2], tip_end[2]],
        color="#6b7280",
        linewidth=2.5,
        label="rigid tip",
    )

    xb, yb = ws["xy_base"]
    z_ws1 = ws["z_base"] + float(solver._L2p_min + solver.translation.d_min)
    z_ws2 = ws["z_base"] + float(solver._L2p_max + solver.translation.d_max)
    ax.plot(
        [xb, xb],
        [yb, yb],
        [z_ws1, z_ws2],
        color="#10b981",
        linewidth=5.0,
        alpha=0.5,
        label="WS(z)",
    )

    z_loc1 = ws["z_base"] + float(solver._L2p_min)
    z_loc2 = ws["z_base"] + float(solver._L2p_max)
    ax.plot(
        [xb, xb],
        [yb, yb],
        [z_loc1, z_loc2],
        color="#22d3ee",
        linewidth=3.0,
        alpha=0.7,
        label="WS(z)|d=0",
    )

    end_p_no_d = sol.end_p
    end_p_w = sol.meta.get(
        "end_p_world", end_p_no_d + np.array([0, 0, sol.translation])
    )
    ax.scatter(
        [end_p_w[0]],
        [end_p_w[1]],
        [end_p_w[2]],
        s=55,
        c="#ef4444",
        marker="^",
        label="EE (with d)",
    )

    drew_cloud = False
    if (
        (cloud_P is not None)
        and (cloud_mask is not None)
        and (cloud_reasons is not None)
    ):
        cloud_P = np.asarray(cloud_P, float).reshape(-1, 3)
        cloud_mask = np.asarray(cloud_mask, bool).reshape(-1)
        cloud_reasons = np.asarray(cloud_reasons, int).reshape(-1)
        if len(cloud_P) == len(cloud_mask) == len(cloud_reasons) and len(cloud_P) > 0:
            drew_cloud = True
            col_reach = "#10b981"
            cmap_unreach = {
                1: "#fb923c",  # xy_miss orange
                2: "#60a5fa",  # sum_interval blue
                3: "#ef4444",  # bevel_tol red
                4: "#9ca3af",  # other grey
            }
            ok = cloud_mask
            if np.any(ok):
                Pi = cloud_P[ok]
                ax.scatter(
                    Pi[:, 0],
                    Pi[:, 1],
                    Pi[:, 2],
                    s=cloud_size,
                    alpha=cloud_alpha,
                    c=col_reach,
                    marker=".",
                    depthshade=False,
                    label="reachable",
                )
            for code, color in cmap_unreach.items():
                idx = (~cloud_mask) & (cloud_reasons == code)
                if np.any(idx):
                    Pi = cloud_P[idx]
                    ax.scatter(
                        Pi[:, 0],
                        Pi[:, 1],
                        Pi[:, 2],
                        s=cloud_size,
                        alpha=min(0.9, cloud_alpha + 0.1),
                        c=color,
                        marker=".",
                        depthshade=False,
                        label={
                            1: "xy_miss",
                            2: "sum_interval",
                            3: "bevel_tol",
                            4: "other",
                        }[code],
                    )

    draw_axes_triad(ax, np.zeros(3), scale=0.02)

    style_axes(ax)

    stack_list = [
        pts_pas,
        pts_act,
        tip_end.reshape(1, 3),
        P.reshape(1, 3),
        end_p_w.reshape(1, 3),
        np.array([[xb, yb, z_ws1], [xb, yb, z_ws2]]),
    ]
    if drew_cloud:
        stack_list.append(cloud_P)

    all_pts = np.vstack(stack_list)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = maxs - mins
    center = 0.5 * (maxs + mins)
    r = max(0.05, 1.2 * span.max())

    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    draw_equal_xyz(ax)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(
            uniq.values(), uniq.keys(), loc="upper right", fontsize=9, frameon=False
        )


def run_case_elegant(
    solver: PCCIKClosedFormFast, name: str, touch: TouchPointSpec
) -> None:
    print(f"Touch point {name}: {touch}")
    # IK solve
    t = Timer()
    t0 = t()
    sols = solver.solve(touch)
    t1 = t()
    print(f"IK solve time: {t1 - t0:.4f} s")

    if not sols:
        print("  -> Not reachable under current tolerances / d-range.")
        return

    sol = sols[0]
    ws = build_workspace_card(solver, np.array(touch.normal, float))

    tmp_path = mkstemp(prefix="inner_ws_", suffix=".npz")[1]
    card = prepare_ws_card_from_solver(solver, np.array(touch.normal, float), tmp_path)

    cloud_P = make_cloud_from_card(
        card,
        pos_tol=float(solver.opts.pos_tol),
        N=30000,
        radius_scale=1.03,
        z_margin=0.0,
        seed=0,
    )
    cloud_mask, cloud_sols, cloud_reasons = reach_and_solve_many(card, cloud_P)

    fig = plt.figure(figsize=(8.6, 7.4), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    plot_elegant(
        ax,
        solver,
        touch,
        sol,
        ws,
        cloud_P=cloud_P,
        cloud_mask=cloud_mask,
        cloud_reasons=cloud_reasons,
        cloud_size=3.5,
        cloud_alpha=0.35,
    )

    rpt = sol.meta.get("report", {})
    o = rpt.get("outer", {})
    ii = rpt.get("inner", {})
    print("  -> Solution:")
    print(
        "     [outer] bend={:.2f} deg, rot={:.2f} deg, d={:.3f} m".format(
            o.get("bending_deg", float("nan")),
            o.get("rotation_deg", float("nan")),
            sol.translation,
        )
    )
    print(
        "     [inner] bend={:.2f} deg, rot={:.2f} deg, Ltot={:.3f} m (L2p={:.3f} m)".format(
            ii.get("bending_deg", float("nan")),
            ii.get("rotation_deg", float("nan")),
            ii.get("translation", float("nan")),
            sol.segments[1].L_passive,
        )
    )

    fig = plt.figure(figsize=(8.6, 7.4), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    plot_elegant(ax, solver, touch, sol, ws)
    fig.suptitle(
        f"IK @ {name}  |  bevel={getattr(solver.seg2,'bevel_angle_deg',45)}°  |  d∈[{solver.translation.d_min:.3f},{solver.translation.d_max:.3f}] m",
        fontsize=12,
        fontweight="semibold",
        color="#111827",
    )
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Elegant closed-form IK demo (manual pick)"
    )
    ap.add_argument(
        "--set",
        choices=[
            "pos",
            "need_d",
            "unreachable",
            "pos_neg",
            "need_d_neg",
            "unreachable_neg",
            "all",
        ],
        default="pos",
        help="Test set to browse",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Run a single case by name (e.g., N01, ND03, NU01, N01_NEG)",
    )
    ap.add_argument("--dmin", type=float, default=0.0)
    ap.add_argument("--dmax", type=float, default=0.02)
    args = ap.parse_args()

    solver = build_solver(args.dmin, args.dmax)

    sets = {
        "pos": tc.TEST_POINTS_POS,
        "need_d": tc.TEST_POINTS_NEED_D,
        "unreachable": tc.TEST_POINTS_UNREACHABLE,
        "pos_neg": tc.TEST_POINTS_POS_NEG,
        "need_d_neg": tc.TEST_POINTS_NEED_D_NEG,
        "unreachable_neg": tc.TEST_POINTS_UNREACHABLE_NEG,
        "all": tc.TEST_POINTS_ALL,
    }
    chosen = sets[args.set]

    if args.name is not None:
        name_map = {nm: pair for nm, pair in tc.NAME_TO_PAIR.items()}
        if args.name not in name_map:
            print("Unknown test name:", args.name)
            print("Available keys, sample:", sorted(name_map.keys())[:20], "...")
            return
        nm, tp = name_map[args.name]
        run_case_elegant(solver, nm, tp)
        return

    items = list(chosen)
    print(f"\nAvailable in set '{args.set}': {len(items)} candidates")
    for i, (nm, tp) in enumerate(items):
        P = np.array(tp.coordinates, float)
        n = np.array(tp.normal, float)
        print(f"  [{i:02d}] {nm}  P={tuple(np.round(P,4))}  n={tuple(np.round(n,3))}")
    try:
        sel = input("\nPick an index to run (e.g., 0): ")
        idx = int(sel)
    except Exception:
        print("Invalid input. Abort.")
        return
    if idx < 0 or idx >= len(items):
        print("Index out of range. Abort.")
        return
    nm, tp = items[idx]
    run_case_elegant(solver, nm, tp)


if __name__ == "__main__":
    main()
