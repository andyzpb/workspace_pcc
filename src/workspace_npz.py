from __future__ import annotations
import math, os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from pccik_native.pcc_workspace.specs import (
    IKSolution,
    SegmentSolution,
    TranslationSpec,
    SegmentSpec,
    TouchPointSpec,
    IKOptions,
)

from numba import njit


@njit(cache=True, fastmath=True)
def _wrap_0_2pi_njit(a: float) -> float:
    two_pi = 2.0 * math.pi
    x = a % two_pi
    if x < 0.0:
        x += two_pi
    return x


@njit(cache=True, fastmath=True)
def _normalize_njit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < eps:
        return v * 0.0
    return v / n


@njit(cache=True, fastmath=True)
def _R_cc_scalar_njit(phi: float, th: float) -> np.ndarray:
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
        dtype=np.float64,
    )


@njit(cache=True, fastmath=True)
def _A_B_scalar_njit(t: float) -> Tuple[float, float]:
    at = abs(t)
    if at < 1e-6:
        A = 0.5 * t - (t**3) / 24.0
        B = 1.0 - (t**2) / 6.0 + (t**4) / 120.0
    else:
        A = (1.0 - math.cos(t)) / t
        B = math.sin(t) / t
    return A, B


@njit(cache=True, fastmath=True)
def _inner_angles_from_orientation_njit(
    n_star: np.ndarray, alpha: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    n_prime = _normalize_njit(n_star)
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    phi2 = math.atan2(n_prime[1], n_prime[0] - sa)
    c = math.cos(phi2)
    s = math.sin(phi2)
    u = np.array([sa * c, -sa * s, ca], dtype=np.float64)
    w = np.array(
        [c * n_prime[0] + s * n_prime[1], -s * n_prime[0] + c * n_prime[1], n_prime[2]],
        dtype=np.float64,
    )
    ux, uz = u[0], u[2]
    wx, wz = w[0], w[2]
    den = ux * ux + uz * uz
    if den < 1e-16:
        den = 1e-16
    cos_th = (ux * wx + uz * wz) / den
    sin_th = (uz * wx - ux * wz) / den
    theta2 = math.atan2(sin_th, cos_th)
    R2 = _R_cc_scalar_njit(phi2, theta2)
    z2 = R2[:, 2].copy()
    return _wrap_0_2pi_njit(phi2), theta2, z2, R2


@dataclass
class InnerWorkspaceConfig:
    s2_fixed: float
    Lrig: float
    bevel_deg: float
    theta2_min: float
    theta2_max: float
    phi2_min: Optional[float]
    phi2_max: Optional[float]
    L2p_min: float
    L2p_max: float
    d_min: float
    d_max: float
    pos_tol: float
    bevel_tol_deg: float
    angle_target_deg: float = 45.0
    outer_name: str = "outer"
    inner_name: str = "inner"
    outer_L1p: float = 0.0
    outer_s1: float = 0.0


def prepare_ws_card_from_solver(
    solver, n_star: np.ndarray, npz_path: str
) -> Dict[str, Any]:
    n_star = np.asarray(n_star, dtype=np.float64).reshape(3)
    cfg = InnerWorkspaceConfig(
        s2_fixed=float(solver.s2_fixed),
        Lrig=float(solver.inner_rigid_tip),
        bevel_deg=float(np.degrees(solver.seg2.bevel_angle_deg)),
        theta2_min=float(getattr(solver.seg2, "theta_min", -1e9)),
        theta2_max=float(getattr(solver.seg2, "theta_max", +1e9)),
        phi2_min=(
            float(getattr(solver.seg2, "phi_min", 0.0))
            if getattr(solver.seg2, "phi_min", None) is not None
            else None
        ),
        phi2_max=(
            float(getattr(solver.seg2, "phi_max", 2.0 * math.pi))
            if getattr(solver.seg2, "phi_max", None) is not None
            else None
        ),
        L2p_min=float(solver._L2p_min),
        L2p_max=float(solver._L2p_max),
        d_min=float(solver.translation.d_min),
        d_max=float(solver.translation.d_max),
        pos_tol=float(solver.opts.pos_tol),
        bevel_tol_deg=float(getattr(solver.opts, "bevel_tol_deg", 1.0)),
        angle_target_deg=float(getattr(solver.opts, "angle_target_deg", 45.0)),
        outer_name=str(solver.seg1.name),
        inner_name=str(solver.seg2.name),
        outer_L1p=float(solver._L1p),
        outer_s1=float(solver._s1),
    )

    alpha = math.radians(cfg.bevel_deg)
    phi2, theta2, z2, R2 = _inner_angles_from_orientation_njit(n_star, alpha)

    if not (cfg.theta2_min - 1e-12 <= theta2 <= cfg.theta2_max + 1e-12):
        card = {"schema": "inner_ws_v1", "reachable": False, "reason": "angle_box"}
        np.savez(npz_path, **{k: np.array(v) for k, v in card.items()})
        return card
    if (cfg.phi2_min is not None) and (cfg.phi2_max is not None):
        pw = _wrap_0_2pi_njit(phi2)
        lo = _wrap_0_2pi_njit(cfg.phi2_min)
        hi = _wrap_0_2pi_njit(cfg.phi2_max)
        span = (hi - lo) % (2.0 * math.pi)
        if span < 2.0 * math.pi - 1e-9:
            x = (pw - lo) % (2.0 * math.pi)
            if x > span + 1e-9:
                card = {
                    "schema": "inner_ws_v1",
                    "reachable": False,
                    "reason": "angle_box",
                }
                np.savez(npz_path, **{k: np.array(v) for k, v in card.items()})
                return card

    A2, B2 = _A_B_scalar_njit(theta2)
    c2, s2 = math.cos(phi2), math.sin(phi2)
    v2 = np.array(
        [c2 * cfg.s2_fixed * A2, s2 * cfg.s2_fixed * A2, cfg.s2_fixed * B2],
        dtype=np.float64,
    )
    u = R2 @ (cfg.Lrig * np.array([0.0, 0.0, 1.0], dtype=np.float64))
    base = v2 + u

    Smin = cfg.L2p_min + cfg.d_min
    Smax = cfg.L2p_max + cfg.d_max

    b0 = np.array([math.sin(alpha), 0.0, math.cos(alpha)], dtype=np.float64)
    b_world = R2 @ b0
    cos_bevel_tol = math.cos(math.radians(cfg.bevel_tol_deg))

    card = {
        "schema": "inner_ws_v1",
        "reachable": True,
        "contact_normal": n_star.astype(np.float64),
        "phi2": float(_wrap_0_2pi_njit(phi2)),
        "theta2": float(theta2),
        "xy_base": base[:2].astype(np.float64),  # (v2+u)_xy
        "z_base": float(base[2]),  # (v2+u)_z
        "sum_interval": np.array([Smin, Smax], dtype=np.float64),
        "cos_bevel_tol": float(cos_bevel_tol),
        "s2_fixed": float(cfg.s2_fixed),
        "Lrig": float(cfg.Lrig),
        "bevel_deg": float(cfg.bevel_deg),
        "theta2_min": float(cfg.theta2_min),
        "theta2_max": float(cfg.theta2_max),
        "phi2_has_box": int((cfg.phi2_min is not None) and (cfg.phi2_max is not None)),
        "phi2_min": float(cfg.phi2_min) if cfg.phi2_min is not None else 0.0,
        "phi2_max": float(cfg.phi2_max) if cfg.phi2_max is not None else 2.0 * math.pi,
        "L2p_min": float(cfg.L2p_min),
        "L2p_max": float(cfg.L2p_max),
        "d_min": float(cfg.d_min),
        "d_max": float(cfg.d_max),
        "pos_tol": float(cfg.pos_tol),
        "bevel_tol_deg": float(cfg.bevel_tol_deg),
        "angle_target_deg": float(cfg.angle_target_deg),
        "outer_name": cfg.outer_name,
        "inner_name": cfg.inner_name,
        "outer_L1p": float(cfg.outer_L1p),
        "outer_s1": float(cfg.outer_s1),
    }

    np.savez(npz_path, **{k: np.array(v) for k, v in card.items()})
    return card


def load_ws_card(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    card: Dict[str, Any] = {}
    for k in data.files:
        val = data[k]
        if val.shape == ():
            try:
                card[k] = val.item()
            except Exception:
                card[k] = float(val)
        else:
            # 字符串/数组
            if val.dtype.kind in ("U", "S"):
                card[k] = str(val)
            else:
                card[k] = np.array(val)
    return card


def reach_and_solve(
    card: Dict[str, Any], P_star: np.ndarray
) -> Tuple[bool, Optional[str], Optional[IKSolution]]:

    if not card.get("reachable", True):
        return False, card.get("reason", "unprepared"), None

    P_star = np.asarray(P_star, dtype=np.float64).reshape(3)

    xy_base = card["xy_base"].astype(np.float64)
    if np.linalg.norm(P_star[:2] - xy_base) > float(card["pos_tol"]) + 1e-12:
        return False, "xy_miss", None

    r = float(P_star[2] - float(card["z_base"]))
    Smin, Smax = float(card["sum_interval"][0]), float(card["sum_interval"][1])
    if (r < Smin - 1e-12) or (r > Smax + 1e-12):
        return False, "sum_interval", None

    d_min, d_max = float(card["d_min"]), float(card["d_max"])
    L2p_min, L2p_max = float(card["L2p_min"]), float(card["L2p_max"])

    d_star = min(max(r, d_min), d_max)
    L2p_star = r - d_star
    if L2p_star < L2p_min or L2p_star > L2p_max:
        L2p_star = min(max(L2p_star, L2p_min), L2p_max)
        d_star = r - L2p_star
        if d_star < d_min or d_star > d_max:
            return False, "sum_interval", None

    alpha = math.radians(float(card["bevel_deg"]))
    b0 = np.array([math.sin(alpha), 0.0, math.cos(alpha)], dtype=np.float64)
    phi2 = float(card["phi2"])
    theta2 = float(card["theta2"])
    R2 = _R_cc_scalar_njit(phi2, theta2)
    b_world = R2 @ b0
    b_world = _normalize_njit(b_world)
    cos_bevel_tol = float(card["cos_bevel_tol"])
    n_star = card["contact_normal"].astype(np.float64)
    if float(b_world @ n_star) < cos_bevel_tol:
        return False, "bevel_tol", None

    s2 = float(card["s2_fixed"])
    Lrig = float(card["Lrig"])
    A2, B2 = _A_B_scalar_njit(theta2)
    c2, s2t = math.cos(phi2), math.sin(phi2)
    v2 = np.array([c2 * s2 * A2, s2t * s2 * A2, s2 * B2], dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    end_p_no_d = (L2p_star * ez) + v2 + (R2 @ (Lrig * ez))
    end_p_world = end_p_no_d + d_star * ez

    segs = [
        SegmentSolution(
            name=str(card.get("outer_name", "outer")),
            theta=0.0,
            phi=0.0,
            L_total=float(card.get("outer_L1p", 0.0) + card.get("outer_s1", 0.0)),
            L_passive=float(card.get("outer_L1p", 0.0)),
            L_active=float(card.get("outer_s1", 0.0)),
        ),
        SegmentSolution(
            name=str(card.get("inner_name", "inner")),
            theta=float(theta2),
            phi=float(phi2),
            L_total=float(s2 + L2p_star),
            L_passive=float(L2p_star),
            L_active=float(s2),
        ),
    ]

    T_tip = np.eye(4, dtype=float)
    T_tip[:3, :3] = R2
    T_tip[:3, 3] = end_p_no_d

    meta = dict(
        end_p_world=end_p_world.copy(),
        bevel_angle_deg=float(card["bevel_deg"]),
        bevel_world=b_world.copy(),
        plane_normal_world=np.array([0.0, 0.0, 0.0], float),
        report=dict(
            outer=dict(
                bending_deg=0.0,
                rotation_deg=0.0,
                translation=float(
                    card.get("outer_L1p", 0.0) + card.get("outer_s1", 0.0)
                ),
            ),
            inner=dict(
                bending_deg=float(np.degrees(theta2)),
                rotation_deg=float(np.degrees(phi2)),
                translation=float(s2 + L2p_star),
            ),
        ),
    )

    sol = IKSolution(
        reachable=True,
        pos_err=0.0,
        ang_err_deg=0.0,
        translation=float(d_star),
        segments=segs,
        end_T=T_tip,
        end_p=end_p_no_d,
        meta=meta,
    )
    return True, None, sol


def reach_and_solve_many(
    card: Dict[str, Any], P_list: np.ndarray
) -> Tuple[np.ndarray, List[Optional[IKSolution]], np.ndarray]:
    P_list = np.asarray(P_list, dtype=np.float64).reshape(-1, 3)
    mask = np.zeros(len(P_list), dtype=bool)
    sols: List[Optional[IKSolution]] = [None] * len(P_list)
    reasons = np.zeros(len(P_list), dtype=np.int32)

    for i, P in enumerate(P_list):
        ok, reason, sol = reach_and_solve(card, P)
        if ok:
            mask[i] = True
            sols[i] = sol
            reasons[i] = 0
        else:
            reasons[i] = {"xy_miss": 1, "sum_interval": 2, "bevel_tol": 3}.get(
                reason or "unprepared", 4
            )
    return mask, sols, reasons
