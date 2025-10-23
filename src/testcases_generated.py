from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Dict
from pccik_native.pcc_workspace.specs import TouchPointSpec

def _rotz(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def _axis_angle_R(u: np.ndarray, th: float) -> np.ndarray:
    u = np.asarray(u, float)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    u = u / n
    c, s = float(np.cos(th)), float(np.sin(th))
    ux, uy, uz = u.tolist()
    K = np.array([[ 0.0, -uz,  uy],
                  [ uz,  0.0, -ux],
                  [-uy,  ux,  0.0]], dtype=float)
    I = np.eye(3, dtype=float)
    return c*I + (1.0-c)*np.outer(u, u) + s*K

def _cc_T(phi: float, kappa: float, theta: float, s_hint: float | None = None) -> np.ndarray:
    """Constant-curvature transform with optional straight fallback s_hint."""
    T = np.eye(4, dtype=float)
    eps = 1e-12
    if abs(theta) < eps or abs(kappa) < eps:
        s = (theta / kappa) if (s_hint is None and abs(kappa) > eps) else (s_hint or 0.0)
        p0 = np.array([0.0, 0.0, float(s)], dtype=float)
        R = _axis_angle_R(np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float), theta)
        p = _rotz(phi) @ p0
    else:
        r = 1.0 / kappa
        c, s_trig = float(np.cos(theta)), float(np.sin(theta))
        p0 = np.array([r*(1.0 - c), 0.0, r*s_trig], dtype=float)
        R  = _axis_angle_R(np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float), theta)
        p  = _rotz(phi) @ p0
    T[:3,:3] = R
    T[:3, 3] = p
    return T

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v*0.0 if n < 1e-12 else (v/n)

# ------------------------ generator ------------------------

def make_touch_inner_only(theta2_deg: float, phi2_deg: float,
                          L2_active: float, L2_passive: float,
                          name: str,
                          *, bevel_angle_deg: float = 45.0,
                          L_rigid: float = 0.003,
                          dz_extra: float = 0.0,
                          dxy_extra: Tuple[float, float] = (0.0, 0.0)) -> Tuple[str, TouchPointSpec]:
    """
    Build a touch point in OUTER-TIP frame (R1=I, p1=0) using inner-only FK.
      - Optionally add dz_extra to z (to emulate needing d)
      - Optionally add dxy_extra to XY (to fabricate unreachable-X/Y cases)
    """
    th2 = float(np.deg2rad(theta2_deg)); ph2 = float(np.deg2rad(phi2_deg))

    T2_pas = np.eye(4); T2_pas[2,3] = float(L2_passive)
    k2 = 0.0 if abs(th2) < 1e-12 else th2 / float(L2_active)
    T2_act = _cc_T(ph2, k2, th2, s_hint=float(L2_active))
    T2_rig = np.eye(4); T2_rig[2,3] = float(L_rigid)

    T_tip = T2_pas @ T2_act @ T2_rig
    R_tip, p_tip = T_tip[:3,:3], T_tip[:3,3]

    # Apply crafted offsets
    p_tip = p_tip + np.array([float(dxy_extra[0]), float(dxy_extra[1]), float(dz_extra)], dtype=float)

    alpha = np.deg2rad(bevel_angle_deg)
    b_world = _normalize(R_tip @ np.array([np.sin(alpha), 0.0, np.cos(alpha)]))
    return (name, TouchPointSpec(coordinates=tuple(p_tip), normal=tuple(b_world)))

# ------------------------ constants ------------------------
L2A_MAX   = 0.006   # inner active ≤ 6 mm
L2P_MAX   = 0.006   # inner passive ≤ 6 mm
L_RIGID   = 0.003   # inner rigid tip 3 mm

# ------------------------ POS (reachable with d=0) ------------------------
# Choose a spread of (theta2, phi2) with feasible L2p
_BASE_POS = [
    (+0.0,    0.0, 0.006, 0.001, "N01"),
    (+18.0,  45.0, 0.006, 0.002, "N02"),
    (+20.0,  90.0, 0.006, 0.004, "N03"),
    (+15.0, 135.0, 0.006, 0.003, "N04"),
    (+22.0, 180.0, 0.006, 0.002, "N05"),
    (+16.0, 225.0, 0.006, 0.003, "N06"),
    (+12.0, 270.0, 0.006, 0.003, "N07"),
    (+24.0, 315.0, 0.006, 0.001, "N08"),
    (+10.0,  30.0, 0.006, 0.003, "N09"),
    (+14.0, 120.0, 0.006, 0.003, "N10"),
    (+18.0, 210.0, 0.006, 0.004, "N11"),
    (+20.0, 300.0, 0.006, 0.002, "N12"),
]

# Strategy: keep same XY but lift z by dz such that L2p would exceed L2P_MAX
# Example: if base L2p=0.002, choose dz=+0.006 -> needs d≈4mm if d-range allows.
_NEED_D = [
    (+18.0,  45.0, 0.006, 0.002, +0.006, "ND01"),
    (+20.0,  90.0, 0.006, 0.001, +0.006, "ND02"),
    (+15.0, 135.0, 0.006, 0.000, +0.007, "ND03"),
    (+22.0, 180.0, 0.006, 0.003, +0.005, "ND04"),
]

# Strategy: inject 8–12 mm XY offsets (>> typical pos_tol=5mm)
_UNREACHABLE = [
    (+18.0,  45.0, 0.006, 0.002, (+0.010,  0.000), "NU01"),
    (+20.0,  90.0, 0.006, 0.004, (-0.012, +0.009), "NU02"),
    (+15.0, 135.0, 0.006, 0.003, (+0.011, -0.011), "NU03"),
]

# ------------------------ builders ------------------------

def _build_pos(case):
    th2, ph2, L2a, L2p, nm = case
    return make_touch_inner_only(th2, ph2, L2a, L2p, nm, bevel_angle_deg=45.0, L_rigid=L_RIGID)

def _build_need_d(case):
    th2, ph2, L2a, L2p, dz, nm = case
    return make_touch_inner_only(th2, ph2, L2a, L2p, nm, bevel_angle_deg=45.0, L_rigid=L_RIGID, dz_extra=float(dz))

def _build_unreach(case):
    th2, ph2, L2a, L2p, dxy, nm = case
    return make_touch_inner_only(th2, ph2, L2a, L2p, nm, bevel_angle_deg=45.0, L_rigid=L_RIGID, dxy_extra=dxy)

STRICT_BEVEL_POINTS_45_POS         = [_build_pos(c) for c in _BASE_POS]
STRICT_BEVEL_POINTS_45_NEED_D      = [_build_need_d(c) for c in _NEED_D]
STRICT_BEVEL_POINTS_45_UNREACHABLE = [_build_unreach(c) for c in _UNREACHABLE]

# NEG (flip normal)

def _flip_normal(pair: Tuple[str, TouchPointSpec], suffix: str = "_NEG") -> Tuple[str, TouchPointSpec]:
    nm, tp = pair
    P = np.array(tp.coordinates, float)
    n = -np.array(tp.normal, float)
    return nm + suffix, TouchPointSpec(coordinates=tuple(P), normal=tuple(n))

STRICT_BEVEL_POINTS_45_POS_NEG         = [_flip_normal(p) for p in STRICT_BEVEL_POINTS_45_POS]
STRICT_BEVEL_POINTS_45_NEED_D_NEG      = [_flip_normal(p) for p in STRICT_BEVEL_POINTS_45_NEED_D]
STRICT_BEVEL_POINTS_45_UNREACHABLE_NEG = [_flip_normal(p) for p in STRICT_BEVEL_POINTS_45_UNREACHABLE]

# Named exports (POS)
(N01, N02, N03, N04, N05, N06, N07, N08, N09, N10, N11, N12) = STRICT_BEVEL_POINTS_45_POS
# Named exports (NEED_D)
(ND01, ND02, ND03, ND04) = STRICT_BEVEL_POINTS_45_NEED_D
# Named exports (UNREACHABLE)
(NU01, NU02, NU03) = STRICT_BEVEL_POINTS_45_UNREACHABLE

# Convenience sets
TEST_POINTS_POS          = STRICT_BEVEL_POINTS_45_POS
TEST_POINTS_NEED_D       = STRICT_BEVEL_POINTS_45_NEED_D
TEST_POINTS_UNREACHABLE  = STRICT_BEVEL_POINTS_45_UNREACHABLE
TEST_POINTS_POS_NEG          = STRICT_BEVEL_POINTS_45_POS_NEG
TEST_POINTS_NEED_D_NEG       = STRICT_BEVEL_POINTS_45_NEED_D_NEG
TEST_POINTS_UNREACHABLE_NEG  = STRICT_BEVEL_POINTS_45_UNREACHABLE_NEG
TEST_POINTS_ALL = (
    STRICT_BEVEL_POINTS_45_POS + STRICT_BEVEL_POINTS_45_NEED_D + STRICT_BEVEL_POINTS_45_UNREACHABLE +
    STRICT_BEVEL_POINTS_45_POS_NEG + STRICT_BEVEL_POINTS_45_NEED_D_NEG + STRICT_BEVEL_POINTS_45_UNREACHABLE_NEG
)

# Name lookup (for CLI)
NAME_TO_PAIR: Dict[str, Tuple[str, TouchPointSpec]] = {
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_POS},
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_NEED_D},
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_UNREACHABLE},
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_POS_NEG},
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_NEED_D_NEG},
    **{nm: (nm, tp) for (nm, tp) in TEST_POINTS_UNREACHABLE_NEG},
}
