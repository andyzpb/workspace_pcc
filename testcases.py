import math
import numpy as np
from typing import Tuple, List
from pcc_workspace.specs import TouchPointSpec

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

def _cc_T(phi: float, kappa: float, theta: float, s: float | None = None) -> np.ndarray:
    """Constant-curvature transform with optional straight-segment fallback length s."""
    T = np.eye(4, dtype=float)
    eps = 1e-12

    if abs(theta) < eps or abs(kappa) < eps:
        if s is None:
            s = (theta / kappa) if abs(kappa) > eps else 0.0
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

def make_touch_prebend_passive(theta1_deg, phi1_deg,
                               theta2_deg, phi2_deg,
                               L2_active, L2_passive,
                               name, bevel_angle_deg=45.0,
                               L1_active=0.05, L_rigid=0.003):
    th1, ph1 = np.deg2rad(theta1_deg), np.deg2rad(phi1_deg)
    th2, ph2 = np.deg2rad(theta2_deg), np.deg2rad(phi2_deg)

    k1 = 0.0 if abs(th1) < 1e-12 else th1 / L1_active
    T1 = _cc_T(ph1, k1, th1, s=L1_active)

    T2_pas  = np.eye(4); T2_pas[2,3] = float(L2_passive)
    k2 = 0.0 if abs(th2) < 1e-12 else th2 / L2_active
    T2_act  = _cc_T(ph2, k2, th2, s=L2_active)
    T2_rig  = np.eye(4); T2_rig[2,3] = float(L_rigid)

    T_tip = T1 @ T2_pas @ T2_act @ T2_rig
    R_tip, p_tip = T_tip[:3,:3], T_tip[:3,3]
    alpha = np.deg2rad(bevel_angle_deg)
    b_world = _normalize(R_tip @ np.array([np.sin(alpha), 0.0, np.cos(alpha)]))
    return (name, TouchPointSpec(coordinates=tuple(p_tip), normal=tuple(b_world)))
L1_ACTIVE = 0.05          # outer bendable 5 cm
L2A_MAX   = 0.006         # inner active ≤ 6 mm
L2P_MAX   = 0.006         # inner passive ≤ 6 mm
L_RIGID   = 0.003         # inner rigid tip 3 mm


_BASE_CASES = [
    #   θ1,  φ1,   θ2,  φ2,   L2a,    L2p,   name
    # NOTE: 1 ≤ θ2/θ2 =0 or no solution
    (+1, 60, +1, 46, 0.006, 0.002, "N01"),
    (-10.0,  30.0, +22.0,  60.0, 0.006, 0.001, "N02"),
    (+14.0,  60.0, +24.0, 120.0, 0.006, 0.003, "N03"),
    (-12.0,  90.0, +20.0, 180.0, 0.005, 0.004, "N04"),
    (+16.0, 120.0, +22.0, 240.0, 0.006, 0.003, "N05"),
    ( +8.0, 150.0, +18.0, 300.0, 0.006, 0.003, "N06"),
    (+10.0, 180.0, +20.0,  30.0, 0.006, 0.003, "N07"),
    (-14.0, 210.0, +24.0,  90.0, 0.006, 0.003, "N08"),
    (+18.0, 240.0, +25.0, 150.0, 0.006, 0.003, "N09"),
    (-16.0, 270.0, +22.0, 210.0, 0.005, 0.004, "N10"),
    (+12.0, 300.0, +20.0, 270.0, 0.006, 0.003, "N11"),
    (+10.0, 330.0, +18.0, 330.0, 0.006, 0.003, "N12"),
]

_EDGE_CASES = [
    (+18.0,   0.0, +26.0,   0.0, 0.006, 0.006, "NE01"),
    (+20.0,  30.0, +28.0,  60.0, 0.006, 0.006, "NE02"),
    (-18.0,  60.0, +26.0, 120.0, 0.006, 0.006, "NE03"),
    (+22.0,  90.0, +28.0, 180.0, 0.006, 0.006, "NE04"),
    (-20.0, 120.0, +26.0, 240.0, 0.006, 0.006, "NE05"),
    (+16.0, 150.0, +24.0, 300.0, 0.006, 0.006, "NE06"),
    (+18.0, 180.0, +26.0,  30.0, 0.006, 0.006, "NE07"),
    (-22.0, 210.0, +28.0,  90.0, 0.006, 0.006, "NE08"),
    (+24.0, 240.0, +30.0, 150.0, 0.006, 0.006, "NE09"),
    (-24.0, 270.0, +28.0, 210.0, 0.006, 0.006, "NE10"),
    (+20.0, 300.0, +26.0, 270.0, 0.006, 0.006, "NE11"),
    (+22.0, 330.0, +28.0, 330.0, 0.006, 0.006, "NE12"),
]

def _build_one(case):
    th1, ph1, th2, ph2, L2a, L2p, nm = case
    return make_touch_prebend_passive(
        th1, ph1, th2, ph2, float(L2a), float(L2p), nm,
        bevel_angle_deg=45.0, L1_active=L1_ACTIVE, L_rigid=L_RIGID
    )

STRICT_BEVEL_POINTS_45_NEW      = [_build_one(c) for c in _BASE_CASES]
STRICT_BEVEL_POINTS_45_EDGE_NEW = [_build_one(c) for c in _EDGE_CASES]

(N01, N02, N03, N04, N05, N06, N07, N08, N09, N10, N11, N12) = STRICT_BEVEL_POINTS_45_NEW
(NE01, NE02, NE03, NE04, NE05, NE06, NE07, NE08, NE09, NE10, NE11, NE12) = STRICT_BEVEL_POINTS_45_EDGE_NEW

B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12 = (
    N01, N02, N03, N04, N05, N06, N07, N08, N09, N10, N11, N12
)

TEST_POINTS = STRICT_BEVEL_POINTS_45_NEW + STRICT_BEVEL_POINTS_45_EDGE_NEW
