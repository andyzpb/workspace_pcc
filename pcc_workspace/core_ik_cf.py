from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pcc_workspace.specs import (
    IKSolution,
    SegmentSolution,
    TranslationSpec,
    SegmentSpec,
    TouchPointSpec,
    IKOptions,
)

# ---------------- small utils ----------------
EPS = 1e-9
DEG = math.pi / 180.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)


def _wrap_0_2pi(a: float) -> float:
    a = float(a) % (2.0 * math.pi)
    return a if a >= 0.0 else (a + 2.0 * math.pi)


def _canon_theta_phi(theta: float, phi: float) -> Tuple[float, float]:
    # canonical representation with non-negative bend and φ wrapped to [0, 2π)
    if theta < 0.0:
        theta = -theta
        phi = phi + math.pi
    return float(theta), float(_wrap_0_2pi(phi))


def _rotz(a: float) -> np.ndarray:
    c, s = float(math.cos(a)), float(math.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _roty(a: float) -> np.ndarray:
    c, s = float(math.cos(a)), float(math.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _axis_angle_R(u: np.ndarray, theta: float) -> np.ndarray:
    u = _normalize(np.asarray(u, float))
    c, s = float(math.cos(theta)), float(math.sin(theta))
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=float)
    I = np.eye(3, dtype=float)
    return c * I + (1.0 - c) * np.outer(u, u) + s * K


def _cc_transform(phi: float, kappa: float, theta: float, s_hint: float | None = None) -> np.ndarray:
    eps = 1e-12
    T = np.eye(4, dtype=float)

    R = _rotz(phi) @ _roty(theta) @ _rotz(-phi)

    if (abs(theta) > eps) and (abs(kappa) > eps):
        s = float(theta / kappa)
    else:
        s = float(s_hint or 0.0)

    t = float(theta)
    if abs(t) < 1e-6:
        A = 0.5 * t - (t**3) / 24.0
        B = 1.0 - (t**2) / 6.0 + (t**4) / 120.0
    else:
        A = (1.0 - math.cos(t)) / t
        B = math.sin(t) / t

    p0 = np.array([s * A, 0.0, s * B], dtype=float) 
    p  = _rotz(phi) @ p0

    T[:3, :3] = R
    T[:3, 3]  = p
    return T



def _snap_to_box(
    x: float, lo: Optional[float], hi: Optional[float], tol: float
) -> Optional[float]:
    """
    Inclusive box with soft snapping:
      - if x < lo - tol or x > hi + tol -> infeasible (return None)
      - else clamp to [lo, hi] if within tol band
    """
    if lo is not None and x < lo - tol:
        return None
    if hi is not None and x > hi + tol:
        return None
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return float(x)


def _project_angle_to_interval(
    phi: float, lo: Optional[float], hi: Optional[float]
) -> float:
    """
    Project φ to the nearest point in [lo, hi] modulo 2π. If missing or ~2π span, just wrap.
    """
    if lo is None or hi is None:
        return _wrap_0_2pi(phi)
    lo = _wrap_0_2pi(float(lo))
    hi = _wrap_0_2pi(float(hi))
    span = (hi - lo) % (2.0 * math.pi)
    if span >= 2.0 * math.pi - 1e-9:
        return _wrap_0_2pi(phi)
    x = (_wrap_0_2pi(phi) - lo) % (2.0 * math.pi)
    if x <= span:
        return lo + x
    to_lo = (2.0 * math.pi - x) % (2.0 * math.pi)
    to_hi = (x - span) % (2.0 * math.pi)
    return hi if to_hi <= to_lo else lo


def _angle_in_range_snap(
    phi_wrapped: float, lo: Optional[float], hi: Optional[float], tol: float
) -> bool:
    """Check φ is inside [lo, hi] modulo 2π with ±tol."""
    if lo is None or hi is None:
        return True
    lo = float(lo) % (2.0 * math.pi)
    hi = float(hi) % (2.0 * math.pi)
    span = (hi - lo) % (2.0 * math.pi)
    if span >= 2.0 * math.pi - 1e-9:
        return True
    x = (phi_wrapped - lo) % (2.0 * math.pi)
    return (x <= span + tol) or (abs(x - (span + tol)) <= 1e-12)


def _angle_err_deg(cosang: float, angle_target_deg: float) -> float:
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    return abs(ang - angle_target_deg)


def _nms_diversity(
    items: List[Dict[str, Any]],
    theta_thresh_deg: float = 2.0,
    phi_thresh_deg: float = 5.0,
    trans_thresh: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Simple NMS by (θ1, θ2, φ1, φ2, d) approximate closeness."""
    kept: List[Dict[str, Any]] = []

    def _feat(it: Dict[str, Any]):
        segs: List["SegmentSolution"] = it["segments"]
        thetas = [s.theta for s in segs]
        phis = []
        for s in segs:
            phis.append(0.0 if abs(s.theta) < 1e-3 or s.L_active < 1e-4 else s.phi)
        return np.array(
            [thetas[0], thetas[1], phis[0], phis[1], it["translation"]], dtype=float
        )

    def _close(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        va, vb = _feat(a), _feat(b)
        dtheta = np.max(np.abs(np.degrees(va[:2] - vb[:2])))
        dphi = np.max(np.abs(np.degrees(va[2:4] - vb[2:4])))
        dtrans = abs(va[-1] - vb[-1])
        return (
            dtheta <= theta_thresh_deg
            and dphi <= phi_thresh_deg
            and dtrans <= trans_thresh
        )

    for s in items:
        if not any(_close(s, t) for t in kept):
            kept.append(s)
    return kept


# ---------------- helper blocks (outer pose, small-angle series) ----------------
def _A_B_and_derivs(th: float):
    """Stable series for A=(1-cos t)/t, B=sin t/t and their derivs."""
    t = float(th)
    if abs(t) < 1e-6:
        A = 0.5 * t - (t**3) / 24.0
        B = 1.0 - (t**2) / 6.0 + (t**4) / 120.0
        Ap = 0.5 - (t**2) / 8.0
        Bp = -t / 3.0 + (t**3) / 30.0
    else:
        A = (1.0 - math.cos(t)) / t
        B = math.sin(t) / t
        Ap = (t * math.sin(t) - (1.0 - math.cos(t))) / (t * t)
        Bp = (t * math.cos(t) - math.sin(t)) / (t * t)
    return A, B, Ap, Bp


def _build_T1_from_theta_phi(
    s1_active: float, L1_passive: float, theta1: float, phi1: float
) -> Tuple[np.ndarray, np.ndarray]:
    """T1 = T_passive(outer) * T_bend(outer); returns (T1, P1)."""
    k1 = 0.0 if abs(theta1) < 1e-12 or abs(s1_active) < 1e-12 else (theta1 / s1_active)
    T_pass = np.eye(4, dtype=float)
    T_pass[2, 3] = float(L1_passive)
    T_bend = _cc_transform(phi1, k1, theta1, s_hint=s1_active)
    T1 = T_pass @ T_bend
    P1 = T1[:3, 3].copy()
    return T1, P1


# ---------------- IK Solver ----------------
class PCCIKClosedForm:
    """
    Two-section PCC: Closed-form core + small GN polish
    - inner: Passive(before bend) -> Active(arc) -> Rigid tip (e.g. 3mm)
    - ACTIVE-FIRST + TOTAL-LENGTH enforced
    - bevel determined by (θ1,φ1,θ2,φ2); no roll DOF
    - translation 'd' is along world +z; end_T/end_p exclude d
    """

    def __init__(
        self,
        segments: List[SegmentSpec],
        translation: Optional[TranslationSpec] = None,
        opts: Optional[IKOptions] = None,
        inner_rigid_tip: float = 0.003,
        debug: bool = False,
    ):
        if len(segments) != 2:
            raise ValueError("This solver expects exactly TWO segments.")
        # identify inner/outer
        inner_idx = None
        for i, s in enumerate(segments):
            if getattr(s, "is_inner", False):
                inner_idx = i
                break
        if inner_idx is None:
            # fallback: smaller active cap is inner
            a0 = float(getattr(segments[0], "active_L_max", segments[0].L_max or 1e9))
            a1 = float(getattr(segments[1], "active_L_max", segments[1].L_max or 1e9))
            inner_idx = 0 if a0 <= a1 else 1
        outer_idx = 1 - inner_idx

        self.seg1: SegmentSpec = segments[outer_idx]  # outer
        self.seg2: SegmentSpec = segments[inner_idx]  # inner

        self.translation = translation or TranslationSpec(0.0, 0.0, samples=1)
        self.opts = opts or IKOptions()
        self.debug = bool(debug)

        # if outer total/passive fixed, active is fixed
        wL = abs(self.seg1.L_max - self.seg1.L_min)
        wP = abs(self.seg1.passive_L_max - self.seg1.passive_L_min)
        self.s1_active_fixed: Optional[float] = None
        if wL <= 1e-9 and wP <= 1e-9:
            self.s1_active_fixed = float(self.seg1.L_max - self.seg1.passive_L_max)

        self.inner_rigid_tip = float(inner_rigid_tip)
        if self.inner_rigid_tip < 0:
            raise ValueError("inner_rigid_tip must be non-negative")

    # ---------- small matrix blocks for Jacobians ----------
    def _skew(self, v: np.ndarray) -> np.ndarray:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)

    def _Rz(self, a: float) -> np.ndarray:
        return _rotz(a)

    def _Ry(self, a: float) -> np.ndarray:
        return _roty(a)

    def _dR_cc_dphi(self, phi: float, th: float) -> np.ndarray:
        R = self._Rz(phi) @ self._Ry(th) @ self._Rz(-phi)
        S = self._skew(np.array([0.0, 0.0, 1.0], dtype=float))
        return S @ R - R @ S

    def _dR_cc_dtheta(self, phi: float, th: float) -> np.ndarray:
        R = self._Rz(phi) @ self._Ry(th) @ self._Rz(-phi)
        S_ey = self._skew(np.array([0.0, 1.0, 0.0], dtype=float))
        M = self._Rz(phi) @ S_ey @ self._Rz(-phi)
        return R @ M

    # ---------- Jacobian (p, bevel) with d ----------
    def jacobian_analytic(self, x):
        th1, ph1, th2, ph2, s2, L2p, d = [float(v) for v in x]

        ez = np.array([0.0, 0.0, 1.0], float)
        alpha = float(np.deg2rad(getattr(self.seg2, "bevel_angle_deg", 45.0)))
        b0 = np.array([math.sin(alpha), 0.0, math.cos(alpha)], float)
        Lrig = float(getattr(self.seg2, "rigid_tip_length", self.inner_rigid_tip))

        # outer effective lengths (s1, L1p)
        if self.s1_active_fixed is not None:
            s1 = float(self.s1_active_fixed)
        else:
            s1 = max(
                0.0,
                0.5
                * (
                    (self.seg1.L_min + self.seg1.L_max)
                    - (self.seg1.passive_L_min + self.seg1.passive_L_max)
                ),
            )
        L1p = float(
            self.seg1.passive_L_max
            if abs(self.seg1.passive_L_max - self.seg1.passive_L_min) <= 1e-9
            else 0.5 * (self.seg1.passive_L_min + self.seg1.passive_L_max)
        )

        # rotations
        R1 = self._Rz(ph1) @ self._Ry(th1) @ self._Rz(-ph1)
        R2 = self._Rz(ph2) @ self._Ry(th2) @ self._Rz(-ph2)
        dR1_dth = self._dR_cc_dtheta(ph1, th1)
        dR1_dph = self._dR_cc_dphi(ph1, th1)
        dR2_dth = self._dR_cc_dtheta(ph2, th2)
        dR2_dph = self._dR_cc_dphi(ph2, th2)

        # arc small-angle series
        A1, B1, A1p, B1p = _A_B_and_derivs(th1)
        v1 = np.array([s1 * A1, 0.0, s1 * B1], float)
        dv1_dth = np.array([s1 * A1p, 0.0, s1 * B1p], float)

        A2, B2, A2p, B2p = _A_B_and_derivs(th2)
        v2 = np.array([s2 * A2, 0.0, s2 * B2], float)
        dv2_dth = np.array([s2 * A2p, 0.0, s2 * B2p], float)
        dv2_ds = np.array([A2, 0.0, B2], float)

        # FK blocks
        p1 = L1p * ez + _rotz(ph1) @ v1
        q = (L2p * ez) + (_rotz(ph2) @ v2) + (R2 @ (Lrig * ez))
        p = p1 + R1 @ q + d * ez
        b = _normalize((R1 @ R2) @ b0)

        # --- Jacobians ---
        Jp = np.zeros((3, 7), float)
        Jp[:, 0] = (_rotz(ph1) @ dv1_dth) + (dR1_dth @ q)  # d/d θ1
        Jp[:, 1] = (_rotz(ph1) @ (self._skew(ez) @ v1)) + (dR1_dph @ q)  # d/d φ1
        Jp[:, 2] = R1 @ ((_rotz(ph2) @ dv2_dth) + (dR2_dth @ (Lrig * ez)))  # d/d θ2
        Jp[:, 3] = R1 @ ((_rotz(ph2) @ (self._skew(ez) @ v2)) + (dR2_dph @ (Lrig * ez)))  # d/d φ2
        Jp[:, 4] = R1 @ (_rotz(ph2) @ dv2_ds)  # d/d s2
        Jp[:, 5] = R1 @ ez  # d/d L2p
        Jp[:, 6] = ez  # d/d d

        Jb = np.zeros((3, 7), float)
        Rtip = R1 @ R2
        Jb[:, 0] = (dR1_dth @ R2) @ b0
        Jb[:, 1] = (dR1_dph @ R2) @ b0
        Jb[:, 2] = (R1 @ dR2_dth) @ b0
        Jb[:, 3] = (R1 @ dR2_dph) @ b0
        # project to tangent space of unit vector b
        Pb = (np.eye(3) - np.outer(b, b))
        Jb = Pb @ Jb

        return p, b, Jp, Jb

    # ---------- φ1 local refinement (golden-section) ----------
    def _refine_phi1_local(
        self,
        theta1_val: float,
        phi1_seed: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        pos_tol: float,
        L2a_max: float,
        L2p_min: float,
        L2p_max: float,
        L_rigid: float,
        alpha: float,
        d0: float,
        halfspan_deg: float = 8.0,
        iters: int = 80,
    ):
        """1D golden-section search on φ1 around seed; returns (best_phi, best_cand)."""
        def score(ph):
            cand = self._evaluate_once_cidgik(
                theta1_val, ph, P_star, n_star, pos_tol,
                L2a_max, L2p_min, L2p_max, L_rigid, alpha, d0,
                do_qp=False,
            )
            if cand is None:
                return 1e9, None
            J = cand["pos_err"] + 1e-6 * cand["ang_err_deg"]
            return J, cand

        gr = 0.6180339887498949
        span = math.radians(halfspan_deg)
        a = phi1_seed - span
        b = phi1_seed + span
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, cand_c = score(c)
        fd, cand_d = score(d)
        for _ in range(iters):
            if (b - a) < 1e-7:
                break
            if fc <= fd:
                b, d, fd, cand_d = d, c, fc, cand_c
                c = b - gr * (b - a)
                fc, cand_c = score(c)
            else:
                a, c, fc, cand_c = c, d, fd, cand_d
                d = a + gr * (b - a)
                fd, cand_d = score(d)
        if fc <= fd:
            return c, cand_c
        else:
            return d, cand_d

    # ---------- A. Orientation closed-form for inner (no-roll) ----------
    @staticmethod
    def _inner_angles_from_orientation_closed(
        R1: np.ndarray, n_star: np.ndarray, alpha: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Given R1 (outer), target normal n*, bevel angle alpha, solve inner (φ2, θ2).
        """
        n_prime = R1.T @ _normalize(n_star)
        sa, ca = float(math.sin(alpha)), float(math.cos(alpha))
        # φ2
        phi2 = math.atan2(float(n_prime[1]), float(n_prime[0]) - sa)
        c, s = float(math.cos(phi2)), float(math.sin(phi2))
        # u = Rz(-φ2) b0, w = Rz(-φ2) n'
        u = np.array([sa * c, -sa * s, ca], float)
        w = np.array(
            [
                c * float(n_prime[0]) + s * float(n_prime[1]),
                -s * float(n_prime[0]) + c * float(n_prime[1]),
                float(n_prime[2]),
            ],
            float,
        )
        ux, uz = float(u[0]), float(u[2])
        wx, wz = float(w[0]), float(w[2])
        den = (ux * ux + uz * uz) if (ux * ux + uz * uz) > 1e-16 else 1e-16
        cos_th = (ux * wx + uz * wz) / den
        sin_th = (uz * wx - ux * wz) / den
        theta2 = math.atan2(sin_th, cos_th)
        if abs(theta2) < np.deg2rad(0.5):
            R2 = np.eye(3)
            z2 = np.array([0.0, 0.0, 1.0], float)
            return float(_wrap_0_2pi(phi2)), 0.0, z2, R2
        R2 = _rotz(phi2) @ _roty(theta2) @ _rotz(-phi2)
        z2 = R2[:, 2].copy()
        return float(_wrap_0_2pi(phi2)), float(theta2), z2, R2

    # ---------- B. Inner lengths from target position (with rigid tip) ----------
    @staticmethod
    def _inner_lengths_from_position_prebend_rigid(
        R1, p1, phi2, theta2, P_star, L_rigid,
        L2a_max, L2p_min, L2p_max, active_first=True,
        th_free_eps: float = np.deg2rad(7),
    ):
        q  = R1.T @ (P_star - p1)
        a  = _rotz(phi2) @ np.array([1.0 - math.cos(theta2), 0.0, math.sin(theta2)], float)
        R2 = _rotz(phi2) @ _roty(theta2) @ _rotz(-phi2)
        z2 = R2[:, 2]
        q2 = q - L_rigid * z2

        a_xy = a[:2]; q2_xy = q2[:2]
        denom = float(a_xy @ a_xy)

        if (denom >= 1e-16) and (abs(theta2) >= th_free_eps):
            r = float(q2_xy @ a_xy) / denom
            s2 = float(theta2 * r)
            L2p = float(q2[2] - r * a[2])
            return s2, L2p

        rem = float(q2[2])           
        rem = max(0.0, rem)          
        if active_first:
            s2  = min(rem, max(0.0, L2a_max))
            L2p = rem - s2
        else:
            L2p = min(max(L2p_min, rem), L2p_max)
            s2  = rem - L2p
        return float(s2), float(L2p)


    # ---------- C. Gauss-Newton + Armijo line search + hard projections ----------
    def _refine_qp(
        self,
        theta1,
        phi1,
        theta2,
        phi2,
        s2,
        L2p,
        d,
        P_star: np.ndarray,
        n_star: np.ndarray,
        L2a_max: float,
        L2p_min: float,
        L2p_max: float,
        L_rigid: float,
        alpha: float,
        d_min: float,
        d_max: float,
        iters: int = 5,
    ):
        wb, lam = 20.0, 1e-4
        tol_th = np.deg2rad(1e-4)
        tol_L = 5e-7
        tol_d = 1e-7

        def obj(xvec):
            p, b, _, _ = self.jacobian_analytic(xvec)
            rp = P_star - p
            rb = n_star - b
            return float(rp @ rp + wb * (rb @ rb))

        def project_inplace(xv: np.ndarray):
            def _hard_clip(val, lo, hi):
                if lo is not None and val < lo: return float(lo)
                if hi is not None and val > hi: return float(hi)
                return float(val)

            # θ1
            tmp = _snap_to_box(xv[0], getattr(self.seg1, "theta_min", None), getattr(self.seg1, "theta_max", None), tol_th)
            xv[0] = _hard_clip(xv[0], getattr(self.seg1, "theta_min", None), getattr(self.seg1, "theta_max", None)) if tmp is None else tmp

            # φ1
            if getattr(self.seg1, "phi_min", None) is not None and getattr(self.seg1, "phi_max", None) is not None:
                xv[1] = _project_angle_to_interval(xv[1], float(self.seg1.phi_min), float(self.seg1.phi_max))
            else:
                xv[1] = _wrap_0_2pi(xv[1])

            # θ2
            tmp = _snap_to_box(xv[2], getattr(self.seg2, "theta_min", None), getattr(self.seg2, "theta_max", None), tol_th)
            xv[2] = _hard_clip(xv[2], getattr(self.seg2, "theta_min", None), getattr(self.seg2, "theta_max", None)) if tmp is None else tmp

            # φ2
            if getattr(self.seg2, "phi_min", None) is not None and getattr(self.seg2, "phi_max", None) is not None:
                xv[3] = _project_angle_to_interval(xv[3], float(self.seg2.phi_min), float(self.seg2.phi_max))
            else:
                xv[3] = _wrap_0_2pi(xv[3])

            tmp = _snap_to_box(xv[4], 0.0, L2a_max, tol_L)
            xv[4] = _hard_clip(xv[4], 0.0, L2a_max) if tmp is None else tmp

            tmp = _snap_to_box(xv[5], L2p_min, L2p_max, tol_L)
            xv[5] = _hard_clip(xv[5], L2p_min, L2p_max) if tmp is None else tmp

            L2_total_max = float(self.seg2.L_max if self.seg2.L_max is not None else 1e9)
            if (xv[4] + xv[5]) > L2_total_max + tol_L:
                xv[4] = min(xv[4], L2a_max, L2_total_max)
                xv[5] = max(0.0, min(L2p_max, L2_total_max - xv[4]))

            # ACTIVE-FIRST: if s2 < cap - tol, force L2p = 0; otherwise leave L2p clamped
            margin = float(getattr(self.opts, "active_first_tol", 5e-4))
            margin = float(min(margin, 0.5 * L2a_max))

            if getattr(self.opts, "active_first", True):
                if xv[4] < (L2a_max - margin):
                    xv[5] = 0.0

            # d
            ds = _snap_to_box(xv[6], d_min, d_max, tol_d)
            if ds is not None:
                xv[6] = ds

        x = np.array([theta1, phi1, theta2, phi2, s2, L2p, d], float)
        project_inplace(x)

        for _ in range(iters):
            p, b, Jp, Jb = self.jacobian_analytic(x.copy())
            rp = P_star - p
            rb = n_star - b
            A = Jp.T @ Jp + wb * (Jb.T @ Jb) + lam * np.eye(7)
            g = Jp.T @ rp + wb * (Jb.T @ rb)

            try:
                delta = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, g, rcond=None)[0]

            f0 = obj(x)
            t = 1.0
            accepted = False
            for _ls in range(8):
                x_try = x + t * delta
                project_inplace(x_try)
                f1 = obj(x_try)
                if f1 <= f0 - 1e-4 * t * float(delta @ delta):
                    x = x_try
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                x += 0.1 * delta
                project_inplace(x)
                break

        return tuple(x.tolist())

    # ---------- seeds for θ1 ----------
    def _theta1_candidates_adaptive(
        self,
        P_world,
        n_star,
        th_lo,
        th_hi,
        coarse_n=61,
        keep_top=8,
        refine_halfspan_deg=6.0,
        refine_n=25,
    ):
        seg2 = self.seg2
        L2_total_max = float(seg2.L_max if seg2.L_max is not None else 1e9)
        act_cap = float(getattr(seg2, "active_L_max", L2_total_max))
        L2a_max = min(act_cap, L2_total_max)
        L2p_min, L2p_max = float(seg2.passive_L_min), float(seg2.passive_L_max)
        L_rigid = float(getattr(seg2, "rigid_tip_length", self.inner_rigid_tip))
        alpha = float(np.deg2rad(getattr(seg2, "bevel_angle_deg", 45.0)))
        theta_grid = np.linspace(th_lo, th_hi, int(max(3, coarse_n)), dtype=float)

        def score(theta1: float) -> float:
            best = float("+inf")
            d0 = 0.5 * (self.translation.d_min + self.translation.d_max)
            for phi1 in np.linspace(-math.pi, math.pi, 360, endpoint=False):
                cand = self._evaluate_once_cidgik(
                    theta1,
                    phi1,
                    P_world,
                    n_star,
                    float(self.opts.pos_tol),
                    L2a_max,
                    L2p_min,
                    L2p_max,
                    L_rigid,
                    alpha,
                    d0,
                    do_qp=False,
                )
                if cand is None:
                    continue
                J = cand["pos_err"] + 0.001 * cand["ang_err_deg"]
                best = min(best, J)
            return best

        scores = np.array([score(th) for th in theta_grid], float)
        order = np.argsort(scores)
        seeds = theta_grid[order[: int(max(1, keep_top))]]
        out = set()
        half = float(np.deg2rad(refine_halfspan_deg))
        for s in seeds:
            a = max(th_lo, s - half)
            b = min(th_hi, s + half)
            for x in np.linspace(a, b, int(max(3, refine_n))):
                out.add(float(x))
        out.update([+1e-3, -1e-3])  # cover near-straight
        return sorted(out)

    # ---------- A + optional B: evaluate one candidate ----------
    def _evaluate_once_cidgik(
        self,
        theta1,
        phi1,
        P_target_world,
        n_star_world,
        pos_tol,
        L2a_max,
        L2p_min,
        L2p_max,
        L_rigid,
        alpha,
        d: float,
        do_qp: bool,
    ):
        seg1 = self.seg1
        if self.s1_active_fixed is not None:
            s1 = float(self.s1_active_fixed)
        else:
            s1 = max(
                0.0,
                0.5
                * (
                    (seg1.L_min + seg1.L_max)
                    - (seg1.passive_L_min + seg1.passive_L_max)
                ),
            )
        Lpass1 = float(
            seg1.passive_L_max
            if abs(seg1.passive_L_max - seg1.passive_L_min) <= 1e-9
            else 0.5 * (seg1.passive_L_min + seg1.passive_L_max)
        )
        T1, P1 = _build_T1_from_theta_phi(s1, Lpass1, theta1, phi1)
        R1 = T1[:3, :3]
        z1 = _normalize(R1[:, 2])

        # orientation closed-form for inner
        phi2, theta2, z2, R2 = self._inner_angles_from_orientation_closed(
            R1, n_star_world, alpha
        )

        # inner angle boxes
        seg2 = self.seg2
        tol_th = np.deg2rad(1e-4)
        theta2_snapped = _snap_to_box(
            theta2, getattr(seg2, "theta_min", None), getattr(seg2, "theta_max", None), tol_th
        )
        if theta2_snapped is None:
            return None
        theta2 = theta2_snapped

        phi2_wrapped = _wrap_0_2pi(phi2)
        tol_phi = np.deg2rad(1e-4)
        if (getattr(seg2, "phi_min", None) is not None) and (getattr(seg2, "phi_max", None) is not None):
            if not _angle_in_range_snap(phi2_wrapped, float(seg2.phi_min), float(seg2.phi_max), tol_phi):
                return None

        # position closed-form (subtract d on target side)
        P_star_d = P_target_world - np.array([0.0, 0.0, d], float)
        s2, L2_passive = self._inner_lengths_from_position_prebend_rigid(
            R1, P1, phi2_wrapped, theta2, P_star_d, L_rigid,
            L2a_max, L2p_min, L2p_max, getattr(self.opts, "active_first", True)  # ★
        )

        # length boxes + ACTIVE-FIRST + TOTAL-LENGTH
        tol_L = 5e-4
        s2 = _snap_to_box(s2, 0.0, L2a_max, tol_L)
        if s2 is None:
            return None
        L2_passive = _snap_to_box(L2_passive, L2p_min, L2p_max, tol_L)
        if L2_passive is None:
            return None

        margin = float(getattr(self.opts, "active_first_tol", 5e-4))
        margin = float(min(margin, 0.5 * L2a_max))
        if getattr(self.opts, "active_first", True):
            if s2 < (L2a_max - margin):
                L2_passive = 0.0

        L2_total_max = float(seg2.L_max if seg2.L_max is not None else 1e9)
        if (s2 + L2_passive) > L2_total_max + tol_L:
            s2 = min(s2, L2a_max, L2_total_max)
            L2_passive = max(0.0, min(L2p_max, L2_total_max - s2))

        # compose and check
        k2 = 0.0 if abs(theta2) < 1e-12 or abs(s2) < 1e-12 else (theta2 / s2)
        T2_act = _cc_transform(phi2_wrapped, k2, theta2,s_hint=s2)
        T_pass = np.eye(4); T_pass[2, 3] = float(L2_passive)  # pre-bend passive
        T_rig  = np.eye(4); T_rig[2, 3]  = float(L_rigid)     # post-bend rigid
        T_tip = T1 @ T_pass @ T2_act @ T_rig

        end_p = T_tip[:3, 3]
        pos_err = float(np.linalg.norm(end_p - P_star_d))

        # gate for QP
        gate = (pos_tol) if do_qp else pos_tol
        if pos_err > gate + 1e-9:
            return None

        # bevel & axis checks
        b0_tip = np.array([math.sin(alpha), 0.0, math.cos(alpha)], float)
        b_world = _normalize(T_tip[:3, :3] @ b0_tip)
        cb = float(np.clip(b_world @ n_star_world, -1.0, 1.0))
        cos_bevel_tol = float(np.cos(np.deg2rad(getattr(self.opts, "bevel_tol_deg", 1.0))))
        if cb < cos_bevel_tol:
            return None

        z_hat = _normalize(T_tip[:3, 2])
        cos_axis = float(np.clip(z_hat @ n_star_world, -1.0, 1.0))
        ang_err_deg = _angle_err_deg(cos_axis, getattr(self.opts, "angle_target_deg", 45.0))
        bevel_err_deg = float(np.degrees(np.arccos(cb)))

        # optional QP polish (short-circuit BEFORE calling QP)
        if do_qp:
            if (pos_err <= max(1e-12, 1e-6 * pos_tol)) and (bevel_err_deg <= 1e-9):
                pass  # already perfect; keep as-is
            else:
                theta1, phi1, theta2, phi2_wrapped, s2, L2_passive, d = self._refine_qp(
                    theta1, phi1, theta2, phi2_wrapped, s2, L2_passive, d,
                    P_target_world, n_star_world,
                    L2a_max, L2p_min, L2p_max, L_rigid, alpha,
                    d_min=float(self.translation.d_min), d_max=float(self.translation.d_max),
                    iters=5,
                )
                # recompute with refined x
                T1, P1 = _build_T1_from_theta_phi(s1, Lpass1, theta1, phi1)
                k2 = 0.0 if abs(theta2) < 1e-12 or abs(s2) < 1e-12 else (theta2 / s2)
                T2_act = _cc_transform(phi2_wrapped, k2, theta2)
                T_pass = np.eye(4); T_pass[2, 3] = float(L2_passive)
                T_tip = T1 @ T_pass @ T2_act @ T_rig
                end_p = T_tip[:3, 3]
                P_star_d = P_target_world - np.array([0.0, 0.0, d], float)
                pos_err = float(np.linalg.norm(end_p - P_star_d))
                b_world = _normalize(T_tip[:3, :3] @ b0_tip)
                cb = float(np.clip(b_world @ n_star_world, -1.0, 1.0))
                z_hat = _normalize(T_tip[:3, 2])
                cos_axis = float(np.clip(z_hat @ n_star_world, -1.0, 1.0))
                ang_err_deg = _angle_err_deg(cos_axis, getattr(self.opts, "angle_target_deg", 45.0))
                bevel_err_deg = float(np.degrees(np.arccos(cb)))

        # canonicalize output angles for human-readable/report consistency
        theta1_out, phi1_out = _canon_theta_phi(theta1, phi1)
        theta2_out, phi2_out = _canon_theta_phi(theta2, phi2_wrapped)

        # pack
        seg_solutions = [
            SegmentSolution(
                name=self.seg1.name,
                theta=float(theta1_out),
                phi=float(phi1_out),
                L_total=float(s1 + Lpass1),
                L_passive=float(Lpass1),
                L_active=float(s1),
            ),
            SegmentSolution(
                name=self.seg2.name,
                theta=float(theta2_out),
                phi=float(phi2_out),
                L_total=float(s2 + L2_passive),
                L_passive=float(L2_passive),
                L_active=float(s2),
            ),
        ]

        meta = dict(
            cos_angle=cb,
            z_axis=z_hat.copy(),
            inner_rigid_tip=float(L_rigid),
            end_p_world=(end_p + np.array([0.0, 0.0, d], float)).copy(),
            bevel_angle_deg=float(np.degrees(alpha)),
            bevel_err_deg=bevel_err_deg,
            bevel_world=b_world.copy(),
            plane_normal_world=_normalize(np.cross(z1, _normalize(n_star_world))),
            report=dict(
                outer=dict(
                    bending_deg=float(np.degrees(theta1_out)),
                    rotation_deg=float(np.degrees(phi1_out)),
                    translation=0.0,
                ),
                inner=dict(
                    bending_deg=float(np.degrees(theta2_out)),
                    rotation_deg=float(np.degrees(phi2_out)),
                    translation=float(s2 + L2_passive),
                ),
            ),
        )

        return dict(
            pos_err=pos_err,
            ang_err_deg=ang_err_deg,
            translation=float(d),
            segments=seg_solutions,
            end_T=T_tip.copy(),
            end_p=end_p.copy(),
            meta=meta,
        )

    # ---------------- sampling helpers ----------------
    def _sample_translations(self, tr: "TranslationSpec") -> List[float]:
        if tr.samples <= 1:
            return [0.5 * (tr.d_min + tr.d_max)]
        if tr.samples == 2:
            return [tr.d_min, tr.d_max]
        if tr.d_step is not None and tr.d_step > 0:
            n = int(np.floor((tr.d_max - tr.d_min) / tr.d_step + 1e-9)) + 1
            vs = tr.d_min + tr.d_step * np.arange(n, dtype=float)
            if len(vs) == 0 or abs(vs[-1] - tr.d_max) > 1e-6:
                vs = np.concatenate([vs, np.array([tr.d_max])])
            return vs.tolist()
        return np.linspace(tr.d_min, tr.d_max, tr.samples).tolist()

    # ---------------- main solve ----------------
    def solve(self, target: "TouchPointSpec") -> List["IKSolution"]:
        P_star = np.asarray(target.coordinates, float).reshape(3)
        n_star = _normalize(np.asarray(target.normal, float).reshape(3))

        pos_tol = float(self.opts.pos_tol)
        topk = int(self.opts.topk)
        alpha = float(np.deg2rad(getattr(self.seg2, "bevel_angle_deg", 45.0)))

        # inner caps
        L2_total_max = float(self.seg2.L_max if self.seg2.L_max is not None else 1e9)
        act_cap = float(getattr(self.seg2, "active_L_max", L2_total_max))
        L2a_max = min(act_cap, L2_total_max)
        L2p_min, L2p_max = float(self.seg2.passive_L_min), float(self.seg2.passive_L_max)
        L_rigid = float(getattr(self.seg2, "rigid_tip_length", self.inner_rigid_tip))

        results: List[Dict[str, Any]] = []
        d_seeds = self._sample_translations(self.translation)

        # theta1 sampling window
        seg = self.seg1
        th_lo = float(seg.theta_min if seg.theta_min is not None else -seg.theta_max)
        th_hi = float(seg.theta_max)
        theta_seeds = self._theta1_candidates_adaptive(
            P_star, n_star, th_lo, th_hi,
            coarse_n=61, keep_top=8, refine_halfspan_deg=6.0, refine_n=25,
        )
        theta_seeds = sorted(set(theta_seeds + [th_lo, th_hi]))

        for d0 in d_seeds:
            for th_seed in theta_seeds:
                # local refine θ1 (golden section on best φ1 score)
                def score_at_theta(theta1_val: float):
                    best = +1e9
                    for ph in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                        cand = self._evaluate_once_cidgik(
                            theta1_val, ph,
                            P_star, n_star, pos_tol,
                            L2a_max, L2p_min, L2p_max,
                            L_rigid, alpha, d0, do_qp=False,
                        )
                        if cand is None:
                            continue
                        J = cand["pos_err"] + 0.001 * cand["ang_err_deg"]
                        best = min(best, J)
                    return best

                def golden_min(f, a, b, tol=1e-4, mi=60):
                    gr = 0.6180339887498949
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    fc = f(c)
                    fd = f(d)
                    it = 0
                    while (b - a) > tol and it < mi:
                        it += 1
                        if fc <= fd:
                            b, d, fd = d, c, fc
                            c = b - gr * (b - a)
                            fc = f(c)
                        else:
                            a, c, fc = c, d, fd
                            d = a + gr * (b - a)
                            fd = f(d)
                    return 0.5 * (a + b)

                half = math.radians(6.0)
                a = max(th_lo, th_seed - half)
                b = min(th_hi, th_seed + half)
                theta1_ref = golden_min(score_at_theta, a, b)

                # φ1 coarse scan (denser near θ1 bounds)
                near_bound = (abs(theta1_ref - th_lo) < np.deg2rad(0.2)) or (abs(th_hi - theta1_ref) < np.deg2rad(0.2))
                n_phi = 360 if near_bound else 96
                phi_list = np.linspace(-math.pi, math.pi, n_phi, endpoint=False)

                pool: List[Tuple[float, Dict[str, Any]]] = []
                for ph in phi_list:
                    cand = self._evaluate_once_cidgik(
                        theta1_ref, ph,
                        P_star, n_star, pos_tol,
                        L2a_max, L2p_min, L2p_max,
                        L_rigid, alpha, d0, do_qp=False,
                    )
                    if cand is None:
                        continue
                    J = cand["pos_err"] + 0.001 * cand["ang_err_deg"]
                    pool.append((J, cand))

                # locally refine φ1 for the best few
                refined_pool = []
                for _, base in sorted(pool, key=lambda t: t[0])[:8]:
                    th1 = base["segments"][0].theta
                    ph1 = base["segments"][0].phi
                    _, cand_ref = self._refine_phi1_local(
                        th1, ph1,
                        P_star, n_star, pos_tol,
                        L2a_max, L2p_min, L2p_max,
                        L_rigid, alpha, d0,
                        halfspan_deg=8.0, iters=80
                    )
                    if cand_ref is not None:
                        refined_pool.append((cand_ref["pos_err"] + 1e-6 * cand_ref["ang_err_deg"], cand_ref))
                if refined_pool:
                    refined_pool.sort(key=lambda t: t[0])
                    pool = refined_pool  # replace with refined candidates

                # >>> append polished candidates to results <<<
                if not pool:
                    continue
                pool.sort(key=lambda t: t[0])
                for _, base in pool[:5]:
                    th1 = base["segments"][0].theta
                    ph1 = base["segments"][0].phi
                    d_init = base["translation"]
                    polished = self._evaluate_once_cidgik(
                        th1, ph1,
                        P_star, n_star, pos_tol,
                        L2a_max, L2p_min, L2p_max,
                        L_rigid, alpha, d_init, do_qp=True,
                    )
                    if polished is not None:
                        results.append(polished)

        if not results:
            return []

        results.sort(key=lambda x: (x["pos_err"], x["ang_err_deg"]))
        if self.opts.nms_enable:
            results = _nms_diversity(
                results,
                theta_thresh_deg=float(self.opts.nms_theta_deg),
                phi_thresh_deg=float(self.opts.nms_th_phi_deg),
                trans_thresh=float(self.opts.nms_translation),
            )
        results = results[:topk]

        sols: List[IKSolution] = []
        for it in results:
            sols.append(
                IKSolution(
                    reachable=True,
                    pos_err=float(it["pos_err"]),
                    ang_err_deg=float(it["ang_err_deg"]),
                    translation=float(it["translation"]),  # d
                    segments=it["segments"],
                    end_T=it["end_T"],
                    end_p=it["end_p"],
                    meta=it["meta"],
                )
            )
        return sols
