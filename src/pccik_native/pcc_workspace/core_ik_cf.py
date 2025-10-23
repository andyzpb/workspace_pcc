from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import numpy as np
from .specs import (
    IKSolution,
    SegmentSolution,
    TranslationSpec,
    SegmentSpec,
    TouchPointSpec,
    IKOptions,
)


# =============================== Utils & Math ================================
EPS = 1e-12
DEG = math.pi / 180.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)


def _wrap_0_2pi(a: float) -> float:
    a = float(a) % (2.0 * math.pi)
    return a if a >= 0.0 else (a + 2.0 * math.pi)


def _canon_theta_phi(theta: float, phi: float) -> Tuple[float, float]:
    """Canonicalize: bend >= 0, wrap phi into [0, 2π)."""
    if theta < 0.0:
        theta = -theta
        phi = phi + math.pi
    return float(theta), float(_wrap_0_2pi(phi))


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _rotz(a: float) -> np.ndarray:
    c, s = float(math.cos(a)), float(math.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _roty(a: float) -> np.ndarray:
    c, s = float(math.cos(a)), float(math.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _A_B_and_derivs_scalar(th: float) -> Tuple[float, float, float, float]:
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


def _A_B_and_derivs_vec(
    th: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    th = np.asarray(th, float)
    out_A = np.empty_like(th)
    out_B = np.empty_like(th)
    out_Ap = np.empty_like(th)
    out_Bp = np.empty_like(th)

    mask = np.abs(th) < 1e-6
    m2 = ~mask

    # small-angle
    t = th[mask]
    out_A[mask] = 0.5 * t - (t**3) / 24.0
    out_B[mask] = 1.0 - (t**2) / 6.0 + (t**4) / 120.0
    out_Ap[mask] = 0.5 - (t**2) / 8.0
    out_Bp[mask] = -t / 3.0 + (t**3) / 30.0

    # regular
    t = th[m2]
    out_A[m2] = (1.0 - np.cos(t)) / t
    out_B[m2] = np.sin(t) / t
    out_Ap[m2] = (t * np.sin(t) - (1.0 - np.cos(t))) / (t * t)
    out_Bp[m2] = (t * np.cos(t) - np.sin(t)) / (t * t)

    return out_A, out_B, out_Ap, out_Bp


def _bend_translation_world(phi: np.ndarray, s: float, theta: float) -> np.ndarray:
    """Vectorized p = Rz(phi) @ [s*A(theta), 0, s*B(theta)].
    Returns shape (N, 3) for phi shape (N,).
    """
    A, B, _, _ = _A_B_and_derivs_scalar(theta)
    x = s * A
    z = s * B
    phi = np.asarray(phi, float)
    c = np.cos(phi)
    sphi = np.sin(phi)
    px = c * x
    py = sphi * x
    pz = np.full_like(phi, z)
    return np.stack([px, py, pz], axis=-1)


def _R_cc(phi: np.ndarray, theta: float) -> np.ndarray:
    """Vectorized R(phi, theta) = Rz(phi) Ry(theta) Rz(-phi).
    Returns (N, 3, 3) for phi shape (N,).
    """
    phi = np.asarray(phi, float)
    c = np.cos(phi)
    s = np.sin(phi)
    ct = math.cos(theta)
    st = math.sin(theta)
    # closed-form elements
    r00 = ct * (c * c) + (s * s)
    r01 = (ct - 1.0) * c * s
    r02 = st * c
    r10 = (ct - 1.0) * c * s
    r11 = ct * (s * s) + (c * c)
    r12 = st * s
    r20 = -st * c
    r21 = -st * s
    r22 = np.full_like(phi, ct)
    R = np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )
    return R  # (N, 3, 3)


# ===================== PCC closed-form blocks (inner) ========================
class _InnerClosedForm:
    @staticmethod
    def angles_from_orientation(
        R1: np.ndarray, n_star: np.ndarray, alpha: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Closed-form inner (phi2, theta2) given outer R1 and target normal.
        Returns (phi2_wrapped, theta2, z2_world, R2).
        """
        n_prime = R1.T @ _normalize(n_star)
        sa, ca = float(math.sin(alpha)), float(math.cos(alpha))
        # phi2
        phi2 = math.atan2(float(n_prime[1]), float(n_prime[0]) - sa)
        c, s = float(math.cos(phi2)), float(math.sin(phi2))
        u = np.array([sa * c, -sa * s, ca], float)  # Rz(-phi2) b0
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

    @staticmethod
    def L2p_from_position_fixed_s2(
        R1: np.ndarray,
        p1: np.ndarray,
        phi2: float,
        theta2: float,
        P_star: np.ndarray,
        L_rigid: float,
        s2: float,
    ) -> float:
        R2 = _rotz(phi2) @ _roty(theta2) @ _rotz(-phi2)
        q = R1.T @ (P_star - p1) - (R2 @ (L_rigid * np.array([0.0, 0.0, 1.0], float)))
        _, B2, _, _ = _A_B_and_derivs_scalar(theta2)
        return float(q[2] - s2 * B2)


# ============================== Helper: Bounds ===============================


def _snap_to_box(
    x: float, lo: Optional[float], hi: Optional[float], tol: float
) -> Optional[float]:
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


# ============================= Brent 1D (bound) ==============================
class _Brent:
    """Simple bounded Brent (no SciPy dependency)."""

    @staticmethod
    def minimize(
        f, a: float, b: float, tol: float = 2e-6, maxiter: int = 60
    ) -> Tuple[float, float]:
        # Implementation inspired by classic Brent; assumes f is well-behaved on [a,b].
        invphi = (math.sqrt(5.0) - 1.0) / 2.0
        x = w = v = a + invphi * (b - a)
        fx = fw = fv = f(x)
        d = e = 0.0
        for _ in range(maxiter):
            m = 0.5 * (a + b)
            tol1 = math.sqrt(np.finfo(float).eps) * abs(x) + tol / 3.0
            tol2 = 2.0 * tol1
            # Convergence check
            if abs(x - m) <= tol2 - 0.5 * (b - a):
                break
            p = q = r = 0.0
            if abs(e) > tol1:
                # fit parabola
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0:
                    p = -p
                q = abs(q)
                if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                    # parabolic step
                    d = p / q
                    u = x + d
                    if (u - a) < tol2 or (b - u) < tol2:
                        d = tol1 if x < m else -tol1
                else:
                    e = (a - x) if x >= m else (b - x)
                    d = invphi * e
            else:
                e = (a - x) if x >= m else (b - x)
                d = invphi * e
            u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
            fu = f(u)
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v, w, x = w, x, u
                fv, fw, fx = fw, fx, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, w = w, u
                    fv, fw = fw, fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
        return x, fx


def _pareto_front(
    items: List[Dict[str, Any]],
    keys: Tuple[str, str, str] = ("pos_err", "ang_err_deg", "abs_d"),
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            better_or_equal = all(b[k] <= a[k] + 1e-12 for k in keys)
            strictly_better = any(b[k] < a[k] - 1e-12 for k in keys)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


class PCCIKClosedForm:
    def __init__(
        self,
        segments: List[SegmentSpec],
        translation: Optional[TranslationSpec] = None,
        opts: Optional[IKOptions] = None,
        inner_rigid_tip: float = 0.003,
        s2_fixed: float = 0.006,
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
            a0 = float(getattr(segments[0], "active_L_max", segments[0].L_max or 1e9))
            a1 = float(getattr(segments[1], "active_L_max", segments[1].L_max or 1e9))
            inner_idx = 0 if a0 <= a1 else 1
        outer_idx = 1 - inner_idx

        self.seg1: SegmentSpec = segments[outer_idx]  # outer
        self.seg2: SegmentSpec = segments[inner_idx]  # inner

        self.translation = translation or TranslationSpec(0.0, 0.0, samples=1)
        self.opts = opts or IKOptions()
        self.debug = bool(debug)

        # cache fixed outer active length if both totals fixed
        wL = abs(self.seg1.L_max - self.seg1.L_min)
        wP = abs(self.seg1.passive_L_max - self.seg1.passive_L_min)
        self.s1_active_fixed: Optional[float] = None
        if wL <= 1e-9 and wP <= 1e-9:
            self.s1_active_fixed = float(self.seg1.L_max - self.seg1.passive_L_max)

        self.inner_rigid_tip = float(inner_rigid_tip)
        if self.inner_rigid_tip < 0:
            raise ValueError("inner_rigid_tip must be non-negative")

        self.s2_fixed = float(s2_fixed)
        if self.s2_fixed < 0.0:
            raise ValueError("s2_fixed must be non-negative")

        # fast-access constants
        self._alpha_rad = float(np.deg2rad(getattr(self.seg2, "bevel_angle_deg", 45.0)))
        self._b0 = np.array(
            [math.sin(self._alpha_rad), 0.0, math.cos(self._alpha_rad)], float
        )
        self._Lrig = float(getattr(self.seg2, "rigid_tip_length", self.inner_rigid_tip))
        # inner bounds
        L2_total_max = float(self.seg2.L_max if self.seg2.L_max is not None else 1e9)
        act_cap = float(getattr(self.seg2, "active_L_max", L2_total_max))
        self._L2a_max = min(act_cap, L2_total_max)
        self._L2p_min, self._L2p_max = float(self.seg2.passive_L_min), float(
            self.seg2.passive_L_max
        )

        # outer cached lengths used in FK
        if self.s1_active_fixed is not None:
            self._s1 = float(self.s1_active_fixed)
        else:
            self._s1 = max(
                0.0,
                0.5
                * (
                    (self.seg1.L_min + self.seg1.L_max)
                    - (self.seg1.passive_L_min + self.seg1.passive_L_max)
                ),
            )
        self._L1p = float(
            self.seg1.passive_L_max
            if abs(self.seg1.passive_L_max - self.seg1.passive_L_min) <= 1e-9
            else 0.5 * (self.seg1.passive_L_min + self.seg1.passive_L_max)
        )

        # quick feasibility for s2
        if self.s2_fixed > self._L2a_max + 1e-9:
            raise ValueError(
                f"s2_fixed={self.s2_fixed:.6f} exceeds inner active cap {self._L2a_max:.6f}"
            )

        # tolerances and weights
        self._pos_tol = float(self.opts.pos_tol)
        self._bevel_tol_deg = float(getattr(self.opts, "bevel_tol_deg", 1.0))
        self._wb = 20.0  # orientation weight inside LM (can expose to opts)

    # ---------------------- small helpers ----------------------
    def _build_T1_components(
        self, theta1: float, phi1: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s1 = self._s1
        L1p = self._L1p
        R1 = _rotz(phi1) @ _roty(theta1) @ _rotz(-phi1)
        # translation of bend in world + passive along z
        A1, B1, _, _ = _A_B_and_derivs_scalar(theta1)
        p_bend = np.array(
            [s1 * A1 * math.cos(phi1), s1 * A1 * math.sin(phi1), s1 * B1], float
        )
        p1 = np.array([0.0, 0.0, L1p], float) + p_bend
        return R1, p1, p_bend

    @staticmethod
    def _best_d_along_z(
        P_star: np.ndarray, p_no_d: np.ndarray, dmin: float, dmax: float
    ) -> float:
        dz = float(P_star[2] - p_no_d[2])
        return max(dmin, min(dmax, dz))

    # ---------------------- single evaluation ----------------------
    def _evaluate_once(
        self, theta1: float, phi1: float, P_star: np.ndarray, n_star: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        # bounds for theta1/phi1
        tol_th = np.deg2rad(1e-4)
        th1 = _snap_to_box(
            theta1,
            getattr(self.seg1, "theta_min", None),
            getattr(self.seg1, "theta_max", None),
            tol_th,
        )
        if th1 is None:
            return None
        theta1 = th1
        if (
            getattr(self.seg1, "phi_min", None) is not None
            and getattr(self.seg1, "phi_max", None) is not None
        ):
            phi1 = _project_angle_to_interval(
                phi1, float(self.seg1.phi_min), float(self.seg1.phi_max)
            )
        else:
            phi1 = _wrap_0_2pi(phi1)

        # outer pose
        R1, p1, _ = self._build_T1_components(theta1, phi1)

        # inner angles from orientation (closed form)
        alpha = self._alpha_rad
        phi2, theta2, z2, R2 = _InnerClosedForm.angles_from_orientation(
            R1, n_star, alpha
        )

        # inner angle boxes
        tol_th2 = np.deg2rad(1e-4)
        th2s = _snap_to_box(
            theta2,
            getattr(self.seg2, "theta_min", None),
            getattr(self.seg2, "theta_max", None),
            tol_th2,
        )
        if th2s is None:
            return None
        theta2 = th2s
        if (
            getattr(self.seg2, "phi_min", None) is not None
            and getattr(self.seg2, "phi_max", None) is not None
        ):
            if not (
                float(self.seg2.phi_min) - 1e-9
                <= phi2
                <= float(self.seg2.phi_max) + 1e-9
            ):
                # project; if projected too far, reject later via pos error/bevel
                phi2 = _project_angle_to_interval(
                    phi2, float(self.seg2.phi_min), float(self.seg2.phi_max)
                )

        # L2p from position (fixed s2)
        L2p_min, L2p_max = self._L2p_min, self._L2p_max
        s2 = self.s2_fixed
        L_rigid = self._Lrig
        L2_passive = _InnerClosedForm.L2p_from_position_fixed_s2(
            R1, p1, phi2, theta2, P_star, L_rigid, s2
        )

        tol_L = 5e-4
        L2p = _snap_to_box(L2_passive, L2p_min, L2p_max, tol_L)
        if L2p is None:
            return None
        L2_total_max = float(self.seg2.L_max if self.seg2.L_max is not None else 1e9)
        if (s2 + L2p) > L2_total_max + tol_L:
            L2p = max(0.0, min(L2p_max, L2_total_max - s2))

        # Compose end position/orientation without d
        # q = L2p*ez + Rz(phi2)@[s2*A2,0,s2*B2] + R2@(Lrig*ez)
        A2, B2, _, _ = _A_B_and_derivs_scalar(theta2)
        v2_local = np.array([s2 * A2, 0.0, s2 * B2], float)
        q = (
            (L2p * np.array([0.0, 0.0, 1.0], float))
            + (_rotz(phi2) @ v2_local)
            + (R2 @ (L_rigid * np.array([0.0, 0.0, 1.0], float)))
        )
        end_p_no_d = p1 + (R1 @ q)

        # best d (clip to allowed translation range)
        d_best = self._best_d_along_z(
            P_star,
            end_p_no_d,
            float(self.translation.d_min),
            float(self.translation.d_max),
        )
        end_p = end_p_no_d + np.array([0.0, 0.0, d_best], float)

        # orientation checks
        b_world = _normalize((R1 @ R2) @ self._b0)
        cb = float(np.clip(b_world @ n_star, -1.0, 1.0))
        cos_bevel_tol = float(np.cos(np.deg2rad(self._bevel_tol_deg)))

        # position error
        pos_err = float(np.linalg.norm(end_p - P_star))
        if (pos_err > self._pos_tol + 1e-9) or (cb < cos_bevel_tol):
            return None

        # axis vs normal diagnostic (same as old meta)
        z_hat = _normalize((R1 @ R2) @ np.array([0.0, 0.0, 1.0], float))
        angle_target_deg = float(getattr(self.opts, "angle_target_deg", 45.0))
        cos_axis = float(np.clip(z_hat @ n_star, -1.0, 1.0))
        ang_err_deg = abs(math.degrees(math.acos(cos_axis)) - angle_target_deg)
        bevel_err_deg = float(np.degrees(math.acos(cb)))

        # canonicalized output angles
        th1_out, ph1_out = _canon_theta_phi(theta1, phi1)
        th2_out, ph2_out = _canon_theta_phi(theta2, phi2)

        seg_solutions = [
            SegmentSolution(
                name=self.seg1.name,
                theta=float(th1_out),
                phi=float(ph1_out),
                L_total=float(self._s1 + self._L1p),
                L_passive=float(self._L1p),
                L_active=float(self._s1),
            ),
            SegmentSolution(
                name=self.seg2.name,
                theta=float(th2_out),
                phi=float(ph2_out),
                L_total=float(self.s2_fixed + L2p),
                L_passive=float(L2p),
                L_active=float(self.s2_fixed),
            ),
        ]

        # meta (keep keys used by your viz stack)
        z1 = _normalize(R1[:, 2])
        meta = dict(
            cos_angle=cb,
            z_axis=z_hat.copy(),
            inner_rigid_tip=float(L_rigid),
            end_p_world=end_p.copy(),
            bevel_angle_deg=float(np.degrees(self._alpha_rad)),
            bevel_err_deg=bevel_err_deg,
            bevel_world=b_world.copy(),
            plane_normal_world=_normalize(np.cross(z1, _normalize(n_star))),
            inner_axis_world=z_hat.copy(),
            report=dict(
                outer=dict(
                    bending_deg=float(np.degrees(th1_out)),
                    rotation_deg=float(np.degrees(ph1_out)),
                    translation=0.0,
                ),
                inner=dict(
                    bending_deg=float(np.degrees(th2_out)),
                    rotation_deg=float(np.degrees(ph2_out)),
                    translation=float(self.s2_fixed + L2p),
                ),
            ),
        )

        # pack
        T_tip = np.eye(4, dtype=float)
        T_tip[:3, :3] = R1 @ _rotz(phi2) @ _roty(theta2) @ _rotz(-phi2)
        # end_p in tip frame excludes d; keep same convention as before
        T_tip[:3, 3] = end_p_no_d.copy()

        return dict(
            pos_err=pos_err,
            ang_err_deg=ang_err_deg,
            translation=float(d_best),
            abs_d=abs(float(d_best)),
            segments=seg_solutions,
            end_T=T_tip.copy(),
            end_p=end_p_no_d.copy(),
            meta=meta,
        )

    def _phi_scan_vectorized(
        self,
        theta1: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        phi_list: np.ndarray,
        k_keep: int = 6,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        out: List[Tuple[float, Dict[str, Any]]] = []
        for ph in phi_list:  # simple loop keeps code clear; (batching is optional)
            cand = self._evaluate_once(theta1, float(ph), P_star, n_star)
            if cand is None:
                continue
            J = cand["pos_err"] + 1e-3 * cand["ang_err_deg"]
            out.append((J, cand))
        out.sort(key=lambda t: t[0])
        return out[:k_keep]

    def _refine_phi1_local(
        self,
        theta1: float,
        phi_seed: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        halfspan_deg: float = 6.0,
        maxiter: int = 60,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        span = math.radians(halfspan_deg)
        a = phi_seed - span
        b = phi_seed + span

        def score(ph: float) -> float:
            cand = self._evaluate_once(theta1, ph, P_star, n_star)
            if cand is None:
                return 1e9
            return cand["pos_err"] + 1e-6 * cand["ang_err_deg"]

        phi_best, _ = _Brent.minimize(score, a, b, tol=2e-6, maxiter=maxiter)
        cand_best = self._evaluate_once(theta1, phi_best, P_star, n_star)
        return phi_best, cand_best

    def _lm_polish(
        self,
        theta1: float,
        phi1: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        iters: int = 8,
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        lam = 1e-2  # initial damping
        dmin, dmax = float(self.translation.d_min), float(self.translation.d_max)

        def residual(x: np.ndarray) -> np.ndarray:
            th, ph = float(x[0]), float(x[1])
            cand = self._evaluate_once(th, ph, P_star, n_star)
            if cand is None:
                # big penalty residual
                return np.array([1e3, 1e3, 1e3, 1e2, 1e2, 1e2], float)
            rp = P_star - cand["meta"]["end_p_world"]  # already includes d
            rb = n_star - _normalize(
                cand["meta"]["bevel_world"]
            )  # project not needed for small deviations
            return np.concatenate([rp, math.sqrt(self._wb) * rb])

        def jac_fd(x: np.ndarray, hth=1e-6, hph=1e-6) -> np.ndarray:
            r0 = residual(x)
            J = np.zeros((r0.size, 2), float)
            for i, h in enumerate([hth, hph]):
                xp = x.copy()
                xp[i] += h
                xm = x.copy()
                xm[i] -= h
                rp = residual(xp)
                rm = residual(xm)
                J[:, i] = (rp - rm) / (2.0 * h)
            return J

        x = np.array([theta1, phi1], float)
        best = self._evaluate_once(x[0], x[1], P_star, n_star)
        if best is None:
            return theta1, phi1, None
        f_best = (
            float(best["pos_err"])
            + 1e-3 * float(best["ang_err_deg"])
            + 1e-6 * best["abs_d"]
        )

        for _ in range(iters):
            r = residual(x)
            J = jac_fd(x)
            A = J.T @ J + lam * np.eye(2)
            g = -J.T @ r
            try:
                step = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A, g, rcond=None)[0]

            x_try = x + step
            # project to bounds
            tol_th = np.deg2rad(1e-4)
            x_try[0] = _snap_to_box(
                float(x_try[0]),
                getattr(self.seg1, "theta_min", None),
                getattr(self.seg1, "theta_max", None),
                tol_th,
            ) or float(x_try[0])
            if (
                getattr(self.seg1, "phi_min", None) is not None
                and getattr(self.seg1, "phi_max", None) is not None
            ):
                x_try[1] = _project_angle_to_interval(
                    float(x_try[1]), float(self.seg1.phi_min), float(self.seg1.phi_max)
                )
            else:
                x_try[1] = _wrap_0_2pi(float(x_try[1]))

            cand_try = self._evaluate_once(x_try[0], x_try[1], P_star, n_star)
            f_try = 1e9
            if cand_try is not None:
                f_try = (
                    float(cand_try["pos_err"])
                    + 1e-3 * float(cand_try["ang_err_deg"])
                    + 1e-6 * cand_try["abs_d"]
                )

            # trust-region style acceptance
            if f_try < f_best:
                x = x_try
                best = cand_try
                f_best = f_try
                lam *= 0.5
            else:
                lam *= 2.0

        return float(x[0]), float(x[1]), best

    # ---------------------- theta1 candidates ----------------------
    def _theta1_candidates(
        self,
        P_world: np.ndarray,
        n_star: np.ndarray,
        th_lo: float,
        th_hi: float,
        coarse_n: int = 41,
        keep_top: int = 6,
        refine_halfspan_deg: float = 5.0,
        refine_n: int = 19,
    ) -> List[float]:
        def score(theta1: float) -> float:
            best = float("+inf")
            for phi1 in np.linspace(-math.pi, math.pi, 48, endpoint=False):
                cand = self._evaluate_once(theta1, phi1, P_world, n_star)
                if cand is None:
                    continue
                J = cand["pos_err"] + 0.001 * cand["ang_err_deg"]
                if J < best:
                    best = J
                    if best <= max(1e-6, 1e-3 * self._pos_tol):
                        break
            return best

        theta_grid = np.linspace(th_lo, th_hi, int(max(3, coarse_n)), dtype=float)
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

    # ---------------------- main solve ----------------------
    def solve(self, target: TouchPointSpec) -> List[IKSolution]:
        P_star = np.asarray(target.coordinates, float).reshape(3)
        n_star = _normalize(np.asarray(target.normal, float).reshape(3))

        results: List[Dict[str, Any]] = []
        # theta1 window from outer bounds
        seg = self.seg1
        th_lo = float(seg.theta_min if seg.theta_min is not None else -seg.theta_max)
        th_hi = float(seg.theta_max)

        theta_seeds = self._theta1_candidates(
            P_star,
            n_star,
            th_lo,
            th_hi,
            coarse_n=41,
            keep_top=6,
            refine_halfspan_deg=5.0,
            refine_n=19,
        )
        theta_seeds = sorted(set(theta_seeds + [th_lo, th_hi]))

        early_pos = max(1e-6, 0.05 * self._pos_tol)
        early_ang = 0.5 * self._bevel_tol_deg

        for th1 in theta_seeds:
            # coarse φ1 grid size near bounds
            near_bound = (abs(th1 - th_lo) < np.deg2rad(0.2)) or (
                abs(th_hi - th1) < np.deg2rad(0.2)
            )
            n_phi = 144 if near_bound else 72
            phi_list = np.linspace(-math.pi, math.pi, n_phi, endpoint=False)

            pool = self._phi_scan_vectorized(th1, P_star, n_star, phi_list, k_keep=6)
            if not pool:
                continue

            # local Brent refine on a few best
            refined: List[Tuple[float, Dict[str, Any]]] = []
            for _, base in pool[:5]:
                ph_seed = base["segments"][0].phi
                _, cand_ref = self._refine_phi1_local(
                    th1, ph_seed, P_star, n_star, halfspan_deg=6.0, maxiter=60
                )
                if cand_ref is not None:
                    refined.append(
                        (cand_ref["pos_err"] + 1e-6 * cand_ref["ang_err_deg"], cand_ref)
                    )
                    if (
                        cand_ref["pos_err"] <= early_pos
                        and cand_ref["ang_err_deg"] <= early_ang
                    ):
                        break

            if refined:
                refined.sort(key=lambda t: t[0])
                pool = refined
            else:
                pool = sorted(pool, key=lambda t: t[0])[:5]

            # LM polish a few best
            for _, base in pool[:3]:
                th_seed = base["segments"][0].theta
                ph_seed = base["segments"][0].phi
                th_fin, ph_fin, polished = self._lm_polish(
                    th_seed, ph_seed, P_star, n_star, iters=8
                )
                cand = polished if polished is not None else base
                if isinstance(cand, tuple):
                    cand = cand[1]
                if cand is not None:
                    results.append(cand)
                    if (
                        cand["pos_err"] <= early_pos
                        and cand["ang_err_deg"] <= early_ang
                        and len(results) >= int(self.opts.topk)
                    ):
                        break

            if len(results) >= int(self.opts.topk):
                break

        if not results:
            return []

        # Pareto filter then sort
        pf = _pareto_front(results, keys=("pos_err", "ang_err_deg", "abs_d"))
        pf.sort(key=lambda c: (c["pos_err"], c["ang_err_deg"]))
        pf = pf[: int(self.opts.topk)]

        sols: List[IKSolution] = []
        for it in pf:
            sols.append(
                IKSolution(
                    reachable=True,
                    pos_err=float(it["pos_err"]),
                    ang_err_deg=float(it["ang_err_deg"]),
                    translation=float(it["translation"]),
                    segments=it["segments"],
                    end_T=it["end_T"],
                    end_p=it["end_p"],
                    meta=it["meta"],
                )
            )
        return sols
