from typing import Iterable, Sequence, List, Optional, Dict, Any, Tuple
import cupy as cp
import numpy as np
from .specs import (
    SegmentSpec,
    TranslationSpec,
    TouchPointSpec,
    IKOptions,
    IKSolution,
    SegmentSolution,
)
try:
    import cupy as cp  # type: ignore
    _USING_CUPY = True
except Exception:  # noqa
    import numpy as cp  # type: ignore
    _USING_CUPY = False

# ------------ FK Solver ------------
class PCCFKSolver:
    """
    Batched FK sampler for PCC robots (no-roll). Produces a set of tip poses.

    - Respects per-segment L_total / passive_L ranges.
    - θ≈0 still advances by s (straight-segment fix).
    - φ_coupling: None -> free; "lock_prev" -> φ_i = φ_{i-1};
                  float -> φ_i = wrap(φ_{i-1} + offset).
    - Sampling strategy: random Latin-like within bounds (budget-based), not full grid.
      This avoids combinatorial explosion but preserves coverage statistics.
    """

    def __init__(
        self,
        segments: Sequence[SegmentSpec],
        translation: Optional[TranslationSpec] = None,
        small_L_eps: float = 1e-6,
    ):
        if len(segments) == 0:
            raise ValueError("need at least one SegmentSpec")
        self.segments: List[SegmentSpec] = list(segments)
        self.translation = translation or TranslationSpec(0.0, 0.0, 1)
        self.small_L_eps = float(small_L_eps)

    # ------------ utils ------------
    @staticmethod
    def _wrap_pi(a: cp.ndarray) -> cp.ndarray:
        return (a + cp.pi) % (2.0 * cp.pi) - cp.pi

    @staticmethod
    def _linspace_cp(a: float, b: float, n: int) -> cp.ndarray:
        if n <= 1:
            return cp.asarray([0.5 * (a + b)], dtype=cp.float32)
        return cp.linspace(a, b, n, dtype=cp.float32)

    @staticmethod
    def _arange_with_endpoint(a: float, b: float, step: float) -> cp.ndarray:
        if step <= 0:
            raise ValueError("step must be > 0")
        n = int(np.floor((b - a) / step + 1e-9)) + 1
        arr = a + step * cp.arange(n, dtype=cp.float32)
        if arr.size == 0:
            return cp.asarray([0.5 * (a + b)], dtype=cp.float32)
        if arr[-1] < b - 1e-6:
            arr = cp.concatenate([arr, cp.asarray([b], dtype=cp.float32)])
        if arr[-1] > b + 1e-6:
            arr[-1] = b
        return arr.astype(cp.float32)

    @staticmethod
    def _sample_range(a: float, b: float, step: Optional[float], samples: int) -> cp.ndarray:
        if step is not None:
            return PCCFKSolver._arange_with_endpoint(a, b, float(step))
        return PCCFKSolver._linspace_cp(a, b, max(samples, 1))

    @staticmethod
    def _phi_default_samples(n: int) -> cp.ndarray:
        if n <= 1:
            return cp.asarray([0.0], dtype=cp.float32)
        return cp.linspace(0.0, 2 * cp.pi, n, endpoint=False, dtype=cp.float32)

    # ------------ CC transform (batched) ------------
    @staticmethod
    def _rot(phi: cp.ndarray, kappa: cp.ndarray, theta: cp.ndarray, L: cp.ndarray) -> cp.ndarray:
        """
        Batched single-arc FK with straight-segment fix:
        if |theta|≈0 -> translate by L along +z, R=I.
        """
        eps = 1e-8
        k_safe = cp.where(cp.abs(kappa) < eps, eps, kappa)

        cos_t = cp.cos(theta)
        sin_t = cp.sin(theta)
        one_minus_cos_t = 1.0 - cos_t

        cos_phi = cp.cos(phi)
        sin_phi = cp.sin(phi)
        cos2 = cos_phi * cos_phi
        sin2 = sin_phi * sin_phi
        sincos = sin_phi * cos_phi

        # rotation (Rodrigues around u=[-sinφ,cosφ,0])
        R = cp.stack(
            [
                cp.stack(
                    [cos2 * (cos_t - 1.0) + 1.0, sincos * (cos_t - 1.0), cos_phi * sin_t],
                    axis=-1,
                ),
                cp.stack(
                    [sincos * (cos_t - 1.0), sin2 * (cos_t - 1.0) + cos_t, sin_phi * sin_t],
                    axis=-1,
                ),
                cp.stack([-cos_phi * sin_t, -sin_phi * sin_t, cos_t], axis=-1),
            ],
            axis=1,
        )

        # translation of CC arc expressed in world after Rz(phi)
        px = cos_phi * one_minus_cos_t / k_safe
        py = sin_phi * one_minus_cos_t / k_safe
        pz = sin_t / k_safe

        # straight-segment limit (θ≈0) uses series point [0.5*s*θ, 0, s]
        small = cp.abs(theta) < 1e-6
        if small.any():
            px_small = cos_phi * 0.5 * theta * L
            py_small = sin_phi * 0.5 * theta * L
            pz_small = L
            px = cp.where(small, px_small, px)
            py = cp.where(small, py_small, py)
            pz = cp.where(small, pz_small, pz)

        p = cp.stack([px, py, pz], axis=-1)

        T = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (phi.shape[0], 4, 4)).copy()
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T

    # ------------ random workspace sampler ------------
    def sample_workspace(
        self,
        n_samples: int = 200_000,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomized Latin-like sampling in joint space under SegmentSpec bounds.

        Returns:
            pts: (N,3) end-effector points in world (numpy)
            Ts:  (N,4,4) end-effector transforms (numpy)
        """
        if rng is None:
            rng = np.random.default_rng()

        S = len(self.segments)
        if S < 1:
            raise ValueError("no segments")

        # Latin-like: draw in [0,1) then affine-map to each bound.
        u = rng.random((n_samples, 4 * S + 1), dtype=np.float64)  # per segment: L_total, L_passive, theta, phi; + translation
        # base translation along +z
        d = self.translation
        d_vals = d.d_min + (d.d_max - d.d_min) * u[:, 0]
        d_vals = d_vals.astype(np.float32)

        # parameter holders
        thetas: List[np.ndarray] = []
        phis: List[np.ndarray] = []
        L_totals: List[np.ndarray] = []
        L_passives: List[np.ndarray] = []

        col = 1
        for i, seg in enumerate(self.segments):
            # lengths
            L_total = seg.L_min + (seg.L_max - seg.L_min) * u[:, col]; col += 1
            L_pass  = seg.passive_L_min + (seg.passive_L_max - seg.passive_L_min) * u[:, col]; col += 1
            L_total = L_total.astype(np.float32)
            L_pass  = L_pass.astype(np.float32)

            # θ bounds
            if seg.theta_min is not None:
                th_lo, th_hi = float(seg.theta_min), float(seg.theta_max)
            else:
                if seg.theta_sign is True:
                    th_lo, th_hi = 0.0, float(seg.theta_max)
                elif seg.theta_sign is False:
                    th_lo, th_hi = -float(seg.theta_max), float(seg.theta_max)
                else:
                    th_lo, th_hi = 0.0, float(seg.theta_max)

            theta = th_lo + (th_hi - th_lo) * u[:, col]; col += 1
            theta = theta.astype(np.float32)

            # φ sampling (independent); coupling applied after loop
            if (seg.phi_min is not None) and (seg.phi_max is not None):
                phi = float(seg.phi_min) + (float(seg.phi_max) - float(seg.phi_min)) * u[:, col]
            else:
                phi = 2.0 * np.pi * u[:, col]
            col += 1
            phi = ((phi + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

            L_totals.append(L_total)
            L_passives.append(L_pass)
            thetas.append(theta)
            phis.append(phi)

        # apply φ coupling (no roll)
        for i, seg in enumerate(self.segments):
            if seg.phi_coupling is None:
                continue
            if i == 0:
                continue
            if seg.phi_coupling == "lock_prev":
                phis[i] = phis[i - 1].copy()
            elif isinstance(seg.phi_coupling, (int, float)):
                phis[i] = ((phis[i - 1] + float(seg.phi_coupling) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
            else:
                raise ValueError(f"Unsupported phi_coupling: {seg.phi_coupling}")

        # ---- build batched transforms ----
        N = int(n_samples)
        T = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (N, 4, 4)).copy()

        # base translation
        if (abs(self.translation.d_max - self.translation.d_min) > 1e-12) or (abs(self.translation.d_min) > 0):
            T_ins = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (N, 4, 4)).copy()
            T_ins[:, 2, 3] = cp.asarray(d_vals)
            T = T @ T_ins

        for i, _seg in enumerate(self.segments):
            Lp = cp.asarray(L_passives[i], dtype=cp.float32)
            Lt = cp.asarray(L_totals[i], dtype=cp.float32)
            La = cp.maximum(Lt - Lp, 0.0).astype(cp.float32)

            # passive straight
            if float(cp.max(Lp)) > self.small_L_eps:
                T_pass = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (N, 4, 4)).copy()
                T_pass[:, 2, 3] = Lp
                T = T @ T_pass

            # active arc (θ≈0 -> straight advance by La)
            th = cp.asarray(thetas[i], dtype=cp.float32)
            # if La==0 and |θ|>0, force θ=0 to avoid singular curvature
            th = cp.where(La < self.small_L_eps, cp.zeros_like(th), th)
            kap = cp.where(La < self.small_L_eps, cp.zeros_like(th), th / cp.maximum(La, 1e-9))
            ph = cp.asarray(phis[i], dtype=cp.float32)

            T_arc = self._rot(ph, kap, th, La)
            T = T @ T_arc

        pts = T[:, :3, 3]
        # to numpy
        pts_np = cp.asnumpy(pts) if _USING_CUPY else pts  # type: ignore
        Ts_np  = cp.asnumpy(T)   if _USING_CUPY else T    # type: ignore
        return np.asarray(pts_np, dtype=np.float64), np.asarray(Ts_np, dtype=np.float64)


# ============================================================
#             ------------ Dexterity metrics ------------
# ============================================================
def _orientation_bin_ids(
    dirs: np.ndarray, n_az: int, n_el: int
) -> Tuple[np.ndarray, int]:
    """
    Bin index for each direction on S^2 using (azimuth, elevation).
    az in [0, 2π), el in [-π/2, π/2]; index = az_bin + n_az * el_bin
    """
    if n_az <= 0 or n_el <= 0:
        raise ValueError("n_az and n_el must be positive")

    ux = dirs[:, 0]; uy = dirs[:, 1]; uz = dirs[:, 2]
    az = np.mod(np.arctan2(uy, ux), 2.0 * np.pi)                 # [0, 2π)
    el = np.clip(np.arcsin(np.clip(uz, -1.0, 1.0)), -np.pi/2, np.pi/2)  # [-π/2, π/2]

    az_id = np.minimum((az / (2.0 * np.pi) * n_az).astype(int), n_az - 1)
    el_id = np.minimum(((el + np.pi/2) / np.pi * n_el).astype(int), n_el - 1)
    TOT = int(n_az * n_el)
    return (az_id + n_az * el_id).astype(np.int64), TOT



# ------------ IK Solver ------------
class PCCIKSolver:
    """Batched grid-search IK with optional bevel alignment and robust φ coupling,
    now supporting 'bending-plane coplanar with surface normal' constraint.

    Requirements on `workspace`:
      - workspace._build_parameter_grid() ->
            (params_per_seg, translation_vals)
            Each segment dict must contain CuPy vectors for keys:
               "theta", "phi", "L_total", "L_passive", and coupling meta either
               "phi_coupling" or "phi_mode".
      - workspace._compose_fk_from_params(per_seg_samples, translation_vals) ->
            (points_cp[B,3], transforms_cp[B,4,4], extras: dict)
            extras MUST contain at least:
               "R_before_inner": cp.ndarray[B,3,3]
            (for backward compatibility, returning only two values is allowed but
             'enforce_coplanar_plane' will raise unless disabled)
      - workspace.segments: iterable of SegmentSpec-like objects with 'name' and 'is_inner'
    """

    def __init__(
        self,
        workspace: "PCCFKSolver",
        opts: Optional[IKOptions] = None,
        batch_size: int = 200_000,
    ) -> None:
        self.ws = workspace
        self.opts = opts or IKOptions()
        self.batch_size = int(batch_size)

    # ------------------------------- Public API -------------------------------

    def solve(self, target: TouchPointSpec) -> List[IKSolution]:
        """Return up to Top-K feasible solutions ranked lexicographically (pos_err, ang_err_deg)."""
        grids_flat, sizes, strides, seg_param_specs = self._build_linear_indexer()
        if not sizes:
            return []
        N_total = int(np.prod(sizes, dtype=np.int64))
        if N_total == 0:
            return []

        # Target and normalized normal.
        tgt = cp.asarray(target.coordinates, dtype=cp.float32).reshape(1, 3)
        nvec = cp.asarray(target.normal, dtype=cp.float32).reshape(1, 3)
        n_hat = self._normalize_cp(nvec, axis=1)  # (1,3)

        # ---- Options ----
        pos_tol = float(self.opts.pos_tol)
        ang_tol_deg = float(self.opts.ang_tol_deg)
        angle_target_deg = float(self.opts.angle_target_deg)
        require_front = bool(self.opts.require_frontside)
        front_safe_cos = float(getattr(self.opts, "front_safe_cos", 0.0))
        use_bevel = bool(getattr(self.opts, "use_bevel_alignment", False))
        bevel_tol_deg = float(getattr(self.opts, "bevel_tol_deg", 2.0))
        enforce_axis_band = bool(getattr(self.opts, "enforce_axis_band", True))
        topk = int(self.opts.topk)

        enforce_plane = bool(getattr(self.opts, "enforce_coplanar_plane", True))
        plane_mode: str = getattr(self.opts, "plane_mode", "snap")
        phi_copl_tol_deg = float(getattr(self.opts, "phi_coplanar_tol_deg", 2.0))
        phi_copl_tol = float(np.deg2rad(phi_copl_tol_deg))

        # Precompute cosine band for the axis target ± tolerance.
        cmin, cmax = self._cos_band_deg(angle_target_deg, ang_tol_deg)

        # Bevel constants (only used if enabled).
        alpha = float(np.deg2rad(self._get_bevel_angle_deg_default()))
        sa, ca = float(np.sin(alpha)), float(np.cos(alpha))
        cos_bevel_tol = float(np.cos(np.deg2rad(bevel_tol_deg)))

        # Identify inner segment index.
        inner_idx = self._find_inner_index()
        if inner_idx is None and enforce_plane:
            # 如果没有标记 inner，就默认最后一段是 inner（以兼容两段结构）
            inner_idx = len(self.ws.segments) - 1 if len(self.ws.segments) > 0 else None

        best_pool: List[Dict[str, Any]] = []
        B = self.batch_size
        start = 0

        while start < N_total:
            count = int(min(B, N_total - start))

            # Gather parameters for this batch, applying φ coupling.
            per_seg_samples, translation_vals = self._gather_batch_params(
                grids_flat, sizes, strides, seg_param_specs, start, count
            )

            # ---------- 第一次 FK（粗评估，拿到 R_before_inner） ----------
            fk_ret = self.ws._compose_fk_from_params(per_seg_samples, translation_vals)
            if isinstance(fk_ret, tuple) and len(fk_ret) == 3:
                pts_cp, Ts_cp, extras = fk_ret
            elif isinstance(fk_ret, tuple) and len(fk_ret) == 2:
                if enforce_plane:
                    raise RuntimeError(
                        "workspace._compose_fk_from_params must return extras['R_before_inner'] "
                        "to enable coplanar-plane constraint. Disable 'enforce_coplanar_plane' "
                        "or extend workspace accordingly."
                    )
                pts_cp, Ts_cp = fk_ret
                extras = {}
            else:
                raise RuntimeError("Unexpected return from _compose_fk_from_params")

            # Position error.
            dpos = pts_cp - tgt
            pos_err = cp.linalg.norm(dpos, axis=1)
            pos_ok = pos_err <= pos_tol

            # Tool axis and cosine to target normal.
            Rb = Ts_cp[:, :3, :3]  # (B,3,3)
            z_axes = Ts_cp[:, :3, 2]  # (B,3)
            z_hat = self._normalize_cp(z_axes, axis=1)
            n_b = cp.broadcast_to(n_hat, z_hat.shape)
            cosang = cp.clip(cp.sum(z_hat * n_b, axis=1), -1.0, 1.0)

            # Front-side with safety margin.
            hemi_ok = (
                (cosang > front_safe_cos)
                if require_front
                else cp.ones_like(cosang, dtype=cp.bool_)
            )

            # Axis-band constraint (legacy).
            ang_band_ok = (cosang >= cmin) & (cosang <= cmax)

            # ---------- 共面约束：计算 φ* 并处理 ----------
            plane_ok = cp.ones_like(pos_ok, dtype=cp.bool_)
            phi_star = None
            R_pre_inner = None

            if enforce_plane and inner_idx is not None:
                # 拿到 inner 段基坐标系的旋转
                if "R_before_inner" in extras:
                    R_pre_inner = extras["R_before_inner"]  # (B,3,3)
                elif "R_before_each" in extras:
                    R_pre_inner = extras["R_before_each"][:, inner_idx]  # (B,3,3)
                else:
                    raise RuntimeError(
                        "extras must contain 'R_before_inner' or 'R_before_each'."
                    )

                # a = R_pre^T * n_hat
                a = cp.einsum(
                    "bij,j->bi", R_pre_inner.transpose(0, 2, 1), n_hat.reshape(3)
                )
                ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
                rho = cp.sqrt(ax * ax + ay * ay)

                phi_star_raw = cp.arctan2(ay, ax)  # (-pi, pi]
                phi_star = cp.mod(phi_star_raw, np.pi)  # [0, pi)

                inner_phi = per_seg_samples[inner_idx]["phi"]

                phi_star_alt = phi_star + np.pi
                d0 = self._wrap_to_pi(inner_phi - phi_star)
                d1 = self._wrap_to_pi(inner_phi - phi_star_alt)
                use_alt = cp.abs(d1) < cp.abs(d0)
                phi_star_adj = cp.where(use_alt, phi_star_alt, phi_star)

                if plane_mode == "filter":
                    plane_ok = (rho <= 1e-6) | (
                        cp.abs(self._wrap_to_pi(inner_phi - phi_star_adj))
                        <= phi_copl_tol
                    )

            if use_bevel:
                a_b = cp.einsum("bij,bj->bi", Rb.transpose(0, 2, 1), n_b)  # (B,3)
                ax_b, ay_b, az_b = a_b[:, 0], a_b[:, 1], a_b[:, 2]
                rho_b = cp.sqrt(ax_b * ax_b + ay_b * ay_b)
                c_bev_max = rho_b * sa + az_b * ca
                bev_ok = c_bev_max >= cos_bevel_tol

                cond0 = bev_ok & hemi_ok & pos_ok
                if enforce_axis_band:
                    cond0 = cond0 & ang_band_ok
            else:
                cond0 = ang_band_ok & hemi_ok & pos_ok

            if plane_mode == "filter" and enforce_plane and inner_idx is not None:
                cond0 = cond0 & plane_ok

            feasible_idx = cp.where(cond0)[0]

            if (
                enforce_plane
                and inner_idx is not None
                and plane_mode == "snap"
                and feasible_idx.size > 0
            ):
                sub_idx = feasible_idx
                sub_per_seg = []
                for si, d in enumerate(per_seg_samples):
                    sub = {
                        k: v[sub_idx] if isinstance(v, cp.ndarray) else v
                        for k, v in d.items()
                    }
                    sub_per_seg.append(sub)
                sub_trans = translation_vals[sub_idx]

                rho_sub = cp.sqrt(a[sub_idx, 0] ** 2 + a[sub_idx, 1] ** 2)
                inner_phi_sub = sub_per_seg[inner_idx]["phi"]
                phi_star_sub = cp.mod(cp.arctan2(a[sub_idx, 1], a[sub_idx, 0]), np.pi)
                phi_star_alt_sub = phi_star_sub + np.pi
                d0 = self._wrap_to_pi(inner_phi_sub - phi_star_sub)
                d1 = self._wrap_to_pi(inner_phi_sub - phi_star_alt_sub)
                use_alt = cp.abs(d1) < cp.abs(d0)
                phi_star_adj_sub = cp.where(use_alt, phi_star_alt_sub, phi_star_sub)
                sub_per_seg[inner_idx]["phi"] = cp.where(
                    rho_sub <= 1e-6, inner_phi_sub, phi_star_adj_sub
                ).astype(cp.float32)

                pts2, Ts2, _extras2 = self.ws._compose_fk_from_params(
                    sub_per_seg, sub_trans
                )

                dpos2 = pts2 - tgt
                pos_err2 = cp.linalg.norm(dpos2, axis=1)
                pos_ok2 = pos_err2 <= pos_tol

                Rb2 = Ts2[:, :3, :3]
                z2 = Ts2[:, :3, 2]
                z2_hat = self._normalize_cp(z2, axis=1)
                cos2 = cp.clip(cp.sum(z2_hat * n_b[sub_idx], axis=1), -1.0, 1.0)
                hemi_ok2 = (
                    (cos2 > front_safe_cos)
                    if require_front
                    else cp.ones_like(cos2, dtype=cp.bool_)
                )
                ang_band_ok2 = (cos2 >= cmin) & (cos2 <= cmax)

                if use_bevel:
                    a_b2 = cp.einsum("bij,bj->bi", Rb2.transpose(0, 2, 1), n_b[sub_idx])
                    ax_b2, ay_b2, az_b2 = a_b2[:, 0], a_b2[:, 1], a_b2[:, 2]
                    rho_b2 = cp.sqrt(ax_b2 * ax_b2 + ay_b2 * ay_b2)
                    c_bev_max2 = rho_b2 * sa + az_b2 * ca
                    bev_ok2 = c_bev_max2 >= cos_bevel_tol
                    cond2 = bev_ok2 & hemi_ok2 & pos_ok2
                    if enforce_axis_band:
                        cond2 = cond2 & ang_band_ok2
                else:
                    cond2 = ang_band_ok2 & hemi_ok2 & pos_ok2

                feasible2_local = cp.where(cond2)[0]
                if feasible2_local.size > 0:
                    pos_sel = pos_err2[feasible2_local]
                    ang_deg2 = cp.degrees(
                        cp.arccos(cp.clip(cos2[feasible2_local], -1.0, 1.0))
                    )
                    ang_err_deg = cp.abs(ang_deg2 - angle_target_deg)
                    order = cp.lexsort(cp.stack([ang_err_deg, pos_sel], axis=0))
                    idx_sorted = feasible2_local[order]
                    take = int(min(len(idx_sorted), max(3 * topk, topk)))
                    cand_local = idx_sorted[:take].get().tolist()
                    self._try_insert_candidates(
                        best_pool=best_pool,
                        cand_batch=cand_local,
                        per_seg_samples=sub_per_seg,
                        translation_vals=sub_trans,
                        pts_cp=pts2,
                        Ts_cp=Ts2,
                        pos_err=pos_err2,
                        angle_target_deg=angle_target_deg,
                        cosang=cos2,
                        bevel_enabled=use_bevel,
                        c_bev_max=(c_bev_max2 if use_bevel else None),
                        Rb=Rb2,
                        n_hat=n_hat,
                        alpha=alpha,
                        extra_meta={
                            "phi_star_used": True,
                            "inner_index": inner_idx,
                        },
                    )
            else:
                if feasible_idx.size > 0:
                    pos_sel = pos_err[feasible_idx]
                    ang_deg = cp.degrees(
                        cp.arccos(cp.clip(cosang[feasible_idx], -1.0, 1.0))
                    )
                    ang_err_deg = cp.abs(ang_deg - angle_target_deg)
                    order = cp.lexsort(cp.stack([ang_err_deg, pos_sel], axis=0))
                    idx_sorted = feasible_idx[order]
                    take = int(min(len(idx_sorted), max(3 * topk, topk)))
                    cand = idx_sorted[:take].get().tolist()

                    self._try_insert_candidates(
                        best_pool=best_pool,
                        cand_batch=cand,
                        per_seg_samples=per_seg_samples,
                        translation_vals=translation_vals,
                        pts_cp=pts_cp,
                        Ts_cp=Ts_cp,
                        pos_err=pos_err,
                        angle_target_deg=angle_target_deg,
                        cosang=cosang,
                        bevel_enabled=use_bevel,
                        c_bev_max=(c_bev_max if use_bevel else None),
                        Rb=Rb,
                        n_hat=n_hat,
                        alpha=alpha,
                        extra_meta={
                            "phi_star_used": False,
                            "inner_index": inner_idx,
                            "phi_star": (
                                phi_star.get()
                                if enforce_plane and phi_star is not None
                                else None
                            ),
                        },
                    )

            if len(best_pool) > 4 * topk:
                best_pool.sort(key=lambda x: (x["pos_err"], x["ang_err_deg"]))
                best_pool[:] = best_pool[: 2 * topk]

            start += count

        if not best_pool:
            return []

        # Final sort + NMS for diversity.
        best_pool.sort(key=lambda x: (x["pos_err"], x["ang_err_deg"]))
        best_pool = self._nms_solutions(best_pool, max_keep=topk)

        # Pack IKSolution list.
        sols: List[IKSolution] = []
        for item in best_pool[:topk]:
            sols.append(
                IKSolution(
                    reachable=True,
                    pos_err=item["pos_err"],
                    ang_err_deg=item["ang_err_deg"],
                    translation=item["translation"],
                    segments=item["segments"],
                    end_T=item["end_T"],
                    end_p=item["end_p"],
                    meta=item["meta"],
                )
            )
        return sols

    def reachable(self, target: TouchPointSpec) -> bool:
        return len(self.solve(target)) > 0

    # ----------------------------- Internal helpers ---------------------------

    @staticmethod
    def _normalize_cp(v: cp.ndarray, axis: int = -1, eps: float = 1e-12) -> cp.ndarray:
        n = cp.linalg.norm(v, axis=axis, keepdims=True)
        n = cp.maximum(n, eps)
        return v / n

    @staticmethod
    def _wrap_to_pi(x: cp.ndarray) -> cp.ndarray:
        """Wrap angle (rad) into (-pi, pi]."""
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _cos_band_deg(
        angle_target_deg: float, ang_tol_deg: float
    ) -> Tuple[float, float]:
        t = float(angle_target_deg)
        tol = float(ang_tol_deg)
        cmin = float(np.cos(np.deg2rad(t + tol)))
        cmax = float(np.cos(np.deg2rad(t - tol)))
        return cmin, cmax

    def _get_bevel_angle_deg_default(self) -> float:
        try:
            inners = [
                s
                for s in getattr(self.ws, "segments", [])
                if getattr(s, "is_inner", False)
            ]
            if inners:
                return float(getattr(inners[0], "bevel_angle_deg", 45.0))
        except Exception:
            pass
        return 45.0

    def _find_inner_index(self) -> Optional[int]:
        for i, s in enumerate(getattr(self.ws, "segments", [])):
            if getattr(s, "is_inner", False):
                return i
        return None

    # ---- Grid Indexer (row-major) ----

    def _build_linear_indexer(
        self,
    ) -> Tuple[List[cp.ndarray], List[int], List[int], List[Dict[str, Any]]]:
        params_per_seg, translation_vals = self.ws._build_parameter_grid()

        grids_flat: List[cp.ndarray] = []
        seg_param_specs: List[Dict[str, Any]] = []

        for d in params_per_seg:
            grids_flat.extend(
                [
                    cp.asarray(d["theta"], dtype=cp.float32),
                    cp.asarray(d["phi"], dtype=cp.float32),
                    cp.asarray(d["L_total"], dtype=cp.float32),
                    cp.asarray(d["L_passive"], dtype=cp.float32),
                ]
            )
            seg_param_specs.append(
                {
                    "phi_coupling": d.get("phi_coupling", None),
                    "phi_mode": d.get("phi_mode", None),
                }
            )

        grids_flat.append(cp.asarray(translation_vals, dtype=cp.float32))

        sizes = [int(g.size) for g in grids_flat]
        if any(s <= 0 for s in sizes):
            return grids_flat, sizes, [1] * len(sizes), seg_param_specs

        strides: List[int] = []
        prod = 1
        for j in range(len(sizes) - 1, -1, -1):
            strides.append(prod)
            prod *= sizes[j]
        strides = list(reversed(strides))
        return grids_flat, sizes, strides, seg_param_specs

    def _gather_batch_params(
        self,
        grids_flat: List[cp.ndarray],
        sizes: List[int],
        strides: List[int],
        seg_param_specs: List[Dict[str, Any]],
        start: int,
        count: int,
    ) -> Tuple[List[Dict[str, cp.ndarray]], cp.ndarray]:
        if count <= 0:
            return [], cp.asarray([], dtype=cp.float32)

        lin = cp.arange(start, start + count, dtype=cp.int64)

        idx_each_dim: List[cp.ndarray] = []
        for sz, st in zip(sizes, strides):
            if sz == 1:
                idx_each_dim.append(cp.zeros_like(lin))
            else:
                idx_each_dim.append((lin // st) % sz)

        gathered: List[cp.ndarray] = []
        for arr, idx in zip(grids_flat, idx_each_dim):
            gathered.append(cp.take(arr, idx).astype(cp.float32, copy=False))

        per_seg_samples: List[Dict[str, cp.ndarray]] = []
        k = 0
        for spec in seg_param_specs:
            th = gathered[k + 0]
            ph = gathered[k + 1]
            ltot = gathered[k + 2]
            lpas = gathered[k + 3]
            per_seg_samples.append(
                {
                    "theta": th,
                    "phi": ph,
                    "L_total": ltot,
                    "L_passive": lpas,
                    "phi_mode": spec.get("phi_mode", None),
                    "phi_coupling": spec.get("phi_coupling", None),
                }
            )
            k += 4
        translation_vals = gathered[-1]

        # φ coupling across segments.
        prev_phi_vec: Optional[cp.ndarray] = None
        two_pi = float(2.0 * np.pi)

        def map_coupling(mode_raw: Any) -> Any:
            if mode_raw is None:
                return None
            if isinstance(mode_raw, (float, int)):
                return float(mode_raw)
            s = str(mode_raw).lower()
            if s in ("none", "free"):
                return None
            if s in ("lock_prev", "same"):
                return "same"
            if s in ("opposite", "opp"):
                return "opposite"
            return s

        for d in per_seg_samples:
            mode_raw = d.get("phi_coupling", None)
            if mode_raw is None:
                mode_raw = d.get("phi_mode", None)
            mode = map_coupling(mode_raw)

            if mode is None:
                prev_phi_vec = d["phi"]
            elif mode == "same":
                if prev_phi_vec is not None:
                    d["phi"] = prev_phi_vec
                prev_phi_vec = d["phi"]
            elif mode == "opposite":
                if prev_phi_vec is None:
                    d["phi"] = (d["phi"] + np.pi).astype(cp.float32) % two_pi
                else:
                    d["phi"] = (prev_phi_vec + np.pi).astype(cp.float32) % two_pi
                prev_phi_vec = d["phi"]
            elif isinstance(mode, float):
                if prev_phi_vec is None:
                    d["phi"] = cp.asarray(mode, dtype=cp.float32).repeat(
                        d["phi"].shape[0]
                    )
                else:
                    d["phi"] = (prev_phi_vec + float(mode)).astype(cp.float32) % two_pi
                prev_phi_vec = d["phi"]
            else:
                raise ValueError(f"Unsupported phi coupling: {mode_raw!r}")
        return per_seg_samples, translation_vals

    # ---- Candidate insertion & NMS ----

    def _try_insert_candidates(
        self,
        best_pool: List[Dict[str, Any]],
        cand_batch: Iterable[int],
        per_seg_samples: List[Dict[str, cp.ndarray]],
        translation_vals: cp.ndarray,
        pts_cp: cp.ndarray,
        Ts_cp: cp.ndarray,
        pos_err: cp.ndarray,
        angle_target_deg: float,
        cosang: cp.ndarray,
        bevel_enabled: bool = False,
        c_bev_max: Optional[cp.ndarray] = None,
        Rb: Optional[cp.ndarray] = None,
        n_hat: Optional[cp.ndarray] = None,
        alpha: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        for idx in cand_batch:
            i = int(idx)

            cos_i = float(cp.clip(cosang[i], -1.0, 1.0).get())
            ang_deg = float(np.degrees(np.arccos(cos_i)))
            ang_err_deg = abs(ang_deg - float(angle_target_deg))

            meta: Dict[str, Any] = {
                "cos_angle": cos_i,
                "z_axis": cp.asnumpy(Ts_cp[i, :3, 2]),
            }
            if extra_meta:
                meta.update(extra_meta)

            if (
                bevel_enabled
                and (c_bev_max is not None)
                and (Rb is not None)
                and (n_hat is not None)
            ):
                a = cp.matmul(Rb[i].T, n_hat.reshape(3, 1)).reshape(3)
                ax, ay, az = float(a[0].get()), float(a[1].get()), float(a[2].get())
                gamma_star = float(np.arctan2(ay, ax)) % (2.0 * np.pi)

                cbev = float(c_bev_max[i].get())
                bevel_err_deg = float(np.degrees(np.arccos(np.clip(cbev, -1.0, 1.0))))

                meta.update(
                    {
                        "tip_roll": gamma_star,
                        "bevel_cos_max": cbev,
                        "bevel_err_deg": bevel_err_deg,
                        "bevel_angle_deg": (
                            float(np.degrees(alpha)) if alpha is not None else 45.0
                        ),
                        "normal_unit": cp.asnumpy(n_hat.reshape(-1)),
                    }
                )
            else:
                meta["normal_unit"] = (
                    cp.asnumpy(n_hat.reshape(-1)) if n_hat is not None else None
                )

            seg_solutions: List[SegmentSolution] = []
            for seg_spec, d in zip(self.ws.segments, per_seg_samples):
                if "L_active" in d:
                    l_act = float(d["L_active"][i].get())
                else:
                    l_act = float((d["L_total"][i].get() - d["L_passive"][i].get()))
                    l_act = max(l_act, 0.0)

                seg_solutions.append(
                    SegmentSolution(
                        name=getattr(seg_spec, "name", "seg"),
                        theta=float(d["theta"][i].get()),
                        phi=float(d["phi"][i].get()),
                        L_total=float(d["L_total"][i].get()),
                        L_passive=float(d["L_passive"][i].get()),
                        L_active=l_act,
                    )
                )

            best_pool.append(
                {
                    "pos_err": float(pos_err[i].get()),
                    "ang_err_deg": ang_err_deg,
                    "translation": float(translation_vals[i].get()),
                    "segments": seg_solutions,
                    "end_T": cp.asnumpy(Ts_cp[i]),
                    "end_p": cp.asnumpy(pts_cp[i]),
                    "meta": meta,
                }
            )

    def _nms_solutions(
        self, items: List[Dict[str, Any]], max_keep: int
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        if not bool(getattr(self.opts, "nms_enable", True)):
            return items[:max_keep]

        theta_thresh = float(getattr(self.opts, "nms_theta_deg", 2.0))
        phi_thresh = float(getattr(self.opts, "nms_th_phi_deg", 5.0))
        trans_thresh = float(getattr(self.opts, "nms_translation", 1e-3))

        def _feat(sol, eps_theta=1e-6, eps_L=1e-6):
            thetas, phis = [], []
            for s in sol["segments"]:
                thetas.append(s.theta)
                if (abs(s.theta) > eps_theta) and (s.L_active > eps_L):
                    phis.append(s.phi)
                else:
                    phis.append(0.0)
            return np.array(thetas + phis + [sol["translation"]], dtype=np.float64)

        def _close(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            va, vb = _feat(a), _feat(b)
            n = len(a["segments"])
            d_theta = np.max(np.abs(np.degrees(va[:n] - vb[:n])))
            d_phi = np.max(np.abs(np.degrees(va[n : 2 * n] - vb[n : 2 * n])))
            d_trans = abs(va[-1] - vb[-1])
            return (
                (d_theta <= theta_thresh)
                and (d_phi <= phi_thresh)
                and (d_trans <= trans_thresh)
            )

        kept: List[Dict[str, Any]] = []
        for s in items:
            if len(kept) >= max_keep:
                break
            if not any(_close(s, t) for t in kept):
                kept.append(s)

        if len(kept) < max_keep:
            for s in items:
                if s not in kept:
                    kept.append(s)
                    if len(kept) >= max_keep:
                        break
        return kept[:max_keep]
