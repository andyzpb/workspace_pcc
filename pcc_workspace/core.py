from typing import Sequence, List, Optional
import cupy as cp
import numpy as np

from .specs import SegmentSpec, TranslationSpec


class PCCWorkspace:
    def __init__(
        self,
        segments: Sequence[SegmentSpec],
        translation: Optional[TranslationSpec] = None,
        small_L_eps: float = 1e-4,
    ):
        if len(segments) == 0:
            raise ValueError("need at least one SegmentSpec")
        self.segments: List[SegmentSpec] = list(segments)
        self.translation = translation or TranslationSpec(0.0, 0.0, 1)
        self.small_L_eps = float(small_L_eps)

    @staticmethod
    def _rot(
        phi: cp.ndarray, kappa: cp.ndarray, theta: cp.ndarray, L: cp.ndarray
    ) -> cp.ndarray:
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

        R = cp.stack(
            [
                cp.stack(
                    [
                        cos2 * (cos_t - 1.0) + 1.0,
                        sincos * (cos_t - 1.0),
                        cos_phi * sin_t,
                    ],
                    axis=-1,
                ),
                cp.stack(
                    [
                        sincos * (cos_t - 1.0),
                        sin2 * (cos_t - 1.0) + cos_t,
                        sin_phi * sin_t,
                    ],
                    axis=-1,
                ),
                cp.stack([-cos_phi * sin_t, -sin_phi * sin_t, cos_t], axis=-1),
            ],
            axis=1,
        )

        px = cos_phi * one_minus_cos_t / k_safe
        py = sin_phi * one_minus_cos_t / k_safe
        pz = sin_t / k_safe

        small = cp.abs(theta) < 1e-4
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

    @staticmethod
    def _linspace_cp(a: float, b: float, n: int) -> cp.ndarray:
        if n <= 1:
            return cp.asarray([0.5 * (a + b)], dtype=cp.float32)
        return cp.linspace(a, b, n, dtype=cp.float32)

    @staticmethod
    def _phi_samples(n: int) -> cp.ndarray:
        if n <= 1:
            return cp.asarray([0.0], dtype=cp.float32)
        return cp.linspace(0.0, 2 * cp.pi, n, endpoint=False, dtype=cp.float32)

    def _build_parameter_grid(self):
        params_per_seg = []
        prev_phi = None

        for i, seg in enumerate(self.segments):
            theta_vals = (
                self._linspace_cp(0.0, seg.theta_max, max(seg.samples_theta, 1))
                if seg.theta_sign
                else self._linspace_cp(
                    -seg.theta_max, seg.theta_max, max(seg.samples_theta, 1)
                )
            )
            L_vals = self._linspace_cp(seg.L_min, seg.L_max, max(seg.samples_length, 1))

            if seg.phi_coupling is None:
                phi_vals = self._phi_samples(max(seg.samples_phi, 1))
            elif seg.phi_coupling == "lock_prev":
                if prev_phi is None:
                    phi_vals = self._phi_samples(max(seg.samples_phi, 1))
                else:
                    phi_vals = cp.asarray([0.0], dtype=cp.float32)
            elif isinstance(seg.phi_coupling, (float, int)):
                phi_vals = cp.asarray([float(seg.phi_coupling)], dtype=cp.float32)
            else:
                raise ValueError(f"Unsupported phi_coupling: {seg.phi_coupling}")

            params_per_seg.append(
                {
                    "theta": theta_vals,
                    "phi": phi_vals,
                    "L": L_vals,
                    "phi_mode": seg.phi_coupling,
                }
            )

            prev_phi = params_per_seg[-1]["phi"]

        translation_vals = self._linspace_cp(
            self.translation.d_min,
            self.translation.d_max,
            max(self.translation.samples, 1),
        )
        return params_per_seg, translation_vals

    def _compose_fk_from_params(
        self,
        per_seg_samples: List[cp.ndarray],
        translation_vals: cp.ndarray,
    ):
        N = translation_vals.size
        T_ins = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (N, 4, 4)).copy()
        T_ins[:, 2, 3] = translation_vals

        T = T_ins

        for i, d in enumerate(per_seg_samples):
            th = d["theta"]
            ph = d["phi"]
            L = d["L"]

            L_eps = self.small_L_eps

            mask_zero = L < L_eps
            mask_pos = ~mask_zero

            N = L.shape[0]
            Ti = cp.broadcast_to(cp.eye(4, dtype=cp.float32), (N, 4, 4)).copy()

            if mask_pos.any():
                th_pos = th[mask_pos]
                L_pos = L[mask_pos]
                ph_pos = ph[mask_pos]
                k_pos = th_pos / L_pos
                Ti_pos = self._rot(ph_pos, k_pos, th_pos, L_pos)  # (M, 4, 4)
                Ti[mask_pos] = Ti_pos

            T = T @ Ti

        pts = T[:, :3, 3]
        return pts, T

    def sample(self):
        params_per_seg, traslation_vals = self._build_parameter_grid()

        grids = []
        grid_names = []

        for i, d in enumerate(params_per_seg):
            grids.extend([d["theta"], d["phi"], d["L"]])
            grid_names.extend([f"th{i}", f"ph{i}", f"L{i}"])

        grids.append(traslation_vals)
        grid_names.append("trans")

        mesh = cp.meshgrid(*grids, indexing="ij")
        flat = [m.ravel() for m in mesh]

        per_seg_samples = []
        idx = 0
        for i, d in enumerate(params_per_seg):
            th = flat[idx + 0]
            ph = flat[idx + 1]
            L = flat[idx + 2]
            idx += 3
            per_seg_samples.append(
                {"theta": th, "phi": ph, "L": L, "phi_mode": d["phi_mode"]}
            )

        ins = flat[-1]

        prev_phi_vec = None
        for i, d in enumerate(per_seg_samples):
            mode = d["phi_mode"]
            if mode is None:
                prev_phi_vec = d["phi"]
            elif mode == "lock_prev":
                if prev_phi_vec is not None:
                    d["phi"] = prev_phi_vec
                prev_phi_vec = d["phi"]
            elif isinstance(mode, (float, int)):
                if prev_phi_vec is None:
                    d["phi"] = cp.asarray((float(mode),), dtype=cp.float32).repeat(
                        d["phi"].shape[0]
                    )
                else:
                    d["phi"] = (prev_phi_vec + float(mode)) % (2 * np.pi)
                prev_phi_vec = d["phi"]
            else:
                raise ValueError(f"Unsupported phi_coupling: {mode}")

        pts, Ts = self._compose_fk_from_params(per_seg_samples, ins)
        return cp.asnumpy(pts), cp.asnumpy(Ts)
