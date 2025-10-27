from __future__ import annotations
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from pccik_native.pcc_workspace.core_ik_cf import PCCIKClosedForm, _Brent

from pccik_native.pcc_workspace.specs import (
    IKSolution,
    SegmentSpec,
    TranslationSpec,
    TouchPointSpec,
    IKOptions,
    SegmentSolution,
)


from pccik_native import _core as _native

_HAVE_NATIVE = True


class PCCIKClosedFormFast(PCCIKClosedForm):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._seg1b = None
        self._seg2b = None
        self._consts = None

    @staticmethod
    def _as_c_float64(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float64, order="C")

    def _native_objs(self):
        if not _HAVE_NATIVE:
            return None, None, None
        if self._seg1b is None:
            self._seg1b = self._to_native_seg(self.seg1, outer=True)
        if self._seg2b is None:
            self._seg2b = self._to_native_seg(self.seg2, outer=False)
        if self._consts is None:
            self._consts = self._to_native_consts()
        return self._seg1b, self._seg2b, self._consts

    def _to_native_seg(self, seg: SegmentSpec, outer: bool):
        if not _HAVE_NATIVE:
            return None
        b = _native.SegmentBounds()
        theta_max = float(getattr(seg, "theta_max", math.pi) or math.pi)
        b.theta_max = theta_max
        theta_min = getattr(seg, "theta_min", None)
        b.theta_min = float(theta_min if theta_min is not None else -theta_max)
        has_phi = (
            getattr(seg, "phi_min", None) is not None
            and getattr(seg, "phi_max", None) is not None
        )
        b.has_phi_bounds = bool(has_phi)
        b.phi_min = float(getattr(seg, "phi_min", 0.0) if has_phi else 0.0)
        b.phi_max = float(
            getattr(seg, "phi_max", 2.0 * math.pi) if has_phi else 2.0 * math.pi
        )
        b.L_min = float(getattr(seg, "L_min", 0.0) or 0.0)
        b.L_max = float(getattr(seg, "L_max", 1e9) or 1e9)
        b.passive_L_min = float(seg.passive_L_min)
        b.passive_L_max = float(seg.passive_L_max)
        b.active_L_max = float(getattr(seg, "active_L_max", b.L_max))
        b.bevel_angle_deg = float(getattr(seg, "bevel_angle_deg", 45.0))
        b.rigid_tip_length = float(
            getattr(seg, "rigid_tip_length", self.inner_rigid_tip if not outer else 0.0)
        )
        return b

    def _to_native_consts(self):
        if not _HAVE_NATIVE:
            return None
        c = _native.SolverConst()
        c.s1 = float(self._s1)
        c.L1p = float(self._L1p)
        c.s2_fixed = float(self.s2_fixed)
        c.d_min = float(self.translation.d_min)
        c.d_max = float(self.translation.d_max)
        c.pos_tol = float(self._pos_tol)
        c.bevel_tol_deg = float(getattr(self.opts, "bevel_tol_deg", 1.0))
        c.angle_target_deg = float(getattr(self.opts, "angle_target_deg", 45.0))
        return c

    def _native_eval(
        self, theta1: float, phi1: float, P_star: np.ndarray, n_star: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        if not _HAVE_NATIVE or not hasattr(_native, "evaluate_once"):
            return None
        seg1b, seg2b, consts = self._native_objs()
        out = _native.evaluate_once(
            float(theta1),
            float(phi1),
            self._as_c_float64(P_star),
            self._as_c_float64(n_star),
            seg1b,
            seg2b,
            consts,
        )
        return out

    def _phi_scan_vectorized(
        self,
        theta1: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        phi_list: np.ndarray,
        k_keep: int = 6,
    ) -> List[Tuple[float, Dict[str, Any]]]:

        if not _HAVE_NATIVE:
            return super()._phi_scan_vectorized(
                theta1, P_star, n_star, phi_list, k_keep
            )

        seg1b, seg2b, consts = self._native_objs()

        res = _native.phi_scan(
            float(theta1),
            self._as_c_float64(phi_list),
            self._as_c_float64(P_star),
            self._as_c_float64(n_star),
            seg1b,
            seg2b,
            consts,
            int(k_keep),
        )

        tmp: List[Tuple[float, Dict[str, Any]]] = []
        for d in res:
            pos_err = float(d["pos_err"])
            ang_err = float(d["ang_err_deg"])
            J = pos_err + 1e-3 * ang_err

            seg0 = SegmentSolution(
                name=self.seg1.name,
                theta=float(d["theta1"]),
                phi=float(d["phi1"]),
                L_total=float(self._s1 + self._L1p),
                L_passive=float(self._L1p),
                L_active=float(self._s1),
            )
            cand = dict(
                pos_err=pos_err,
                ang_err_deg=ang_err,
                translation=float(d["translation"]),
                abs_d=float(d["abs_d"]),
                segments=[seg0],
            )
            tmp.append((J, cand))

        tmp.sort(key=lambda t: t[0])
        return tmp

    def _refine_phi1_local(
        self,
        theta1: float,
        phi_seed: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        halfspan_deg: float = 6.0,
        maxiter: int = 60,
    ):
        if not _HAVE_NATIVE or not hasattr(_native, "brent_refine_phi"):
            return super()._refine_phi1_local(
                theta1, phi_seed, P_star, n_star, halfspan_deg, maxiter
            )

        seg1b, seg2b, consts = self._native_objs()
        out = _native.brent_refine_phi(
            float(theta1),
            float(phi_seed),
            float(halfspan_deg),
            int(maxiter),
            self._as_c_float64(P_star),
            self._as_c_float64(n_star),
            seg1b,
            seg2b,
            consts,
        )
        if out is None:
            return float(phi_seed), None

        phi_best = float(out["phi1"])
        seg0 = SegmentSolution(
            name=self.seg1.name,
            theta=float(out["theta1"]) if "theta1" in out else float(theta1),
            phi=float(out["phi1_eval"]) if "phi1_eval" in out else float(phi_best),
            L_total=float(self._s1 + self._L1p),
            L_passive=float(self._L1p),
            L_active=float(self._s1),
        )
        cand_best = dict(
            pos_err=float(out["pos_err"]),
            ang_err_deg=float(out["ang_err_deg"]),
            translation=float(out["translation"]),
            abs_d=float(out["abs_d"]),
            segments=[seg0],
        )
        return phi_best, cand_best

    def _lm_polish(
        self,
        theta1: float,
        phi1: float,
        P_star: np.ndarray,
        n_star: np.ndarray,
        iters: int = 8,
    ):
        if not _HAVE_NATIVE or not hasattr(_native, "lm_polish"):
            return super()._lm_polish(theta1, phi1, P_star, n_star, iters)

        seg1b = self._to_native_seg(self.seg1, outer=True)
        seg2b = self._to_native_seg(self.seg2, outer=False)
        consts = self._to_native_consts()
        out = _native.lm_polish(
            float(theta1),
            float(phi1),
            int(iters),
            np.asarray(P_star, float),
            np.asarray(n_star, float),
            seg1b,
            seg2b,
            consts,
        )
        th_fin = float(out["theta1"])
        ph_fin = float(out["phi1"])
        polished = self._evaluate_once(th_fin, ph_fin, P_star, n_star)
        return th_fin, ph_fin, polished

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
    ) -> list[float]:
        if not _HAVE_NATIVE:
            return super()._theta1_candidates(
                P_world,
                n_star,
                th_lo,
                th_hi,
                coarse_n=coarse_n,
                keep_top=keep_top,
                refine_halfspan_deg=refine_halfspan_deg,
                refine_n=refine_n,
            )

        seg1b = self._to_native_seg(self.seg1, outer=True)
        seg2b = self._to_native_seg(self.seg2, outer=False)
        consts = self._to_native_consts()

        phi_grid = np.linspace(-math.pi, math.pi, 48, endpoint=False, dtype=float)

        def score(theta1: float) -> float:
            res = _native.phi_scan(
                float(theta1),
                phi_grid,
                np.asarray(P_world, float),
                np.asarray(n_star, float),
                seg1b,
                seg2b,
                consts,
                1,
            )
            if len(res) == 0:
                return float("+inf")
            r0 = res[0]
            return float(r0["pos_err"]) + 0.001 * float(r0["ang_err_deg"])

        theta_grid = np.linspace(th_lo, th_hi, int(max(3, coarse_n)), dtype=float)
        scores = np.array([score(th) for th in theta_grid], dtype=float)
        order = np.argsort(scores)
        seeds = theta_grid[order[: int(max(1, keep_top))]]

        out = set()
        half = float(np.deg2rad(refine_halfspan_deg))
        for s in seeds:
            a = max(th_lo, s - half)
            b = min(th_hi, s + half)
            finer = np.linspace(a, b, int(max(3, refine_n)), dtype=float)
            pairs = [(score(x), float(x)) for x in finer]
            pairs.sort(key=lambda t: t[0])
            for _, x in pairs[:3]:
                out.add(x)

        out.update([+1e-3, -1e-3])

        return sorted(out)


def create_solver(
    segments: List[SegmentSpec],
    translation: Optional[TranslationSpec] = None,
    opts: Optional[IKOptions] = None,
    inner_rigid_tip: float = 0.003,
    s2_fixed: float = 0.006,
    debug: bool = False,
) -> PCCIKClosedForm:
    cls = PCCIKClosedFormFast if _HAVE_NATIVE else PCCIKClosedForm
    return cls(
        segments=segments,
        translation=translation,
        opts=opts,
        inner_rigid_tip=inner_rigid_tip,
        s2_fixed=s2_fixed,
        debug=debug,
    )
