from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
import numpy as np
from pccik_native.pcc_workspace.core_ik_cf import PCCIKClosedForm
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

class PCCIKClosedFormSingle(PCCIKClosedForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seg1b = None
        self._seg2b = None
        self._consts = None

    @staticmethod
    def _as_c_float64(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float64, order="C")

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
        b.phi_max = float(getattr(seg, "phi_max", 2.0 * math.pi) if has_phi else 2.0 * math.pi)
        b.L_min = float(getattr(seg, "L_min", 0.0) or 0.0)
        b.L_max = float(getattr(seg, "L_max", 1e9) or 1e9)
        b.passive_L_min = float(seg.passive_L_min)
        b.passive_L_max = float(seg.passive_L_max)
        b.active_L_max = float(getattr(seg, "active_L_max", b.L_max))
        b.bevel_angle_deg = float(getattr(seg, "bevel_angle_deg", 45.0))
        b.rigid_tip_length = float(getattr(seg, "rigid_tip_length", self.inner_rigid_tip if not outer else 0.0))
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

    def solve(self, touch: TouchPointSpec):
        P = self._as_c_float64(np.asarray(touch.coordinates))
        n = self._as_c_float64(np.asarray(touch.normal))
        seg1b, seg2b, consts = self._native_objs()
        topk = int(getattr(self.opts, "topk", 1) or 1)
        res = _native.solve(P, n, seg1b, seg2b, consts, topk)
        if res is None or len(res) == 0:
            return []
        sols: List[IKSolution] = []
        for d in res:
            th1 = float(d["theta1"])
            ph1 = float(d["phi1"])
            th2 = float(d["theta2"])
            ph2 = float(d["phi2"])
            L2p = float(d["l2p"])
            dtra = float(d["translation"])
            pos_err = float(d["pos_err"])
            ang_err = float(d["ang_err_deg"])
            end_T = np.asarray(d["end_T"], dtype=np.float64)
            end_p = np.asarray(d["end_p_world"], dtype=np.float64)
            seg_outer = SegmentSolution(
                name=self.seg1.name,
                theta=th1,
                phi=ph1,
                L_total=float(self._s1 + self._L1p),
                L_passive=float(self._L1p),
                L_active=float(self._s1),
            )
            seg_inner = SegmentSolution(
                name=self.seg2.name,
                theta=th2,
                phi=ph2,
                L_total=float(self.s2_fixed + L2p),
                L_passive=L2p,
                L_active=float(self.s2_fixed),
            )
            report = {
                "outer": {
                    "bending_deg": float(np.degrees(th1)),
                    "rotation_deg": float(np.degrees(ph1)),
                    "translation": 0.0,
                },
                "inner": {
                    "bending_deg": float(np.degrees(th2)),
                    "rotation_deg": float(np.degrees(ph2)),
                    "translation": dtra,
                },
            }
            sols.append(
                IKSolution(
                    segments=[seg_outer, seg_inner],
                    translation=dtra,
                    reachable=True,
                    end_T=end_T,
                    end_p=end_p,
                    pos_err=pos_err,
                    ang_err_deg=ang_err,
                    meta={"report": report, "bevel_world": np.asarray(d["bevel_world"], np.float64)},
                )
            )
        sols.sort(key=lambda s: (s.pos_err, s.ang_err_deg))
        return sols[:topk]

def create_solver(
    segments: List[SegmentSpec],
    translation: Optional[TranslationSpec] = None,
    opts: Optional[IKOptions] = None,
    inner_rigid_tip: float = 0.003,
    s2_fixed: float = 0.006,
    debug: bool = False,
) -> PCCIKClosedForm:
    return PCCIKClosedFormSingle(
        segments=segments,
        translation=translation,
        opts=opts,
        inner_rigid_tip=inner_rigid_tip,
        s2_fixed=s2_fixed,
        debug=debug,
    )
