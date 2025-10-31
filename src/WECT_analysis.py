import time
import numpy as np

from pccik_native.pcc_workspace.specs import (
    SegmentSpec,
    TranslationSpec,
    IKOptions,
)
import pccik_native.testcases as tc
import os
try:
    os.nice(-20)
except:
    pass
from IK_pipeline import create_solver 
import gc
# --------------------------- Segment definitions ---------------------------
outer = SegmentSpec(
    name="outer",
    L_min=0.05,
    L_max=0.05,
    samples_length=1,
    passive_L_min=0.0,
    passive_L_max=0.0,
    samples_passive_length=1,
    theta_min=np.deg2rad(-135.2),
    theta_max=np.deg2rad(135.2),
    phi_min=np.deg2rad(-180.0),
    phi_max=np.deg2rad(180.0) - 1e-5,
    samples_theta=270,
    samples_phi=720,
)

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

tr = TranslationSpec(d_min=0.0, d_max=0.0, samples=1)

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

solver = create_solver([inner, outer], tr, opts)
solver.inner_rigid_tip = 0.003


import types, math

def _theta1_candidates_fast(self, P_world, n_star, th_lo, th_hi,
                            coarse_n=33, keep_top=4,
                            refine_halfspan_deg=5.0, refine_n=11):
    seg1b, seg2b, consts = self._native_objs()
    phi_grid = np.linspace(-math.pi, math.pi, 24, endpoint=False, dtype=float)

    def score(theta1: float) -> float:
        res = self._native_eval_phi_scan(theta1, P_world, n_star, phi_grid, k_keep=1)
        if len(res) == 0:
            return float("+inf")
        r0 = res[0]
        return float(r0["pos_err"]) + 1e-3 * float(r0["ang_err_deg"])

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

def _native_eval_phi_scan(self, theta1, P_star, n_star, phi_list, k_keep=1):
    seg1b, seg2b, consts = self._native_objs()
    import pccik_native._core as _core
    res = _core.phi_scan(
        float(theta1),
        np.asarray(phi_list, np.float64),
        np.asarray(P_star, np.float64),
        np.asarray(n_star, np.float64),
        seg1b,
        seg2b,
        consts,
        int(k_keep),
    )
    return res

solver._theta1_candidates = types.MethodType(_theta1_candidates_fast, solver)
solver._native_eval_phi_scan = types.MethodType(_native_eval_phi_scan, solver)


def bench_native_evaluate_once(solver, loops=50000):
    import pccik_native._core as _core
    def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, float)
        n = float(np.linalg.norm(v))
        return v * 0.0 if n < 1e-12 else (v / n)
    
    seg1b, seg2b, consts = solver._native_objs()

    touch = tc.N01[1]
    P = np.asarray(touch.coordinates, float).reshape(3)
    n = _normalize(np.asarray(touch.normal, float).reshape(3))

    theta1 = 0.2
    phi1 = 1.0

    for _ in range(200):
        _core.evaluate_once(theta1, phi1, P, n, seg1b, seg2b, consts)

    t0 = time.perf_counter()
    for _ in range(loops):
        _core.evaluate_once(theta1, phi1, P, n, seg1b, seg2b, consts)
    t1 = time.perf_counter()

    per_call = (t1 - t0) / loops
    print(f"[native evaluate_once] {per_call*1e6:.2f} us / call  ({loops} loops)")
    return per_call


def bench_full_solve(solver, loops=20000):
    touch = tc.N01[1]
    for _ in range(100):
        solver.solve(touch)

    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        solver.solve(touch)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times)
    print(f"[full solve] mean={arr.mean()*1e3:.3f} ms, "
          f"p95={np.percentile(arr,95)*1e3:.3f} ms, "
          f"max={arr.max()*1e3:.3f} ms, n={loops}")

if __name__ == "__main__":
    gc.disable()
    t_eval = bench_native_evaluate_once(solver)
    bench_full_solve(solver)
    gc.enable()
