# force_scan_bench.py
import os
import gc
import math
import time
import types
import numpy as np

from pccik_native.pcc_workspace.specs import SegmentSpec, TranslationSpec, IKOptions
import pccik_native.testcases as tc
from IK_pipeline import create_solver

try:
    os.nice(-20)
except Exception:
    pass

import numpy as np, math
PHI_GRID_24 = np.linspace(-math.pi, math.pi, 16, endpoint=False, dtype=np.float64)
LOCAL_SPAN = float(np.deg2rad(4.0))
LOCAL_OFFSETS_25 = np.linspace(-LOCAL_SPAN, +LOCAL_SPAN, 25, dtype=np.float64)



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
if hasattr(solver, "inner_rigid_tip"):
    solver.inner_rigid_tip = 0.003

# --------------------------- Core instrumentation (counts) ---------------------------
import pccik_native._core as _core
PHI_SCAN_CALLS = 0
EVAL_ONCE_CALLS = 0

_core_phi_scan_orig = _core.phi_scan
_core_evaluate_once_orig = _core.evaluate_once

def _phi_scan_counter(*args, **kwargs):
    global PHI_SCAN_CALLS
    PHI_SCAN_CALLS += 1
    return _core_phi_scan_orig(*args, **kwargs)

def _evaluate_once_counter(*args, **kwargs):
    global EVAL_ONCE_CALLS
    EVAL_ONCE_CALLS += 1
    return _core_evaluate_once_orig(*args, **kwargs)

def enable_core_counters():
    _core.phi_scan = _phi_scan_counter
    _core.evaluate_once = _evaluate_once_counter

def disable_core_counters():
    _core.phi_scan = _core_phi_scan_orig
    _core.evaluate_once = _core_evaluate_once_orig

# --------------------------- Helpers ---------------------------
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)

# Baselines (same style you used earlier)
def bench_native_evaluate_once(solver, loops=50_000):
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

def bench_phi_scan(solver, loops=20_000, theta1=0.2, M=24):
    seg1b, seg2b, consts = solver._native_objs()
    touch = tc.N01[1]
    P = np.asarray(touch.coordinates, float).reshape(3)
    n = _normalize(np.asarray(touch.normal, float).reshape(3))
    phi_grid = np.linspace(-math.pi, math.pi, int(M), endpoint=False, dtype=float)
    for _ in range(200):
        _core.phi_scan(float(theta1), phi_grid, P, n, seg1b, seg2b, consts, 1)
    t0 = time.perf_counter()
    for _ in range(loops):
        _core.phi_scan(float(theta1), phi_grid, P, n, seg1b, seg2b, consts, 1)
    t1 = time.perf_counter()
    per_call = (t1 - t0) / loops
    print(f"[phi_scan({M})] {per_call*1e6:.2f} us / call  ({loops} loops)")
    return per_call

# --------------------------- Pure-Scan θ×φ pipeline ---------------------------
def theta_candidates_scan(P_world, n_star, th_lo, th_hi, seg1b, seg2b, consts,
                          M=16, coarse_n=15, keep_top=3, refine_halfspan_deg=5.0, refine_n=9):
    phi_grid = np.linspace(-math.pi, math.pi, int(M), endpoint=False, dtype=float)
    def score(theta1: float) -> float:
        res = _core.phi_scan(float(theta1), phi_grid,
                             np.asarray(P_world, np.float64),
                             np.asarray(n_star, np.float64),
                             seg1b, seg2b, consts, 1)
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

def forced_scan_solve(solver, touch,
                      M=24, coarse_n=33, keep_top=4,
                      refine_halfspan_deg=5.0, refine_n=11,
                      local_refine_phi_deg=4.0, local_refine_phi_points=25):
    import pccik_native._core as _core
    seg1b, seg2b, consts = solver._native_objs()
    P = np.asarray(touch.coordinates, np.float64).reshape(3)
    n = np.asarray(touch.normal, np.float64).reshape(3); n /= (np.linalg.norm(n)+1e-12)

    phi_grid = PHI_GRID_24 if M == 24 else np.linspace(-math.pi, math.pi, int(M), endpoint=False, dtype=np.float64)
    span = float(np.deg2rad(local_refine_phi_deg))
    local_offsets = LOCAL_OFFSETS_25 if (abs(span-LOCAL_SPAN)<1e-12 and local_refine_phi_points==25) \
        else np.linspace(-span, +span, int(max(5, local_refine_phi_points)), dtype=np.float64)

    th_lo = float(seg1b.theta_min); th_hi = float(seg1b.theta_max)

    def score_theta(theta1: float) -> float:
        res = _core.phi_scan(float(theta1), phi_grid, P, n, seg1b, seg2b, consts, 1)
        if len(res) == 0: return float("+inf")
        r0 = res[0]
        return float(r0["pos_err"]) + 1e-3*float(r0["ang_err_deg"])

    theta_grid = np.linspace(th_lo, th_hi, int(max(3, coarse_n)), dtype=np.float64)
    scores = np.array([score_theta(th) for th in theta_grid], dtype=np.float64)
    seeds = theta_grid[np.argsort(scores)[:int(max(1, keep_top))]]

    ths = set()
    half = float(np.deg2rad(refine_halfspan_deg))
    for s in seeds:
        a = max(th_lo, s - half); b = min(th_hi, s + half)
        finer = np.linspace(a, b, int(max(3, refine_n)), dtype=np.float64)
        pairs = [(score_theta(x), float(x)) for x in finer]
        pairs.sort(key=lambda t: t[0])
        for _, x in pairs[:3]:
            ths.add(x)
    ths.update([+1e-3, -1e-3])
    ths = sorted(ths)

    best = None; bestJ = float("+inf")
    for th in ths:
        top_phis = _core.phi_scan(float(th), phi_grid, P, n, seg1b, seg2b, consts, 3)
        for cand in top_phis:
            phi_c = float(cand["phi1"])
            phi_local = (phi_c + local_offsets).astype(np.float64, copy=False)
            loc_res = _core.phi_scan(float(th), phi_local, P, n, seg1b, seg2b, consts, 1)
            if len(loc_res) == 0:
                continue
            d = loc_res[0]
            J = float(d["pos_err"]) + 1e-3*float(d["ang_err_deg"]) + 1e-7*float(d["abs_d"])
            if J < bestJ:
                bestJ = J; best = d

    return best

# --------------------------- Bench forced-scan ---------------------------
def bench_forced_scan(solver, loops=1000, **scan_kwargs):
    touch = tc.N01[1]
    times = []
    for _ in range(50):
        forced_scan_solve(solver, touch, **scan_kwargs)
    enable_core_counters()
    try:
        global PHI_SCAN_CALLS, EVAL_ONCE_CALLS
        PHI_SCAN_CALLS = 0; EVAL_ONCE_CALLS = 0
        t0 = time.perf_counter()
        for _ in range(loops):
            forced_scan_solve(solver, touch, **scan_kwargs)
        t1 = time.perf_counter()
    finally:
        disable_core_counters()
    per = (t1 - t0) / loops
    print(f"[forced-scan solve] {per*1e3:.3f} ms / call  ({loops} loops)")
    print(f"  calls/solve: phi_scan={PHI_SCAN_CALLS/loops:.2f}, evaluate_once={EVAL_ONCE_CALLS/loops:.2f}")
    return per, PHI_SCAN_CALLS/loops, EVAL_ONCE_CALLS/loops

# --------------------------- Derive WCET from measured constants ---------------------------
def compute_wcet_from_counts(t_eval, t_phiM, m_bar, e_bar):
    wcet = m_bar * t_phiM + e_bar * t_eval
    print(f"[WCET (from counts)] m̄*t_phi + ē*t_eval = {wcet*1e3:.3f} ms "
          f"(m̄={m_bar:.2f}, ē={e_bar:.2f})")
    return wcet

# --------------------------- Entry ---------------------------
if __name__ == "__main__":
    gc.disable()
    try:
        t_eval = bench_native_evaluate_once(solver)             # seconds / evaluate_once
        t_phi24 = bench_phi_scan(solver, M=16)                  # seconds / phi_scan(24)

        # Force-scan pipeline (θ×φ only)
        per, mbar, ebar = bench_forced_scan(
            solver, loops=500,
            M=16, coarse_n=15, keep_top=3,
            refine_halfspan_deg=5.0, refine_n=9,
            local_refine_phi_deg=4.0, local_refine_phi_points=25
        )

        compute_wcet_from_counts(t_eval, t_phi24, mbar, ebar)

    finally:
        gc.enable()
