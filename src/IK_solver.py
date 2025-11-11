from pccik_native.pcc_workspace.specs import (
    SegmentSpec,
    TranslationSpec,
    TouchPointSpec,
    IKOptions,
)
from pccik_native.pcc_workspace.viz import visualize_touch_solutions_segmented
from IK_pipeline import create_solver

from matplotlib import pyplot as plt
import pccik_native.testcases as tc
import numpy as np
from time import time as _time
import os
import gc
try:
    os.nice(-20)
except:
    pass


class Timer:

    #: The raw function to get the CPU time.
    clock = _time

    def __call__(self):
        return self.clock()

    def run(self, profiler):
        yield


# --------------------------- Segment definitions ---------------------------

# Outer tube: bendable = 0.05 m, keep total 1.0 -> passive = 0.95 m
outer = SegmentSpec(
    name="outer",
    L_min=0.05,
    L_max=0.05,
    samples_length=1,  # fixed total length
    passive_L_min=0.0,  # fixed passive portion
    passive_L_max=0.0,
    samples_passive_length=1,
    theta_min=np.deg2rad(-135.2),
    theta_max=np.deg2rad(135.2),
    phi_min=np.deg2rad(-180.0),
    phi_max=np.deg2rad(180.0) - 1e-5,
    samples_theta=270,
    samples_phi=720,
)

# Inner tube: active <= 0.006 m, rigid tip = 0.003 m (added in solver),
# remaining passive <= 0.006 m, so (L_passive + L_active) <= 0.012 m
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
    active_L_max=0.006,  # <-- active 6 mm
    is_inner=True,
    bevel_angle_deg=45.0,
    roll_offset_deg=0.0,
)


def main():
    tr = TranslationSpec(d_min=0.0, d_max=0.0, samples=1)

    # ws = PCCFKSolver([outer, inner], translation=tr, small_L_eps=1e-4)

    touch = tc.N01[1]

    # IK options
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
    time = Timer()
    # IK solve (Closed-Form + 1D root on φ1)
    solver = create_solver([outer, inner], tr, opts)
    solver.debug = False

    solver.inner_rigid_tip = 0.003  # meters
    print(touch)
    t0 = Timer()()
    solutions = solver.solve(touch)
    t1 = Timer()()
    print(f"IK solve time: {(t1 - t0)*1e3:.3f} ms")
    if not solutions:
        print("Not reachable under current grid/tolerances.")
        return

    print(f"Found {len(solutions)} solution(s). Showing Top-{opts.topk}:")
    for i, sol in enumerate(solutions, start=1):
        rpt = sol.meta.get("report", {})
        o = rpt.get("outer", {})
        ii = rpt.get("inner", {})
        print(f"\n--- Solution #{i} (compact) ---")
        print(
            "[outer]  bend={:.2f} deg, rot={:.2f} deg, translation={:.3f} m".format(
                o.get("bending_deg", float("nan")),
                o.get("rotation_deg", float("nan")),
                o.get("translation", float("nan")),
            )
        )
        print(
            "[inner]  bend={:.2f} deg, rot={:.2f} deg, translation={:.3f} m".format(
                ii.get("bending_deg", float("nan")),
                ii.get("rotation_deg", float("nan")),
                ii.get("translation", float("nan")),
            )
        )

    figs = visualize_touch_solutions_segmented(
        solver,
        touch,
        topk=opts.topk,
        samples_per_passive=4,
        samples_per_active=50,
    )

    plt.show()


if __name__ == "__main__":
    gc.disable()
    main()
    gc.enable()
