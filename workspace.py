from pcc_workspace import PCCWorkspace as ws
from pcc_workspace import viz
from pcc_workspace import SegmentSpec as ss
from pcc_workspace import TranslationSpec as ts
import numpy as np


def demo_spec_for_two_seg_robot() -> ws:
    outer = ss(
        name="outer",
        L_min=0.15,
        L_max=0.15,
        theta_max=np.deg2rad(135.0),
        theta_sign=False,
        samples_theta=10,
        samples_phi=1,
        samples_length=1,
    )
    inner = ss(
        name="inner",
        L_min=0.0,
        L_max=0.015,
        theta_max=np.deg2rad(90.0),
        theta_sign=False,
        samples_theta=10,
        samples_phi=100,
        samples_length=1,
    )
    translation = ts(0.0, 0.0, 1)
    return ws(segments=[outer, inner], translation=translation, small_L_eps=1e-4)


def main() -> None:
    ws = demo_spec_for_two_seg_robot()
    pts, Ts = ws.sample()
    print(f"[INFO] Sampled points: {pts.shape[0]}")

    viz.plot_with_directions(pts, Ts, voxel=0.001, max_quivers=500)


if __name__ == "__main__":
    main()
