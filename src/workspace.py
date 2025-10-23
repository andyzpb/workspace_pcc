import numpy as np
from pcc_workspace.core import PCCFKSolver as ws
from pcc_workspace.specs import SegmentSpec as ss, TranslationSpec as ts
from pcc_workspace import viz


def demo_spec_for_two_seg_robot() -> ws:
    outer = ss(
        name="outer",
        L_min=0.05,
        L_max=0.05,
        L_step=None,
        samples_length=1,
        passive_L_min=0.0,
        passive_L_max=0.0,
        passive_L_step=None,
        theta_min=np.deg2rad(-135.0),
        theta_max=np.deg2rad(135.0) - 1e-6,
        theta_step=np.deg2rad(27.0),
        phi_min=np.deg2rad(-180.0),
        phi_max=np.deg2rad(180.0) - 1e-6,
        phi_step=np.deg2rad(360.0 / 16),
    )
    inner = ss(
        name="inner",
        L_min=0.015,
        L_max=0.015,
        L_step=None,
        passive_L_min=0.00,
        passive_L_max=0.00,
        passive_L_step=None,
        theta_min=np.deg2rad(-90.0),
        theta_max=np.deg2rad(90.0) - 1e-6,
        theta_step=np.deg2rad(20.0),
        phi_min=np.deg2rad(-180.0),
        phi_max=np.deg2rad(180.0) - 1e-6,
        phi_step=np.deg2rad(360.0 / 16),
    )
    translation = ts(0.0, 0.0, samples=1, d_step=None)
    return ws(segments=[outer, inner], translation=translation, small_L_eps=1e-4)


def main():
    w = demo_spec_for_two_seg_robot()
    pts, Ts = w.sample_workspace()

    viz.plot_with_directions(
        pts,
        Ts,
        voxel=0.003,
        equalize_azimuth=True,
        equalize_bins=48,
        equalize_per_bin=150,
        title="Color by passive ratio",
        max_quivers=800,
    )

    viz.plot_with_directions(
        pts,
        Ts,
        voxel=(0.003, 0.003, 1e9),
        equalize_azimuth=True,
        equalize_bins=36,
        equalize_per_bin=100,
        title="Groups by active bending",
        label_names={0: "Passive-only", 1: "Active present"},
        max_quivers=800,
    )

    # viz.plot_with_directions(
    #     pts,
    #     Ts,
    #     slice_axis="z",
    #     slice_at=0.96,
    #     slice_thickness=0.002,
    #     voxel=0.0015,
    #     title="Slice at z=0.96 m",
    #     equalize_azimuth=True,
    #     equalize_bins=2,
    #     equalize_per_bin=None,
    #     equalize_plane_point=np.array([0.0, 0.0, 1.0]),
    #     equalize_plane_normal=np.array([1.0, 0.0, 0.0]),
    # )

    phi_deg = 30.0
    phi = np.deg2rad(phi_deg)

    viz.plot_with_directions(
        pts,
        Ts,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([-np.sin(phi), np.cos(phi), 0.0]),
        plane_thickness=0.03, 
        voxel=None,
        title=f"Radial slice @ φ = {phi_deg}°",
    )

    viz.plot_dexterity_pointcloud3d(pts, Ts, voxel=0.01, orientation_bins=(24,12))

    viz.plot_dexterity_pointcloud2d(pts, Ts, slice_axis="y", slice_at=0.0, slice_thickness=0.1,
                            grid_size=(200,200), orientation_bins=(36,18))
    

if __name__ == "__main__":
    main()
