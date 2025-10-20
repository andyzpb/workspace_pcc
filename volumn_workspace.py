import numpy as np
from pcc_workspace.core import PCCFKSolver as ws
from pcc_workspace.specs import SegmentSpec as ss, TranslationSpec as ts
import pcc_workspace.viz as viz
from pcc_workspace.volume_sampling import sample_volume_random_balanced


def volumetric_spec() -> ws:
    outer = ss(
        name="outer",
        L_min=1,
        L_max=1,
        L_step=None,
        samples_length=1,
        passive_L_min=0.90,
        passive_L_max=0.90,
        passive_L_step=None,
        theta_min=np.deg2rad(-135.0),
        theta_max=np.deg2rad(135.0),
        theta_step=None,
        phi_min=np.deg2rad(-180.0),
        phi_max=np.deg2rad(180.0) - 1e-5,
        phi_step=None,
    )
    inner = ss(
        name="inner",
        L_min=0.00,
        L_max=0.15,
        samples_length=15,
        passive_L_min=0.00,
        passive_L_max=0.00,
        passive_L_step=None,
        theta_min=np.deg2rad(-90.0),
        theta_max=np.deg2rad(90.0),
        theta_step=None,
        phi_coupling="lock_prev",
        phi_step=None,
    )
    translation = ts(d_min=0.0, d_max=0.0, samples=1, d_step=None)
    return ws(segments=[outer, inner], translation=translation, small_L_eps=1e-4)


def main():
    w = volumetric_spec()
    pts, Ts, info = sample_volume_random_balanced(
        w,
        target_per_dir=4000,
        bins_az_el=(24, 12),
        batch=2500000,
        max_batches=8,
        seed=42,
    )
    print("[INFO] Equalized:", info)

    viz.plot_with_directions(
        pts,
        Ts,
        title="Volumetric workspace",
        max_quivers=1500,
        show_colorbar=False,
    )
    phi_deg = 30.0
    phi = np.deg2rad(phi_deg)
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
    viz.plot_with_directions(
        pts,
        Ts,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([-np.sin(phi), np.cos(phi), 0.0]),
        plane_thickness=0.03,
        voxel=None,
        title=f"Radial slice @ φ = {phi_deg}°",
    )
    viz.plot_dexterity_pointcloud(
        pts,
        Ts,
        radius=0.01,
        orientation_bins=(24, 12),
        min_neighbors=12,
        subsample=25000,
        max_points=20000,
        title="Dexterity",
    )


if __name__ == "__main__":
    main()
