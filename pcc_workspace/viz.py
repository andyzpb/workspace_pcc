import numpy as np
import matplotlib.pyplot as plt


def _voxel_downsample(pts: np.ndarray, vox: float):
    if vox is None or vox <= 0:
        return np.arange(pts.shape[0])
    base = pts.min(axis=0)
    keys = np.floor((pts - base) / float(vox)).astype(np.int64)
    keys_view = keys.view([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    _, idx = np.unique(keys_view, return_index=True)
    return np.sort(idx)


def _set_axes_equal(ax):
    xlim = np.array(ax.get_xlim3d())
    ylim = np.array(ax.get_ylim3d())
    zlim = np.array(ax.get_zlim3d())
    xyz = np.vstack([xlim, ylim, zlim])
    centers = xyz.mean(axis=1)
    spans = xyz[:, 1] - xyz[:, 0]
    radius = spans.max() / 2.0
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_with_directions(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: float = 0.004,
    max_quivers: int = 1200,
    point_size: int = 2,
    title: str = ""
):
    assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be (N,3)"
    assert Ts.ndim == 3 and Ts.shape[1:] == (4, 4), "Ts must be (N,4,4)"
    assert pts.shape[0] == Ts.shape[0], "pts and Ts length mismatch"

    keep = _voxel_downsample(pts, voxel)
    pts_ds = pts[keep]
    Ts_ds = Ts[keep]

    if len(pts_ds) > max_quivers:
        step = max(1, len(pts_ds) // max_quivers)
        q_idx = np.arange(0, len(pts_ds), step)
    else:
        q_idx = np.arange(len(pts_ds))

    pos = pts_ds[q_idx]
    dirs = Ts_ds[q_idx, :3, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    c = pts_ds[:, 2]
    sc = ax.scatter(
        pts_ds[:, 0],
        pts_ds[:, 1],
        pts_ds[:, 2],
        s=point_size,
        alpha=0.5,
        c=c,
        cmap="viridis",
        depthshade=False,
    )

    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    dirs_n = dirs / norm
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        dirs_n[:, 0],
        dirs_n[:, 1],
        dirs_n[:, 2],
        length=0.003,
        normalize=False,
        linewidth=0.6,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "PCC Workspace")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()
