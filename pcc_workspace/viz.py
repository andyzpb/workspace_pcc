from curses import meta
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional, Sequence, Tuple, Union, Dict, List
from .specs import IKSolution, TouchPointSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .core import PCCFKSolver, PCCIKSolver  # for type hint only
import math
import cupy as cp

def _voxel_downsample_indices(
    pts: np.ndarray,
    vox: Optional[Union[float, Tuple[float, float, float]]],
) -> np.ndarray:
    if vox is None:
        return np.arange(pts.shape[0])
    if isinstance(vox, (int, float)):
        if vox <= 0:
            return np.arange(pts.shape[0])
        vx = vy = vz = float(vox)
    else:
        vx, vy, vz = [float(x) for x in vox]
        if vx <= 0 or vy <= 0 or vz <= 0:
            return np.arange(pts.shape[0])

    base = pts.min(axis=0)
    keys = np.stack(
        [
            np.floor((pts[:, 0] - base[0]) / vx).astype(np.int64),
            np.floor((pts[:, 1] - base[1]) / vy).astype(np.int64),
            np.floor((pts[:, 2] - base[2]) / vz).astype(np.int64),
        ],
        axis=1,
    )

    perm = np.random.permutation(len(pts))
    keys_view = keys[perm].view([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    _, idx = np.unique(keys_view, return_index=True)
    keep = np.sort(perm[idx])
    return keep


def _equalize_by_azimuth(
    pts: np.ndarray,
    Ts: np.ndarray,
    color_ds: Optional[np.ndarray],
    labels_ds: Optional[np.ndarray],
    *,
    bins: int = 32,
    per_bin: Optional[int] = None,
    plane_point: Optional[np.ndarray] = None,
    plane_normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if (plane_point is None) or (plane_normal is None):
        u = pts[:, 0]
        v = pts[:, 1]
    else:
        p0 = np.asarray(plane_point).reshape(1, 3)
        n = np.asarray(plane_normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(n, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rel = pts - p0
        u = rel @ e1
        v = rel @ e2

    phi = (np.arctan2(v, u) + 2 * np.pi) % (2 * np.pi)
    edges = np.linspace(0.0, 2 * np.pi, bins + 1, endpoint=True)
    which = np.digitize(phi, edges) - 1
    which[which == bins] = bins - 1

    counts = np.bincount(which, minlength=bins)
    if per_bin is None:
        nz = counts[counts > 0]
        per_bin = int(nz.min()) if len(nz) else 0

    if per_bin <= 0:
        return pts, Ts, color_ds, labels_ds

    rng = np.random.default_rng()
    keep_idx = []
    for k in range(bins):
        I = np.where(which == k)[0]
        if len(I) >= per_bin:
            choose = rng.choice(I, per_bin, replace=False)
            keep_idx.append(choose)

    if len(keep_idx) == 0:
        return pts, Ts, color_ds, labels_ds

    sel = np.sort(np.concatenate(keep_idx))
    pts_eq = pts[sel]
    Ts_eq = Ts[sel]
    col_eq = None if color_ds is None else color_ds[sel]
    lab_eq = None if labels_ds is None else labels_ds[sel]
    return pts_eq, Ts_eq, col_eq, lab_eq


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
    slice_axis: Optional[str] = None,
    slice_at: Optional[float] = None,
    slice_thickness: float = 0.0,
    plane_point: Optional[np.ndarray] = None,
    plane_normal: Optional[np.ndarray] = None,
    plane_thickness: float = 0.0,
    voxel: Union[None, float, Tuple[float, float, float]] = 0.004,
    equalize_azimuth: bool = False,
    equalize_bins: int = 32,
    equalize_per_bin: Optional[int] = None,
    equalize_plane_point: Optional[np.ndarray] = None,
    equalize_plane_normal: Optional[np.ndarray] = None,
    max_quivers: int = 1200,
    quiver_stride: Optional[int] = None,
    quiver_length: Optional[float] = None,
    point_size: int = 2,
    title: str = "",
    color_by: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    cmap: str = "viridis",
    equal_axes: bool = True,
    show_colorbar: bool = True,
):

    assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be (N,3)"
    assert Ts.ndim == 3 and Ts.shape[1:] == (4, 4), "Ts must be (N,4,4)"
    assert pts.shape[0] == Ts.shape[0], "pts and Ts length mismatch"
    N = pts.shape[0]

    if slice_axis is not None:
        assert slice_axis in ("x", "y", "z"), "slice_axis must be 'x'/'y'/'z'"
        assert slice_at is not None, "slice_at must be provided when using slice_axis"
        ax_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
        half = max(0.0, float(slice_thickness) * 0.5)
        lo, hi = slice_at - half, slice_at + half
        mask = (pts[:, ax_idx] >= lo) & (pts[:, ax_idx] <= hi)
        if not np.any(mask):
            raise ValueError(
                "Slice filter (axis-aligned) removed all points; increase slice_thickness."
            )
        pts = pts[mask]
        Ts = Ts[mask]
        if color_by is not None:
            color_by = np.asarray(color_by)[mask]
        if labels is not None:
            labels = np.asarray(labels)[mask]
        N = pts.shape[0]

    if (plane_point is not None) and (plane_normal is not None):
        p0 = np.asarray(plane_point, dtype=float).reshape(1, 3)
        n = np.asarray(plane_normal, dtype=float).reshape(1, 3)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("plane_normal must be non-zero")
        n = n / n_norm
        d = np.abs((pts - p0) @ n.T).reshape(-1)
        half = max(0.0, float(plane_thickness) * 0.5)
        mask = d <= half
        if not np.any(mask):
            raise ValueError(
                "Slice filter (arbitrary plane) removed all points; increase plane_thickness."
            )
        pts = pts[mask]
        Ts = Ts[mask]
        if color_by is not None:
            color_by = np.asarray(color_by)[mask]
        if labels is not None:
            labels = np.asarray(labels)[mask]
        N = pts.shape[0]

    keep = _voxel_downsample_indices(pts, voxel)
    pts_ds = pts[keep]
    Ts_ds = Ts[keep]
    color_ds = None
    labels_ds = None
    if color_by is not None and labels is None:
        cb = np.asarray(color_by).reshape(-1)
        assert cb.shape[0] == N, "color_by must have length N (after slicing)"
        color_ds = cb[keep]
    if labels is not None:
        lb = np.asarray(labels).reshape(-1)
        assert lb.shape[0] == N, "labels must have length N (after slicing)"
        labels_ds = lb[keep]

    if equalize_azimuth:
        pts_ds, Ts_ds, color_ds, labels_ds = _equalize_by_azimuth(
            pts_ds,
            Ts_ds,
            color_ds,
            labels_ds,
            bins=int(equalize_bins),
            per_bin=equalize_per_bin,
            plane_point=equalize_plane_point,
            plane_normal=equalize_plane_normal,
        )

    if quiver_stride is None:
        step = max(1, len(pts_ds) // max_quivers) if len(pts_ds) > max_quivers else 1
    else:
        step = max(1, int(quiver_stride))
    nq = min(max_quivers, len(pts_ds))
    if nq > 0:
        q_idx = np.random.choice(len(pts_ds), nq, replace=False)
        pos = pts_ds[q_idx]
        dirs = Ts_ds[q_idx, :3, 2]
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        dirs_n = dirs / norm
    else:
        pos = np.zeros((0, 3))
        dirs_n = np.zeros((0, 3))

    if quiver_length is None:
        if len(pts_ds) > 0:
            bbox_min = pts_ds.min(axis=0)
            bbox_max = pts_ds.max(axis=0)
            diag = np.linalg.norm(bbox_max - bbox_min)
        else:
            diag = 1.0
        quiver_length = 0.02 * (diag if np.isfinite(diag) and diag > 0 else 1.0)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    scatter_kwargs = dict(s=point_size, alpha=0.5, depthshade=False)
    mappable = None
    cb_label = None

    if labels_ds is not None:
        uniq = np.unique(labels_ds)
        import matplotlib.cm as cm

        colormap = cm.get_cmap("viridis", len(uniq))
        for i, lab in enumerate(uniq):
            mask = labels_ds == lab
            c_this = np.full(
                mask.sum(), i / max(len(uniq) - 1, 1) if len(uniq) > 1 else 0.5
            )
            ax.scatter(
                pts_ds[mask, 0],
                pts_ds[mask, 1],
                pts_ds[mask, 2],
                c=c_this,
                **scatter_kwargs,
            )
        handles, texts = [], []
        for i, lab in enumerate(uniq):
            handles.append(
                plt.Line2D([0], [0], marker="o", linestyle="", color=colormap(i))
            )
            name = (label_names or {}).get(int(lab), f"label {int(lab)}")
            texts.append(name)
        ax.legend(handles, texts, title="Groups", loc="upper right")
    else:
        if color_ds is None:
            c_vals = pts_ds[:, 2]
            cb_label = "Z (m)"
        else:
            c_vals = color_ds
            cb_label = "Color By"
        sc = ax.scatter(
            pts_ds[:, 0],
            pts_ds[:, 1],
            pts_ds[:, 2],
            c=c_vals,
            **scatter_kwargs,
        )
        mappable = sc

    # if len(pos) > 0:
    # ax.quiver(
    #     pos[:, 0],
    #     pos[:, 1],
    #     pos[:, 2],
    #     dirs_n[:, 0],
    #     dirs_n[:, 1],
    #     dirs_n[:, 2],
    #     length=float(quiver_length),
    #     normalize=False,
    #     linewidth=0.6,
    # )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "PCC Workspace")
    if equal_axes:
        _set_axes_equal(ax)

    if (labels_ds is None) and show_colorbar and (mappable is not None):
        cb = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(cb_label or "")

    plt.tight_layout()
    plt.show()


def _orientation_bin_ids(dirs: np.ndarray, n_az: int, n_el: int):
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    u = dirs / n
    az = (np.arctan2(u[:, 1], u[:, 0]) + 2 * np.pi) % (2 * np.pi)
    el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
    az_bin = np.floor(az / (2 * np.pi) * n_az).astype(int)
    az_bin = np.clip(az_bin, 0, n_az - 1)
    el_bin = np.floor((el + 0.5 * np.pi) / np.pi * n_el).astype(int)
    el_bin = np.clip(el_bin, 0, n_el - 1)
    ids = az_bin + n_az * el_bin
    return ids, n_az * n_el


def compute_dexterity_heatmap2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], np.ndarray]:
    """
    Compute a 2D heatmap of dexterity on an axis-aligned slice.
    Dexterity metric = fraction of orientation bins reachable in each cell.

    Returns:
      H: (Ny, Nx) array of coverage in [0,1] (NaN for under-sampled cells)
      extent: (xmin, xmax, ymin, ymax) suitable for plt.imshow
      counts: (Ny, Nx) array of sample counts per cell
    """
    assert pts.shape[0] == Ts.shape[0], "pts/Ts length mismatch"
    assert slice_axis in ("x", "y", "z")

    # Slice filter
    ax_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    half = max(0.0, float(slice_thickness) * 0.5)
    lo, hi = slice_at - half, slice_at + half
    mask = (pts[:, ax_idx] >= lo) & (pts[:, ax_idx] <= hi)
    if not np.any(mask):
        raise ValueError("Slice removed all points; increase slice_thickness.")
    P = pts[mask]
    D = Ts[mask, :3, 2]

    # Project to plane
    other = [i for i in range(3) if i != ax_idx]
    U = P[:, other[0]]  # horizontal
    V = P[:, other[1]]  # vertical

    # Grid extents
    if extent is None:
        pad = 1e-6
        xmin, xmax = float(U.min()) - pad, float(U.max()) + pad
        ymin, ymax = float(V.min()) - pad, float(V.max()) + pad
    else:
        xmin, xmax, ymin, ymax = extent

    Nx, Ny = int(grid_size[0]), int(grid_size[1])
    dx = (xmax - xmin) / Nx if Nx > 0 else 1.0
    dy = (ymax - ymin) / Ny if Ny > 0 else 1.0

    # Cell indices (row-major with y the fast-varying in reshape below)
    # Cell indices (row-major with y the fast-varying in reshape below)
    ix = np.floor((U - xmin) / dx).astype(int)
    iy = np.floor((V - ymin) / dy).astype(int)
    ix = np.clip(ix, 0, Nx - 1)
    iy = np.clip(iy, 0, Ny - 1)
    cell_id = ix + Nx * iy
    cell_id = ix + Nx * iy

    # Orientation bin ids
    ori_ids, TOT = _orientation_bin_ids(D, orientation_bins[0], orientation_bins[1])

    # Unique (cell, orientation) pairs → coverage per cell
    # Unique (cell, orientation) pairs → coverage per cell
    combo = cell_id.astype(np.int64) * np.int64(TOT) + ori_ids.astype(np.int64)
    uniq_combo = np.unique(combo)
    cells_from_combo = (uniq_combo // np.int64(TOT)).astype(int)
    cells_from_combo = (uniq_combo // np.int64(TOT)).astype(int)
    unique_count = np.bincount(cells_from_combo, minlength=Nx * Ny)


    # Raw sample count per cell
    sample_count = np.bincount(cell_id, minlength=Nx * Ny)

    # Coverage metric
    coverage = unique_count.astype(float) / float(TOT)
    H = coverage.reshape(Ny, Nx)
    C = sample_count.reshape(Ny, Nx).astype(int)

    # Under-sampled cells -> NaN
    # Under-sampled cells -> NaN
    H[C < int(min_samples_per_cell)] = np.nan
    return H, (xmin, xmax, ymin, ymax), C


# -------------------- 3D voxelized coverage --------------------
# -------------------- 3D voxelized coverage --------------------
def compute_dexterity_voxels3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[None, float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a voxelized 3D dexterity metric (orientation coverage) and return:
      centers: (M,3) voxel centers
      coverage: (M,) coverage in [0,1]
      counts: (M,) sample counts per voxel

    Implementation aligns coverage/counts to the *unique* occupied voxels to avoid
    length mismatches.
    Implementation aligns coverage/counts to the *unique* occupied voxels to avoid
    length mismatches.
    """
    if voxel is None or (isinstance(voxel, (int, float)) and float(voxel) <= 0):
        raise ValueError("voxel must be a positive scalar or (vx,vy,vz)")
    if isinstance(voxel, (int, float)):
        vx = vy = vz = float(voxel)
    else:
        vx, vy, vz = [float(x) for x in voxel]
        if min(vx, vy, vz) <= 0:
            raise ValueError("voxel sizes must be > 0")

    base = pts.min(axis=0)
    ix = np.floor((pts[:, 0] - base[0]) / vx).astype(np.int64)
    iy = np.floor((pts[:, 1] - base[1]) / vy).astype(np.int64)
    iz = np.floor((pts[:, 2] - base[2]) / vz).astype(np.int64)
    nx = int(ix.max() + 1)
    ny = int(iy.max() + 1)
    nid = ix + nx * (iy + ny * iz)
    nid = ix + nx * (iy + ny * iz)
    max_id = int(nid.max())

    # Orientation bins per sample
    ori_ids, TOT = _orientation_bin_ids(
        Ts[:, :3, 2], orientation_bins[0], orientation_bins[1]
    )

    # Unique (voxel, orientation-bin)
    # Unique (voxel, orientation-bin)
    combo = nid.astype(np.int64) * np.int64(TOT) + ori_ids.astype(np.int64)
    uniq_combo = np.unique(combo)
    vox_from_combo = (uniq_combo // np.int64(TOT)).astype(np.int64)

    unique_count_full = np.bincount(
        vox_from_combo, minlength=max_id + 1
    )  # bins per voxel
    sample_count_full = np.bincount(nid, minlength=max_id + 1)  # raw samples per voxel
    coverage_full = unique_count_full.astype(float) / float(TOT)

    # Restrict to actually occupied voxels (stable order)
    uniq_vox, first_idx = np.unique(nid, return_index=True)
    # Restrict to actually occupied voxels (stable order)
    uniq_vox, first_idx = np.unique(nid, return_index=True)
    vx_i = (uniq_vox % nx).astype(np.int64)
    vy_i = ((uniq_vox // nx) % ny).astype(np.int64)
    vz_i = (uniq_vox // (nx * ny)).astype(np.int64)
    centers = np.stack(
        [
            base[0] + (vx_i + 0.5) * vx,
            base[1] + (vy_i + 0.5) * vy,
            base[2] + (vz_i + 0.5) * vz,
        ],
        axis=1,
    )

    cov = coverage_full[uniq_vox]
    cnt = sample_count_full[uniq_vox]

    mask = cnt >= int(min_samples_per_voxel)
    return centers[mask], cov[mask], cnt[mask]


def plot_dexterity_heatmap2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
    title: str = "Dexterity heatmap",
    show_colorbar: bool = True,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    annotate_counts: bool = False,
):
    """
    Render the 2D dexterity heatmap on an axis-aligned slice.

    NOTE: requires `compute_dexterity_heatmap2d` to be available in scope.
    """
    H, extent, C = compute_dexterity_heatmap2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
        extent=None,
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()

    # mask NaNs for nicer rendering
    Hm = np.ma.masked_invalid(H)
    im = ax.imshow(
        Hm,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # axis labels based on slice axis
    xlab = {"x": "Y (m)", "y": "X (m)", "z": "X (m)"}[slice_axis]
    ylab = {"x": "Z (m)", "y": "Z (m)", "z": "Y (m)"}[slice_axis]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("Orientation coverage (0–1)")

    if annotate_counts:
        Ny, Nx = H.shape
        xmin, xmax, ymin, ymax = extent
        dx = (xmax - xmin) / Nx
        dy = (ymax - ymin) / Ny
        for iy in range(Ny):
            for ix in range(Nx):
                if not np.isnan(H[iy, ix]):
                    ax.text(
                        xmin + (ix + 0.5) * dx,
                        ymin + (iy + 0.5) * dy,
                        str(int(C[iy, ix])),
                        color="w",
                        ha="center",
                        va="center",
                        fontsize=6,
                        alpha=0.8,
                    )

    plt.tight_layout()
    plt.show()
    return fig, ax, im


# -----------------------------------------------------------
# 3D voxel scatter (uses your compute_dexterity_voxels3d)
# -----------------------------------------------------------
def plot_dexterity_heatmap2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
    title: str = "Dexterity heatmap",
    show_colorbar: bool = True,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    annotate_counts: bool = False,
):
    """
    Render the 2D dexterity heatmap on an axis-aligned slice.

    NOTE: requires `compute_dexterity_heatmap2d` to be available in scope.
    """
    H, extent, C = compute_dexterity_heatmap2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
        extent=None,
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()

    # mask NaNs for nicer rendering
    Hm = np.ma.masked_invalid(H)
    im = ax.imshow(
        Hm,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # axis labels based on slice axis
    xlab = {"x": "Y (m)", "y": "X (m)", "z": "X (m)"}[slice_axis]
    ylab = {"x": "Z (m)", "y": "Z (m)", "z": "Y (m)"}[slice_axis]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("Orientation coverage (0–1)")

    if annotate_counts:
        Ny, Nx = H.shape
        xmin, xmax, ymin, ymax = extent
        dx = (xmax - xmin) / Nx
        dy = (ymax - ymin) / Ny
        for iy in range(Ny):
            for ix in range(Nx):
                if not np.isnan(H[iy, ix]):
                    ax.text(
                        xmin + (ix + 0.5) * dx,
                        ymin + (iy + 0.5) * dy,
                        str(int(C[iy, ix])),
                        color="w",
                        ha="center",
                        va="center",
                        fontsize=6,
                        alpha=0.8,
                    )

    plt.tight_layout()
    plt.show()
    return fig, ax, im


# -----------------------------------------------------------
# 3D voxel scatter (uses your compute_dexterity_voxels3d)
# -----------------------------------------------------------
def plot_dexterity_voxels3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[None, float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
    title: str = "Voxelized dexterity (3D)",
    max_points: int = 20000,
    show_colorbar: bool = True,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    size_by_count: bool = False,
    
):
    """
    Scatter plot of 3D voxel dexterity coverage.
    """
    centers, cov, cnt = compute_dexterity_voxels3d(
        pts,
        Ts,
        voxel=voxel,
        orientation_bins=orientation_bins,
        min_samples_per_voxel=min_samples_per_voxel,
    )
    if centers.shape[0] == 0:
        raise ValueError(
            "No voxels after filtering; enlarge `voxel` or lower `min_samples_per_voxel`."
        )

    # decimation for big clouds
    M = centers.shape[0]
    if M > max_points:
        idx = np.random.choice(M, max_points, replace=False)
        centers = centers[idx]
        cov = cov[idx]
        cnt = cnt[idx]

    # point size
    if size_by_count:
        s = 4.0 + 0.6 * (cnt - np.min(cnt)) / max(1, (np.max(cnt) - np.min(cnt)))
        s = 10.0 * s
    else:
        s = 8.0
        cnt = cnt[idx]

    # point size
    if size_by_count:
        s = 4.0 + 0.6 * (cnt - np.min(cnt)) / max(1, (np.max(cnt) - np.min(cnt)))
        s = 10.0 * s
    else:
        s = 8.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=cov,
        s=s,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    if show_colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("Orientation coverage (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, ax, sc


# -----------------------------------------------------------
# Convenience: plot multiple slices at once
# -----------------------------------------------------------
def compute_dexterity_pointcloud3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
) -> np.ndarray:
    if isinstance(voxel, (int, float)):
        vx = vy = vz = float(voxel)
    else:
        vx, vy, vz = [float(x) for x in voxel]
        if min(vx, vy, vz) <= 0:
            raise ValueError("voxel sizes must be > 0")

    base = pts.min(axis=0)
    ix = np.floor((pts[:, 0] - base[0]) / vx).astype(np.int64)
    iy = np.floor((pts[:, 1] - base[1]) / vy).astype(np.int64)
    iz = np.floor((pts[:, 2] - base[2]) / vz).astype(np.int64)
    nx = int(ix.max() + 1)
    ny = int(iy.max() + 1)
    nid = ix + nx * (iy + ny * iz)
    max_id = int(nid.max())

    ori_ids, TOT = _orientation_bin_ids(
        Ts[:, :3, 2], orientation_bins[0], orientation_bins[1]
    )

    combo = nid.astype(np.int64) * np.int64(TOT) + ori_ids.astype(np.int64)
    uniq_combo = np.unique(combo)
    vox_from_combo = (uniq_combo // np.int64(TOT)).astype(np.int64)

    unique_count_full = np.bincount(vox_from_combo, minlength=max_id + 1)
    sample_count_full = np.bincount(nid, minlength=max_id + 1)
    coverage_full = unique_count_full.astype(float) / float(TOT)

    cov_pts = coverage_full[nid]
    cov_pts[sample_count_full[nid] < int(min_samples_per_voxel)] = np.nan
    return cov_pts


def plot_dexterity_pointcloud3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
    max_points: int = 100000,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    point_size: float = 3.5,
    alpha: float = 0.95,
    title: str = "Dexterity point cloud (3D)",
):

    cov_pts = compute_dexterity_pointcloud3d(
        pts,
        Ts,
        voxel=voxel,
        orientation_bins=orientation_bins,
        min_samples_per_voxel=min_samples_per_voxel,
    )
    m = ~np.isnan(cov_pts)
    if not np.any(m):
        raise ValueError(
            "All points are under-sampled; relax min_samples_per_voxel or enlarge voxel."
        )

    P = pts[m]
    C = cov_pts[m]

    if P.shape[0] > max_points:
        idx = np.random.choice(P.shape[0], max_points, replace=False)
        P = P[idx]
        C = C[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        c=C,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Orientation coverage (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, ax, sc


# -------------------- 2D: per-sample dexterity (slice) --------------------
def compute_dexterity_pointcloud2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:

    H, extent, counts = compute_dexterity_heatmap2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
    )

    ax_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    half = max(0.0, float(slice_thickness) * 0.5)
    lo, hi = slice_at - half, slice_at + half
    mask = (pts[:, ax_idx] >= lo) & (pts[:, ax_idx] <= hi)
    P = pts[mask]

    other = [i for i in range(3) if i != ax_idx]
    U = P[:, other[0]]
    V = P[:, other[1]]

    xmin, xmax, ymin, ymax = extent
    Nx, Ny = int(grid_size[0]), int(grid_size[1])
    dx = (xmax - xmin) / Nx if Nx > 0 else 1.0
    dy = (ymax - ymin) / Ny if Ny > 0 else 1.0

    ix = np.clip(np.floor((U - xmin) / dx).astype(int), 0, Nx - 1)
    iy = np.clip(np.floor((V - ymin) / dy).astype(int), 0, Ny - 1)

    Cgrid = H[iy, ix]
    ok = ~np.isnan(Cgrid)
    P2 = np.stack([U[ok], V[ok]], axis=1)
    return P2, Cgrid[ok]


def plot_dexterity_pointcloud2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
    max_points: int = 150000,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    point_size: float = 1.8,
    alpha: float = 0.9,
    title: str = "Dexterity point cloud (2D slice)",
):

    P2, C = compute_dexterity_pointcloud2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
    )
    if P2.shape[0] == 0:
        raise ValueError("No valid points on slice; loosen thickness or sampling.")

    if P2.shape[0] > max_points:
        idx = np.random.choice(P2.shape[0], max_points, replace=False)
        P2 = P2[idx]
        C = C[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()
    sc = ax.scatter(
        P2[:, 0],
        P2[:, 1],
        c=C,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel({"x": "Y (m)", "y": "X (m)", "z": "X (m)"}[slice_axis])
    ax.set_ylabel({"x": "Z (m)", "y": "Z (m)", "z": "Y (m)"}[slice_axis])
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Orientation coverage (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, ax, sc
    return fig, ax, sc


# -----------------------------------------------------------
# Convenience: plot multiple slices at once
# -----------------------------------------------------------
def compute_dexterity_pointcloud3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
) -> np.ndarray:
    if isinstance(voxel, (int, float)):
        vx = vy = vz = float(voxel)
    else:
        vx, vy, vz = [float(x) for x in voxel]
        if min(vx, vy, vz) <= 0:
            raise ValueError("voxel sizes must be > 0")

    base = pts.min(axis=0)
    ix = np.floor((pts[:, 0] - base[0]) / vx).astype(np.int64)
    iy = np.floor((pts[:, 1] - base[1]) / vy).astype(np.int64)
    iz = np.floor((pts[:, 2] - base[2]) / vz).astype(np.int64)
    nx = int(ix.max() + 1)
    ny = int(iy.max() + 1)
    nid = ix + nx * (iy + ny * iz)
    max_id = int(nid.max())

    ori_ids, TOT = _orientation_bin_ids(
        Ts[:, :3, 2], orientation_bins[0], orientation_bins[1]
    )

    combo = nid.astype(np.int64) * np.int64(TOT) + ori_ids.astype(np.int64)
    uniq_combo = np.unique(combo)
    vox_from_combo = (uniq_combo // np.int64(TOT)).astype(np.int64)

    unique_count_full = np.bincount(vox_from_combo, minlength=max_id + 1)
    sample_count_full = np.bincount(nid, minlength=max_id + 1)
    coverage_full = unique_count_full.astype(float) / float(TOT)

    cov_pts = coverage_full[nid]
    cov_pts[sample_count_full[nid] < int(min_samples_per_voxel)] = np.nan
    return cov_pts


def plot_dexterity_pointcloud3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    voxel: Union[float, Tuple[float, float, float]] = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_samples_per_voxel: int = 6,
    max_points: int = 100000,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    point_size: float = 3.5,
    alpha: float = 0.95,
    title: str = "Dexterity point cloud (3D)",
):

    cov_pts = compute_dexterity_pointcloud3d(
        pts,
        Ts,
        voxel=voxel,
        orientation_bins=orientation_bins,
        min_samples_per_voxel=min_samples_per_voxel,
    )
    m = ~np.isnan(cov_pts)
    if not np.any(m):
        raise ValueError(
            "All points are under-sampled; relax min_samples_per_voxel or enlarge voxel."
        )

    P = pts[m]
    C = cov_pts[m]

    # 可选降采样
    if P.shape[0] > max_points:
        idx = np.random.choice(P.shape[0], max_points, replace=False)
        P = P[idx]
        C = C[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        c=C,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Orientation coverage (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, ax, sc


# -------------------- 2D: per-sample dexterity (slice) --------------------
def compute_dexterity_pointcloud2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:

    H, extent, counts = compute_dexterity_heatmap2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
    )

    ax_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    half = max(0.0, float(slice_thickness) * 0.5)
    lo, hi = slice_at - half, slice_at + half
    mask = (pts[:, ax_idx] >= lo) & (pts[:, ax_idx] <= hi)
    P = pts[mask]

    other = [i for i in range(3) if i != ax_idx]
    U = P[:, other[0]]
    V = P[:, other[1]]

    xmin, xmax, ymin, ymax = extent
    Nx, Ny = int(grid_size[0]), int(grid_size[1])
    dx = (xmax - xmin) / Nx if Nx > 0 else 1.0
    dy = (ymax - ymin) / Ny if Ny > 0 else 1.0

    ix = np.clip(np.floor((U - xmin) / dx).astype(int), 0, Nx - 1)
    iy = np.clip(np.floor((V - ymin) / dy).astype(int), 0, Ny - 1)

    Cgrid = H[iy, ix]
    ok = ~np.isnan(Cgrid)
    P2 = np.stack([U[ok], V[ok]], axis=1)
    return P2, Cgrid[ok]


def plot_dexterity_pointcloud2d(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_at: float = 0.0,
    slice_thickness: float = 0.005,
    grid_size: Tuple[int, int] = (200, 200),
    orientation_bins: Tuple[int, int] = (36, 18),
    min_samples_per_cell: int = 8,
    max_points: int = 150000,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    point_size: float = 1.8,
    alpha: float = 0.9,
    title: str = "Dexterity point cloud (2D slice)",
):

    P2, C = compute_dexterity_pointcloud2d(
        pts,
        Ts,
        slice_axis=slice_axis,
        slice_at=slice_at,
        slice_thickness=slice_thickness,
        grid_size=grid_size,
        orientation_bins=orientation_bins,
        min_samples_per_cell=min_samples_per_cell,
    )
    if P2.shape[0] == 0:
        raise ValueError("No valid points on slice; loosen thickness or sampling.")

    if P2.shape[0] > max_points:
        idx = np.random.choice(P2.shape[0], max_points, replace=False)
        P2 = P2[idx]
        C = C[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot()
    sc = ax.scatter(
        P2[:, 0],
        P2[:, 1],
        c=C,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel({"x": "Y (m)", "y": "X (m)", "z": "X (m)"}[slice_axis])
    ax.set_ylabel({"x": "Z (m)", "y": "Z (m)", "z": "Y (m)"}[slice_axis])
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Orientation coverage (0–1)")
    plt.tight_layout()
    plt.show()
    return fig, ax, sc


# ------------- orientation binning (azimuth/elevation) -------------


def _orientation_bin_ids(
    dirs: np.ndarray, n_az: int, n_el: int
) -> Tuple[np.ndarray, int]:
    """
    Map 3D unit vectors `dirs` (N,3) to azimuth/elevation bin ids in [0, n_az*n_el).
    """
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    u = dirs / n
    az = (np.arctan2(u[:, 1], u[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)  # [0, 2π)
    el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))  # [-π/2, π/2]

    az_bin = np.floor(az / (2.0 * np.pi) * n_az).astype(np.int64)
    az_bin = np.clip(az_bin, 0, n_az - 1)
    el_bin = np.floor((el + 0.5 * np.pi) / np.pi * n_el).astype(np.int64)
    el_bin = np.clip(el_bin, 0, n_el - 1)
    bid = az_bin + n_az * el_bin
    return bid, n_az * n_el


# ------------- hash-grid neighbor search (radius) -------------


def _build_hash_grid(pts: np.ndarray, h: float):
    """
    Build a 3D hash grid with cell size h (uniform) for fast radius queries.
    Returns:
      base, h, nx, ny, keys (int64), order (argsort), uniq_keys, starts, counts
    """
    base = pts.min(axis=0)
    g = np.floor((pts - base) / h).astype(np.int64)
    nx = int(g[:, 0].max() + 1)
    ny = int(g[:, 1].max() + 1)
    keys = g[:, 0] + nx * (g[:, 1] + ny * g[:, 2])
    order = np.argsort(keys, kind="mergesort")  # stable
    ks = keys[order]
    uniq_keys, starts, counts = np.unique(ks, return_index=True, return_counts=True)
    return base, float(h), nx, ny, keys, order, uniq_keys, starts, counts


def _neighbors_in_radius(
    idx: int,
    pts: np.ndarray,
    base: np.ndarray,
    h: float,
    nx: int,
    ny: int,
    keys: np.ndarray,
    order: np.ndarray,
    uniq_keys: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    r: float,
):
    """
    Return candidate neighbor indices for point idx by scanning 27 neighbor cells,
    then filter by Euclidean radius r.
    """
    # locate cell of idx
    p = pts[idx]
    g = np.floor((p - base) / h).astype(np.int64)
    # collect ranges from 27 cells
    Nlist = []
    for dx in (-1, 0, 1):
        x = g[0] + dx
        if x < 0:
            continue
        for dy in (-1, 0, 1):
            y = g[1] + dy
            if y < 0:
                continue
            for dz in (-1, 0, 1):
                z = g[2] + dz
                if z < 0:
                    continue
                key = x + nx * (y + ny * z)
                # binary search in uniq_keys
                j = np.searchsorted(uniq_keys, key)
                if j < len(uniq_keys) and uniq_keys[j] == key:
                    s = starts[j]
                    c = counts[j]
                    Nlist.append(order[s : s + c])
    if len(Nlist) == 0:
        return np.empty((0,), dtype=np.int64)
    cand = np.unique(np.concatenate(Nlist))
    # true radius filter
    d2 = np.sum((pts[cand] - p) ** 2, axis=1)
    return cand[d2 <= r * r]


# ------------- point-cloud dexterity (local orientation coverage) -------------


def compute_dexterity_pointcloud(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    radius: float = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_neighbors: int = 12,
    subsample: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    For each selected point, compute local dexterity = fraction of orientation bins
    covered by neighbors within a spatial radius.
    Returns (indices, dex) where `indices` are point indices in the original arrays,
    and `dex` are the per-point coverage values (NaN if under-sampled).
    """
    assert pts.shape[0] == Ts.shape[0], "pts/Ts length mismatch"
    N = pts.shape[0]
    rng = np.random.default_rng(seed)

    # Choose which points to evaluate
    if subsample is not None and subsample < N:
        I = np.sort(rng.choice(N, int(subsample), replace=False))
    else:
        I = np.arange(N, dtype=np.int64)

    # Build hash grid for neighbor queries
    h = float(radius)  # cell size ~= radius
    base, h, nx, ny, keys, order, uniq_keys, starts, counts = _build_hash_grid(pts, h)

    # Pre-extract approach directions
    dirs = Ts[:, :3, 2].copy()

    dex = np.full(I.shape[0], np.nan, dtype=np.float32)
    n_az, n_el = orientation_bins
    TOT = n_az * n_el

    for ii, idx in enumerate(I):
        cand = _neighbors_in_radius(
            idx, pts, base, h, nx, ny, keys, order, uniq_keys, starts, counts, radius
        )
        if cand.size < int(min_neighbors):
            continue  # keep NaN
        bids, TOT = _orientation_bin_ids(dirs[cand], n_az, n_el)
        cov = np.unique(bids).size / float(TOT)
        dex[ii] = cov

    return I, dex


def plot_dexterity_pointcloud(
    pts: np.ndarray,
    Ts: np.ndarray,
    *,
    radius: float = 0.01,
    orientation_bins: Tuple[int, int] = (24, 12),
    min_neighbors: int = 12,
    subsample: Optional[int] = None,
    max_points: Optional[int] = 200000,
    title: str = "Dexterity (point-cloud, local orientation coverage)",
    show_colorbar: bool = True,
):
    """
    Scatter the original points, colored by per-point local dexterity.
    By default it computes dexterity on a (possibly) subsampled set and plots those.
    """
    I, dex = compute_dexterity_pointcloud(
        pts,
        Ts,
        radius=radius,
        orientation_bins=orientation_bins,
        min_neighbors=min_neighbors,
        subsample=subsample,
    )

    # Mask out NaNs (under-sampled neighborhoods)
    m = np.isfinite(dex)
    I = I[m]
    dex = dex[m]
    if I.size == 0:
        raise ValueError(
            "All points under-sampled. Increase radius or lower min_neighbors."
        )

    # Optional plot downsample
    if (max_points is not None) and (I.size > max_points):
        sel = np.random.default_rng().choice(I.size, int(max_points), replace=False)
        I = I[sel]
        dex = dex[sel]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        pts[I, 0], pts[I, 1], pts[I, 2], c=dex, s=6, alpha=0.9, depthshade=False
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    if show_colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("Local orientation coverage (0-1)")
    plt.tight_layout()
    plt.show()

# ---------------- basics ----------------
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)

def _name_points(item):
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        raise ValueError(
            f"Bad polyline item: {type(item)} with len={len(item) if hasattr(item,'__len__') else 'N/A'}"
        )
    name = item[0]
    arr = np.asarray(item[1], dtype=float)
    return name, arr

def _get_segment_colors(num_segments: int) -> List[str]:
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    return (prop_cycle * ((num_segments + len(prop_cycle) - 1) // len(prop_cycle)))[:num_segments]

def _set_axes_equal_3d(ax) -> None:
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
    radius = 0.5 * max(x_range, y_range, z_range, 1e-9)
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])

def _autoscale_arrow_lengths(points_stack: np.ndarray, min_frac: float = 0.12) -> float:
    span = np.max(points_stack, axis=0) - np.min(points_stack, axis=0)
    return float(np.linalg.norm(span)) * max(min_frac, 1e-3)

def _create_empty_figures() -> List[plt.Figure]:
    figs: List[plt.Figure] = []
    fig3d = plt.figure(); ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.set_title("No IK solution"); ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    figs.append(fig3d)
    fig_xy = plt.figure(); ax_xy = fig_xy.add_subplot(111)
    ax_xy.set_title("No IK solution (XY)"); ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y"); ax_xy.set_aspect("equal", adjustable="box")
    figs.append(fig_xy)
    fig_xz = plt.figure(); ax_xz = fig_xz.add_subplot(111)
    ax_xz.set_title("No IK solution (XZ)"); ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z"); ax_xz.set_aspect("equal", adjustable="box")
    figs.append(fig_xz)
    return figs

def _prepare_common_scale(polylines_per_sol: List[List[Tuple[str, np.ndarray]]], target_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    stacks = []
    for seg_lines in polylines_per_sol:
        for _, arr in seg_lines:
            stacks.append(arr)
    stacks.append(target_xyz.reshape(1, 3))
    all_pts = np.vstack(stacks)
    arrow_len = _autoscale_arrow_lengths(all_pts, min_frac=0.15)
    return all_pts, arrow_len

# ---------------- minimal FK blocks ----------------
def _rotz(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s, 0.0],[s, c, 0.0],[0.0, 0.0, 1.0]], dtype=float)

def _axis_angle_R(u: np.ndarray, theta: float) -> np.ndarray:
    u = np.array(u, dtype=float); n = np.linalg.norm(u)
    if n < 1e-12: return np.eye(3, dtype=float)
    u = u / n
    c, s = float(np.cos(theta)), float(np.sin(theta))
    ux, uy, uz = u
    K = np.array([[0.0, -uz,  uy],[ uz, 0.0, -ux],[-uy,  ux,  0.0]], dtype=float)
    I = np.eye(3, dtype=float)
    return c*I + (1.0-c)*np.outer(u,u) + s*K

def _cc_partial_transform(phi: float, kappa: float, theta_partial: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    u = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    if abs(theta_partial) < 1e-12:
        s = 0.0
        R = np.eye(3, dtype=float)
        p0 = np.array([0.0, 0.0, s], dtype=float)
    elif abs(kappa) < 1e-12:
        R = _axis_angle_R(u, theta_partial)
        p0 = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        r = 1.0 / kappa
        R = _axis_angle_R(u, theta_partial)
        x = r * (1.0 - float(np.cos(theta_partial))); z = r * float(np.sin(theta_partial))
        p0 = np.array([x, 0.0, z], dtype=float)
    T[:3, :3] = R; T[:3, 3] = _rotz(phi) @ p0
    return T

def reconstruct_segment_polylines_numpy(
    solution: "IKSolution",
    samples_per_passive: int = 2,
    samples_per_active: int = 30,
    small_L_eps: float = 1e-6,
) -> List[Tuple[str, np.ndarray]]:
    seg_lines: List[Tuple[str, np.ndarray]] = []
    T_cur = np.eye(4, dtype=float)

    # base insertion (+z)
    if abs(getattr(solution, "translation", 0.0)) > 0:
        T_ins = np.eye(4, dtype=float); T_ins[2, 3] = float(solution.translation)
        T_cur = T_cur @ T_ins

    for idx, seg_sol in enumerate(solution.segments):
        Lp = float(seg_sol.L_passive)
        La = float(seg_sol.L_active)
        th = float(seg_sol.theta)
        ph = float(seg_sol.phi)

        pts = [T_cur[:3, 3].copy()]

        # 1) passive straight
        if Lp > small_L_eps and samples_per_passive > 0:
            s_list = np.linspace(0.0, Lp, int(max(2, samples_per_passive)))
            for s in s_list[1:]:
                T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                T_tmp = T_cur @ T_step
                pts.append(T_tmp[:3, 3].copy())
            T_pass = np.eye(4, dtype=float); T_pass[2, 3] = Lp
            T_cur = T_cur @ T_pass

        # 2) active
        if La > small_L_eps and samples_per_active > 0:
            if abs(th) <= 1e-12:
                # straight active
                s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                for s in s_list[1:]:
                    T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                    T_tmp = T_cur @ T_step
                    pts.append(T_tmp[:3, 3].copy())
                T_lin = np.eye(4, dtype=float); T_lin[2, 3] = La
                T_cur = T_cur @ T_lin
            else:
                # circular arc
                kappa = th / La
                s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                for s in s_list[1:]:
                    theta_s = kappa * s
                    Ti = _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=theta_s)
                    T_tmp = T_cur @ Ti
                    pts.append(T_tmp[:3, 3].copy())
                Ti_full = _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=th)
                T_cur = T_cur @ Ti_full

        seg_lines.append((seg_sol.name, np.vstack(pts)))

    stub = float((getattr(solution, "meta", {}) or {}).get("inner_rigid_tip", 0.0))
    if stub > small_L_eps:
        p0 = T_cur[:3, 3].copy()
        T_stub = np.eye(4, dtype=float); T_stub[2, 3] = stub
        T_end = T_cur @ T_stub
        p1 = T_end[:3, 3].copy()
        seg_lines.append(("inner_rigid_stub", np.vstack([p0, p1])))

    return seg_lines

# ---------------- outer-end pose T1 (robustly pick outer by L_total max) ----------------
def _fk_outer_end_from_solution(sol: "IKSolution") -> np.ndarray:
    T = np.eye(4, dtype=float)
    if abs(getattr(sol, "translation", 0.0)) > 0:
        T_ins = np.eye(4, dtype=float); T_ins[2, 3] = float(sol.translation)
        T = T @ T_ins

    # pick outer by L_total max
    outer_idx = int(np.argmax([s.L_total for s in sol.segments]))
    seg0 = sol.segments[outer_idx]

    if seg0.L_passive > 1e-9:
        T_pass = np.eye(4, dtype=float); T_pass[2, 3] = float(seg0.L_passive)
        T = T @ T_pass

    if seg0.L_active > 1e-9:
        if abs(seg0.theta) <= 1e-12:
            T_lin = np.eye(4, dtype=float); T_lin[2, 3] = float(seg0.L_active)
            T = T @ T_lin
        else:
            kappa = seg0.theta / seg0.L_active
            Ti = _cc_partial_transform(phi=seg0.phi, kappa=kappa, theta_partial=seg0.theta)
            T = T @ Ti
    return T
def _format_mm(x: float) -> str:
    return f"{x*1e3:.3f} mm"

def _get_inner_outer_indices(sol: "IKSolution") -> Tuple[int, int]:
    names = [s.name.lower() for s in sol.segments]
    if "inner" in names and "outer" in names:
        inner_idx = names.index("inner")
        outer_idx = names.index("outer")
        return inner_idx, outer_idx
    lens = [s.L_total for s in sol.segments]
    outer_idx = int(np.argmax(lens))
    inner_idx = 1 - outer_idx
    return inner_idx, outer_idx


def _tip_pose_with_d(sol: "IKSolution") -> Tuple[np.ndarray, np.ndarray]:
    p_tip = np.asarray(sol.end_T[:3, 3], float)
    R_tip = np.asarray(sol.end_T[:3, :3], float)
    d = float(getattr(sol, "translation", 0.0) or 0.0)
    return p_tip + np.array([0.0, 0.0, d], float), R_tip

def _tip_pose_with_d(sol: "IKSolution") -> Tuple[np.ndarray, np.ndarray]:
    p_tip = np.asarray(sol.end_T[:3, 3], float)
    R_tip = np.asarray(sol.end_T[:3, :3], float)
    d = float(getattr(sol, "translation", 0.0) or 0.0)
    p_tip_d = p_tip + np.array([0.0, 0.0, d], float)
    return p_tip_d, R_tip
def reconstruct_segment_polylines_numpy(
    solution: "IKSolution",
    samples_per_passive: int = 2,
    samples_per_active: int = 30,
    small_L_eps: float = 1e-6,
) -> List[Tuple[str, np.ndarray]]:
    seg_lines: List[Tuple[str, np.ndarray]] = []
    T_cur = np.eye(4, dtype=float)

    if abs(getattr(solution, "translation", 0.0)) > 0:
        T_ins = np.eye(4, dtype=float); T_ins[2, 3] = float(solution.translation)
        T_cur = T_cur @ T_ins

    inner_idx, outer_idx = _get_inner_outer_indices(solution)

    for idx, seg_sol in enumerate(solution.segments):
        name = seg_sol.name
        Lp = float(seg_sol.L_passive)
        La = float(seg_sol.L_active)
        th = float(seg_sol.theta)
        ph = float(seg_sol.phi)

        if idx == outer_idx:
            pts = [T_cur[:3, 3].copy()]
            # passive
            if Lp > small_L_eps and samples_per_passive > 0:
                s_list = np.linspace(0.0, Lp, int(max(2, samples_per_passive)))
                for s in s_list[1:]:
                    T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                    T_tmp = T_cur @ T_step
                    pts.append(T_tmp[:3, 3].copy())
                T_cur = T_cur @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,Lp],[0,0,0,1]], float)
            # active
            if La > small_L_eps and samples_per_active > 0:
                if abs(th) <= 1e-12:
                    s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                    for s in s_list[1:]:
                        T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                        T_tmp = T_cur @ T_step
                        pts.append(T_tmp[:3, 3].copy())
                    T_cur = T_cur @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,La],[0,0,0,1]], float)
                else:
                    kappa = th / La
                    s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                    for s in s_list[1:]:
                        Ti = _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=kappa*s)
                        T_tmp = T_cur @ Ti
                        pts.append(T_tmp[:3, 3].copy())
                    T_cur = T_cur @ _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=th)
            seg_lines.append(("outer", np.vstack(pts)))
            continue

        # 1) inner_passive
        if Lp > small_L_eps and samples_per_passive > 0:
            pts_p = [T_cur[:3, 3].copy()]
            s_list = np.linspace(0.0, Lp, int(max(2, samples_per_passive)))
            for s in s_list[1:]:
                T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                T_tmp = T_cur @ T_step
                pts_p.append(T_tmp[:3, 3].copy())
            T_cur = T_cur @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,Lp],[0,0,0,1]], float)
            seg_lines.append(("inner_passive", np.vstack(pts_p)))

        # 2) inner_active
        if La > small_L_eps and samples_per_active > 0:
            pts_a = [T_cur[:3, 3].copy()]
            if abs(th) <= 1e-12:
                s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                for s in s_list[1:]:
                    T_step = np.eye(4, dtype=float); T_step[2, 3] = float(s)
                    T_tmp = T_cur @ T_step
                    pts_a.append(T_tmp[:3, 3].copy())
                T_cur = T_cur @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,La],[0,0,0,1]], float)
            else:
                kappa = th / La
                s_list = np.linspace(0.0, La, int(max(2, samples_per_active)))
                for s in s_list[1:]:
                    Ti = _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=kappa*s)
                    T_tmp = T_cur @ Ti
                    pts_a.append(T_tmp[:3, 3].copy())
                T_cur = T_cur @ _cc_partial_transform(phi=ph, kappa=kappa, theta_partial=th)
            seg_lines.append(("inner_active", np.vstack(pts_a)))

        # 3) inner_rigid（stub）
        stub = float((getattr(solution, "meta", {}) or {}).get("inner_rigid_tip", 0.003))
        if stub > small_L_eps:
            p0 = T_cur[:3, 3].copy()
            T_end = T_cur @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,stub],[0,0,0,1]], float)
            p1 = T_end[:3, 3].copy()
            seg_lines.append(("inner_rigid", np.vstack([p0, p1])))
            T_cur = T_end 
    return seg_lines

def _draw_inner_plane_glass(ax3d, sol: "IKSolution", arrow_len: float,
                            face_rgba=(0.70, 0.90, 1.00, 0.18),
                            edge_rgba=(1.00, 1.00, 1.00, 0.80),
                            hl_rgba=(1.00, 1.00, 1.00, 0.10)) -> None:
    T1 = _fk_outer_end_from_solution(sol) 
    P1 = T1[:3, 3].copy()
    _, R_tip = _tip_pose_with_d(sol)
    z_tip = _normalize(R_tip[:, 2])

    meta = sol.meta or {}
    n_plane = meta.get("plane_normal_world", None)
    if n_plane is None and len(sol.segments) >= 2:
        phi2 = float(sol.segments[_get_inner_outer_indices(sol)[0]].phi)
        R1 = T1[:3, :3]
        n_plane = R1 @ np.array([-math.sin(phi2), math.cos(phi2), 0.0], float)
    if n_plane is None:
        n_plane = np.array([0.0, 0.0, 1.0], float)
    n_plane = _normalize(np.asarray(n_plane, float))

    e1 = np.cross(n_plane, z_tip)
    if np.linalg.norm(e1) < 1e-12:
        alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(alt, n_plane)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n_plane, alt)
    e1 = _normalize(e1); e2 = _normalize(np.cross(n_plane, e1))

    L1 = 0.55 * arrow_len
    corners = np.stack([
        P1 + L1*e1 + 0.75*L1*e2,
        P1 - L1*e1 + 0.75*L1*e2,
        P1 - L1*e1 - 0.75*L1*e2,
        P1 + L1*e1 - 0.75*L1*e2,
    ])
    plane = Poly3DCollection([corners], facecolors=[face_rgba], edgecolors=[edge_rgba], linewidths=1.2, zorder=5)
    ax3d.add_collection3d(plane)
    inner = P1 + 0.85*(corners - P1)
    hl = Poly3DCollection([inner], facecolors=[hl_rgba], edgecolors=[(1,1,1,0.35)], linewidths=0.8, zorder=6)
    ax3d.add_collection3d(hl)

def _draw_bevel(ax3d, sol: "IKSolution", arrow_len: float, color: str = "red") -> np.ndarray:
    p_tip_d, R_tip = _tip_pose_with_d(sol)
    meta = sol.meta or {}
    b = meta.get("bevel_world", None)
    if b is None:
        alpha = float(np.deg2rad(meta.get("bevel_angle_deg", 45.0)))
        b0_tip = np.array([np.sin(alpha), 0.0, np.cos(alpha)], float)
        b = R_tip @ b0_tip
    b = _normalize(np.asarray(b, float))
    ax3d.quiver(p_tip_d[0], p_tip_d[1], p_tip_d[2], b[0], b[1], b[2],
                length=arrow_len*0.8, normalize=True, color=color, zorder=22)
    return b

def visualize_touch_solutions_segmented(
    solver,
    target: "TouchPointSpec",
    topk: int = 3,
    samples_per_passive: int = 2,
    samples_per_active: int = 30,
    show_coplanarity_check: bool = True,
    coplanarity_tol: float = 1e-6,
):
    sols = solver.solve(target)
    if not sols:
        return _create_empty_figures()
    sols = sols[:max(1, int(topk))]

    polylines_per_sol = []
    for sol in sols:
        seg_lines = reconstruct_segment_polylines_numpy(
            sol, samples_per_passive=samples_per_passive, samples_per_active=samples_per_active
        )
        polylines_per_sol.append(seg_lines)

    tx, ty, tz = target.coordinates
    n_hat = _normalize(np.asarray(target.normal, float))
    all_pts, arrow_len = _prepare_common_scale(polylines_per_sol, np.array([tx, ty, tz], float))

    color_map = {
        "outer":         "C0",
        "inner_passive": "C1",
        "inner_active":  "C3",
        "inner_rigid":   "C2",
    }
    linestyles = ["-", "--", ":", "-."]
    alphas = [1.0, 0.95, 0.9, 0.85]

    # ------- 3D -------
    fig3d = plt.figure(); ax3d = fig3d.add_subplot(111, projection="3d")
    for i, (seg_lines, sol) in enumerate(zip(polylines_per_sol, sols), start=1):
        ls = linestyles[(i-1) % len(linestyles)]
        a = alphas[(i-1) % len(alphas)]
        lw = 2.2 if i == 1 else 1.6

        for name, arr in seg_lines:
            xs, ys, zs = arr[:,0], arr[:,1], arr[:,2]
            ax3d.plot(xs, ys, zs, color=color_map.get(name, "k"), linestyle=ls, alpha=a, lw=lw,
                      label=name if i == 1 else None)

        p_tip_d, R_tip = _tip_pose_with_d(sol)
        u_axis = _normalize(R_tip[:, 2])
        ax3d.quiver(p_tip_d[0], p_tip_d[1], p_tip_d[2], u_axis[0], u_axis[1], u_axis[2],
                    length=arrow_len, normalize=True, color="blue")

        _draw_inner_plane_glass(ax3d, sol, arrow_len)
        b_world = _draw_bevel(ax3d, sol, arrow_len, color="red")

        if show_coplanarity_check:
            meta = sol.meta or {}
            n2 = meta.get("plane_normal_world", None)
            if n2 is None and len(sol.segments) >= 2:
                T1 = _fk_outer_end_from_solution(sol)
                R1 = T1[:3,:3]
                phi2 = float(sol.segments[_get_inner_outer_indices(sol)[0]].phi)
                n2 = R1 @ np.array([-math.sin(phi2), math.cos(phi2), 0.0], float)
            if n2 is not None:
                nb = float(abs(np.dot(_normalize(n2), _normalize(b_world))))
                cb = float(np.dot(_normalize(b_world), n_hat))
                pos_err = float(np.linalg.norm(p_tip_d - np.array([tx,ty,tz], float)))
                print(f"[viz check] |n_plane·bevel|={nb:.3e} (tol={coplanarity_tol:g}) "
                      f"[{'OK' if nb<=coplanarity_tol else 'FAIL'}],  "
                      f"bevel·target={cb:.6f},  |tip-target|={pos_err*1e3:.3f} mm")

    # target
    ax3d.scatter([tx],[ty],[tz], s=24, color="k")
    ax3d.quiver(tx,ty,tz, n_hat[0], n_hat[1], n_hat[2], length=arrow_len, normalize=True, color="green")
    ax3d.set_title("IK Top solutions (3D) — outer + inner(passive/active/rigid 3 mm) + plane + bevel")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    pad = np.linalg.norm(np.max(all_pts, axis=0) - np.min(all_pts, axis=0)) * 0.1 + 1e-9
    ax3d.set_xlim(np.min(all_pts[:,0])-pad, np.max(all_pts[:,0])+pad)
    ax3d.set_ylim(np.min(all_pts[:,1])-pad, np.max(all_pts[:,1])+pad)
    ax3d.set_zlim(np.min(all_pts[:,2])-pad, np.max(all_pts[:,2])+pad)
    _set_axes_equal_3d(ax3d)

    sol0 = sols[0]
    inner_idx, outer_idx = _get_inner_outer_indices(sol0)
    outer_seg = sol0.segments[outer_idx]
    inner_seg = sol0.segments[inner_idx]
    stub_len = float((sol0.meta or {}).get("inner_rigid_tip", 0.003))

    legend_labels = [
        (color_map["outer"],         f"outer (total={_format_mm(outer_seg.L_total)})"),
        (color_map["inner_passive"], f"inner_passive (L={_format_mm(inner_seg.L_passive)})"),
        (color_map["inner_active"],  f"inner_active (L={_format_mm(inner_seg.L_active)})"),
        (color_map["inner_rigid"],   f"inner_rigid (L={_format_mm(stub_len)})"),
        ("blue",  "Tool axis (+z)"),
        ("green", "Target normal"),
        ("red",   "Bevel (exact)"),
        ("#66CCFF", "Inner bending plane"),
    ]
    legend_handles = []
    for col, lab in legend_labels:
        if col == "#66CCFF":
            legend_handles.append(Line2D([0],[0], color=col, lw=6, alpha=0.3, label=lab))
        else:
            legend_handles.append(Line2D([0],[0], color=col, lw=2, label=lab))
    ax3d.legend(handles=legend_handles, loc="best")

    # ------- XY -------
    fig_xy = plt.figure(); ax_xy = fig_xy.add_subplot(111)
    for i, seg_lines in enumerate(polylines_per_sol, start=1):
        ls = linestyles[(i-1)%len(linestyles)]; a = alphas[(i-1)%len(alphas)]; lw=2.0 if i==1 else 1.4
        for name, arr in seg_lines:
            ax_xy.plot(arr[:,0], arr[:,1], color=color_map.get(name, "k"), linestyle=ls, alpha=a, lw=lw)
    ax_xy.scatter([tx],[ty], s=22, color="k")
    ax_xy.quiver([tx],[ty],[n_hat[0]*arrow_len],[n_hat[1]*arrow_len], angles="xy", scale_units="xy", scale=1, color="green")
    ax_xy.set_title("IK Top solutions (XY) — outer + inner(passive/active/rigid)")
    ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
    ax_xy.set_aspect("equal", adjustable="box")

    # ------- XZ -------
    fig_xz = plt.figure(); ax_xz = fig_xz.add_subplot(111)
    for i, seg_lines in enumerate(polylines_per_sol, start=1):
        ls = linestyles[(i-1)%len(linestyles)]; a = alphas[(i-1)%len(alphas)]; lw=2.0 if i==1 else 1.4
        for name, arr in seg_lines:
            ax_xz.plot(arr[:,0], arr[:,2], color=color_map.get(name, "k"), linestyle=ls, alpha=a, lw=lw)
    ax_xz.scatter([tx],[tz], s=22, color="k")
    ax_xz.quiver([tx],[tz],[n_hat[0]*arrow_len],[n_hat[2]*arrow_len], angles="xy", scale_units="xy", scale=1, color="green")
    ax_xz.set_title("IK Top solutions (XZ) — outer + inner(passive/active/rigid)")
    ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z")
    ax_xz.set_aspect("equal", adjustable="box")

    return [fig3d, fig_xy, fig_xz]
