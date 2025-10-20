import numpy as np
import cupy as cp
from typing import Optional, Tuple, List, Dict, Any

# ---------- Volumetric random sampler for PCCWorkspace ----------

def sample_volume_random(workspace, N: int, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    per_seg_samples: List[Dict[str, cp.ndarray]] = []

    # --- per-section random sampling
    for seg in workspace.segments:
        # theta range
        if seg.theta_min is not None:
            theta_lo = float(seg.theta_min); theta_hi = float(seg.theta_max)
        else:
            if seg.theta_sign is True:
                theta_lo, theta_hi = 0.0, float(seg.theta_max)
            elif seg.theta_sign is False:
                theta_lo, theta_hi = -float(seg.theta_max), float(seg.theta_max)
            else:
                theta_lo, theta_hi = 0.0, float(seg.theta_max)
        theta = cp.asarray(rng.uniform(theta_lo, theta_hi, size=N), dtype=cp.float32)

        # L_total & L_passive
        L_total = cp.asarray(rng.uniform(float(seg.L_min), float(seg.L_max), size=N), dtype=cp.float32)
        L_pass  = cp.asarray(rng.uniform(float(seg.passive_L_min), float(seg.passive_L_max), size=N), dtype=cp.float32)

        # phi (may be overridden by coupling below)
        if seg.phi_coupling is None:
            if (seg.phi_min is not None) and (seg.phi_max is not None):
                phi = rng.uniform(float(seg.phi_min), float(seg.phi_max), size=N)
            else:
                phi = rng.uniform(0.0, 2*np.pi, size=N)
            phi = cp.asarray(phi, dtype=cp.float32)
        elif seg.phi_coupling in ("lock_prev",) or isinstance(seg.phi_coupling, (float, int)):
            # placeholder, will be filled after we know prev segment
            phi = cp.zeros((N,), dtype=cp.float32)
        else:
            raise ValueError(f"Unsupported phi_coupling: {seg.phi_coupling}")

        per_seg_samples.append({
            "theta": theta, "phi": phi, "L_total": L_total, "L_passive": L_pass,
            "phi_mode": seg.phi_coupling,
        })

    # base translation
    tr = workspace.translation
    d = rng.uniform(float(tr.d_min), float(tr.d_max), size=N).astype(np.float32)
    d = cp.asarray(d, dtype=cp.float32)

    # --- apply phi coupling
    prev_phi_vec = None
    for dseg in per_seg_samples:
        mode = dseg["phi_mode"]
        if mode is None:
            prev_phi_vec = dseg["phi"]
        elif mode == "lock_prev":
            if prev_phi_vec is not None:
                dseg["phi"] = prev_phi_vec
            prev_phi_vec = dseg["phi"]
        elif isinstance(mode, (float, int)):
            if prev_phi_vec is None:
                dseg["phi"] = cp.asarray((float(mode),), dtype=cp.float32).repeat(N)
            else:
                dseg["phi"] = (prev_phi_vec + float(mode)) % (2*np.pi)
            prev_phi_vec = dseg["phi"]
        else:
            raise ValueError(f"Unsupported phi_coupling: {mode}")

    # --- compose FK
    pts_cp, Ts_cp = workspace._compose_fk_from_params(per_seg_samples, d)

    # build aux same as workspace.sample(return_aux=True)
    Ntot = pts_cp.shape[0]
    per_segment_aux: List[Dict[str, np.ndarray]] = []
    total_active = cp.zeros((Ntot,), dtype=cp.float32)
    total_passive = cp.zeros((Ntot,), dtype=cp.float32)

    for dseg in per_seg_samples:
        seg_aux = {
            "theta": cp.asnumpy(dseg["theta"]),
            "phi": cp.asnumpy(dseg["phi"]),
            "L_total": cp.asnumpy(dseg["L_total"]),
            "L_passive": cp.asnumpy(dseg["L_passive"]),
            "L_active": cp.asnumpy(dseg["L_active"]),
        }
        per_segment_aux.append(seg_aux)
        total_active += dseg["L_active"]
        total_passive += dseg["L_passive"]

    total_active = cp.asnumpy(total_active)
    total_passive = cp.asnumpy(total_passive)
    total_len = total_active + total_passive
    passive_ratio = total_passive / np.maximum(total_len, 1e-8)

    labels = {
        "has_active": (total_active > workspace.small_L_eps).astype(np.int32),
        "last_active": (per_segment_aux[-1]["L_active"] > workspace.small_L_eps).astype(np.int32),
    }
    aux: Dict[str, Any] = {
        "per_segment": per_segment_aux,
        "translation": cp.asnumpy(d),
        "total_active_len": total_active,
        "total_passive_len": total_passive,
        "passive_ratio": passive_ratio,
        "labels": labels,
    }
    return cp.asnumpy(pts_cp), cp.asnumpy(Ts_cp), aux


# ---------- Equalize sampling by 3D approach direction ----------

def equalize_by_direction3d(
    pts: np.ndarray,
    Ts: np.ndarray,
    color: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    bins_az_el: Tuple[int, int] = (36, 18),
    per_bin: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    Bin by (azimuth, elevation) of the tool z-axis (Ts[:,:3,2]) and keep the same
    number of samples in every non-empty bin.
    If per_bin is None, use the minimum population among non-empty bins.
    """
    if rng is None:
        rng = np.random.default_rng()

    dirs = Ts[:, :3, 2]
    n = np.linalg.norm(dirs, axis=1, keepdims=True); n[n == 0] = 1.0
    u = dirs / n
    az = (np.arctan2(u[:, 1], u[:, 0]) + 2*np.pi) % (2*np.pi)      # [0, 2π)
    el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))                    # [-π/2, π/2]

    na, ne = bins_az_el
    az_bin = np.floor(az / (2*np.pi) * na).astype(int); az_bin = np.clip(az_bin, 0, na-1)
    el_bin = np.floor((el + 0.5*np.pi) / np.pi * ne).astype(int);  el_bin = np.clip(el_bin, 0, ne-1)
    bid = az_bin + na * el_bin

    counts = np.bincount(bid, minlength=na*ne)
    nz_bins = np.where(counts > 0)[0]
    if len(nz_bins) == 0:
        raise ValueError("All direction bins are empty; check your data.")
    if per_bin is None:
        per_bin = int(counts[nz_bins].min())

    keep_idx = []
    for b in nz_bins:
        I = np.where(bid == b)[0]
        if len(I) >= per_bin:
            keep_idx.append(rng.choice(I, per_bin, replace=False))
        # bins with < per_bin are skipped to enforce exact equality

    if len(keep_idx) == 0:
        raise ValueError("No bins have enough samples for the requested per_bin. Lower per_bin or generate more samples.")

    sel = np.sort(np.concatenate(keep_idx))
    pts_e = pts[sel]
    Ts_e = Ts[sel]
    color_e = None if color is None else np.asarray(color)[sel]
    labels_e = None if labels is None else np.asarray(labels)[sel]
    return pts_e, Ts_e, color_e, labels_e, per_bin, len(nz_bins)


# ---------- Balanced volumetric sampler ----------

def sample_volume_random_balanced(workspace, *, target_per_dir: int = 200,
                                  bins_az_el: Tuple[int,int] = (36,18),
                                  batch: int = 20000, max_batches: int = 20,
                                  seed: Optional[int] = None):
    """
    Keep sampling random configurations until each (az,el) direction bin
    has at least 'target_per_dir' samples (or we hit max_batches).
    Returns a direction-equalized subset (same count per non-empty bin).
    """
    rng = np.random.default_rng(seed)
    pts_all = []; Ts_all = []

    for _ in range(max_batches):
        pts, Ts, _ = sample_volume_random(workspace, batch, seed=int(rng.integers(1<<30)))
        pts_all.append(pts); Ts_all.append(Ts)

        # Try equalize: if enough, break
        P = np.concatenate(pts_all, axis=0)
        T = np.concatenate(Ts_all, axis=0)
        try:
            _, _, _, _, per_bin, nbins = equalize_by_direction3d(P, T, bins_az_el=bins_az_el, per_bin=target_per_dir, rng=rng)
            # enough coverage achieved
            break
        except Exception:
            continue

    P = np.concatenate(pts_all, axis=0)
    T = np.concatenate(Ts_all, axis=0)
    P_eq, T_eq, _, _, per_bin, nbins = equalize_by_direction3d(P, T, bins_az_el=bins_az_el, per_bin=target_per_dir, rng=rng)
    return P_eq, T_eq, dict(per_bin=per_bin, nbins=nbins, total=len(P_eq))