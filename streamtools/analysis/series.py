"""
Build per-star time series from a SnapshotData bundle.

"""
import numpy as np


def energy_series(data, tracked_idx, i_start=0, i_end=None):
    """E(t) for tracked stars. Returns (times, E) with E shape (n_snaps, n_tracked)."""
    i_end = data.n_snaps if i_end is None else i_end + 1
    return data.times[i_start:i_end], data.E[i_start:i_end][:, tracked_idx]


def radius_series(data, tracked_idx, i_start=0, i_end=None):
    """Cartesian radius r(t) for tracked stars."""
    i_end = data.n_snaps if i_end is None else i_end + 1
    r = data.radius()
    return data.times[i_start:i_end], r[i_start:i_end][:, tracked_idx]


def v_rad_series(data, tracked_idx, i_start=0, i_end=None):
    """Radial velocity v_rad(t) for tracked stars."""
    i_end = data.n_snaps if i_end is None else i_end + 1
    return data.times[i_start:i_end], data.v_rad[i_start:i_end][:, tracked_idx]


def position_series(data, tracked_idx, i_start=0, i_end=None):
    """rotx(t), roty(t), rotz(t) for tracked stars."""
    i_end = data.n_snaps if i_end is None else i_end + 1
    sl = slice(i_start, i_end)
    return (
        data.times[sl],
        data.rotx[sl][:, tracked_idx],
        data.roty[sl][:, tracked_idx],
        data.rotz[sl][:, tracked_idx],
    )


def angular_momentum_series(data, tracked_idx, i_start=0, i_end=None):
    """
    Lz(t) for tracked stars, read directly from the SnapshotData cache
    (Lz and L_cluster_z are precomputed by save_snapshot_data -- this no
    longer needs raw Snapshot objects).

    Parameters
    ----------
    data : SnapshotData
    tracked_idx : array
    i_start, i_end : int

    Returns
    -------
    times : array (n_snaps,)
    Lz : array (n_snaps, n_tracked)
    L_cluster_z : array (n_snaps,) cluster orbital angular momentum z-component
    """
    i_end = data.n_snaps if i_end is None else i_end + 1
    sl = slice(i_start, i_end)
    return data.times[sl], data.Lz[sl][:, tracked_idx], data.L_cluster_z[sl]
