"""
Build per-star time series from a SnapshotData bundle.

"""
import numpy as np

from . import selection

# Properties derivable directly from the precomputed cache, for the
# escape-property helpers below. Notably absent: "v_y" (velocity along the
# orbit-tangential basis vector) and "L_mag" (3D angular-momentum
# magnitude) -- both would need raw per-star velocity vectors, which are
# intentionally *not* part of the cache (only derived scalars are). Lz is
# cached and used instead of L_mag; it's also the physically relevant
# quantity for prograde/retrograde classification (see
# selection.classify_escape_state).
PROPERTY_LABELS = {
    "energy": "Energy",
    "v_rad": "Radial velocity",
    "r": "Radial distance [kpc]",
    "theta": r"$\theta$ [deg]",
    "phi": r"$\phi$ [deg]",
    "Lz": r"$L_z$",
}


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


def _property_values_at(data, property, t_idx, star_idx):
    """Look up `property` for (t_idx[k], star_idx[k]) pairs, vectorized."""
    if property == "energy":
        return data.E[t_idx, star_idx]
    if property == "v_rad":
        return data.v_rad[t_idx, star_idx]
    if property == "r":
        return data.radius()[t_idx, star_idx]
    if property == "theta":
        return data.theta[t_idx, star_idx]
    if property == "phi":
        return data.phi[t_idx, star_idx]
    if property == "Lz":
        return data.Lz[t_idx, star_idx]
    raise ValueError(
        f"Unknown property {property!r}; expected one of {sorted(PROPERTY_LABELS)}"
    )


def escape_property_values(data, property="energy", escape_idx=None):
    """
    For every star that permanently escapes (see
    `selection.escape_time_index`), look up `property` at the snapshot
    where it escapes.

    Parameters
    ----------
    data : SnapshotData
    property : str
        One of PROPERTY_LABELS (energy, v_rad, r, theta, phi, Lz).
    escape_idx : array, optional
        Precomputed via `selection.escape_time_index(data)`; computed
        on the fly if not given (cheap, but pass it in if you're calling
        this repeatedly for several properties on the same dataset).

    Returns
    -------
    values : array (n_escaped,)
    times : array (n_escaped,) -- escape time [Gyr] for each entry
    star_idx : array (n_escaped,) -- which stars these values belong to
    """
    if escape_idx is None:
        escape_idx = selection.escape_time_index(data)

    star_idx = np.where(escape_idx >= 0)[0]
    t_idx = escape_idx[star_idx]
    values = _property_values_at(data, property, t_idx, star_idx)
    return values, data.times[t_idx], star_idx


def escape_property_by_time_bin(data, property="theta", bin_width=1.0, escape_idx=None):
    """
    Group `escape_property_values` into bin_width-Gyr-wide windows of
    escape time -- e.g. "what did the escape-angle distribution look like
    for stars that left between t=4-5 Gyr, vs 5-6 Gyr, ...".

    Parameters
    ----------
    data : SnapshotData
    property : str
    bin_width : float
        Width of each escape-time window, in Gyr.
    escape_idx : array, optional
        See `escape_property_values`.

    Returns
    -------
    bin_edges : array (n_bins + 1,)
    values_per_bin : list of arrays, length n_bins
        `property` values for stars escaping within each window (empty
        array if no stars escaped in that window).
    labels : list of str, length n_bins
        "t0-t1 Gyr | N=<count>" labels, ready to use as subplot titles.
    """
    values, times, _ = escape_property_values(data, property, escape_idx=escape_idx)

    if len(times) == 0:
        bin_edges = np.array([data.times.min(), data.times.max()])
        return bin_edges, [np.array([])], [f"{bin_edges[0]:.1f}-{bin_edges[1]:.1f} Gyr | N=0"]

    t_min, t_max = data.times.min(), data.times.max()
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    n_bins = len(bin_edges) - 1

    values_per_bin, labels = [], []
    for i in range(n_bins):
        t0, t1 = bin_edges[i], bin_edges[i + 1]
        in_bin = (times >= t0) & (times < t1)
        values_per_bin.append(values[in_bin])
        labels.append(f"{t0:.1f}-{t1:.1f} Gyr | N={int(np.sum(in_bin))}")

    return bin_edges, values_per_bin, labels


def along_orbit_position(data, i_start=0, i_end=None, nskip=1):
    """
    Decompose each star's cluster-relative position into a coordinate
    along the cluster's instantaneous velocity ("leading/trailing" along
    the orbit) and the distance perpendicular to it (transverse spread of
    the stream).

    Reconstructed entirely from the precomputed cache: rotx/roty/rotz are
    the star's position in the (e_r, e_y, e_L) rotating frame (see
    Snapshot.rotating_basis), and rc3/vc3 let us rebuild that frame --
    and the cluster's velocity direction -- at every snapshot. No raw
    per-star position/velocity arrays are needed.

    Note this redefines the perpendicular axis relative to the cluster's
    *velocity* direction (v_hat), not the raw galactocentric y/z axes --
    geometrically the right choice for "spread transverse to the orbit",
    and consistent regardless of how the orbital plane happens to be
    oriented.

    Parameters
    ----------
    data : SnapshotData
    i_start, i_end : int
        Snapshot index range (inclusive of i_end).
    nskip : int
        Use every nskip-th star (for speed on large star counts).

    Returns
    -------
    times : array (n_sel_snaps,)
    s_orbit : array (n_sel_snaps, n_sel_stars)
        Signed distance along the cluster's velocity direction.
    r_perp : array (n_sel_snaps, n_sel_stars)
        Distance perpendicular to the cluster's velocity direction.
    """
    i_end = data.n_snaps if i_end is None else i_end + 1
    sl = slice(i_start, i_end)

    rc3 = data.rc3[sl]
    vc3 = data.vc3[sl]
    rotx = data.rotx[sl][:, ::nskip]
    roty = data.roty[sl][:, ::nskip]
    rotz = data.rotz[sl][:, ::nskip]

    e_r = rc3 / np.linalg.norm(rc3, axis=1, keepdims=True)
    L = np.cross(rc3, vc3)
    e_L = L / np.linalg.norm(L, axis=1, keepdims=True)
    e_y = np.cross(e_L, e_r)
    v_hat = vc3 / np.linalg.norm(vc3, axis=1, keepdims=True)

    # rs_rel = rotx*e_r + roty*e_y + rotz*e_L, broadcast over stars.
    rs_rel = (
        rotx[..., None] * e_r[:, None, :]
        + roty[..., None] * e_y[:, None, :]
        + rotz[..., None] * e_L[:, None, :]
    )  # (n_sel_snaps, n_sel_stars, 3)

    s_orbit = np.sum(rs_rel * v_hat[:, None, :], axis=-1)
    r_perp = np.linalg.norm(rs_rel - s_orbit[..., None] * v_hat[:, None, :], axis=-1)

    return data.times[sl], s_orbit, r_perp
