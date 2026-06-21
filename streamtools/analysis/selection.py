"""
Star selection and classification helpers, operating on SnapshotData.

"""
import numpy as np


def select_below_threshold(data, i_snap, E_threshold):
    """Indices of stars with E < E_threshold at snapshot i_snap."""
    return np.where(data.E[i_snap] < E_threshold)[0]


def select_above_threshold(data, i_snap, E_threshold):
    """Indices of stars with E > E_threshold at snapshot i_snap."""
    return np.where(data.E[i_snap] > E_threshold)[0]


def sample_indices(candidate_idx, n, rng=None):
    """
    Randomly sample up to n indices from candidate_idx (no replacement).
    Returns all of candidate_idx if there are fewer than n candidates.
    """
    rng = np.random.default_rng() if rng is None else rng
    if len(candidate_idx) > n:
        return rng.choice(candidate_idx, n, replace=False)
    return candidate_idx


def tracked_stars_below_threshold(data, i_snap, E_threshold, nstars, rng=None):
    """
    Convenience wrapper: candidate selection + random sampling in one call.
    This is the pattern used by plot_energyvstime / plot_energyvstime_with_orbits.
    """
    candidates = select_below_threshold(data, i_snap, E_threshold)
    return sample_indices(candidates, nstars, rng=rng)


def classify_escape_state(E_final, Lz_final, L_cluster_z_final):
    """
    Classify each tracked star's final state.

    Parameters
    ----------
    E_final : array (n_tracked,)
        Final binding energy per tracked star.
    Lz_final : array (n_tracked,)
        Final z-angular-momentum per tracked star.
    L_cluster_z_final : float
        Cluster orbital angular momentum z-component at the final snapshot.

    Returns
    -------
    dict mapping array index (0..n_tracked-1) -> "bound" / "prograde escape" / "retrograde escape"
    """
    result = {}
    for i, (E, Lz) in enumerate(zip(E_final, Lz_final)):
        if E > 0:
            result[i] = "prograde escape" if Lz * L_cluster_z_final > 0 else "retrograde escape"
        else:
            result[i] = "bound"
    return result


def escape_time_index(data):
    """
    Snapshot index at which each star becomes *permanently* unbound
    (E > 0 at every subsequent snapshot, with no later rebinding).

    Vectorized equivalent of "walk forward from the last bound snapshot and
    check everything after it is positive". Because the set of starting
    indices for which E stays positive forever is always a suffix of the
    snapshot range, the first index where a reverse running minimum of E
    turns positive is exactly that escape index -- no per-star Python loop
    needed.

    Parameters
    ----------
    data : SnapshotData

    Returns
    -------
    escape_idx : array (n_stars,) of int
        Snapshot index of permanent escape. 0 if the star is unbound at
        every snapshot in `data`. -1 if the star never permanently
        escapes within the covered time range (e.g. still bound at the
        final snapshot, or it rebinds after every excursion to E > 0).
    """
    E = data.E
    n_snaps = E.shape[0]
    # rev_min[i, s] = min(E[i:, s]); non-decreasing in i for fixed s.
    rev_min = np.minimum.accumulate(E[::-1], axis=0)[::-1]
    positive_forever = rev_min > 0
    has_escape = positive_forever.any(axis=0)
    escape_idx = np.where(has_escape, np.argmax(positive_forever, axis=0), -1)
    return escape_idx.astype(int)


def unbinding_time_index(data):
    """
    Snapshot index of the *first* time each star has E > 0, regardless of
    whether it later rebinds. Contrast with `escape_time_index`, which
    only counts permanent escapes. Used together to measure how long a
    star "flirts" with the escape threshold before permanently leaving
    (see analysis.series for the delay-time helper).

    Returns
    -------
    unbind_idx : array (n_stars,) of int
        First snapshot index with E > 0, or -1 if the star is never
        unbound in `data`.
    """
    unbound = data.E > 0
    ever_unbound = unbound.any(axis=0)
    unbind_idx = np.where(ever_unbound, np.argmax(unbound, axis=0), -1)
    return unbind_idx.astype(int)


def energy_oscillation_flips(data, min_flips=4):
    """
    Identify "potential escapers": stars whose binding energy crosses
    zero (bound <-> unbound) repeatedly before settling down, rather than
    escaping (or rebinding) cleanly on the first crossing.

    Parameters
    ----------
    data : SnapshotData
    min_flips : int
        Minimum number of bound/unbound sign changes in E(t) required to
        flag a star as oscillating.

    Returns
    -------
    osc_idx : array
        Star indices with >= min_flips sign changes in E(t).
    flip_counts : array (n_stars,)
        Number of sign changes for every star, aligned with `data`'s star
        indexing (not just `osc_idx`).
    """
    bound = data.E < 0
    flip_counts = np.sum(bound[1:] != bound[:-1], axis=0)
    osc_idx = np.where(flip_counts >= min_flips)[0]
    return osc_idx, flip_counts
