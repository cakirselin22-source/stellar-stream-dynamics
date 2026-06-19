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
