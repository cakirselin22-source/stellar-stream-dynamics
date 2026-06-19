import numpy as np
from ..physics import dynamics

def escape_properties(snapshots):
    """
    Track where stars escape in cluster frame.
    """
    esc_times = {i: [] for i in range(len(snapshots) + 1)}
    esc_pos_series = {i: [] for i in range(len(snapshots) + 1)}
    c_esc_pos_series = {i: [] for i in range(len(snapshots) + 1)}
    for i, snap in enumerate(snapshots):
       # --- Cluster-frame energy ---
        E = dynamics.calculate_energy(snap.rs3, snap.vs3, snap.rc3, snap.vc3, snap.time,"binding")

        # --- Escapers ---
        mask = E > 0
        n_esc = np.sum(mask)
        if n_esc == 0:
            continue  # no escapers in this snapshot

        esc_times[i] = snap.time * np.ones(n_esc)
        esc_pos_series[i] = np.atleast_2d(snap.rs3[mask])         # star positions
        c_esc_pos_series[i] = np.atleast_2d(snap.rc3)   # cluster position repeated if needed

    return esc_times, esc_pos_series, c_esc_pos_series
