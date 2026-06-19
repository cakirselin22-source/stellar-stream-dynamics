import numpy as np
import matplotlib.pyplot as plt

from ..physics import dynamics
from ..core.snapshot import Snapshot
from ..analysis import analysis

def plot_energyvstime(snapshots, ifile_start, ifile_end, E_threshold, nstars=10,
                       t_xlim=(3, 14), e_ylim=(-10, 30), output="energyvstime_random"):
    """
    Plot cluster-frame energy and radius evolution for a random sample of stars
    that start below a given binding-energy threshold.

    Parameters
    ----------
    snapshots : list of Snapshot
    ifile_start, ifile_end : int
        Snapshot index range to plot over.
    E_threshold : float
        Only stars with initial binding energy below this value are eligible.
    nstars : int
        Number of stars to randomly sample from eligible candidates.
    t_xlim, e_ylim : tuple
        Axis limits for the time and energy panels.
    output : str
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ifile_end = min(ifile_end, len(snapshots) - 1)
    snap0 = snapshots[ifile_start]

    E0 = dynamics.calculate_energy(
        snap0.rs3, snap0.vs3, snap0.rc3, snap0.vc3, snap0.time, "binding"
    )

    candidate_idx = np.where(E0 < E_threshold)[0]
    tracked_idx = (
        np.random.choice(candidate_idx, nstars, replace=False)
        if len(candidate_idx) > nstars else candidate_idx
    )

    times = []
    E_series = {idx: [] for idx in tracked_idx}
    R_series = {idx: [] for idx in tracked_idx}

    for snap in snapshots[ifile_start:ifile_end + 1]:
        E_snap = dynamics.calculate_energy(
            snap.rs3, snap.vs3, snap.rc3, snap.vc3, snap.time, "binding"
        )
        rotx, roty, rotz = snap.project_cluster_frame()
        R_snap = np.sqrt(rotx**2 + roty**2 + rotz**2)

        times.append(snap.time)
        for idx in tracked_idx:
            E_series[idx].append(E_snap[idx])
            R_series[idx].append(R_snap[idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for idx in tracked_idx:
        ax1.plot(times, E_series[idx], label=f"Star {idx}")
        ax2.plot(times, R_series[idx])

    ax1.set(xlabel="Time [Gyr]", ylabel="Energy", xlim=t_xlim, ylim=e_ylim,
            title=f"Energy evolution (E < {E_threshold})")
    ax2.set(xlabel="Time [Gyr]", ylabel="Radius", xlim=t_xlim)
    ax1.legend(fontsize=8)

    fig.savefig(output)
    return fig
