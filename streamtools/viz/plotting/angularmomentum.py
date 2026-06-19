"""
Angular momentum plots, reading Lz/L_cluster_z straight from the
SnapshotData cache (precomputed by save_snapshot_data).
"""
import numpy as np
import matplotlib.pyplot as plt

from ..analysis import series
from .style import savefig


def angular_momentum_stars(data, tracked_idx=None, n_track=20, outfile="angular_momentum_stars.png", rng=None):
    """
    Plot Lz(t) for a tracked sample of stars, |L|(t) for the cluster, and
    E(t) for the same tracked stars.

    Parameters
    ----------
    data : SnapshotData
    tracked_idx : array, optional
        Specific stars to track. If None, `n_track` random stars are chosen.
    n_track : int
        Number of stars to randomly track if tracked_idx is not given.
    """
    if tracked_idx is None:
        rng = np.random.default_rng() if rng is None else rng
        tracked_idx = rng.choice(data.n_stars, size=n_track, replace=False)
    n_track = len(tracked_idx)

    times, Lz, L_cluster_z = series.angular_momentum_series(data, tracked_idx)
    _, E = series.energy_series(data, tracked_idx)

    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

    for j in range(n_track):
        ax[0].plot(times, Lz[:, j], alpha=0.6, label=f"Star {tracked_idx[j]}")
    ax[0].legend(fontsize=7)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("L (stars)")
    ax[0].set_title("Tracked Stars Angular Momentum")

    ax[1].plot(times, L_cluster_z, color="black")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("|L| (cluster)")
    ax[1].set_title("Cluster Angular Momentum")

    for j in range(n_track):
        ax[2].plot(times, E[:, j], alpha=0.6, label=f"Star {tracked_idx[j]}")
    ax[2].legend(fontsize=7)
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Energy")
    ax[2].set_title("Energy")

    fig.tight_layout()
    savefig(fig, outfile)
    return fig


def plot_Lz_nstars(data, nstars=5, outfile="Lz_nstars.png"):
    """
    Track and plot L_z vs time for the first `nstars` particles (by raw
    star index, not energy-selected).
    """
    star_indices = np.arange(nstars)
    times = data.times
    Lz = data.Lz[:, star_indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    for j in range(nstars):
        ax.plot(times, Lz[:, j], lw=1.5, label=f"Star {j}")
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel(r"$L_z$")
    ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, outfile)
    return fig
