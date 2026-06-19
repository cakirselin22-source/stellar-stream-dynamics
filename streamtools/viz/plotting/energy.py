"""
Energy-focused plots: E vs time, E vs radius, spatial E maps.

All functions here take a SnapshotData bundle (precomputed theta/phi/E/
v_rad/rotx/roty/rotz/times) 
"""
import numpy as np
import matplotlib.pyplot as plt

from ..analysis import selection, series
from .style import savefig


def plot_energyvstime(
    data, snapshots, i_start, i_end, E_threshold, nstars=10,
    t_xlim=(3, 14), e_ylim=(-10, 30), outfile="energyvstime_random.png", rng=None,
):
    """
    Plot cluster-frame energy and radius evolution for a random sample of stars
    that start below a given binding-energy threshold.

    Parameters
    ----------
    data : SnapshotData
        Precomputed bundle from load_snapshot_data().
    snapshots : list of Snapshot
        Raw snapshots -- only used here to clip i_end to the available range;
        pass `len(snapshots)` directly if you'd rather not carry snapshots around.
    i_start, i_end : int
        Snapshot index range to plot over.
    E_threshold : float
        Only stars with initial binding energy below this value are eligible.
    nstars : int
        Number of stars to randomly sample from eligible candidates.
    t_xlim, e_ylim : tuple
        Axis limits for the time and energy panels.
    outfile : str
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    i_end = min(i_end, len(snapshots) - 1)

    tracked_idx = selection.tracked_stars_below_threshold(
        data, i_start, E_threshold, nstars, rng=rng
    )

    times, E = series.energy_series(data, tracked_idx, i_start, i_end)
    _, R = series.radius_series(data, tracked_idx, i_start, i_end)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for k, idx in enumerate(tracked_idx):
        ax1.plot(times, E[:, k], label=f"Star {idx}")
        ax2.plot(times, R[:, k])

    ax1.set(xlabel="Time [Gyr]", ylabel="Energy", xlim=t_xlim, ylim=e_ylim,
            title=f"Energy evolution (E < {E_threshold})")
    ax2.set(xlabel="Time [Gyr]", ylabel="Radius", xlim=t_xlim)
    ax1.legend(fontsize=8)

    savefig(fig, outfile)
    return fig


def plot_energyvstime_with_orbits(
    data, snapshots, i_start, i_end, E_threshold, nstars=10,
    e_ylim=(-10, 30), r_ylim=(-1, 10), outfile="energyvstime_orbits.png", rng=None,
):
    """
    Plot energy, radius, angular momentum, and radial velocity evolution
    for a random sample of stars in the cluster frame.

    Classifies each tracked star's final state as "bound", "prograde escape",
    or "retrograde escape" based on energy and angular momentum direction
    at the final snapshot.

    Lz and L_cluster_z come straight from the SnapshotData cache

    Parameters
    ----------
    data : SnapshotData
    snapshots : list of Snapshot
    i_start, i_end : int
    E_threshold : float
    nstars : int
    e_ylim, r_ylim : tuple
    outfile : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    escape_type : dict
        Maps array position (0..nstars-1) -> "bound" / "prograde escape" / "retrograde escape"
    tracked_idx : array
        Star indices corresponding to escape_type's keys.
    """
    i_end = min(i_end, len(snapshots) - 1)

    tracked_idx = selection.tracked_stars_below_threshold(
        data, i_start, E_threshold, nstars, rng=rng
    )

    times, E = series.energy_series(data, tracked_idx, i_start, i_end)
    _, R = series.radius_series(data, tracked_idx, i_start, i_end)
    _, v_rad = series.v_rad_series(data, tracked_idx, i_start, i_end)
    _, Lz, L_cluster_z = series.angular_momentum_series(data, tracked_idx, i_start, i_end)

    escape_type = selection.classify_escape_state(E[-1], Lz[-1], L_cluster_z[-1])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.plot(times, E)
    ax1.set(ylabel="Energy", ylim=e_ylim, title="Energy evolution")

    ax2.plot(times, R)
    ax2.set(ylabel="Radius", ylim=r_ylim, title="Radius evolution")

    ax3.plot(times, Lz)
    ax3.set(xlabel="Time", ylabel="Lz", title="Angular momentum evolution")

    ax4.plot(times, v_rad)
    ax4.set(xlabel="Time", ylabel="v_rad", title="Radial velocity (cluster frame)")

    plt.tight_layout()
    savefig(fig, outfile)

    return fig, escape_type, tracked_idx


def energy_vs_radius_grid(data, ncols=5, figsize_per_plot=(4, 4), outfile="energyvsradius_grid.png"):
    """
    Grid of E vs (rotx, roty) scatter plots, one panel per snapshot.
    """
    n_snaps = data.n_snaps
    nrows = int(np.ceil(n_snaps / ncols))
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    for i in range(n_snaps):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        sc = ax.scatter(data.rotx[i], data.roty[i], s=1, lw=0, c=data.E[i], cmap="rainbow", alpha=0.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("rotx")
        ax.set_ylabel("roty")
        ax.set_title(f"Snapshot {i}, t={data.times[i]:.2f} Gyr")
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.grid(True)
        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    savefig(fig, outfile)
    return fig


def energy_vs_radius_tracks(data, i_start, i_end, nstars=5, start="bound", outfile="energyvsradius_tracks.png"):
    """
    Plot radial distance of the *same* tracked stars through time

    Parameters
    ----------
    data : SnapshotData
    i_start, i_end : int
        Snapshot indices to track.
    nstars : int
        Number of stars to track (first nstars matching `start`).
    start : str
        "bound" or "unbound" -- selects stars from the initial snapshot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    i_end = min(i_end, data.n_snaps - 1)

    if start == "bound":
        idx = selection.select_below_threshold(data, i_start, 0)
    elif start == "unbound":
        idx = selection.select_above_threshold(data, i_start, 0)
    else:
        raise ValueError(f"start must be 'bound' or 'unbound', got {start!r}")

    tracked_idx = idx[:nstars]
    times, R = series.radius_series(data, tracked_idx, i_start, i_end)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, sid in enumerate(tracked_idx):
        ax.plot(times, R[:, k], label=f"Star {sid}", linewidth=0.8)
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("Radial distance [kpc]")
    ax.set_yscale("log")
    ax.set_title(f"Radius evolution of {start} stars")
    ax.grid(True)
    ax.legend()

    savefig(fig, outfile)
    return fig


def energyxy(data, i_snap, outfile="energyxy.png"):
    """Single-snapshot scatter of E colored over (rotx, roty)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(data.rotx[i_snap], data.roty[i_snap], s=1, lw=0,
                     c=data.E[i_snap], cmap="rainbow", alpha=0.5)
    ax.set_xlabel("rotx")
    ax.set_ylabel("roty")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.grid(True)
    fig.colorbar(sc, ax=ax)

    savefig(fig, outfile)
    return fig


