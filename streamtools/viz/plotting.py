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

def plot_energyvstime_with_orbits(
    snapshots, ifile_start, ifile_end, E_threshold, nstars=10,
    e_ylim=(-10, 30), r_ylim=(-1, 10), output="energyvstime_orbits"
):
    """
    Plot energy, radius, angular momentum, and radial velocity evolution
    for a random sample of stars in the cluster frame.

    Classifies each tracked star's final state as "bound", "prograde escape",
    or "retrograde escape" based on energy and angular momentum direction
    at the final snapshot.

    Parameters
    ----------
    snapshots : list of Snapshot
    ifile_start, ifile_end : int
        Snapshot index range to plot over.
    E_threshold : float
        Only stars with initial binding energy below this value are eligible.
    nstars : int
        Number of stars to randomly sample from eligible candidates.
    e_ylim, r_ylim : tuple
        Axis limits for the energy and radius panels.
    output : str
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    escape_type : dict
        Maps tracked star index -> "bound" / "prograde escape" / "retrograde escape"
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
    L_series = {idx: [] for idx in tracked_idx}
    R_series = {idx: [] for idx in tracked_idx}
    v_rad_series = {idx: [] for idx in tracked_idx}

    for snap in snapshots[ifile_start:ifile_end + 1]:
        E_snap = dynamics.calculate_energy(
            snap.rs3, snap.vs3, snap.rc3, snap.vc3, snap.time, "binding"
        )
        rs_rel = snap.rs3 - snap.rc3
        vs_rel = snap.vs3 - snap.vc3
        v_rad = snap.radial_velocity(rs=snap.rs3, vs=snap.vs3)
        rotx, roty, rotz = snap.project_cluster_frame()
        R_snap = np.sqrt(rotx**2 + roty**2 + rotz**2)
        Lz = np.cross(rs_rel, vs_rel)[:, 2]

        times.append(snap.time)

        for idx in tracked_idx:
            E_series[idx].append(E_snap[idx])
            R_series[idx].append(R_snap[idx])
            L_series[idx].append(Lz[idx])
            v_rad_series[idx].append(v_rad[idx])

    # --- classify final state per star ---
    final_snap = snapshots[ifile_end]
    L_cluster_z_final = np.cross(final_snap.rc3, final_snap.vc3)[2]

    escape_type = {}
    for idx in tracked_idx:
        if E_series[idx][-1] > 0:
            escape_type[idx] = (
                "prograde escape" if L_series[idx][-1] * L_cluster_z_final > 0
                else "retrograde escape"
            )
        else:
            escape_type[idx] = "bound"

    # --- plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()

    for idx in tracked_idx:
        ax1.plot(times, E_series[idx])
    ax1.set(ylabel="Energy", ylim=e_ylim, title="Energy evolution")

    for idx in tracked_idx:
        ax2.plot(times, R_series[idx])
    ax2.set(ylabel="Radius", ylim=r_ylim, title="Radius evolution")

    for idx in tracked_idx:
        ax3.plot(times, L_series[idx])
    ax3.set(xlabel="Time", ylabel="Lz", title="Angular momentum evolution")

    for idx in tracked_idx:
        ax4.plot(times, v_rad_series[idx])
    ax4.set(xlabel="Time", ylabel="v_rad", title="Radial velocity (cluster frame)")

    plt.tight_layout()
    fig.savefig(output)

    return fig, escape_type
