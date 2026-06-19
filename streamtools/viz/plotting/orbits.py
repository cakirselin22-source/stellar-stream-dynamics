"""
Orbit and trajectory plots: cluster center orbit, individual star orbits,
cluster trajectory in multiple projections.
"""
import numpy as np
import matplotlib.pyplot as plt

from .style import savefig


def plot_cluster_orbit(data, outfile="cluster_orbit.png"):
    """
    Plot the cluster center's galactocentric radius vs time, and its xy track.
    Prints orbital eccentricity (computed from min/max radius over the range given).

    """
    R = np.linalg.norm(data.rc3, axis=1)
    x, y = data.rc3[:, 0], data.rc3[:, 1]

    R_peri, R_apo = np.min(R), np.max(R)
    e = (R_apo - R_peri) / (R_apo + R_peri)
    print("Eccentricity =", e)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Galactocentric Radius")
    ax[0].plot(data.times, R)

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].plot(x, y)

    savefig(fig, outfile)
    return fig, e


def plot_cluster_trajectory(data, i_start, i_end, size=40, tmin=3, tmax=14, outfile="cluster_trajectory.png"):
    """
    Plot the trajectory of a star cluster in three projections:
      1. xy-projection of the cluster center
      2. radial distance vs time (galactocentric R and |r|)
      3. R vs z projection of the cluster orbit
    """
    sl = slice(i_start, i_end)
    rc = data.rc3[sl]
    ts = data.times[sl]

    xs, ys, zs = rc[:, 0], rc[:, 1], rc[:, 2]
    Rs = np.sqrt(xs**2 + ys**2)
    dm = np.linalg.norm(rc, axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].axis("equal")
    ax[0].set(xlim=(-size, size), ylim=(-size, size))
    ax[0].set_xlabel("x [kpc]", size="x-large")
    ax[0].set_ylabel("y [kpc]", size="x-large")
    ax[0].plot(xs, ys, lw=1)

    ax[1].set_yscale("log")
    ax[1].set_xlabel("time [Gyr]", size="x-large")
    ax[1].set_ylabel("radius [kpc]", size="x-large")
    ax[1].set_xlim(tmin, tmax)
    ax[1].plot(ts, dm, label="r")
    ax[1].plot(ts, Rs, label="R")
    ax[1].legend()

    ax[2].set_xlabel("R [kpc]", size="x-large")
    ax[2].set_ylabel("z [kpc]", size="x-large")
    ax[2].plot(Rs, zs, lw=1)

    plt.tight_layout()
    savefig(fig, outfile)
    return fig, ax


def plot_star_orbits(data, i_end, nskip=4000, size=0.1, outfile="orbits.pdf"):
    """
    Plot xy orbits (cluster-rotating frame) and r(t) for a subsample of
    stars (every `nskip`-th star), from snapshot 0 through i_end.
    """
    n_snaps = i_end + 1
    star_idx = np.arange(0, data.n_stars, nskip)

    times = data.times[:n_snaps]
    x = data.rotx[:n_snaps][:, star_idx]
    y = data.roty[:n_snaps][:, star_idx]
    r = data.radius()[:n_snaps][:, star_idx]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].axis("equal")
    ax[0].set(xlim=(-size, size), ylim=(-size, size))
    ax[0].set_xlabel("xr [kpc]")
    ax[0].set_ylabel("yr [kpc]")

    ax[1].set_yscale("log")
    ax[1].set_xlim(3, 14)
    ax[1].set_xlabel("time [Gyr]")
    ax[1].set_ylabel("r [kpc]")

    for k in range(len(star_idx)):
        ax[0].plot(x[:, k], y[:, k], lw=0.8)
        ax[1].plot(times, r[:, k], lw=0.5)

    savefig(fig, outfile)
    return fig


def plot_cluster_orbits_filtered(
    data, i_start, i_end, nskip=2, rmin=0.1, rmax=0.5, zmax=0.02, size=0.5,
    outfile="cluster_orbits_filtered.png",
):
    """
    Plot stellar orbits in the cluster-rotating frame, with each star's
    track masked to the snapshots where it satisfies the radial/vertical
    selection (rmin < r < rmax, |z| < zmax).

    """
    sl = slice(i_start, i_end + 1)
    star_idx = np.arange(0, data.n_stars, nskip)

    x = data.rotx[sl][:, star_idx]
    y = data.roty[sl][:, star_idx]
    z = data.rotz[sl][:, star_idx]
    r = data.radius()[sl][:, star_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("equal")
    ax.set(xlim=(-size, size), ylim=(-size, size))

    for k in range(len(star_idx)):
        sel = (np.abs(z[:, k]) < zmax) & (r[:, k] > rmin) & (r[:, k] < rmax)
        ax.plot(x[sel, k], y[sel, k], lw=0.5)

    ax.set_xlabel("xr [kpc]")
    ax.set_ylabel("yr [kpc]")

    savefig(fig, outfile)
    return fig, ax


def plot_star_orbits_energy_snapshots(
    data, i_start, i_end, nskip=1, rmin=0.1, rmax=0.5, zmax=0.02, size=0.5,
    E_select_min=-2, r_select_min=0.001, r_select_max=0.01,
    outfile="star_orbits_energy.png",
):
    """
    Plot orbits (xy, r vs t, E vs t) for stars selected at the initial
    snapshot by an energy + radius cut.

    """
    r0 = data.radius()[i_start]
    E0 = data.E[i_start]

    select = (E0 > E_select_min) & (r0 > r_select_min) & (r0 < r_select_max)
    star_ids = np.where(select)[0][::nskip]
    nstars = len(star_ids)

    if nstars == 0:
        raise ValueError("No stars selected -- adjust cuts.")

    sl = slice(i_start, i_end)
    times = data.times[sl]
    x = data.rotx[sl][:, star_ids]
    y = data.roty[sl][:, star_ids]
    z = data.rotz[sl][:, star_ids]
    r = data.radius()[sl][:, star_ids]
    E = data.E[sl][:, star_ids]

    cmap = plt.cm.get_cmap("rainbow", nstars)
    fig, ax = plt.subplots(1, 3, figsize=(16, 7))

    ax[0].axis("equal")
    ax[0].set(xlim=(-size, size), ylim=(-size, size))
    ax[0].set_xlabel("xr [kpc]")
    ax[0].set_ylabel("yr [kpc]")

    ax[1].set_yscale("log")
    ax[1].set_xlim(4, 12)
    ax[1].set_ylim(0.01, 10)
    ax[1].set_xlabel("time [Gyr]")
    ax[1].set_ylabel("r [kpc]")

    ax[2].set_xlabel("time [Gyr]")
    ax[2].set_ylabel("Energy")

    for k in range(nstars):
        sel = (np.abs(z[:, k]) < zmax) & (r[:, k] > rmin) & (r[:, k] < rmax)
        ax[0].plot(x[sel, k], y[sel, k], lw=0.6, color=cmap(k))
        ax[1].plot(times[sel], r[sel, k], lw=0.6, color=cmap(k))
        ax[2].plot(times, E[:, k], lw=0.6, color=cmap(k))

    plt.tight_layout()
    savefig(fig, outfile)
    return fig, ax

