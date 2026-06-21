"""
Animated views of the cluster-rotating frame: energy maps, energy-group
membership, and escape-state evolution over time.

Saving to mp4 requires ffmpeg on PATH; if that's unavailable, pass an
outfile ending in .gif to fall back to Pillow's writer.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, SymLogNorm

from ...analysis import selection


def animate_energy_scatter(
    data, energy_range, rotz_range, log=False, fps=12, frame_step=3,
    panel_lims=(1.0, 5.0), outfile="energy_scatter.mp4",
):
    """
    Animate (rotx, roty) positions colored by binding energy, restricted
    to a thin |rotz| < rotz_range slice. Two panels at different zoom
    levels share one color scale, so the same frame shows both the dense
    cluster core and the wider stream.

    Parameters
    ----------
    data : SnapshotData
    energy_range : (float, float)
        Selection window and color-scale range for binding energy.
    rotz_range : float
        Half-thickness of the |rotz| slice to include.
    log : bool
        Use a symmetric-log color norm instead of a linear one --
        useful since binding energy spans orders of magnitude on both
        sides of zero. (Replaces the original script's `log(E)` color
        mapping, which produced NaNs for every E < 0 star; SymLogNorm
        handles the sign change correctly.)
    fps : int
    frame_step : int
        Use every frame_step-th snapshot -- animations get long otherwise.
    panel_lims : (float, float)
        Half-width of the (zoomed, wide) panels' axis limits.
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    if log:
        linthresh = max(abs(energy_range[0]), abs(energy_range[1])) * 1e-2 or 1e-3
        norm = SymLogNorm(linthresh=linthresh, vmin=energy_range[0], vmax=energy_range[1])
    else:
        norm = Normalize(vmin=energy_range[0], vmax=energy_range[1])

    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(12, 5))
    scat_zoom = ax_zoom.scatter([], [], s=1, lw=0, c=[], cmap="rainbow", norm=norm)
    scat_wide = ax_wide.scatter([], [], s=1, lw=0, c=[], cmap="rainbow", norm=norm)

    for ax, lim in zip((ax_zoom, ax_wide), panel_lims):
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="rotx", ylabel="roty")
    fig.colorbar(scat_wide, ax=ax_wide, label="E (symlog)" if log else "E")

    frames = range(0, data.n_snaps, frame_step)

    def update(i):
        E = data.E[i]
        mask = (np.abs(data.rotz[i]) < rotz_range) & (E > energy_range[0]) & (E < energy_range[1])
        xy = np.column_stack((data.rotx[i][mask], data.roty[i][mask]))
        c = E[mask]
        for scat in (scat_zoom, scat_wide):
            scat.set_offsets(xy)
            scat.set_array(c)

        t_str = f"t={data.times[i]:.2f} Gyr"
        ax_zoom.set_title(f"Zoom | {t_str}")
        ax_wide.set_title(f"Wide | {t_str}")
        return scat_zoom, scat_wide

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=True)
    anim.save(outfile, fps=fps)
    plt.close(fig)
    return anim


def animate_energy_groups(
    data, energy_range=(-2, 2), rotz_range=0.2, fps=10,
    panel_lims=(0.05, 0.03), outfile="energy_groups.mp4",
):
    """
    Animate stars split into three energy groups -- bound, transition,
    unbound -- fixed at the first snapshot, over two panels at different
    zoom levels.

    Group membership is fixed at t=0 deliberately: the point is to watch
    where stars that *started* in each energy bracket end up spatially
    over time, not to re-bucket every frame.

    Parameters
    ----------
    data : SnapshotData
    energy_range : (float, float)
        (E_min, E_max) split points: bound is E0 < E_min, unbound is
        E0 > E_max, transition is everything in between.
    rotz_range : float
        Half-thickness of the |rotz| slice to include.
    fps : int
    panel_lims : (float, float)
        Half-width of the (zoomed, wide) panels' axis limits.
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    E0 = data.E[0]
    E_min, E_max = energy_range
    groups = {
        "bound": (np.where(E0 < E_min)[0], "tab:blue"),
        "transition": (np.where((E0 >= E_min) & (E0 <= E_max))[0], "gold"),
        "unbound": (np.where(E0 > E_max)[0], "tab:red"),
    }
    print(", ".join(f"{name}: {len(idx)}" for name, (idx, _) in groups.items()))

    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(12, 5))
    panels = {}
    for ax, lim, title in zip((ax_zoom, ax_wide), panel_lims, ("Zoomed View", "Wide View")):
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=title)
        bg = ax.scatter([], [], s=1, color="gray", alpha=0.05)
        scats = {
            name: ax.scatter([], [], s=4, color=color, label=name if ax is ax_zoom else None)
            for name, (_, color) in groups.items()
        }
        panels[ax] = (bg, scats)
    ax_zoom.legend(loc="upper right", fontsize=8)

    def update(i):
        rotx, roty, rotz = data.rotx[i], data.roty[i], data.rotz[i]
        mask_plane = np.abs(rotz) < rotz_range
        bg_xy = np.column_stack((rotx[mask_plane], roty[mask_plane]))

        artists = []
        for ax, (bg, scats) in panels.items():
            bg.set_offsets(bg_xy)
            artists.append(bg)
            for name, (idx, _) in groups.items():
                grp_mask = np.zeros_like(mask_plane)
                grp_mask[idx] = True
                grp_mask &= mask_plane
                scats[name].set_offsets(np.column_stack((rotx[grp_mask], roty[grp_mask])))
                artists.append(scats[name])
            ax.set_title(f"t = {data.times[i]:.2f} Gyr")

        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=data.n_snaps, interval=1000 // fps)
    anim.save(outfile, fps=fps)
    plt.close(fig)
    return anim


def animate_energy_histogram_grid(
    data, energy_range=(-2, 2), bound_nskip=1000, bins=50, spatial_lim=0.4,
    fps=10, outfile="energy_histogram_grid.mp4",
):
    """
    Combined spatial + property-histogram animation: for each of three
    energy groups (bound/transition/unbound, fixed at t=0), show a
    spatial (rotx, roty) scatter colored by energy alongside histograms
    of energy, Lz, and radius for that group, all evolving together.

    Parameters
    ----------
    data : SnapshotData
    energy_range : (float, float)
        (E_min, E_max) group split points, as in `animate_energy_groups`.
    bound_nskip : int
        Subsample the (typically much larger) bound group by this
        stride, so its scatter/histogram rendering doesn't dominate
        render time.
    bins : int
    spatial_lim : float
        Half-width of the spatial panels' axis limits.
    fps : int
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    E0 = data.E[0]
    E_min, E_max = energy_range
    groups = {
        "bound": np.where(E0 < E_min)[0][::bound_nskip],
        "transition": np.where((E0 >= E_min) & (E0 <= E_max))[0],
        "unbound": np.where(E0 > E_max)[0],
    }
    print(", ".join(f"{name}: {len(idx)}" for name, idx in groups.items()))

    R_all = data.radius()

    fig, axs = plt.subplots(3, 4, figsize=(16, 10))
    row_of = {"bound": 0, "transition": 1, "unbound": 2}

    spatial_axes, hist_axes, scats = {}, {}, {}
    for name, row in row_of.items():
        ax_sp, ax_E, ax_L, ax_R = axs[row]
        ax_sp.set(xlim=(-spatial_lim, spatial_lim), ylim=(-spatial_lim, spatial_lim),
                  title=name.capitalize())
        scat = ax_sp.scatter([], [], s=4, lw=0, c=[], cmap="rainbow")
        fig.colorbar(scat, ax=ax_sp, label="E")
        spatial_axes[name] = ax_sp
        scats[name] = scat
        hist_axes[name] = (ax_E, ax_L, ax_R)

    axs[0, 1].set_title("Energy")
    axs[0, 2].set_title("Angular momentum")
    axs[0, 3].set_title("Radius")

    def update(i):
        rotx, roty = data.rotx[i], data.roty[i]
        E, Lz, r = data.E[i], data.Lz[i], R_all[i]

        artists = []
        for name, idx in groups.items():
            e_vals = E[idx]
            scat = scats[name]
            scat.set_offsets(np.column_stack((rotx[idx], roty[idx])))
            scat.set_array(e_vals)
            if len(e_vals):
                scat.set_clim(e_vals.min(), e_vals.max())
            artists.append(scat)

            ax_E, ax_L, ax_R = hist_axes[name]
            for ax, vals in ((ax_E, e_vals), (ax_L, Lz[idx]), (ax_R, r[idx])):
                ax.clear()
                ax.hist(vals, bins=bins)

        for name, ax in spatial_axes.items():
            ax.set_title(f"{name.capitalize()} | t={data.times[i]:.2f} Gyr")

        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=data.n_snaps, interval=1000 // fps)
    anim.save(outfile, fps=fps)
    plt.close(fig)
    return anim


def animate_escape_state(
    data, rotz_range, fps=12, frame_step=3, panel_lims=(1.0, 5.0),
    outfile="escape_state.mp4",
):
    """
    Animate (rotx, roty) positions colored by escape state: bound (blue),
    potential escaper -- E > 0 but not yet permanently unbound (orange),
    and permanently escaped (red).


    Parameters
    ----------
    data : SnapshotData
    rotz_range : float
        Half-thickness of the |rotz| slice to include.
    fps : int
    frame_step : int
    panel_lims : (float, float)
        Half-width of the (zoomed, wide) panels' axis limits.
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    escape_idx = selection.escape_time_index(data)

    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(12, 5))
    scat_zoom = ax_zoom.scatter([], [], s=1, lw=0)
    scat_wide = ax_wide.scatter([], [], s=1, lw=0)
    for ax, lim in zip((ax_zoom, ax_wide), panel_lims):
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="rotx", ylabel="roty")

    frames = range(0, data.n_snaps, frame_step)

    def update(i):
        rotx, roty, rotz = data.rotx[i], data.roty[i], data.rotz[i]
        E = data.E[i]
        mask = np.abs(rotz) < rotz_range

        escaped_now = (escape_idx != -1) & (escape_idx <= i)
        potential = (E > 0) & ~escaped_now
        colors = np.where(escaped_now, "red", np.where(potential, "orange", "blue"))

        xy = np.column_stack((rotx[mask], roty[mask]))
        c = colors[mask]
        for scat in (scat_zoom, scat_wide):
            scat.set_offsets(xy)
            scat.set_color(c)

        t_str = f"t={data.times[i]:.2f} Gyr"
        ax_zoom.set_title(t_str)
        ax_wide.set_title(t_str)
        return scat_zoom, scat_wide

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=True)
    anim.save(outfile, fps=fps, dpi=200)
    plt.close(fig)
    return anim


def animate_escape_state_3d(
    data, rotz_range, fps=10, frame_step=3, lims=(1.0, 5.0),
    azim_start=30, azim_end=390, elev=25, outfile="escape_state_3d.mp4",
):
    """
    3-D companion to `animate_escape_state`: rotating 3-D scatter at two
    zoom levels (full cluster + wide, to catch escapers) plus a 2-D
    edge-on (X-Z) slice, all colored by escape state (see
    `selection.escape_time_index`).

    Parameters
    ----------
    data : SnapshotData
    rotz_range : float
        Half-thickness of the |roty| slice used for the edge-on panel.
    fps : int
    frame_step : int
    lims : (float, float)
        Half-width of the (zoom, wide) 3-D panels' axis limits; the
        edge-on panel uses the wide limit.
    azim_start, azim_end : float
        Azimuth sweep (degrees) for the 3-D panels over the animation;
        set equal to disable rotation.
    elev : float
        Elevation angle (degrees) for the 3-D panels.
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    escape_idx = selection.escape_time_index(data)

    fig = plt.figure(figsize=(18, 6))
    fig.subplots_adjust(wspace=0.35)
    ax3d_zoom = fig.add_subplot(131, projection="3d")
    ax3d_wide = fig.add_subplot(132, projection="3d")
    ax2d_edge = fig.add_subplot(133)

    for ax, lim in ((ax3d_zoom, lims[0]), (ax3d_wide, lims[1])):
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_xlabel("X", fontsize=8, labelpad=2)
        ax.set_ylabel("Y", fontsize=8, labelpad=2)
        ax.set_zlabel("Z", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim_start)

    ax2d_edge.set(xlim=(-lims[1], lims[1]), ylim=(-lims[1], lims[1]),
                  xlabel="X (rotated)", ylabel="Z (rotated)")

    sc_close = ax3d_zoom.scatter([], [], [], s=1, lw=0)
    sc_wide = ax3d_wide.scatter([], [], [], s=1, lw=0)
    sc_edge = ax2d_edge.scatter([], [], s=1, lw=0)

    frame_list = list(range(0, data.n_snaps, frame_step))
    total_frames = len(frame_list)

    def update(fi):
        i = frame_list[fi]
        rotx, roty, rotz = data.rotx[i], data.roty[i], data.rotz[i]
        E = data.E[i]

        escaped_now = (escape_idx != -1) & (escape_idx <= i)
        potential = (E > 0) & ~escaped_now
        colors = np.where(escaped_now, "red", np.where(potential, "orange", "blue"))

        sc_close._offsets3d = (rotx, roty, rotz)
        sc_close.set_color(colors)
        sc_wide._offsets3d = (rotx, roty, rotz)
        sc_wide.set_color(colors)

        mask_edge = np.abs(roty) < rotz_range
        sc_edge.set_offsets(np.column_stack((rotx[mask_edge], rotz[mask_edge])))
        sc_edge.set_color(colors[mask_edge])

        frac = fi / max(total_frames - 1, 1)
        azim = azim_start + frac * (azim_end - azim_start)
        ax3d_zoom.view_init(elev=elev, azim=azim)
        ax3d_wide.view_init(elev=elev, azim=azim)

        t_str = f"t = {data.times[i]:.2f} Gyr"
        ax3d_zoom.set_title(f"3D (±{lims[0]:g} kpc)\n{t_str}", fontsize=9)
        ax3d_wide.set_title(f"3D (±{lims[1]:g} kpc)\n{t_str}", fontsize=9)
        ax2d_edge.set_title(f"Edge-on XZ slice\n{t_str}", fontsize=9)

        return sc_close, sc_wide, sc_edge

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, blit=False)
    anim.save(outfile, fps=fps, dpi=200)
    plt.close(fig)
    return anim


def animate_potential_escapers(
    data, min_flips=4, fps=15, lim=1.0, outfile="potential_escapers.mp4",
):
    """
    Highlight "potential escapers" -- stars whose energy crosses zero
    repeatedly before settling down (see
    `selection.energy_oscillation_flips`) -- against a faint backdrop of
    the full star field. Each potential escaper is colored red/blue by
    its instantaneous bound/unbound state, and dropped from the
    highlighted set once it permanently escapes (see
    `selection.escape_time_index`).

    Parameters
    ----------
    data : SnapshotData
    min_flips : int
        Minimum number of bound/unbound sign changes to flag a star.
    fps : int
    lim : float
        Half-width of the panel's axis limits.
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    osc_idx, flip_counts = selection.energy_oscillation_flips(data, min_flips=min_flips)
    escape_idx = selection.escape_time_index(data)
    print(f"Found {len(osc_idx)} potential escapers (>= {min_flips} energy sign flips)")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="x (cluster frame)", ylabel="y (cluster frame)")

    scat_all = ax.scatter(data.rotx[0], data.roty[0], s=1, alpha=0.1)
    scat_pe = ax.scatter([], [], s=6, alpha=0.9)
    title = ax.set_title(f"Potential escapers (t={data.times[0]:.3f} Gyr)")

    def update(i):
        scat_all.set_offsets(np.column_stack((data.rotx[i], data.roty[i])))

        still_in = osc_idx[(escape_idx[osc_idx] == -1) | (i < escape_idx[osc_idx])]
        if len(still_in):
            xy = np.column_stack((data.rotx[i][still_in], data.roty[i][still_in]))
            colors = np.where(data.E[i][still_in] > 0, "red", "blue")
            scat_pe.set_offsets(xy)
            scat_pe.set_color(colors)
        else:
            scat_pe.set_offsets(np.empty((0, 2)))

        title.set_text(f"Potential escapers (t={data.times[i]:.3f} Gyr)")
        return scat_all, scat_pe, title

    anim = FuncAnimation(fig, update, frames=data.n_snaps, interval=1000 / fps, blit=False)
    anim.save(outfile, fps=fps)
    plt.close(fig)
    return anim
