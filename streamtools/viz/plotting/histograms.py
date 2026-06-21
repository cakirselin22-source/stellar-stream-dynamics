"""
Histogram-based views of escape properties: distributions at escape,
escape-time/delay-time statistics, time-binned and radius-binned
breakdowns, and a couple of animated variants.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

from ...analysis import selection, series
from .style import savefig


def _mark_peaks(ax, counts, bin_centers, max_peaks=2, distance=3, colors=("red", "blue")):
    """
    Find and annotate up to `max_peaks` tallest local maxima of a
    histogram's bar heights with vertical dashed lines + labels.

    Returns the x-positions of the marked peaks, tallest first.
    """
    peaks, _ = find_peaks(counts, distance=distance)
    if len(peaks) == 0:
        return []

    peaks_sorted = peaks[np.argsort(counts[peaks])[::-1]][:max_peaks]
    ymax = ax.get_ylim()[1] or 1
    positions = []
    for k, p in enumerate(peaks_sorted):
        x = bin_centers[p]
        color = colors[k % len(colors)]
        ax.axvline(x, color=color, linestyle="--", lw=1.5)
        ax.text(x, 0.9 * ymax, f"{x:.1f}", color=color, rotation=90,
                 va="top", ha="right" if k == 0 else "left", fontsize=8)
        positions.append(x)
    return positions


def histogram_at_escape(data, property="energy", bins=30, escape_idx=None, outfile=None):
    """
    Histogram of `property`, evaluated at each star's permanent-escape
    snapshot. Consolidates four near-identical scripts
    (histogram_angular_momentum, histogram_all, histogram_all_theta,
    histogram_all_phi) that each recomputed escape detection from raw
    snapshots and only differed in which property they read off after.

    Parameters
    ----------
    data : SnapshotData
    property : str
        One of series.PROPERTY_LABELS (energy, v_rad, r, theta, phi, Lz).
    bins : int
    escape_idx : array, optional
        Precomputed via selection.escape_time_index(data).
    outfile : str, optional
        Defaults to "<property>_at_escape.png".

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    values, times, star_idx = series.escape_property_values(data, property, escape_idx=escape_idx)
    outfile = outfile or f"{property}_at_escape.png"
    label = series.PROPERTY_LABELS.get(property, property)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, alpha=0.8, color="steelblue")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of stars")
    ax.set_title(f"{label} at escape (N={len(values)})")

    savefig(fig, outfile)
    return fig


def histogram_delay_time(data, bins=10, outfile="delaytime_histogram.png"):
    """
    Histogram of "delay time": Gyr elapsed between a star's first
    crossing into E > 0 (`selection.unbinding_time_index`) and its
    permanent escape (`selection.escape_time_index`) -- i.e. how long
    stars flirt with the escape threshold before actually leaving for
    good. Only stars with both indices defined are included.

    """
    unbind_idx = selection.unbinding_time_index(data)
    escape_idx = selection.escape_time_index(data)

    valid = (unbind_idx != -1) & (escape_idx != -1)
    delay_time = data.times[escape_idx[valid]] - data.times[unbind_idx[valid]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delay_time, bins=bins, range=(delay_time.min(), delay_time.max()))
    ax.set_xlabel("Delay time [Gyr]")
    ax.set_ylabel("Number of stars")
    ax.set_title("Distribution of escape delay times")

    savefig(fig, outfile)
    return fig


def histogram_escape_time(data, bins=20, escape_idx=None, outfile="escapetimes_distribution.png"):
    """
    Distribution of permanent-escape times, plus the cumulative number
    of stars escaped by time t.
    """
    if escape_idx is None:
        escape_idx = selection.escape_time_index(data)
    escape_times = data.times[escape_idx[escape_idx != -1]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].hist(escape_times, bins=bins)
    ax[0].set(xlabel="Escape time [Gyr]", ylabel="Number of stars",
              title="Distribution of escape times")

    sorted_times = np.sort(escape_times)
    ax[1].step(sorted_times, np.arange(1, len(sorted_times) + 1), where="post")
    ax[1].set(xlabel="Time [Gyr]", ylabel="Cumulative stars escaped",
              title="Cumulative escapes vs time")

    savefig(fig, outfile)
    return fig


def histogram_energy_single_snapshot(data, i_snap, bins=100, bound=True, outfile=None):
    """
    Histogram of binding energy for bound (E < 0) or unbound (E > 0)
    stars at a single snapshot.
    """
    E = data.E[i_snap]
    mask = E < 0 if bound else E > 0
    label = "bound" if bound else "unbound"
    outfile = outfile or f"energy_hist_{label}_snap{i_snap}.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(E[mask], bins=bins)
    ax.set_xlabel(f"Energy (E {'< 0' if bound else '> 0'})")
    ax.set_ylabel("Number of stars")
    ax.set_title(f"{label.capitalize()} stars energy distribution (t={data.times[i_snap]:.2f} Gyr)")

    savefig(fig, outfile)
    return fig


def histogram_by_time_bin_grid(
    data, property="theta", bin_width=1.0, bins=30, mark_peaks=True,
    escape_idx=None, outfile=None,
):
    """
    Grid of histograms (one panel per bin_width-Gyr escape-time window)
    of `property` at escape.

    Parameters
    ----------
    data : SnapshotData
    property : str
        One of series.PROPERTY_LABELS.
    bin_width : float
        Width of each escape-time window, in Gyr.
    bins : int
        Histogram bins within each panel.
    mark_peaks : bool
        Annotate the 1-2 tallest peaks per panel (useful for spotting
        e.g. a bimodal leading/trailing theta structure).
    escape_idx : array, optional
        Precomputed via selection.escape_time_index(data).
    outfile : str, optional
        Defaults to "<property>_serialhist.png".

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    outfile = outfile or f"{property}_serialhist.png"
    label = series.PROPERTY_LABELS.get(property, property)
    bin_edges, values_per_bin, labels = series.escape_property_by_time_bin(
        data, property, bin_width=bin_width, escape_idx=escape_idx
    )
    n_bins = len(values_per_bin)

    ncols = 3
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    nonempty = [v for v in values_per_bin if len(v)]
    all_values = np.concatenate(nonempty) if nonempty else np.array([0.0, 1.0])

    for i, (vals, bin_label) in enumerate(zip(values_per_bin, labels)):
        ax = axes[i]
        if len(vals):
            counts, edges, _ = ax.hist(vals, bins=bins, color="tab:blue", alpha=0.7)
            if mark_peaks:
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                _mark_peaks(ax, counts, bin_centers)
        ax.set_ylabel(bin_label, fontsize=9)
        ax.set_xlim(all_values.min(), all_values.max())

    for ax in axes[n_bins:]:
        ax.axis("off")
    if n_bins:
        axes[n_bins - 1].set_xlabel(label)

    fig.suptitle(f"{label} at escape, by escape-time bin", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    savefig(fig, outfile)
    return fig


def animate_histogram_by_time_bin(
    data, property="theta", bin_width=1.0, bins=30, mark_peaks=True,
    fps=2, escape_idx=None, outfile="serial_histogram.mp4",
):
    """
    Animated version of `histogram_by_time_bin_grid`: one frame per
    escape-time bin instead of a static grid 
    """
    label = series.PROPERTY_LABELS.get(property, property)
    bin_edges, values_per_bin, labels = series.escape_property_by_time_bin(
        data, property, bin_width=bin_width, escape_idx=escape_idx
    )
    nonempty = [v for v in values_per_bin if len(v)]
    all_values = np.concatenate(nonempty) if nonempty else np.array([0.0, 1.0])
    xmin, xmax = all_values.min(), all_values.max()

    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        ax.clear()
        vals = values_per_bin[frame]
        ax.set_xlim(xmin - 1, xmax + 1)
        ax.set_title(labels[frame], fontsize=12)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No escapers in this bin", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            return
        counts, edges, _ = ax.hist(vals, bins=bins, alpha=0.7)
        if mark_peaks:
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            _mark_peaks(ax, counts, bin_centers)

    anim = FuncAnimation(fig, update, frames=len(values_per_bin), interval=1000 / fps)
    if str(outfile).endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps)
    else:
        anim.save(outfile, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return anim


def histogram_evolution_peaks(
    data, n_time_bins=20, bins=60, theta_range=(-100, 100), height_frac=0.3,
    distance=5, n_peaks=1, min_stars=10, escape_idx=None,
    outfile="theta_peaks_evolution.png",
):
    """
    Track the dominant escape-theta peak(s) over time: bin escape
    events into `n_time_bins` windows, histogram theta within each, and
    scatter the position of the tallest peak(s) per window against
    time. Useful for watching whether a clean (e.g. bimodal
    leading/trailing) peak structure emerges as the stream matures.

    Returns
    -------
    fig : matplotlib.figure.Figure
    t_centers : list of float
    peak_positions_all : list of arrays
    """
    theta_vals, esc_times, _ = series.escape_property_values(data, "theta", escape_idx=escape_idx)

    bins_time = np.linspace(esc_times.min(), esc_times.max(), n_time_bins)
    digitized = np.digitize(esc_times, bins_time)

    t_centers, peak_positions_all = [], []
    for i in range(1, len(bins_time)):
        mask = digitized == i
        t_centers.append(0.5 * (bins_time[i] + bins_time[i - 1]))
        if np.sum(mask) < min_stars:
            peak_positions_all.append(np.array([]))
            continue

        hist, edges = np.histogram(theta_vals[mask], bins=bins, range=theta_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        peaks, props = find_peaks(hist, height=hist.max() * height_frac, distance=distance)
        if len(peaks) == 0:
            peak_positions_all.append(np.array([]))
            continue

        order = np.argsort(props["peak_heights"])[::-1][:n_peaks]
        peak_positions_all.append(centers[peaks][order])

    fig, ax = plt.subplots(figsize=(6, 4))
    for t, peaks in zip(t_centers, peak_positions_all):
        for p in peaks:
            ax.scatter(t, p, color="red", s=10)
    ax.set(xlabel="Time [Gyr]", ylabel=r"$\theta$ (peak position)",
           title="Evolution of escape directions")

    savefig(fig, outfile)
    return fig, t_centers, peak_positions_all


def histogram_by_radius_cut(
    data, radius_cuts, property="theta", bins=30, escape_idx=None,
    outfile="histogram_by_radius.png",
):
    """
    Overlay histograms of `property` at escape, split into radial-
    distance bands [d, 2d] of escape-snapshot radius for each d in
    `radius_cuts`.

    """
    if escape_idx is None:
        escape_idx = selection.escape_time_index(data)

    values, esc_times, star_idx = series.escape_property_values(data, property, escape_idx=escape_idx)
    r_at_escape = data.radius()[escape_idx[star_idx], star_idx]

    colors = plt.cm.viridis(np.linspace(0, 1, len(radius_cuts)))
    fig, ax = plt.subplots(figsize=(8, 5))

    for color, d in zip(colors, radius_cuts):
        in_band = (r_at_escape >= d) & (r_at_escape < 2 * d)
        ax.hist(values[in_band], bins=bins, color=color, alpha=0.5,
                label=f"{d:g} < r < {2 * d:g}", edgecolor="black")

    label = series.PROPERTY_LABELS.get(property, property)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{label} at escape, by escape radius")
    ax.legend()

    savefig(fig, outfile)
    return fig


def plot_elevation_heatmap(data, n_bins=90, cmap="viridis", log_scale=True, outfile="elevation_heatmap.png"):
    """
    2-D heatmap of "elevation" (the angle out of the cluster's orbital
    plane) vs time, computed from cached rotx/roty/rotz:

        elevation = atan2(rotz, sqrt(rotx**2 + roty**2))

    i.e. how far a star sits above/below the orbital (e_r, e_y) plane.
    This is a distinct quantity from the cached `theta`/`phi` fields
    (which use different reference axes -- see SnapshotData /
    io.precomputed_cache) but only needs the cached rotx/roty/rotz
    components, so no raw per-star position data is required.

    Each snapshot's row is normalized to its own peak density, so
    brightness encodes the *shape* of the angular distribution rather
    than the (rapidly changing) number of stars satisfying any cut.
    """
    elev_bins = np.linspace(-np.pi / 2, np.pi / 2, n_bins + 1)
    elev_centers = 0.5 * (elev_bins[:-1] + elev_bins[1:])

    elevation = np.arctan2(data.rotz, np.sqrt(data.rotx**2 + data.roty**2))
    H = np.zeros((data.n_snaps, n_bins))
    for i in range(data.n_snaps):
        counts, _ = np.histogram(elevation[i], bins=elev_bins)
        if counts.max() > 0:
            H[i] = counts / counts.max()

    fig, ax = plt.subplots(figsize=(12, 7))
    norm = LogNorm(vmin=1e-2, vmax=1) if log_scale else Normalize(0, 1)
    im = ax.pcolormesh(
        data.times, np.degrees(elev_centers), H.T,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized density (per snapshot)", fontsize=11)
    ax.set_xlabel("Time [Gyr]", fontsize=12)
    ax.set_ylabel("Elevation [deg]", fontsize=12)
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_title("Out-of-orbital-plane angle distribution over time", fontsize=13)

    fig.tight_layout()
    savefig(fig, outfile)
    return fig


def animate_property_histograms(
    data, mode="newly_escaped", i_start=0, i_end=None, bins=30, nskip=1,
    fps=15, outfile="property_histograms.mp4",
):
    """
    4-panel (theta, phi, energy, v_rad) histogram animation of escaping
    stars, fully driven by the precomputed cache.

    Parameters
    ----------
    data : SnapshotData
    mode : str
        "newly_escaped" -- only stars that *permanently* escape exactly
            at the current snapshot (selection.escape_time_index).
            Most frames are sparse; this shows the distribution of
            stars as they leave.
        "unbound" -- all stars with E > 0 at the current snapshot,
            including ones that later rebind. Denser; shows the
            instantaneous unbound population rather than permanent
            escapers.
    i_start, i_end : int
    bins : int
    nskip : int
        Use every nskip-th selected star per frame (for speed).
    fps : int
    outfile : str

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation

    Notes
    -----
    Consolidates two originally separate, non-functional scripts: one
    (histogram_escaping_stars_evolution) destructured a single
    SnapshotData return value as a 4-tuple; the other
    (animate_histogram) referenced undefined module-level globals. Both
    are replaced by this single, correct, cache-only implementation.
    """
    if mode not in ("newly_escaped", "unbound"):
        raise ValueError("mode must be 'newly_escaped' or 'unbound'")

    i_end = data.n_snaps - 1 if i_end is None else min(i_end, data.n_snaps - 1)
    snapshot_numbers = list(range(i_start, i_end + 1))

    escape_idx = selection.escape_time_index(data) if mode == "newly_escaped" else None

    theta_frames, phi_frames, E_frames, v_rad_frames = [], [], [], []
    for i in snapshot_numbers:
        mask = (escape_idx == i) if mode == "newly_escaped" else (data.E[i] > 0)
        idx = np.where(mask)[0][::nskip]

        theta_frames.append(data.theta[i][idx])
        phi_frames.append(data.phi[i][idx])
        E_frames.append(data.E[i][idx])
        v_rad_frames.append(data.v_rad[i][idx])

    fig, ax = plt.subplots(1, 4, figsize=(16, 3))

    def update(k):
        real_snap = snapshot_numbers[k]
        for a in ax:
            a.cla()

        if len(theta_frames[k]):
            ax[0].hist(theta_frames[k], bins=bins, color="skyblue", range=(-90, 90))
        ax[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax[0].set_title(rf"$\theta$ (snap {real_snap})")

        if len(phi_frames[k]):
            ax[1].hist(phi_frames[k], bins=bins, color="indianred", range=(-180, 180))
        ax[1].set_title(rf"$\phi$ (snap {real_snap})")

        if len(E_frames[k]):
            e_lo, e_hi = np.percentile(E_frames[k], [1, 99])
            ax[2].hist(E_frames[k], bins=bins, range=(e_lo, e_hi), color="gold")
        ax[2].set_title("Energy")

        if len(v_rad_frames[k]):
            ax[3].hist(v_rad_frames[k], bins=bins, color="seagreen")
        ax[3].set_title(r"$v_{rad}$")

        fig.suptitle(f"Snapshot {real_snap} | t={data.times[real_snap]:.2f} Gyr ({mode})")

    anim = FuncAnimation(fig, update, frames=len(snapshot_numbers), repeat=False)
    anim.save(outfile, writer="ffmpeg", fps=fps, dpi=200)
    plt.close(fig)
    return anim


def leading_trailing_evolution(
    data, i_start=0, i_end=None, nskip=1, bins=50, fps=10,
    outfile="leading_trailing_stream.mp4",
):
    """
    Animate the along-orbit ("leading/trailing") position distribution
    of stars: a histogram of distance along the cluster's orbit, and a
    scatter of that distance against perpendicular spread, evolving
    frame by frame.

    """
    times, s_orbit, r_perp = series.along_orbit_position(data, i_start, i_end, nskip=nskip)
    snapshot_numbers = list(range(i_start, i_start + len(times)))

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4))

    def update(k):
        ax_hist.cla()
        ax_scatter.cla()
        snap_num = snapshot_numbers[k]

        ax_hist.hist(s_orbit[k], bins=bins, color="skyblue")
        ax_hist.axvline(0, color="k", lw=1)
        ax_hist.set(xlabel="Along-orbit distance [kpc]", ylabel="Number of stars",
                    title=f"Snapshot {snap_num} histogram")

        ax_scatter.scatter(s_orbit[k], r_perp[k], s=5, alpha=0.7)
        ax_scatter.axvline(0, color="k", lw=1)
        ax_scatter.set(xlabel="Along-orbit distance [kpc]", ylabel="Perpendicular distance [kpc]",
                       title=f"Snapshot {snap_num} stream shape")

        fig.suptitle(f"Snapshot {snap_num} | t={times[k]:.2f} Gyr")

    anim = FuncAnimation(fig, update, frames=len(times), repeat=False)
    anim.save(outfile, writer="ffmpeg", fps=fps, dpi=200)
    plt.close(fig)
    return anim
