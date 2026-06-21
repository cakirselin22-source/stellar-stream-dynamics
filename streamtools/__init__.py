"""
streamtools: analysis and visualization for the tidal stream of stars
escaping a star cluster as it dissolves while orbiting a host galaxy.

Typical workflow
----------------
    from streamtools.io.cache import load_or_read_snapshots
    from streamtools.io.precomputed_cache import save_snapshot_data, load_snapshot_data

    snapshots = load_or_read_snapshots("g656")                  # read + pickle-cache raw snapshots
    save_snapshot_data(snapshots, filename="g656_data.npz")     # precompute derived scalars, once
    data = load_snapshot_data("g656_data.npz")                  # SnapshotData -- used by everything below

    from streamtools.viz.plotting import energy, orbits, angularmomentum, animation, histograms
    energy.plot_energyvstime(data, snapshots, i_start=0, i_end=50, E_threshold=0)

See README.md for the full architecture, module reference, and usage
examples.
"""
__version__ = "0.1.0"
