"""
Raw-snapshot ingestion with an on-disk pickle cache.

Workflow
--------
    snapshots = load_or_read_snapshots("g656")   # reads + caches, or loads cache
    save_snapshot_data(snapshots, filename="g656_data.npz")   # see io/precomputed_cache.py
    data = load_snapshot_data("g656_data.npz")

`load_or_read_snapshots` returns a list of `core.snapshot.Snapshot` objects,
so it can be passed directly to `io.precomputed_cache.save_snapshot_data`
(which calls `.rs3`, `.project_cluster_frame()`, etc. on each element) --
this used to return raw (rs3, vs3, rc3, vc3, time) tuples with no method of
that shape, which `save_snapshot_data` would have crashed on immediately.
Once you have a `SnapshotData` from `load_snapshot_data`, you shouldn't need
to touch `Snapshot` objects (or this module) again -- everything in
analysis/ and viz/ works off the precomputed cache.
"""
import os, pickle
from pathlib import Path
import numpy as np

from .snapshot_reader import readsnap
from ..core.snapshot import Snapshot

DEFAULT_DATA_DIR = Path(os.environ.get("STREAM_DATA_DIR", "./data"))


def load_or_read_snapshots(snap_dir, ifilend=1111, step=1, cache_file="snapshots.pkl", data_dir=DEFAULT_DATA_DIR):
    """
    Read raw snapshot files (or load a pickled cache of them) and wrap each
    one in a `Snapshot` object.

    Returns
    -------
    list of Snapshot
    """
    cache_file = Path(cache_file)
    if os.path.exists(cache_file):
        print(f"Loading cached snapshots from {cache_file} ...")
        with open(cache_file, "rb") as f:
            snapshots = pickle.load(f)
    else:
        print("Reading snapshots from disk...")
        snapshots = []
        for ifile in np.arange(0, ifilend, step):
            rs3, vs3, rc3, vc3, time = readsnap(snap_dir, ifile, data_dir=data_dir)
            snapshots.append(Snapshot(rs3, vs3, rc3, vc3, time))
        # Save for next time
        with open(cache_file, "wb") as f:
            pickle.dump(snapshots, f)
        print(f"Saved {len(snapshots)} snapshots to {cache_file}")
    return snapshots



