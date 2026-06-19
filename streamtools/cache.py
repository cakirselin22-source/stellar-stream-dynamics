import os, pickle
from pathlib import Path
import numpy as np

from . import dynamics
from .snapshot_reader import readsnap
DEFAULT_DATA_DIR = Path(os.environ.get("STREAM_DATA_DIR", "./data"))

def load_or_read_snapshots(snap_dir, ifilend=1111, step=1, cache_file="snapshots.pkl",  data_dir=DEFAULT_DATA_DIR): 
    cache_file = Path(cache_file)
    if os.path.exists(cache_file):
        print(f"Loading cached snapshots from {cache_file} ...")
        with open(cache_file, "rb") as f:
            snapshots = pickle.load(f)
    else:
        print("Reading snapshots from disk...")
        snapshots = []
        for ifile in np.arange(0, ifilend, step):
            rs3, vs3, rc3, vc3, time = do_read1t.readsnap(snap_dir, ifile)
            snapshots.append((rs3, vs3, rc3, vc3, time))
        # Save for next time
        with open(cache_file, "wb") as f:
            pickle.dump(snapshots, f)
        print(f"Saved {len(snapshots)} snapshots to {cache_file}")
    return snapshots



