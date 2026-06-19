import numpy as np

def cluster_properties(snap, r_cut):
    """
    Compute basic cluster properties using a Snapshot object.

    Parameters
    ----------
    snap : Snapshot
        Snapshot object containing positions, velocities, cluster center.
    r_cut : float
        Radial cutoff for defining the cluster [kpc]

    Returns
    -------
    r_half : float
        Half-mass radius of stars within r_cut
    sigma : float
        3D velocity dispersion of stars within r_cut
    mass : int
        Number of stars within r_cut
    """
    # Cluster-frame positions and velocities
    rs, vs = snap.cluster_frame()
    dr = np.linalg.norm(rs, axis=1)
    dv = np.linalg.norm(vs, axis=1)

    cluster_mask = dr < r_cut
    r_half = np.median(dr[cluster_mask])
    sigma = np.sqrt(np.mean(dv[cluster_mask] ** 2))
    mass = np.sum(cluster_mask)

    return r_half, sigma, mass
