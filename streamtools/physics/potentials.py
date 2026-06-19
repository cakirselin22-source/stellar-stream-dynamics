import numpy as np


Grav = 43018.7   # (km/s)^2 kpc / M_sun
NFW_M  = 92.2541 
NFW_Rs = 20
NFW_c = 13.839
NFW_Tb = 0.97
NFW_Tc = 0.83
MN_M = 6.8
MN_a = 3.0
MN_b = 0.28
MWbulge_M = 0.5
MWbulge_a = 0.25

# -------------------------------
# Galactic potential helpers
# -------------------------------
def plummer_potential(rs, Grav=43018.7, MWbulge_M=0.5, MWbulge_a=0.25):
    r2 = np.sum(rs**2, axis=1)
    return -Grav * MWbulge_M / np.sqrt(r2 + MWbulge_a**2)

def nfw_potential(rs, time, Grav=43018.7, NFW_M=92.2541, NFW_Rs=20,
                  NFW_c=13.839, NFW_Tb=0.97, NFW_Tc=0.83, MHaloT=0.3, MHaloN=1.037):
    ra = np.sqrt(rs[:,0]**2 + (rs[:,1]/NFW_Tb)**2 + (rs[:,2]/NFW_Tc)**2)
    gc = 1.0 / (np.log(1 + NFW_c) - NFW_c / (1 + NFW_c))
    pot = -Grav * gc * NFW_M * np.log(1.0 + ra/NFW_Rs) / ra
    pot *= MHaloN * 2/np.pi * np.arctan((MHaloT*time)**2)
    return pot

def miyamoto_nagai_potential(rs, Grav=43018.7, MN_M=6.8, MN_a=3.0, MN_b=0.28):
    Rc2 = rs[:,0]**2 + rs[:,1]**2
    return -Grav * MN_M / np.sqrt(Rc2 + (np.sqrt(rs[:,2]**2 + MN_b**2) + MN_a)**2)

def cluster_potential(rs, Grav=43018.7, mstar=1e-10):
    dr = np.linalg.norm(rs, axis=1)
    cluster_mask = dr < 0.02
    r_half = np.median(dr[cluster_mask]) if np.any(cluster_mask) else np.median(dr)
    N_inside_half = np.sum(dr <= r_half)
    M_tot = 2 * N_inside_half * mstar
    a = np.sqrt(2**(2/3) - 1) * r_half
    return -Grav * M_tot / np.sqrt(dr**2 + a**2)

def calculate_energy(rs3, vs3, rc3, vc3, time,frame):
    """
    Compute total energy of stars in a galatic frame.
    """
    # Make arrays safe
    rs3, vs3 = np.asarray(rs3), np.asarray(vs3)
    rc3, vc3 = np.asarray(rc3), np.asarray(vc3)

    # Relative coordinates and velocities in cluster frame
    rs_rel = rs3  - rc3
    vs_rel = vs3  - vc3

    # --- Kinetic energy ---
    KE = 0.5 * np.sum(vs_rel**2, axis=1)

    # --- Galactic potentials ---
    P_plum = plummer_potential(rs3)
    P_NFW = nfw_potential(rs3, time)
    P_MN = miyamoto_nagai_potential(rs3)

    # --- Cluster self-potential ---
    P_cluster = cluster_potential(rs_rel)

    # --- Total energy per star ---
    E_stars =  P_cluster + KE  + P_plum + P_NFW +  P_MN
    
     # --- Kinetic energy ---
    c_KE = 0.5 * np.sum(vc3**2)

    # --- Galactic potentials ---
    c_P_plum = plummer_potential(rc3[None, :])[0]
    c_P_NFW  = nfw_potential(rc3[None, :], time)[0]
    c_P_MN   = miyamoto_nagai_potential(rc3[None, :])[0]
   
    
    #E_c = c_P_plum + c_P_NFW + c_P_MN + c_KE 

    E = E_stars 
    
    if frame == "binding":
        E = P_cluster + KE 
        return E
    else:
         return E_stars
