import numpy as np
import matplotlib.pyplot as plt
from ..analysis import cluster_properties.py


def plot_cluster_evolution(snapshots, r_cut=0.1, outfile="cluster_pdf"):
    plt.rcParams["figure.figsize"] = (5, 5)
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    for snap in snapshots:
        r_half, sigma, mass = analysis.cluster_properties(snap, r_cut)
        ax[0].scatter(snap.time, r_half * 1000, s=1)
        ax[1].scatter(snap.time, sigma, s=1)
        ax[2].scatter(snap.time, mass, s=1)

    ax[0].set_ylabel(r"$r_{half}$ [pc]")
    ax[1].set_ylabel(r"$\sigma_{3D}$ [km s$^{-1}$]")
    ax[2].set_ylabel(r"$M(<20\,\mathrm{pc})$")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Time [Gyr]")
    
    ax[0].set_ylim(0,15)
    
    ax[1].set_ylim(0,4.5)
   
  

    plt.savefig(outfile)
    
def plot_cluster_orbit(snapshots, outfile="cluster_orbit"):
    R = []
    times = []
    x = []
    y = []
    for snap in snapshots:
        r = np.linalg.norm(snap.rc3)
        R.append(r)
        times.append(snap.time)
        x.append(snap.rc3[0])
        y.append(snap.rc3[1])
     
    R_peri = np.min(R)
    R_apo  = np.max(R)
    e = (R_apo - R_peri) / (R_apo + R_peri)
    print("Eccentricity =", e)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Galactocentric Radius")
    ax[0].plot(times, R)
    
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].plot(x,y)
    
    fig.savefig(output)
    return fig

def plot_cluster_orbit_properties(snapshots, output="orbit_properties"):
    R = []
    times =[]
    Lmag = []
    Lz = []
    for snap in snapshots:
        R.append(np.linalg.norm(snap.rc3))
        times.append(snap.time)
        L = np.cross(snap.rc3, snap.vc3)
        Lmag.append(np.linalg.norm(L))
        Lz.append(L[2])
    
    R_peri = np.min(R)
    R_apo  = np.max(R)
   
    e = (R_apo - R_peri) / (R_apo + R_peri)
    print("Eccentricity =", e)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))

    ax[0].set_xlabel("Time [Gyr]")
    ax[0].set_ylabel("Galactocentric Radius")
    ax[0].plot(times, R)
    
    ax[1].set_xlabel("Time [Gyr]")
    ax[1].set_ylabel("L_z")
    ax[1].plot(times, Lz)
    
    ax[2].set_xlabel("Time [Gyr]")
    ax[2].set_ylabel("L_mag")
    ax[2].plot(times, Lmag)
    
    fig.savefig(output)
 
    
    return fig 
  
    
