"""Analysis functions for BIFROST simulations.
    
This module includes functions for data reading, manipulation, plotting and analysis.

Functions:
- primary_string: Calculate primary string parameters.
- e_cos: Cosine function for energy fitting.
- fit_energy: Fit energy distribution.
- weighted_cov: Calculate weighted covariance.
- flat_to_symmetric: Convert flat covariance to symmetric matrix.

- reduced_res_plot: Plot reduced resolution data.
- plot_cov_ellipse: Plot covariance ellipse.
- plot_cov_ellipse_3d: Plot 3D covariance ellipse.
- plot_cov_proj: Plot 2D projected covariance ellipse.
- plot_cov_cross: Plot 2D covariance cross-section.

- BIFROST_nD: Class for reading and manipulating BIFROST nD data.
    - read_data: Read data from folder.
    - RT_2_CA: Convert Ring and Tube to Channel and Analyzer.
    - pixel_split: Split pixels based on event charges.
    - E_f_k_f: Calculate final energy and wavevector.
    - psc_timing: Calculate PSC timing parameters.
    - E_i_k_i: Calculate initial energy and wavevector.
    - Q_hw: Calculate Q and hbar omega.
    - a3_rotation: Apply A3 rotation to Q vectors.
    - convert: Convert data to BIFROST format.

- BIFROST_Res: Class for BIFROST resolution analysis.
    - Res_Q_hw: Calculate resolution in Q and hbar omega.
    - pixel_resolution: Calculate pixel resolution and save results.

- Theory_Res: Class for theoretical resolution calculations.
    - resolution: Calculate theoretical resolution.
    - intensity: Calculate intensity based on theoretical resolution.

- compare_resolution: Compare BIFROST resolution with theoretical resolution.

Written by: Nicolai Amin
Date: 2025-06-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import scipy as sp
from tqdm import tqdm
import os
from mcstas_functions import read_data_files
from chopcal import bifrost
from time import time
from neutron_functions import *
import pkg_resources
from scipy.interpolate import interp1d


# Load constants from the package data
L2_const = np.load("consts/dL.npy")
phi_const = np.load("consts/dphi.npy")
E_T_const = np.load("consts/Es_calc.npy")
L1_const = np.loadtxt("consts/L1.dat", delimiter=",")
L1_E = L1_const[:,0]
L1_L = L1_const[:,1]
V_i_const = np.loadtxt("consts/V_i.dat")
V_ei = V_i_const[:,0]
V_Ii = V_i_const[:,1]
interp_func = interp1d(V_ei, V_Ii, kind='cubic')
V_f_const = np.load("consts/V_f.npz", allow_pickle=True)["Is"]











def primary_string(E_i, dt, folder):
    L0 = E2lambda_(E_i)
    BW = 1.808
    lambda_0 = L0+BW/2
    chop_pars = bifrost(0,lambda_min=lambda_0, shaping_time=dt)
    
    return chop_pars["ps1delay"],chop_pars["ps2delay"], dt
    

# Calculation of energy distribution on a tube
def e_cos(x, a, b, c):
    return a/np.cos(b*(x -c))

def fit_energy(x, y):
    # Initial guess for the parameters
    initial_guess = [min(y), 0.02, 50]
    params, covariance = sp.optimize.curve_fit(e_cos, x, y, p0=initial_guess)
    a, b, c = params
    return a, b, c

def weighted_cov(X,w):
    w /= np.sum(w)
    mean = np.average(X, axis=0, weights=w)
    X_centered = X - mean
    
    cov = (X_centered.T * w) @ X_centered / np.sum(w)
    return mean, cov

def flat_to_symmetric(flat,d=4):
    mat = np.zeros((d, d))
    idx = np.tril_indices(d)
    mat[idx] = flat
    mat[(idx[1], idx[0])] = flat
    return mat


# Plotting functions

def reduced_res_plot(data, file_name, folder):
    labels = [r"$\Delta Q_x$ [Å]",r"$\Delta Q_z$ [Å]",r"$\Delta \hbar \omega$ [meV]"]
    cols = ["dQ_x", "dQ_z", "dhbar_w"]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Resolution Plot")
    for i in range(3):
        for j in range(3):
            if i == j:
                sns.histplot(data, x=cols[i], bins=20, ax=axs[i, j], weights="Intensity")
                
                axs[i, j].set_xlabel(labels[i])
                axs[i, j].set_ylabel("Intensity")
            else:
                ax = sns.histplot(data, x=cols[i], y=cols[j], bins=20, ax=axs[i, j], cmap="viridis", cbar=True, weights="Intensity")
                cbar = ax.collections[0].colorbar
                axs[i, j].set_xlabel(labels[i])
                axs[i, j].set_ylabel(labels[j])
                cbar.set_label("Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, file_name))
    plt.close()

def plot_cov_ellipse(cov2d, mean2d, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * np.sqrt(vals)  # 2 sigma
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # 2 factor is since we want the ellipse to cover 95% of the data
    ellipse = Ellipse(xy=mean2d, width=2*width, height=2*height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def plot_cov_ellipse_3d(cov, center, ax=None, color='b', alpha=1, resolution=1000):
    # Eigen-decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors from largest to smallest
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Generate a unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)  # shape (res, res, 3)

    # Scale the unit sphere to the ellipsoid
    radii = 2 * np.sqrt(eigvals)  # n_std = 1 -> 68%, 2 -> 95%, etc.
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T + center

    # Plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
        color=color, alpha=alpha, linewidth=0
    )

    return ax

def plot_cov_proj(
    cov: np.ndarray, pos: np.ndarray, i: int, j: int, ax=None, **kwargs
) -> Ellipse:
    """
    Plot 2D projected covariance ellipse from full covariance matrix by selecting indices i, j.

    Args:
        cov (np.ndarray): Full covariance matrix.
        pos (np.ndarray): Mean position vector.
        i (int): Index for x-axis.
        j (int): Index for y-axis.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        **kwargs: Passed to Ellipse.

    Returns:
        Ellipse: The matplotlib ellipse patch.
    """
    cov2d = cov[np.ix_([i, j], [i, j])]
    pos2d = [pos[i], pos[j]]
    return plot_cov_ellipse(cov2d, pos2d, ax=ax, **kwargs)

def project_cov(cov, *axes):
    """
    Projects the covariance matrix onto the plane defined by p1 and p2.
    
    Args:
        cov (np.ndarray): Covariance matrix to be projected.
        p1 (np.ndarray): First axis.
        p2 (np.ndarray): Second axis.
        
    Returns:
        np.ndarray: Projected covariance matrix.
    """
    cov_proj = cov.copy()
    for axis in sorted(axes, reverse=True):
        cov_proj -= np.outer(cov_proj[:, axis], cov_proj[axis, :]) / cov_proj[axis, axis]
    return cov_proj


def plot_cov_cross(cov, pos, i, j, ax=None, **kwargs):
    """
    Plot 2D covariance cross-section from full covariance matrix by selecting indices i, j.

    Args:
        cov (np.ndarray): Full covariance matrix.
        pos (np.ndarray): Mean position vector.
        i (int): Index for x-axis.
        j (int): Index for y-axis.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        **kwargs: Passed to Ellipse.

    Returns:
        Ellipse: The matplotlib ellipse patch.
    """
    remaining_axes = [k for k in range(len(pos)) if k not in (i, j)]
    cov_proj = project_cov(cov, *remaining_axes)

    return plot_cov_cross(cov_proj, pos, i, j, ax=ax, **kwargs)

# Classes for BIFROST data

class BIFROST_nD:
    """
    Class for BIFROST data generated by several Monitor_nD components.
    
    This class is used to read, manipulate, and plot data from the BIFROST instrument.
    This is specifically only for data that will be gotten from the actual BIFROST instrument
    
    Attributes:
        folder (str): Path to the folder containing the data files.
        
    """
    
    def read_data(self, remove=True, reread=False):
        if os.path.exists(os.path.join(self.folder, "data.csv")) and not reread:
            self.data = pd.read_csv(os.path.join(self.folder, "data.csv"))
            f = np.load(os.path.join(self.folder, "parameters.npy"), allow_pickle=True)
            self.pars = f.item()
            return True
        # Get list of all files in the folder
        files = [f for f in os.listdir(self.folder) if "list" in f]
        if isinstance(files, bytes):
            files = [f.decode("utf-8") for f in files]
        data = pd.DataFrame()
        
        for n, file in enumerate(files):
            data_i, parameters = read_data_files(self.folder, file, D="N")
            # Check for empty data (lines that are only 0s)
            if n == 0:
                idxs = np.where(np.all(data_i == 0, axis=1))[0]
                self.pars = parameters
            data_i = np.delete(data_i, idxs, axis=0)
            if data_i.shape[0] == 0:
                return False

            # Get data names:
            data_names = parameters["title"].split()
            data_col = data.columns.tolist()
            for i in range(data_i.shape[1]):
                if data_names[i] not in data_col:
                    data[data_names[i]] = data_i[:, i]
        # Save the   data to a CSV file
        data.to_csv(os.path.join(self.folder, "data.csv"), index=False)
        np.save(os.path.join(self.folder, "parameters.npy"), parameters)
        if remove:
            for file in files:
                os.remove(os.path.join(self.folder, file))
                
        self.data = data
        return True

    
    def RT_2_CA(self):
        analyzer = self.data["TUBE"] // 3 + 1
        self.data["analyzer"] = analyzer.astype(int)
        channel = 3*(self.data["RING"] // 2) + self.data["TUBE"] % 3 + 1
        self.data["channel"] = channel.astype(int)
        
    def pixel_split(self):
        diff = self.data["event_charge_right"] - self.data["event_charge_left"]
        add = self.data["event_charge_left"] + self.data["event_charge_right"]
        L_x = diff / add
        
        ds = np.linspace(-1, 1, 301)
        
        L_x = [l if l< 1 else 0.999999 for l in L_x]
        bins = np.digitize(L_x, ds) - 1
        
        self.data["D_tube"] = (bins // 100) + 1
        self.data["d_i"] = bins % 100
        
        # Flip second tube
        self.data.loc[self.data["D_tube"] == 2, "d_i"] = 99 - self.data.loc[self.data["D_tube"] == 2, "d_i"]   
        
        return
    
    def E_f_k_f(self):        
        a4 = float(self.pars["a4"])
        
        C = self.data["channel"]
        A = self.data["analyzer"]
        T = self.data["D_tube"]
        d = 99 - self.data["d_i"]
        
        
        L = L2_const[C-1, A-1, T-1, d]
        
        phi = np.deg2rad(phi_const[C-1, A-1, T-1, d]+a4)
        dphi = np.rad2deg(phi) - a4 + 10*(5-C)
        
        
        E = E_T_const[A-1, T-1]/np.cos(np.deg2rad(dphi))
            
        
        self.data["E_f"] = E
        self.data["L_2"] = L
        self.data["t_2"] = L / E2v_(E)
        self.data["stt"] = phi
        
        self.data["k_f"] = E2k_(E)
        self.data["k_fx"] = self.data["k_f"] * np.sin(phi)
        self.data["k_fy"] = np.zeros_like(self.data["k_f"])
        self.data["k_fz"] = self.data["k_f"] * np.cos(phi)
        
    def psc_timing(self):
        
        psc_v = float(self.pars["ps1speed"])
        psc_offset_1 = float(self.pars["ps1phase"])
        psc_offset_2 = float(self.pars["ps2phase"])
        
        psc_off = (psc_offset_1 + psc_offset_2) / (720*psc_v)
        psc_opening = (psc_offset_1 - psc_offset_2+170) / (360*psc_v) 
                
        return psc_off, psc_opening
    
    def E_i_k_i(self):
        t2 = self.data["t_2"]
        t = self.data["event_time"]
        
        t1 = t - t2
        L_MS = 161.3466 # m
        L_PSC = 6.349590 # m
        
        psc_off, psc_opening = self.psc_timing()        
        
        self.data["dt"] = np.ones_like(t1) * psc_opening
        self.dt = psc_opening
        
        t1 = t1 - (psc_off+ psc_opening/2) # m
        self.data["t_1"] = t1
        L1 = (L_MS - L_PSC) # m
        L1 = self.L1_test
        
        v = L1 / t1
        E = v2E_(v)
        k = v2k_(v)
        
        self.data["E_i"] = E
        self.data["k_i"] = k
        self.data["k_ix"] = np.zeros_like(k)
        self.data["k_iy"] = np.zeros_like(k)
        self.data["k_iz"] = k
        
    def Q_hw(self):        
        self.data["Q_x"] = self.data["k_ix"] - self.data["k_fx"]
        self.data["Q_y"] = self.data["k_iy"] - self.data["k_fy"]
        self.data["Q_z"] = self.data["k_iz"] - self.data["k_fz"]
        self.data["hbar_omega"] = self.data["E_i"] - self.data["E_f"]   
        
    def a3_rotation(self):
        a3 = -np.deg2rad(float(self.pars["a3"]))
        qx = self.data["Q_x"]
        qz = self.data["Q_z"]
        self.data["Q_h"] = qz*np.cos(a3) - qx*np.sin(a3)
        self.data["Q_k"] = qz*np.sin(a3) + qx*np.cos(a3)
        self.data["a3"] = np.ones_like(self.data["Q_x"]) * a3
        
    def convert(self):
        self.RT_2_CA()
        self.pixel_split()
        self.E_f_k_f()
        self.E_i_k_i()
        self.Q_hw()
        a3_check = "a3" in self.pars.keys()
        if a3_check:
            self.a3_rotation() 
                    
    def __init__(self, folder, remove=True, reread=False,L1_test=155.575):
        self.any_data = False
        self.folder = folder
        self.L1_test = L1_test
        self.read_data(remove=remove, reread=reread)
        if self.data.empty:
            print(f"No data found in {folder}.")
            return None
        self.convert()
        self.any_data = True
        
    def add_data(self, folder):
        new = BIFROST_nD(folder)
        if new.data.empty:
            print(f"No data found in {folder}.")
            return
        if self.any_data is False:
            self.data = new.data
            self.pars = new.pars
            self.any_data = True
            return
        else:
            new_data = new.data
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.data.reset_index(drop=True, inplace=True)
            return
    
    def save(self, save_folder=None, save_file=None):
        if save_folder is None:
            save_folder = self.folder
        if save_file is None:
            save_file = "data.csv"
        self.data.to_csv(os.path.join(save_folder, save_file), index=False)
        np.save(os.path.join(save_folder, "parameters.npy"), self.pars)
        
class BIFROST_Res(BIFROST_nD):
    """
    Class for BIFROST data generated by several Monitor_nD components.
    
    This class is used to calculate the resolution of the BIFROST instrument.

    Args:
        BIFROST_nD (class): Class for BIFROST data generated by several Monitor_nD components.
    """
    
    def Res_Q_hw(self):

        # Calculate the actual Q, hw
        self.data["Qx"] = self.data["kix"] - self.data["kfx"]
        self.data["Qy"] = self.data["kiy"] - self.data["kfy"]
        self.data["Qz"] = self.data["kiz"] - self.data["kfz"]
        ki = np.sqrt(self.data["kix"]**2 + self.data["kiy"]**2 + self.data["kiz"]**2)
        kf = np.sqrt(self.data["kfx"]**2 + self.data["kfy"]**2 + self.data["kfz"]**2)
        self.data["Ef"] = k2E_(kf)
        self.data["Ei"] = k2E_(ki)
        self.data["hw"] = k2E_(ki) - k2E_(kf)
        
        self.data["dQx"] = self.data["Q_x"] - self.data["Qx"]
        self.data["dQy"] = self.data["Q_y"] - self.data["Qy"]
        self.data["dQz"] = self.data["Q_z"] - self.data["Qz"]
        self.data["dhw"] = self.data["hbar_omega"] - self.data["hw"]
        
    
    def __init__(self, folder, remove=True, reread=False, plot=False, L1_test=155.575):
        super().__init__(folder, remove=remove, reread=reread,L1_test=L1_test)
        self.Res_Q_hw()
        self.pixel_resolution(plot=plot)

    def pixel_resolution(self, save_folder=None, save_file=None, threshold=1e-8, plot=False, data_save=False):
        if save_folder == None:
            save_folder = self.folder+"/pixel_resolution"
        
        Res = pd.DataFrame(columns=["channel", "analyzer", "D_tube", "d_i", "I","Qx_m", "Qy_m", "Qz_m", "hw_m", 
                               "sigma_Qx_2", "sigma_Qx_Qy", "sigma_Qx_Qz", "sigma_Qx_hw", "sigma_Qy_2", 
                               "sigma_Qy_Qz", "sigma_Qy_hw", "sigma_Qz_2", "sigma_Qz_hw", "sigma_hw_2"])
        
        
        for c in range(1, 10):
            if len(self.data[self.data["channel"] == c]) == 0:
                continue
            for a in range(1, 6):
                if len(self.data[self.data["analyzer"] == a]) == 0:
                    continue
                for t in range(1, 4):
                    for d in range(100):
                        
                        data_i = self.data[(self.data["channel"] == c) & (self.data["analyzer"] == a) & (self.data["D_tube"] == t) & (self.data["d_i"] == d)]
                        intensity = data_i["Intensity"].sum()

                        if intensity < threshold:
                            continue
                        if plot:
                            if save_file == None:
                                save_file = f"pixel_{c}_{a}_{t}_{d}.png"
                            reduced_res_plot(data_i, save_folder=save_folder, save_file=save_file)
                            
                        mean, cov = weighted_cov(data_i[["dQx", "dQy", "dQz", "dhw"]].to_numpy(), data_i["Intensity"].to_numpy())
                        res_i = pd.DataFrame([{"channel": c, "analyzer": a, "D_tube": t, "d_i": d, "I": intensity,
                            "Qx_m": mean[0], "Qy_m": mean[1], "Qz_m": mean[2], "hw_m": mean[3], 
                            "sigma_Qx_2": cov[0, 0], "sigma_Qx_Qy": cov[0, 1], "sigma_Qy_2": cov[1, 1], 
                            "sigma_Qx_Qz": cov[0, 2], "sigma_Qy_Qz": cov[1, 2], "sigma_Qz_2": cov[2, 2],
                            "sigma_Qx_hw": cov[0, 3], "sigma_Qy_hw": cov[1, 3], "sigma_Qz_hw": cov[2, 3],
                            "sigma_hw_2": cov[3, 3]}])

                        Res = pd.concat([Res, res_i], ignore_index=True)

        self.Resolution = Res
        if data_save:
            Res.to_csv(os.path.join(self.folder, "Resolution.csv"), index=False)
            
            
    def resolution_ellipse(self, channel, analyzer, tube, pixel):
        Res_i = self.Resolution[(self.Resolution["channel"] == channel) & (self.Resolution["analyzer"] == analyzer) & (self.Resolution["D_tube"] == tube) & (self.Resolution["d_i"] == pixel)]
        
        if len(Res_i) == 0:
            return None
        
        mean = Res_i[["Qx_m", "Qz_m", "hw_m"]].to_numpy()[0]
        cov = Res_i[["sigma_Qx_2", "sigma_Qx_Qz", "sigma_Qz_2", "sigma_Qx_hw", "sigma_Qz_hw", "sigma_hw_2"]].to_numpy()[0]
        cov = flat_to_symmetric(cov, 3)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        pairs = [(0, 1), (0, 2), (1, 2)]
        labels = [r"$\Delta Q_x$ [Å]", r"$\Delta Q_z$ [Å]", r"$\Delta \hbar \omega$ [meV]"]

        for k in range(3):
            ax = axs[k]
            ax.cla()
            i, j = pairs[k]
            sub_cov = cov[np.ix_([i, j], [i, j])]
            sub_mean = [mean[i], mean[j]]
            plot_cov_cross(sub_cov, sub_mean, ax=ax, edgecolor='blue', facecolor='none')
            ax.scatter(*sub_mean, color='blue')
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            #ax.axis('equal')
            ax.grid(True)
        
        fig.suptitle(f"Channel {channel}, Analyzer {analyzer}, Tube {tube}, Pixel {pixel}")
        plt.savefig(os.path.join(self.folder, "resolution_viewer.png"))
        plt.close()        
      
      
def meV2J(E):
    return E*1.6020506e-22 # J
def J2meV(E):
    return E/1.6020506e-22 # meV
def Å2m(L):
    return L*1e-10 # m
def invÅ2invm(k):
    return k*1e10 # m^-1
  
class Theory_Res:
    """
    
    Class for BIFROST Theoretical Resolution function
    
    """
    
    def __init__(self, dt, hw, stt, jaws=None):
        self.dt = dt
        self.hw = hw
        self.stt = stt
        self.jaws = jaws
        
        
    
    def resolution(self, channel, analyzer, tube, pixel):


        phi = phi_const[channel-1, analyzer-1, tube-1, pixel]
        dphi = phi + 10*(5-channel)
        
        Ef = E_T_const[analyzer-1, tube-1]/np.cos(np.deg2rad(dphi))
        self.Ef = Ef
        
        L2 = L2_const[channel-1, analyzer-1, tube-1, pixel]
        
        
        delta_phi = [0.86,0.79,0.74,0.67,0.625]
        delta_theta_a = np.array([0.00295 , 0.00265 , 0.00245, 0.00225 , 0.0022])
        
        
        Ei = Ef + self.hw
        L1 = 162
        #L1 = interp1d(L1_E, L1_L, kind='linear')(Ei)

        self.Ei = Ei
        
        ki = E2k_(Ei)
        kf = E2k_(Ef)
        
        self.Qx = - np.sin(np.deg2rad(self.stt)) * kf
        self.Qy = np.zeros_like(self.Qx)
        self.Qz = ki - kf * np.cos(np.deg2rad(self.stt))
        
        
        g_i = (0.0644022/(Ei**(0.21809942)) -0.01536873)/2.355/1.265
        
        if self.jaws is not None:
            if g_i > np.deg2rad(self.jaws):
                g_i = np.deg2rad(self.jaws)
        
        
        t1 = L1 / E2v_(Ei)
        t2 = L2 / E2v_(Ef)
        
        d = 3.355
        
        theta_a = theta_Bragg(E2lambda_(Ef),d)
        theta_s = np.deg2rad(self.stt)
        
        delta_phi = float(np.deg2rad(delta_phi[analyzer-1]))
        delta_theta_a = float(delta_theta_a[analyzer-1])
        
        delta_deltas = [0.98732578, 1.06461164, 1.0822254,  1.05639024, 1.07277133]
        delta_deltas.reverse()
        delta_delta = delta_deltas[analyzer-1]
        
        
        J = np.array([[0,                                   0,      ki/L1,                                          2*Ei/L1], 
                      [0,                                   0,      ki*t2/(L2*t1),                                  2*Ei*t2/(L2*t1)], 
                      [0,                                   0,      -ki/t1,                                         -2*Ei/t1], 
                      [kf*np.sin(theta_s)/d,                0,      t2/t1*ki/d + kf*np.cos(theta_s)/d,              2*Ei*t2/(d*t1)+2*Ef/d],
                      [kf*np.cos(theta_s)/np.tan(theta_a),  0,      (t2/t1*ki+kf*np.cos(theta_s))/np.tan(theta_a),  (2*Ei/np.tan(theta_a)*t2/t1)*1+2*Ef/np.tan(theta_a)],
                      [ki,                                  0,      0,                                              0],
                      [0,                                   ki,     0,                                              0],
                      [-kf*np.cos(theta_s),                 0,      kf*np.sin(theta_s),                             0],
                      [0,                                   kf,     0,                                              0]
                      ])
        
        if self.dt > 0.0002:
            sigma_dt = np.sqrt((self.dt*0.79742508)**2+2e-5**2)
        elif np.round(self.dt,4) == 0.0002:
            sigma_dt = self.dt*0.87
        elif self.dt < 0.0002:
            sigma_dt = self.dt
                
        sigma = np.diag(np.array([
            0.01/2.355,                                             # L_1
            0.002/2.355,                                            # L_2
            sigma_dt/2.355,         # t_1
            0.0001/d,                                               # d     
            delta_theta_a/2.355,                                    # theta_a  
            (0.0644022/(Ei**(0.21809942)) -0.01536873)/2.355/1.265, # gamma_i
            2.56131010e-02-2.89318559e-04*Ei, # delta_i
            delta_phi/2.355,                                        # theta_s
            delta_delta/2.355                                         # delta_s                       
            ]))                                      
        
        cov = J.T @ sigma**2 @ J
        
        
        self.hw_L1 = (J[0,3]*sigma[0,0])**2
        self.hw_L2 = (J[1,3]*sigma[1,1])**2
        self.hw_t1 = (J[2,3]*sigma[2,2])**2
        self.hw_d = (J[3,3]*sigma[3,3])**2
        self.hw_theta_a = (J[4,3]*sigma[4,4])**2
        
        
        m_n = (2.286e-3/2)**2
        
        K_dt = self.dt*3.5
        K_theta = np.array([0.00095 , 0.00265 , 0.00245, 0.00225 , 0.0022])*5
        
        self.K_tof = ((m_n*L1**2/(t1**3))*K_dt)**2
        self.K_theta_a = ((1+m_n*L1**2*t2/(2*Ef*t1**3))*K_theta[analyzer-1])**2
        self.K_err = self.K_tof + self.K_theta_a
        
        
        
     
        self.cov = cov
        
        p = np.array([self.Qx, self.Qy, self.Qz, self.hw])
        self.mean = p
        
        
    def intensity(self,channel,analyzer):
        
        try:
            V_i = interp_func(self.Ei)
        except ValueError:
            V_i = 0
        
        
        V_f = V_f_const[analyzer-1,(channel-1)%3]
        
        M = np.linalg.inv(self.cov)

        det = np.linalg.det(M)
        
        V_e = 2*(np.pi*np.log(2))**2/(np.sqrt(det))
        
        R0 = np.log(2)**2*V_i*V_f/V_e/2
                
        self.R0 = R0
        
        
        
        

def compare_resolution(folder, stt, hw, channel, analyzer, tube, pixel, plot=True, save_file=None, reread=False):
    """
    Compare the resolution of the BIFROST instrument with the theoretical resolution.
    
    Args:
        folder (str): Path to the folder containing the data files.
        channel (int): Channel number.
        analyzer (int): Analyzer number.
        tube (int): Tube number.
        pixel (int): Pixel number.
        plot (bool): If True, plot the resolution comparison.
        
    Returns:
        None
    """
    
    # Load the data
    data = BIFROST_Res(folder, remove=False, reread=reread)
    print(data.Resolution["d_i"].unique())    
    # Get the resolution data
    Res = data.Resolution[(data.Resolution["channel"] == channel) & (data.Resolution["analyzer"] == analyzer) & (data.Resolution["D_tube"] == tube) & (data.Resolution["d_i"] == 100-pixel)]
    
    sim_cov = Res[["sigma_Qx_2","sigma_Qx_Qy","sigma_Qy_2", "sigma_Qx_Qz","sigma_Qy_Qz", "sigma_Qz_2", "sigma_Qx_hw", "sigma_Qy_hw","sigma_Qz_hw", "sigma_hw_2"]].to_numpy()
    sim_cov = flat_to_symmetric(sim_cov, 4)
    sim_mean = [Res["Qx_m"].values[0],Res["Qy_m"].values[0], Res["Qz_m"].values[0], Res["hw_m"].values[0]]
        
    if len(Res) == 0:
        print("No data found for the given parameters.")
        return
    
    # Get the theoretical resolution
    theory = Theory_Res(data.dt, hw, stt)
    theory.resolution(channel, analyzer, tube, pixel)
    
    theory_cov = theory.cov
    
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        proj_pairs = [(2,3), (1,3), (1,2), (0,3), (0,2), (0,1)]
        axs = axs.flatten()
        labels = [r"$\Delta Q_x$ [Å]", r"$\Delta Q_y$ [Å]", r"$\Delta Q_z$ [Å]", r"$\Delta \hbar \omega$ [meV]"]
        
        for k in range(4):
            ax = axs[k]
            ax.cla()
            i, j = pairs[k]
            n, m = proj_pairs[k]
            
            plot_cov_cross(sim_cov[np.ix_([i, j], [i, j])], sim_mean, ax=ax, edgecolor='blue', facecolor='none', label="Simulated")
            plot_cov_proj(sim_cov, sim_mean, n, m, ax=ax, edgecolor='blue', facecolor='none', label="Simulated Projection")
            plot_cov_cross(theory_cov[np.ix_([i, j], [i, j])], sim_mean, ax=ax, edgecolor='red', facecolor='none', label="Theoretical")
            plot_cov_proj(theory_cov, sim_mean, n, m, ax=ax, edgecolor='red', facecolor='none', label="Theoretical Projection")
            ax.scatter(sim_mean[i], sim_mean[j], color='blue')
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            
        plt.suptitle(f"Resolution hw={Res['hw_m'].values[0]:.2f} dt={data.dt:.5f} stt={stt} {channel}-{analyzer}-{tube}-{pixel}")
        axs[0].legend()
        plt.tight_layout()
        if save_file is None:
            save_file = f"resolution_{channel}_{analyzer}_{tube}_{pixel}.png"
        plt.savefig(os.path.join(folder, save_file))
        plt.close()
    
    # Bhattacharyya distance
    
    det_sim = np.linalg.det(sim_cov)
    det_theory = np.linalg.det(theory_cov)
    det_sum = np.linalg.det((sim_cov + theory_cov)/2)
    
    D_b = np.log(det_sum/np.sqrt(det_sim*det_theory))/2
    
    return D_b, sim_cov, sim_mean, theory_cov   