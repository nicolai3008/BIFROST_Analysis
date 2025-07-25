"""
Module that includes functions for use in McStas data analysis.

Written by: Nicolai Amin
Version: 2.0
Date: 2025-05-06

"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
# Set the matplotlib parameters for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 14,
    "font.weight": "bold",
    "legend.fontsize": 12,
    "legend.edgecolor": [0.2, 0.2, 0.2],
    "axes.linewidth": 1.75,
    "axes.titlesize": 20,
    'text.latex.preamble': r'\boldmath',
    "figure.autolayout": True})
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import scipy.stats as stats


# Read the data files
def read_data_files(folder_path, file_name, D=2):
    """
    Read data files from a specified folder path and extract the data based on the given file name and dimension.

    Parameters:
        folder_path (str): The path to the folder containing the data files.
        file_name (str): The name of the file to be read.
        D (int, optional): The dimension of the data. Default is 2.

    Returns:
        tuple: A tuple containing the extracted data and parameters. The structure of the tuple depends on the dimension (D) parameter.
            - If D=2, the tuple contains:
                - intensity (numpy.ndarray): The intensity data.
                - error (numpy.ndarray): The error data.
                - count (numpy.ndarray): The count data.
                - parameters (dict): A dictionary containing the extracted parameters.
            - If D=1, the tuple contains:
                - lam (numpy.ndarray): The wavelength data.
                - intensity (numpy.ndarray): The intensity data.
                - error (numpy.ndarray): The error data.
                - count (numpy.ndarray): The count data.
                - parameters (dict): A dictionary containing the extracted parameters.
            - If D="N", the tuple contains:
                - data (numpy.ndarray): The data read from the file.
                - parameters (dict): A dictionary containing the extracted parameters.
    """
    parameters = {}
    for root, _, files in os.walk(folder_path):
        file_path = os.path.join(root, file_name)
        if  D==2:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if line.startswith("#"):
                        if line.startswith("# Param"):
                            key, value = line.split("=")[0].strip().split(" ")[-1], line.split("=")[1].strip()
                            parameters[key] = value
                        else:
                            key, value = line.split(":")[0].strip().split(" ")[-1], line.split(":")[1].strip()
                            parameters[key] = value
                lines = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
                data = np.array(lines)
                
                # Separate the data into three chunks
                intensity = data[0:data.shape[0]//3, :]
                error = data[data.shape[0]//3:2*(data.shape[0]//3), :]
                count = data[2*(data.shape[0]//3):, :]
                return intensity, error, count, parameters
        elif D == 1:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if line.startswith("#"):
                        if line.startswith("# Param"):
                            key, value = line.split("=")[0].strip().split(" ")[-1], line.split("=")[1].strip()
                            parameters[key] = value
                        else:
                            key, value = line.split(":")[0].strip().split(" ")[-1], line.split(":")[1].strip()
                            parameters[key] = value
                lines = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
                data = np.array(lines)
                lam = data[:, 0]
                intensity = data[:, 1]
                error = data[:, 2]
                count = data[:, 3]
            return lam, intensity, error, count, parameters
        elif D == "N":
            with open(file_path,'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if line.startswith("#"):
                        if line.startswith("# Param"):
                            key, value = line.split("=")[0].strip().split(" ")[-1], line.split("=")[1].strip()
                            parameters[key] = value
                        else:
                            key, value = line.split(":")[0].strip().split(" ")[-1], line.split(":")[1].strip()
                            parameters[key] = value
                lines = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
                data = np.array(lines)
            return data, parameters
        else:
            raise ValueError("Invalid dimension specified. D should be 1, 2, or 'N'.")

# Define the x and y meshgrid
def xy(parameters, intensity):
    """
    Generate x and y coordinates based on the given parameters and intensity shape.

    Parameters:
        parameters (dict): A dictionary containing the xylimits as a string in the format "x1 x2 y1 y2".
        intensity (ndarray): An array representing the intensity.

    Returns:
        x (ndarray): An array of x-coordinates.
        y (ndarray): An array of y-coordinates.
    """
    xylimits = parameters["xylimits"].split(" ")
    x = np.linspace(float(xylimits[0]), float(xylimits[1]), intensity.shape[1])
    y = np.linspace(float(xylimits[2]), float(xylimits[3]), intensity.shape[0])
    x, y = np.meshgrid(x, y)
    return x, y

# Plot the heatmap
def mcstas_heatmap(data, parameters, logarithm=False, save_folder=None, save_name=None):
    """
    Plot a heatmap of the given data.

    Parameters:
    - data: numpy.ndarray
        The 2D array of data to be plotted.
    - parameters: dict
        A dictionary containing the plot parameters.
        It should have the following keys:
        - 'xlabel': str
            The label for the x-axis.
        - 'ylabel': str
            The label for the y-axis.
        - 'zlabel': str
            The label for the colorbar.
        - 'title': str
            The title of the plot.
    - logarithm: bool, optional
        If True, the data will be plotted in logarithmic scale.
        Default is False.
    - with_fit: bool, optional
        If True, the plot will include a fitted Gaussian curve.
        Default is False.
    """
    if save_name is None and save_folder is not None:
        save_name = "heatmap"
    x, y = xy(parameters,data)
    if logarithm:
        data = np.log(data)
    # Using pcolor to plot the 2D function
    plt.figure()
    plt.pcolor(x, y, data, cmap='viridis')
    plt.xlabel(parameters['xlabel'])
    plt.ylabel(parameters['ylabel'])
    plt.title(parameters['title'])
    plt.colorbar(label=parameters['zlabel'])  # Add the colorbar label
    
    if save_folder:
        plt.savefig(os.path.join(save_folder, save_name+".png"))
    plt.show()
    plt.close()

# Calculate the Pearson's chi-square test
def pearsons_chi_square_test(data, fit_data):
    """
    Perform Pearson's chi-square test of independence.

    Parameters:
    - data (ndarray): Observed data.
    - fit_data (ndarray): Expected data.

    Returns:
    - chi2 (float): The chi-square test statistic.
    - p_value (float): The p-value associated with the test statistic.
    """
    observed = data.ravel()
    expected = fit_data.ravel()
    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return chi2, p_value

# Calculate the R squared value
def r_squared(data, fit_data):
    """
    Calculate the coefficient of determination (R-squared) for a given data set and its corresponding fit data.

    Parameters:
    data (array-like): The actual data.
    fit_data (array-like): The fitted data.

    Returns:
    float: The coefficient of determination (R-squared) value.
    """
    y_mean = np.mean(data)
    ss_total = np.sum((data - y_mean)**2)
    ss_residual = np.sum((data - fit_data)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def mccode_intensity(folder, monitor="l_monitor_0_I"):
    """
    Read and extract data from a mccode.dat file, for a scan of a paratmeter

    Parameters:
    folder (str): The path to the folder containing the mccode.dat file.
    monitor (str, optional): The name of the monitor to extract data from. Defaults to "l_monitor_0_I".

    Returns:
    tuple: A tuple containing two numpy arrays, x and y, representing the x and y coordinates of the monitor data.
    """
    file_path = os.path.join(folder, "mccode.dat")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines] 
        for i in lines:
            if i.startswith("# variables"):
                monitors = i.split(" ")
                j = monitors.index(monitor)-2
        lines = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
        data = np.array(lines)
    x = data[:, 0]
    y = data[:, j]
    return x, y

def wstd(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    try:
        average = np.average(values, weights=weights)
    except ZeroDivisionError:
        return np.nan
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance), average

def distribution_std(intensity, parameters):
    """
    Calculate the average width of the intensity distribution along the x and y axes.

    Parameters:
    intensity (ndarray): 2D array representing the intensity distribution.
    parameters (dict): Dictionary containing the parameters for calculating the width.

    Returns:
    x_std (float): Standard deviation of the intensity distribution along the x axis.
    y_std (float): Standard deviation of the intensity distribution along the y axis.
    """
    xylimits = parameters["xylimits"].split(" ")
    x = np.linspace(float(xylimits[0]), float(xylimits[1]), intensity.shape[1])
    y = np.linspace(float(xylimits[2]), float(xylimits[3]), intensity.shape[0])
    x_hist = np.sum(intensity, axis=0)
    y_hist = np.sum(intensity, axis=1)
    x_std, x_m = wstd(x, x_hist)
    y_std, y_m = wstd(y, y_hist)
    return [x_m, x_std], [y_m, y_std]

def gaussian(x, a, x0, sigma):
    """
    Gaussian function.

    Parameters:
    x (ndarray): The input data.
    a (float): Amplitude of the Gaussian.
    x0 (float): Mean of the Gaussian.
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    ndarray: The Gaussian function evaluated at x.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def gaussian_FWHM(x, y):
    """
    Fit a Gaussian function to the given data and calculate the Full Width at Half Maximum (FWHM).

    Parameters:
    x (ndarray): The x data.
    y (ndarray): The y data.

    Returns:
    tuple: A tuple containing the optimized parameters for the Gaussian fit.
    """
    # Initial guess for the parameters
    initial_guess = [np.max(y), x[np.argmax(y)], 1]
    
    # Perform the curve fitting
    popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
    
    # Calculate FWHM
    FWHM = 2.355* popt[2]
    
    return popt, FWHM

def signal_FWHM(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a signal.
    
    Parameters:
    x (ndarray): The x data.
    y (ndarray): The y data.
    
    Returns:
    float: The FWHM of the signal, assuming only one peak.
    """
    peak = np.argmax(y)
    
    FWHM = peak_widths(y, [peak], rel_height=0.5)[0][0]
    
    dx = x[1] - x[0]
    
    return FWHM * dx

def FWHM(x, y):
    """
    Calculate the FWHM of a distribution using 3 methods:
    1. Calculated Weighted Standard Deviation (WSTD)
    2. Gaussian Fit
    3. Peak Widths
    
    Parameters:
    x (ndarray): The x data.
    y (ndarray): The y data.
    
    Returns:
    tuple: A tuple containing the FWHM values calculated by each method.
    """
    
    wstd_FWHM = wstd(x,y)[0]*2.355
    gaussian_FWHM = gaussian_FWHM(x,y)[1]
    signal_FWHM = signal_FWHM(x,y)
    
    return wstd_FWHM, gaussian_FWHM, signal_FWHM

def powder_angle_fit(theta, U, V, W):
    """
    Calculate the Full Width at Half Maximum (FWHM) angle fit.

    Parameters:
    theta (float): The angle in degrees.
    U (float): Coefficient U.
    V (float): Coefficient V.
    W (float): Coefficient W.

    Returns:
    float: The FWHM angle fit.

    """
    return np.sqrt(W + V*(np.tan(np.deg2rad(theta/2))) + U*(np.tan(np.deg2rad(theta/2)))**2)

def powder_fitting(angle, FWHM):
    """
    Fits the FWHM (Full Width at Half Maximum) values to the given angles using curve fitting.

    Parameters:
    angle (array-like): The array of angles.
    FWHM (array-like): The array of FWHM values.

    Returns:
    tuple: A tuple containing the optimized parameters for the curve fit.
    """
    popt, pcov = curve_fit(powder_angle_fit, angle, FWHM, maxfev=10000, p0=[0.01, -0.1, 0.5], bounds=([-np.inf, -np.inf, 0], [np.inf, 0, np.inf]))
    return popt
