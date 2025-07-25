# BIFROST_Analysis

Source code for masters thesis: **Modelling the 4D Resolution function for the Neutron Spectrometer BIFROST**

This repository contains the source code for my master's thesis, which focuses on modeling the 4D resolution function for the BIFROST neutron spectrometer. The code is written in Python and utilizes various libraries for data analysis and visualization.

## Requirements

To run the code, you will need the following Python packages:

- numpy
- scipy
- matplotlib
- pandas
- seaborn
- tqdm
- chopcal

## Files

- `neutron_functions.py`: Contains function for converting neutron characteristics (e.g. wavelength to energy, wavevector, momentum, etc.).
- `mcstas_functions.py`: Contains functions for reading and processing data from McStas simulations.
- `analysis_functions.py`: Contains functions for calculating the 4D resolution function based on the BIFROST spectrometer geometry and neutron characteristics, and validating the model against simulation data.
