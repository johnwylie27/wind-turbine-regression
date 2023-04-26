# wind-turbine-regression
# Wind Turbine Airfoil Aerodynamic Force Prediction via Regression of Wind Tunnel Data
written by John Wylie (6616262436, wyliej@rpi.edu)

# Overview:
A project for MANE 6962 at RPI to use ML methods to predict wind turbine airfoil forces and moments based on a number of features from real data.

# Description:
This project uses data from [zenodo.org](https://zenodo.org/record/3482801). The authors presented the data in a report to IRPWind detailing a set of ex situ wind tunnel and in situ wind turbine experiments to investigate the effect of surface roughness on wind turbine blade performance. The researchers hail from organizations such as TU Delft, a leading technical institute in the Netherlands, and DTU Wind Energy, a public research institution.

The wind tunnel data, the focus of this project, includes aerodynamic measurements from pitot static probe rakes aft of the airfoil, static presssure ports around the airfoil model, and microphones placed on the model surface. The main data employed involves the mean aerodynamic characteristics integrated from the rakes (i.e., C_l, C_d, and C_m) and the surface pressures from the static pressure ports on the airfoil. Each set of tests includes features such as: 
* Run number
* Angle of attack, alpha
* Lift coefficient, C_l
* Drag coefficient, C_d
* Moment coefficient, C_m
* Reynolds number, Re
* Mach number, M
* Dynamic pressure, Q
* Free stream velocity, V
* Pressure coefficient (at each static pressure port - 40 on the upper suction side and 23 on the lower pressure side.

Additional information about the surface roughness is included in the file names and described in the test matrix. New features can include:
* Average roughness grain size
* Upper leading edge length covered
* Lower leading edge length covered

The majority of preprocessing includes extracting the information from the text files and loading it into python for implementation (readData.py). The features are scaled and normalized as needed. Negative values are changed to positive to avoid issues with activation functions such as ReLu. 

The remainder of the project after data pre-processing includes regression and hyper parameter tuning.

# Getting Started
Ensure that you have Python version 3.10 with the packages listed in piplist.txt. Download the data from the [zenodo.org](https://zenodo.org/record/3482801) link.

# Scripts
* readData.py - reads the data from the raw data files and generates .csv files containing the data of interest for faster loading in the other scripts
* dataRegressor.py - runs a simple ANN for different subsets of the data (i.e., pressure data, test parameters, and both)
* linearRegressor.py - brief code that uses a linear regression implementation from SciKit-Learn to compare to the ANN (unfinished)
* tuneHyperParam.py - shows the hyper parameter tuning and investigation to refine the final ANN architecture and attributes in comparison.py
* comparison.py - Final comparison of the ANNs for the pressure data and the test parameters
