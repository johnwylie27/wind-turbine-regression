# wind-turbine-regression
written by John Wylie (6616262436, wyliej@rpi.edu)

# Overview:
A project for MANE 6962 at RPI to use ML methods to predict wind turbine loads based on a number of features from real data.

# Description:
This project uses data from [zenodo.org](https://zenodo.org/record/3482801). The authors presented the data in a report to IRPWind detailing a set of ex situ wind tunnel and in situ wind turbine experiments to investigate the effect of surface roughness on wind turbine blade performance. The researchers hail from organizations such as TU Delft, a leading technical institute in the Netherlands, and DTU Wind Energy, a public research institution.

The wind tunnel data, the focus of this project, includes aerodynamic measurements from pitot static probe rakes aft of the airfoil, static presssure ports around the airfoil model, and microphones placed on the model surface. The main data that will be employed involves the mean aerodynamic characteristics integrated from the rakes (i.e., C_l, C_d, and C_m) and the surface pressures from the static pressure ports on the airfoil. Each set of tests includes features such as: 
* Run number
* Angle of attack, alpha
* Lift coefficient, C_l
* Drag coefficient, C_d
* Moment coefficient, C_m
* Normal force coefficient, C_n
* Tangential force coefficient, C_t
* Reynolds number, Re
* Mach number, M
* Dynamic pressure, Q
* Free stream velocity, V
* Flap deflection
* Pressure coefficient (at each static pressure port - 40 on the upper suction side and 23 on the lower pressure side.

Additional information about the surface roughness is included in the file names and described in the test matrix. New features can include:
* Average roughness grain size
* Upper leading edge length covered
* Lower leading edge length covered

The majority of preprocessing will include extracting the information from the text files and loading it into python for implementation. The features will be scaled and normalized as needed. Negative values will be changed to positive to avoid issues with activation functions such as ReLU. Data regularization may also be necessary.

The remainder of the project after data pre-processing will include dimensionality reduction and regression. Since the data can contain as many as the 78 features named above, dimensionality reduction is necessary and critical for proper regression to be performed.
