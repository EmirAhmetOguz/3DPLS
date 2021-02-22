# 3DPLS

The 3-Dimensional Probabilistic Landslide Susceptibility (3DPLS) Model:

The 3-Dimensional Probabilistic Landslide Susceptibility Model (3DPLS) is a Python code developed for landslide susceptibility assessment. The 3DPLS model evaluates the landslide susceptibility on a local to a regional scale (i.e. single slope to 10 km2) and allows for the effects of variability of the model parameter on slope stability to be accounted for. 
The 3DPLS model couples the hydrological and slope stability models. The hydrological model calculates the transient pore pressure changes due to rainfall infiltration using Iverson’s linearized solution of the Richards equation assuming tension saturation. The slope stability model calculates the factor of safety (FS) by utilizing the extension of Bishop’s simplified method of slope stability analysis to three dimensions. The 3DPLS model requires topographic data (e.g., DEM, slope, aspect, groundwater depth, depth to bedrock, geological zones), hydrological parameters (e.g., steady background infiltration rate, permeability coefficient, diffusivity), geotechnical parameters (e.g., soil unit weight, cohesion, friction angle), and rainfall data.
The model has a grid containing hundreds or thousands of cells depending on the problem size and refinement. The smallest unit of the grid is called a grid cell having its own model parameters. The developed model calculates the FS of an ellipsoidal sliding surface consisting of grid cells over a discretized problem domain while equivalent cell-based models, as the name states, perform the calculations per each cell individually. The model generates a large number of ellipsoidal sliding surfaces centered at each grid cell over the terrain and calculates the FS of all sliding surfaces. After the calculation, each cell is involved in several ellipsoidal sliding surfaces with different FS values. Among all FS values, the minimum FS representing the critical ellipsoidal sliding surface is assigned to each cell. Each simulation results in a FS map over the terrain. After a number simulation, the 3DPLS model provides the FS map of each simulation, the mean FS, µ(FS), and the probability of failure, Pf, of each cell.

The current version of the 3DPLS model can consider only 1 soil type over the study area. Besides, all the cells should have data (NoData value is not considered). The changes can be made in the main code, but the function code should be kept as it is. 

For your questions and cooperation, 

Emir Ahmet OGUZ

Norwegian University of Science and Technology

Department of Civil and Environmental Engineering

N-7491 Trondheim, Norway

e-mail: emir.a.oguz@ntnu.com / emirahmetoguz@gmail.com
