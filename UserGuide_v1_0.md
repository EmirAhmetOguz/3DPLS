# The 3-Dimensional Probabilistic Landslide Susceptibility (3DPLS) Model
**3DPLS version 1.0 (August 2022)**

The 3-Dimensional Probabilistic Landslide Susceptibility (3DPLS) model is a Python code developed for landslide susceptibility assessment (see Oguz et al. 2022). The 3DPLS model evaluates landslide susceptibility on a local to regional scale (i.e., single slope to 10 km²) and accounts for the effects of variability in model parameters on slope stability.

The 3DPLS model couples hydrological and slope stability models:
- **Hydrological Model**: Calculates transient pore pressure changes due to rainfall infiltration using Iverson’s linearized solution of the Richards equation, assuming tension saturation.
- **Slope Stability Model**: Calculates the factor of safety (FS) by extending Bishop’s simplified method of slope stability analysis to three dimensions.

### Model Requirements
The 3DPLS model requires the following input data:
- **Topographic Data**: DEM, slope, aspect, groundwater depth, depth to bedrock, geological zones.
- **Hydrological Parameters**: Steady background infiltration rate, permeability coefficient, diffusivity.
- **Geotechnical Parameters**: Soil unit weight, cohesion, friction angle.
- **Rainfall Data**: For modeling transient conditions.

### Model Functionality
The model grid comprises hundreds or thousands of cells depending on the problem size and refinement. The smallest unit is a *grid cell*, each with its own model parameters. The model calculates the FS of an ellipsoidal sliding surface consisting of grid cells across a discretized problem domain.

Key functionality includes:
- Generating a large number of ellipsoidal sliding surfaces centered at each grid cell and calculating the FS for each surface.
- Assigning the minimum FS (representing the critical ellipsoidal sliding surface) to each cell.
- Producing an FS map over the terrain for each simulation and, after multiple simulations, generating:
  - FS map for each simulation
  - Mean FS (µ(FS)) map
  - Probability of failure (Pf) map

### Current Version Capabilities
The 3DPLS model (v1.0) allows for multiple soil types and spatial variability (heterogeneity) of geotechnical and hydrological model parameters across the study area. However, it requires complete data for each cell, as cells with no data are not supported. The main code can be modified based on the problem requirements, but the function code should remain unchanged.

---

### For Questions and Cooperation
**Emir Ahmet OGUZ**

Email: [emirahmetoguz@gmail.com](mailto:emirahmetoguz@gmail.com)

---

### Reference
1. Oguz, E.A., Depina, I. & Thakur, V. *Effects of soil heterogeneity on susceptibility of shallow landslides*. Landslides, 19, 67–83 (2022). [https://doi.org/10.1007/s10346-021-01738-x](https://doi.org/10.1007/s10346-021-01738-x)

---

# Input Parameters for the Example Problems

This guide outlines the input parameters for three validation problems designed to validate the 3DPLS slope stability model. In addition, two example problems (a simplified case problem and a case study problem as described in Oguz et al., 2022) are available in the `InputData` folder.

## Overview of Validation and Example Problems

- **Validation Problems**: Three validation problems are provided to test the 3DPLS model.
- **Example Problems**: Two additional problems (simplified case and case study) demonstrate the model’s application. See Oguz et al. (2022) for full details.

### Automatic Modifications

For validation and simplified case problems, a few modifications are automatically applied within the code. Setting the `ProblemName` parameter to any of the following values will trigger these modifications:

- `{'Pr1', 'Pr2', 'Pr3S1Dry', 'Pr3S2Dry', 'Pr3S2Wet', 'SimpCase'}`

Each value corresponds to a specific configuration, and the modifications for each example problem are described in detail below.

### Required Changes in Main Code

Each example problem requires specific changes in the main code. After making the changes outlined below, you can run the code to reproduce the results published in Oguz et al. (2022). No modifications to the function code are necessary, as parameter definitions are already set within the main code.

## Detailed Problem Information

The full details and explanations for each problem are provided in Oguz et al. (2022)².

> **Note**: Line numbers referenced in this document apply to the main code provided for validation problem 1.

---

# Validation Problem – 1

For Validation Problem – 1, adjust the following parameters in the main code:

| Line #       | Parameter               | Value                                                      |
|--------------|-------------------------|------------------------------------------------------------|
| 44           | `GIS_Data_Directory`    | `Main_Directory + '\\InputData\\Validation\\Problem1'`     |
| 127          | `riInp`                 | `np.array(([0.],[86400]))`                                 |
| 137          | `MCnumber`              | `1`                                                        |
| 140          | `AnalysisType`          | `'Drained'`                                                |
| 141          | `FSCalType`             | `'Bishop3D'`                                               |
| 166          | `Mean_cInp`             | `np.array([0.1])`                                          |
| 171          | `Mean_phiInp`           | `np.array([0.0])`                                          |
| 176          | `Mean_uwsInp`           | `np.array([1.0])`                                          |
| 192-216      | `CoV_cInp`<br>`CoV_phiInp`<br>`CoV_uwsInp`<br>`CoV_kSatInp`<br>`CoV_diffusInp` | `np.array([0.0])` |
| 247-296      | `CorrLenX_cInp`<br>`CorrLenY_cInp`<br>`CorrLenX_phiInp`<br>`CorrLenY_phiInp`<br>`CorrLenX_uwsInp`<br>`CorrLenY_uwsInp`<br>`CorrLenX_kSatInp`<br>`CorrLenY_kSatInp`<br>`CorrLenX_diffusInp`<br>`CorrLenY_diffusInp` | `np.array(['inf'])` |
| 380          | `Ella`                  | `1`                                                        |
| 382          | `Ellb`                  | `1`                                                        |
| 384          | `Ellc`                  | `1`                                                        |
| 387          | `EllAlpha`              | `90`                                                       |
| 388          | `EllAlpha_Calc`         | `"No"`                                                     |
| 390          | `Ellz`                  | `0.5`                                                      |
| 398          | `InZone`                | `np.array(([int(nrows/2), int(nrows/2)], [int(ncols/2), int(ncols/2)]))` |
| 407          | `TimeToAnalyse`         | `np.array((0,))`                                           |
| 413          | `SubDisNum`             | `0`                                                        |
| 425          | `ProblemName`           | `'Pr1'`                                                    |
| 451          | `Multiprocessing_Option`| `Multiprocessing_Option_List[0]` or `"C-SP-SP"`            |

### Additional Notes
- **Dry Condition Simulation**: For `ProblemName` defined as `'Pr1'`, pore water pressures are set to zero to simulate dry conditions.
- **Input Files**: Input files with different cell size refinements for Validation Problem – 1 can be found in the `"InputData"` folder.

---

# Validation Problem – 2

For Validation Problem – 2, modify the following parameters in the main code:

| Line #       | Parameter               | Value                                                      |
|--------------|-------------------------|------------------------------------------------------------|
| 44           | `GIS_Data_Directory`    | `Main_Directory + '\\InputData\\Validation\\Problem2'`     |
| 127          | `riInp`                 | `np.array(([0.],[86400]))`                                 |
| 137          | `MCnumber`              | `1`                                                        |
| 140          | `AnalysisType`          | `'Drained'`                                                |
| 141          | `FSCalType`             | `'Bishop3D'`                                               |
| 166          | `Mean_cInp`             | `np.array([0.116])`                                        |
| 171          | `Mean_phiInp`           | `np.array([15.0])`                                         |
| 176          | `Mean_uwsInp`           | `np.array([1.0])`                                          |
| 192-216      | `CoV_cInp`<br>`CoV_phiInp`<br>`CoV_uwsInp`<br>`CoV_kSatInp`<br>`CoV_diffusInp` | `np.array([0.0])` |
| 247-296      | `CorrLenX_cInp`<br>`CorrLenY_cInp`<br>`CorrLenX_phiInp`<br>`CorrLenY_phiInp`<br>`CorrLenX_uwsInp`<br>`CorrLenY_uwsInp`<br>`CorrLenX_kSatInp`<br>`CorrLenY_kSatInp`<br>`CorrLenX_diffusInp`<br>`CorrLenY_diffusInp` | `np.array(['inf'])` |
| 380          | `Ella`                  | `2.02`                                                     |
| 382          | `Ellb`                  | `2.02`                                                     |
| 384          | `Ellc`                  | `2.02`                                                     |
| 387          | `EllAlpha`              | `90`                                                       |
| 388          | `EllAlpha_Calc`         | `"No"`                                                     |
| 390          | `Ellz`                  | `1.556`                                                    |
| 398          | `InZone`                | `np.array(([95,95], [149,149]))`                           |
| 407          | `TimeToAnalyse`         | `np.array((0,))`                                           |
| 413          | `SubDisNum`             | `0`                                                        |
| 425          | `ProblemName`           | `'Pr2'`                                                    |
| 451          | `Multiprocessing_Option`| `Multiprocessing_Option_List[0]` or `"C-SP-SP"`            |

### Additional Notes
- **Dry Condition Simulation**: For `ProblemName` defined as `'Pr2'`, pore water pressures are set to zero to simulate dry conditions.
- **Ellipsoidal Sliding Surface**: The center of the ellipsoidal sliding surface is introduced with an offset perpendicular to the ground surface. A small modification is required in the 3DPLS model to define this center for this problem.
- **Input Files**: Input files with different cell size refinements for Validation Problem – 2 can be found in the `"InputData"` folder.


---

# Validation Problem – 3

For Validation Problem – 3, modify the following parameters in the main code:

| Line #       | Parameter               | Value                                                      |
|--------------|-------------------------|------------------------------------------------------------|
| 44           | `GIS_Data_Directory`    | `Main_Directory + '\\InputData\\Validation\\Problem3\\Slide1'` or<br>`Main_Directory + '\\InputData\\Validation\\Problem3\\Slide2'` |
| 127          | `riInp`                 | `np.array(([0.],[86400]))`                                 |
| 137          | `MCnumber`              | `1`                                                        |
| 140          | `AnalysisType`          | `'Drained'`                                                |
| 141          | `FSCalType`             | `'Bishop3D'`                                               |
| 166          | `Mean_cInp`             | `np.array([28.7])`                                         |
| 171          | `Mean_phiInp`           | `np.array([20.0])`                                         |
| 176          | `Mean_uwsInp`           | `np.array([18.84])`                                        |
| 192-216      | `CoV_cInp`<br>`CoV_phiInp`<br>`CoV_uwsInp`<br>`CoV_kSatInp`<br>`CoV_diffusInp` | `np.array([0.0])` |
| 247-296      | `CorrLenX_cInp`<br>`CorrLenY_cInp`<br>`CorrLenX_phiInp`<br>`CorrLenY_phiInp`<br>`CorrLenX_uwsInp`<br>`CorrLenY_uwsInp`<br>`CorrLenX_kSatInp`<br>`CorrLenY_kSatInp`<br>`CorrLenX_diffusInp`<br>`CorrLenY_diffusInp` | `np.array(['inf'])` |
| 380          | `Ella`                  | `24.38`                                                    |
| 382          | `Ellb`                  | `24.38`                                                    |
| 384          | `Ellc`                  | `24.38`                                                    |
| 387          | `EllAlpha`              | `90`                                                       |
| 388          | `EllAlpha_Calc`         | `"No"`                                                     |
| 390          | `Ellz`                  | `16.35`                                                    |
| 398          | `InZone`                | `np.array(([int(25.5/cellsize), int(25.5/cellsize)], [int(ncols/2-1), int(ncols/2-1)]))` |
| 407          | `TimeToAnalyse`         | `np.array((0,))`                                           |
| 413          | `SubDisNum`             | `0`                                                        |
| 425          | `ProblemName`           | `'Pr3S1Dry'` or `'Pr3S2Dry'` or `'Pr3S2Wet'`              |
| 451          | `Multiprocessing_Option`| `Multiprocessing_Option_List[0]` or `"C-SP-SP"`            |

### Additional Notes

#### For Slide 1 – Dry:
- **Dry Condition Simulation**: Pore water pressures are set to zero when `ProblemName` is defined as `'Pr3S1Dry'`.
- **Ellipsoidal Surface Inclination**: The inclination of the ellipsoidal surface (`EllBeta`) is assigned directly as `arctan(0.5)` instead of calculating for a rectangular area.

#### For Slide 2 – Dry:
- **Dry Condition Simulation**: Pore water pressures are set to zero when `ProblemName` is defined as `'Pr3S2Dry'`.
- **Ellipsoidal Surface Inclination**: The inclination of the ellipsoidal surface (`EllBeta`) is assigned directly as `arctan(0.5)` instead of calculating for a rectangular area.
- **Parameter Modifications for Truncated Cells**: The parameters `{"ThetaAvr", "Theta", "AngleTangentXZE1", "A"}` are modified when cells are truncated. New cohesion and friction angle values are assigned to these truncated cells.

#### For Slide 2 – Wet:
- **Wet Condition Simulation**: Negative pore water pressures are set to zero, and suction is ignored when `ProblemName` is defined as `'Pr3S2Wet'`.
- **Ellipsoidal Surface Inclination**: The inclination of the ellipsoidal surface (`EllBeta`) is assigned directly as `arctan(0.5)` instead of calculating for a rectangular area.
- **Parameter Modifications for Truncated Cells**: The parameters `{"ThetaAvr", "Theta", "AngleTangentXZE1", "A"}` are modified when cells are truncated. New cohesion and friction angle values are assigned to these truncated cells.

- **Input Files**: Input files with different cell size refinements for Validation Problem – 3 can be found in the `"InputData"` folder.


---

# Simplified Case Problem

For the simplified case problem, modify the following parameters in the main code:

| Line #       | Parameter               | Value                                                      |
|--------------|-------------------------|------------------------------------------------------------|
| 44           | `GIS_Data_Directory`    | `Main_Directory + '\\InputData\\SimplifiedCase'`           |
| 127          | `riInp`                 | `np.array(([0.],[86400]))`                                 |
| 137          | `MCnumber`              | `1000`                                                     |
| 140          | `AnalysisType`          | `'Drained'` - `'Undrained'`                                |
| 141          | `FSCalType`             | `'Bishop3D'`                                               |
| 147          | `RanFieldMethod`        | `'CMD'`                                                     |
| 149          | `SaveMat`               | `'YES'`                                                     |
| 166          | `Mean_cInp`             | `np.array([6])`                                            |
| 171          | `Mean_phiInp`           | `np.array([40])`                                           |
| 176          | `Mean_uwsInp`           | `np.array([20])`                                           |
| 181          | `Mean_kSatInp`          | `np.array([1.00E-06])`                                     |
| 186          | `Mean_diffusInp`        | `np.array([5.00E-06])`                                     |
| 192          | `CoV_cInp`              | `np.array([0.1])` - `np.array([0.2])` - `np.array([0.3])`  |
| 197          | `CoV_phiInp`            | `np.array([0.05])` - `np.array([0.10])` - `np.array([0.15])` |
| 202-216      | `CoV_uwsInp`<br>`CoV_kSatInp`<br>`CoV_diffusInp` | `np.array([0.0])` |
| 218          | `Dist_cInp`             | `np.array(['LN'])`                                         |
| 223          | `Dist_phiInp`           | `np.array(['N'])`                                          |
| 228          | `Dist_uwsInp`           | `np.array(['N'])`                                          |
| 233          | `Dist_kSatInp`          | `np.array(['LN'])`                                         |
| 238          | `Dist_diffusInp`        | `np.array(['LN'])`                                         |
| 247-266      | `CorrLenX_cInp`<br>`CorrLenY_cInp`<br>`CorrLenX_phiInp`<br>`CorrLenY_phiInp` | `np.array([0])` - `np.array([10])` - `np.array([20])` - `np.array([50])` - `np.array([100])` - `np.array([200])` - `np.array([500])` - `np.array([1000])` |
| 267-296      | `CorrLenX_uwsInp`<br>`CorrLenY_uwsInp`<br>`CorrLenX_kSatInp`<br>`CorrLenY_kSatInp`<br>`CorrLenX_diffusInp`<br>`CorrLenY_diffusInp` | `np.array(['inf'])` |
| 308          | `Mean_SuInp`            | `40`                                                       |
| 313          | `Mean_uwsInp`           | `20`                                                       |
| 319          | `CoV_SuInp`             | `np.array([0.1])` - `np.array([0.2])` - `np.array([0.3])`  |
| 324          | `CoV_uwsInp`            | `np.array([0.0])`                                          |
| 330          | `Dist_SuInp`            | `np.array(['LN'])`                                         |
| 335          | `Dist_uwsInp`           | `np.array(['N'])`                                          |
| 344-353      | `CorrLenX_SuInp`<br>`CorrLenY_SuInp` | `np.array([0])` - `np.array([10])` - `np.array([20])` - `np.array([50])` - `np.array([100])` - `np.array([200])` - `np.array([500])` - `np.array([1000])` |
| 354-363      | `CorrLenX_uwsInp`<br>`CorrLenY_uwsInp` | `np.array(['inf'])` |
| 380          | `Ella`                  | `20`                                                       |
| 382          | `Ellb`                  | `20`                                                       |
| 384          | `Ellc`                  | `2`                                                        |
| 387          | `EllAlpha`              | `90`                                                       |
| 388          | `EllAlpha_Calc`         | `"No"`                                                     |
| 390          | `Ellz`                  | `0`                                                        |
| 398          | `InZone`                | `np.array(([10,29], [10,29]))`                             |
| 407          | `TimeToAnalyse`         | `np.array((0,))`                                           |
| 413          | `SubDisNum`             | `200`                                                      |
| 425          | `ProblemName`           | `'SimpCase'`                                               |
| 451          | `Multiprocessing_Option`| `Multiprocessing_Option_List[7]` or `"S-MP-MP"`            |
| 459          | `TOTAL_PROCESSES_IndMC` | `4` (depends on the capacity of the computer)              |
| 460          | `TOTAL_PROCESSES_EllGen`| `4` (depends on the capacity of the computer)              |

### Additional Notes
- **Sub-Discretization**: During the sub-discretization, the DEM data is recalculated instead of using `np.kron` when `ProblemName` is defined as `'SimpCase'`.

---

# Kvam Landslides Case Study

For the Kvam Landslides case study, the following parameters in the code should be modified:

| Line #       | Parameter               | Value                                                      |
|--------------|-------------------------|------------------------------------------------------------|
| 44           | `GIS_Data_Directory`    | `Main_Directory + '\\InputData\\KvamCaseStudy'`            |
| 127          | `riInp`                 | `np.array(([7.144e-7], [86400]))`                          |
| 137          | `MCnumber`              | `1000`                                                     |
| 140          | `AnalysisType`          | `'Drained'`                                                |
| 141          | `FSCalType`             | `'Bishop3D'`                                               |
| 147          | `RanFieldMethod`        | `'SCMD'`                                                   |
| 149          | `SaveMat`               | `'YES'`                                                     |
| 152          | `ZmaxVar`               | `'NO'`                                                      |
| 166          | `Mean_cInp`             | `np.array([4.0])`                                          |
| 171          | `Mean_phiInp`           | `np.array([32.0])`                                         |
| 176          | `Mean_uwsInp`           | `np.array([20.0])`                                         |
| 181          | `Mean_kSatInp`          | `np.array([1.00E-06])`                                     |
| 186          | `Mean_diffusInp`        | `np.array([5.00E-06])`                                     |
| 192          | `CoV_cInp`              | `np.array([0.3])`                                          |
| 197          | `CoV_phiInp`            | `np.array([0.2])`                                          |
| 202          | `CoV_uwsInp`            | `np.array([0.0])`                                          |
| 207          | `CoV_kSatInp`           | `np.array([0.0])`                                          |
| 212          | `CoV_diffusInp`         | `np.array([0.0])`                                          |
| 218          | `Dist_cInp`             | `np.array(['LN'])`                                         |
| 223          | `Dist_phiInp`           | `np.array(['N'])`                                          |
| 228          | `Dist_uwsInp`           | `np.array(['N'])`                                          |
| 233          | `Dist_kSatInp`          | `np.array(['LN'])`                                         |
| 238          | `Dist_diffusInp`        | `np.array(['LN'])`                                         |
| 247          | `CorrLenX_cInp`         | `np.array([50])`                                           |
| 252          | `CorrLenY_cInp`         | `np.array([50])`                                           |
| 257          | `CorrLenX_phiInp`       | `np.array([50])`                                           |
| 262          | `CorrLenY_phiInp`       | `np.array([50])`                                           |
| 267-296      | `CorrLenX_uwsInp`<br>`CorrLenY_uwsInp`<br>`CorrLenX_kSatInp`<br>`CorrLenY_kSatInp`<br>`CorrLenX_diffusInp`<br>`CorrLenY_diffusInp` | `np.array(['inf'])` |
| 380          | `Ella`                  | `100`                                                      |
| 382          | `Ellb`                  | `20`                                                       |
| 384          | `Ellc`                  | `2.5`                                                      |
| 387          | `EllAlpha`              | `0`                                                        |
| 388          | `EllAlpha_Calc`         | `"No"`                                                     |
| 390          | `Ellz`                  | `0`                                                        |
| 398          | `InZone`                | `np.array(([10,107], [20,77]))`                            |
| 407          | `TimeToAnalyse`         | `np.array((0,86400))`                                      |
| 413          | `SubDisNum`             | `100`                                                      |
| 425          | `ProblemName`           | `' '` (Empty space)                                        |
| 451          | `Multiprocessing_Option`| `Multiprocessing_Option_List[7]` or `"S-MP-MP"`            |
| 459          | `TOTAL_PROCESSES_IndMC` | `4` (depends on the capacity of the computer)              |
| 460          | `TOTAL_PROCESSES_EllGen`| `4` (depends on the capacity of the computer)              |

### Additional Notes
- **Line 121-124 Activation**: The lines `{121-124}` should be activated to run the case study for Kvam Landslides.


