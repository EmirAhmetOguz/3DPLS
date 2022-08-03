"""
#####
## 3DPLS version 1.0
## Main code
#####

The 3-Dimensional Probabilistic Landslide Susceptibility (3DPLS) model is a Python code developed for landslide susceptibility assessment. 
The 3DPLS model evaluates the landslide susceptibility on a local to a regional scale (i.e. single slope to 10 km2) 
and allows for the effects of variability of model parameter on slope stability to be accounted for.
The 3DPLS model couples the hydrological and the slope stability models. The hydrological model calculates the
transient pore pressure changes due to rainfall infiltration using Iverson’s linearized solution of the Richards equation
(Iverson 2000) assuming tension saturation. The slope stability model calculates the factor of safety by utilizing the extension of
Bishop’s simplified method of slope stability analysis (Bishop 1955) to three dimensions, proposed in the study of Hungr (1987). 
The 3DPLS model requires topographic data (e.g., DEM, slope, aspect, groundwater depth, depth to bedrock, geological zones), 
hydrological parameters (e.g., steady background infiltration rate, permeability coefficient, diffusivity), 
geotechnical parameters (e.g., soil unit weight, cohesion, friction angle), and rainfall data.

Developed by: Emir Ahmet Oguz

The code was first developed in 2020 April-December for research purposes (3DPLS version 0.0). 
The code has been improved in 2022 through a project supported by NTNU's Innovation Grant (3DPLS version 1.0).

Paper: Oguz, E.A., Depina, I. & Thakur, V. Effects of soil heterogeneity on susceptibility of shallow landslides. Landslides 19, 67–83 (2022). https://doi.org/10.1007/s10346-021-01738-x

"""

## Libraries 
import os
import numpy as np  
import time
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process, Queue, Array,cpu_count  ## For Parallelization 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  ##To remove the warnings (due to FS calculation iterations)

## Directories
# os.path.dirname(os.path.realpath(__file__)) ## The folder is assigned as the current working directory first. 
# Main_Directory      = os.path.dirname(os.getcwd()) ## The main folder
Main_Directory      = r'C:\\3DPLS_v1.0' ## The main folder
Code_Directory      = Main_Directory + '\\Codes' ## The folder including the code
Maxrix_Directory    = Main_Directory + '\\Codes\\Matrix'
GIS_Data_Directory  = Main_Directory + '\\InputData\\Validation\\Problem1'  ## The folder including the input data
## Create directories if they do not exist. 
if (os.path.exists(Main_Directory + '\\Codes\\Matrix') == False):
    os.makedirs(Main_Directory + '\\Codes\\Matrix')
if (os.path.exists(Main_Directory+ '\\Results') == False):
    os.makedirs(Main_Directory+ '\\Results')
if (os.path.exists(Main_Directory+ '\\InputData') == False):
    os.makedirs(Main_Directory+ '\\InputData')
if (os.path.exists(Main_Directory+ '\\Results_3DPLS') == False):
    os.makedirs(Main_Directory+ '\\Results_3DPLS')
Results_Directory   = Main_Directory+ '\\Results_3DPLS' ## The folder to store results

os.chdir(Main_Directory) ## Change directory 

'''
#####################################################################################
## Part - 1 
## Get the data from GIS_Data_Directory (ASCII grid format files)
## Arrange the data if needed
#####################################################################################
'''

## Import the function to read the ASCII grid format files
os.chdir(Code_Directory) ## Change directory 
from Functions_3DPLS_v1_0 import ReadData 

## Dimensions of the problem domain 
## The user can define the dimensions or it can be read from the dem file. 
## Manual entry
# nrows, ncols         = 
# xllcorner, yllcorner = 
# cellsize             = 
# nel                  = nrows*ncols
## Entry from dem file (or any other file with ASCII format)
os.chdir(GIS_Data_Directory) ## Change directory 
with open('dem.asc') as f: ## read information from dem file
    Temp = f.readlines()[0:7]
    ncols = int(Temp[0].split()[1])             ## Number of columns
    nrows = int(Temp[1].split()[1])             ## Number of rows
    xllcorner = int(float(Temp[2].split()[1]))  ## Corner coodinate
    yllcorner = int(float(Temp[3].split()[1]))  ## Corner coodinate
    cellsize = (float(Temp[4].split()[1]))      ## Cell size
    NoData = int(float(Temp[5].split()[1]))     ## No Data value
    nel = nrows*ncols                           ## Number of cells
f.close()

## Find, read and store the input data with '.asc' extension
DataFiles = os.listdir(GIS_Data_Directory)
DataFiles = [i for i in DataFiles if i.endswith('.asc')] ## Name of the files with '.asc' extension in GIS_Data_Directory.
NumberData = ReadData (DataFiles)                        ## Read files 
NumberDataAndNames = []                                  ## Allocate. File names and data together
for i in range(np.size(DataFiles)):
    NumberDataAndNames.append([DataFiles[i]]+ [NumberData [i]])

## Input data (change the names if necessary)
for i in range(np.size(DataFiles)):
    if (DataFiles[i] == 'dem.asc'):
        DEMInput       = NumberDataAndNames[i][1]   ## Digital elevation map data
    elif (DataFiles[i] == 'dir.asc'):
        DirectionInput = NumberDataAndNames[i][1]   ## Direction of steepest slope (From TopoIndex or QGIS) (if needed)
    elif (DataFiles[i] == 'rizero.asc'):        
        rizeroInput    =  NumberDataAndNames[i][1]  ## Steady, background infiltration 
    elif (DataFiles[i] == 'slope.asc'):
        SlopeInput    =  NumberDataAndNames[i][1]   ## Slope angle 
    elif (DataFiles[i] == 'zones.asc'):
        ZoneInput    =  NumberDataAndNames[i][1].astype(int)   ## Zones, converted to int
    elif (DataFiles[i] == 'aspect.asc'):
        AspectInput    =  NumberDataAndNames[i][1]  ## Aspect (if needed)
    elif (DataFiles[i] == 'zmax.asc'):
        ZmaxInput    =  NumberDataAndNames[i][1]    ## Depth to bedrock (can be obtained later)
    elif (DataFiles[i] == 'source.asc'):
        SourceInput    =  NumberDataAndNames[i][1]  ## Initiation cells, source (if needed)
    elif (DataFiles[i] == 'depthwt.asc'):
        HwInput        =  NumberDataAndNames[i][1]  ## Depth to ground water table (can be obtained later)
  
## Import the function to arrange the data (Note: Currently the function is for the case study in Oguz et al. (2022): Kvam Lansdlides)
## Depending on the problem and needs, the function can be modified.
# os.chdir(Code_Directory)
# from Functions_3DPLS_v1_0 import DataArrange
# ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput,ZmaxInput,HwInput = \
#     DataArrange(ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput, NoData)

## Rainfall input data (m/sec). 
riInp =  np.array(([0.],[86400]))   ##np.array(([7.144e-7],[86400]))

'''
#####################################################################################
## Part - 2
## Defining the parameters 
#####################################################################################
'''

## Monte Carlo number (suggested: 1000)
MCnumber= 1

## Analysis type and soil parameters
AnalysisType = 'Drained'    ## 'Drained'-'Undrained' 
FSCalType    = 'Bishop3D'   ## 'Normal3D' - 'Bishop3D' - 'Janbu3D'

## Select the method for the random field generation
## The covariance matrix decomposition method, CMD (Fenton and Griffiths 2008) with the ellipsoidal autocorrelation function
## The stepwise covariance matrix decomposition method, SCMD (Li et al. 2019) with the separable autocorrelation function
## Suggestion: Select SCMD for the problems with more than 400 elements.
RanFieldMethod = 'SCMD'  ## 'CMD' - 'SCMD' 
## Save matrix to save time for the CMD  !!! check what is saved? lower triangle or all correlation
SaveMat = 'YES' ## 'YES' - 'NO'

## Variability of Zmax (Mean is calculated by a function based on slope.)
ZmaxVar = 'NO'
if (ZmaxVar=='YES'): 
    CoV_Zmax = 0.3
    MinZmax  = 0.4
else:
    CoV_Zmax = 0
    MinZmax  = 0
ZmaxArg = (ZmaxVar, CoV_Zmax, MinZmax)

## Define the parameters with their variability 
if (AnalysisType == 'Drained'): ## Soil properties for drained case
    
    ## Zones
    ## Mean, CoV and distribution type ("LN" or "N") of parameters for each zone. 
    Mean_cInp        = np.array([0.1   , ])  ## Mean cohesion
    Mean_phiInp      = np.array([0.0   , ])  ## Mean friction angle
    Mean_uwsInp      = np.array([1.0   , ])  ## Mean unit weight of soil
    Mean_kSatInp     = np.array([1.00E-06, ])  ## Mean saturated conductivity
    Mean_diffusInp   = np.array([5.00E-06, ])  ## Mean diffusivity

    CoV_cInp         = np.array([0.0 ,])  ## CoV of cohesion
    CoV_phiInp       = np.array([0.0 ,])  ## CoV of friction angle
    CoV_uwsInp       = np.array([0.0 ,])  ## CoV of unit weight of soil    
    CoV_kSatInp      = np.array([0.0 ,])  ## CoV of diffusivity
    CoV_diffusInp    = np.array([0.0 ,])  ## CoV of saturated conductivity
    
    Dist_cInp        = np.array(['LN', ])  ## Distribution type of cohesion
    Dist_phiInp      = np.array(['N' , ])  ## Distribution type of friction angle
    Dist_uwsInp      = np.array(['N' , ])  ## Distribution type of unit weight of soil    
    Dist_kSatInp     = np.array(['LN', ])  ## Distribution type of saturated conductivity
    Dist_diffusInp   = np.array(['LN', ])  ## Distribution type of saturated conductivity
    
    ## Correlation lengths in X and Y directions
    ## 'inf' will model the variables as homogeneous over space but employing random values using the distribution.
    ## Parameter will be modelled as homogeneous if correlation length is even 'inf' in one direction. 
    CorrLenX_cInp      = np.array(['inf', ]) 
    CorrLenY_cInp      = np.array(['inf', ]) 
    CorrLenX_phiInp    = np.array(['inf', ]) 
    CorrLenY_phiInp    = np.array(['inf', ])  
    CorrLenX_uwsInp    = np.array(['inf', ]) 
    CorrLenY_uwsInp    = np.array(['inf', ]) 
    CorrLenX_kSatInp   = np.array(['inf', ]) 
    CorrLenY_kSatInp   = np.array(['inf', ]) 
    CorrLenX_diffusInp = np.array(['inf', ]) 
    CorrLenY_diffusInp = np.array(['inf', ])     
    
    ## Allocate all parameter information
    Parameter_Means    = np.array([ [Mean_cInp]     , [Mean_phiInp]     , [Mean_uwsInp]     , [Mean_kSatInp]     , [Mean_diffusInp]     ])
    Parameter_CoVs     = np.array([ [CoV_cInp]      , [CoV_phiInp]      , [CoV_uwsInp]      , [CoV_kSatInp]      , [CoV_diffusInp]      ])
    Parameter_Dist     = np.array([ [Dist_cInp]     , [Dist_phiInp]     , [Dist_uwsInp]     , [Dist_kSatInp]     , [Dist_diffusInp]     ])
    Parameter_CorrLenX = np.array([ [CorrLenX_cInp] , [CorrLenX_phiInp] , [CorrLenX_uwsInp] , [CorrLenX_kSatInp] , [CorrLenX_diffusInp] ])
    Parameter_CorrLenY = np.array([ [CorrLenY_cInp] , [CorrLenY_phiInp] , [CorrLenY_uwsInp] , [CorrLenY_kSatInp] , [CorrLenY_diffusInp] ])
    
elif (AnalysisType == 'Undrained'): ## Soil properties for undrained case
        
    ## Mean, CoV and distribution type ("LN" or "N") of parameters for each zone. 
    Mean_SuInp       = np.array([40.0  , ])  ## Mean undrained shear strength
    Mean_uwsInp      = np.array([20   , ])  ## Mean unit weight of soil

    CoV_SuInp        = np.array([0.0  ,])   ## CoV of undrained shear strengt
    CoV_uwsInp       = np.array([0.0 ,])   ## CoV of unit weight of soil

    Dist_SuInp       = np.array(['LN', ])   ## CoV of undrained shear strengt
    Dist_uwsInp      = np.array(['N' , ])   ## CoV of unit weight of soil

    ## Correlation lengths in X and Y directions
    ## 'inf' will model the variables as homogeneous over space but employing random values using the distribution. 
    ## Parameter will be modelled as homogeneous if correlation length is even 'inf' in one direction.
    CorrLenX_SuInp      = np.array(['inf', ])
    CorrLenY_SuInp      = np.array(['inf', ])
    CorrLenX_uwsInp     = np.array(['inf', ]) 
    CorrLenY_uwsInp     = np.array(['inf', ]) 

    ## Allocate all parameter information
    Parameter_Means    = np.array([ [Mean_SuInp] , [Mean_uwsInp]  ])
    Parameter_CoVs     = np.array([ [CoV_SuInp]  , [CoV_uwsInp]  ])
    Parameter_Dist     = np.array([ [Dist_SuInp] , [Dist_uwsInp] , ])
    Parameter_CorrLenX = np.array([ [CorrLenX_SuInp] , [CorrLenX_uwsInp] ])
    Parameter_CorrLenY = np.array([ [CorrLenY_SuInp] , [CorrLenY_uwsInp] ])

## Ellipsoidal parameters
'''
## For the ellipsoidal parameters see Oguz et al. (2022)
## EllAlpha can be either assigned by the user or calculated as average for a circular area.
## An average values of slope in a rectangular zone will be assigned as beta. 
## The dimensions of the ellipsoid are decided based on the landslide observations. 
'''
## The dimension of the ellipdoid in the direction of motion. 
Ella        = 1.0
## The dimension of the ellipdoid perpendicular to the direction of motion. 
Ellb        = 1.0
## The dimension of the ellipdoid perpendicular to the other two directions. 
Ellc        = 1.0
## The aspect of the direction of motion. 
## If EllAlpha_Calc = "Yes", the aspect of the motion will be averaged for a circular area with a radius of  Ella.
EllAlpha    = 90.0
EllAlpha_Calc = "No"  ##"Yes" - "No" 
##Offset of the ellipsoid.
Ellz        = 0.5
# EllParam = np.array((Ella, Ellb, Ellc, EllAlpha, Ellz,EllAlpha_Calc)) ## All ellipsoidal parameters.
EllParam = [Ella, Ellb, Ellc, EllAlpha, Ellz,EllAlpha_Calc]  ## All ellipsoidal parameters.

## The investigation zone for the ellipsoidal sliding surfaces. 
## Note that the ellipsodal sliding surfaces should not be truncated at the edges. 
## Row start-end and column start-end (including the ends) as rectangular zone. 
## Then, it will be transfered to the lists of cells (row, column) for generation of ellipsoidal sliding surfaces. 
InZone = np.array(([int(nrows/2),int(nrows/2)],[int(ncols/2),int(ncols/2)]))
from Functions_3DPLS_v1_0 import InZone_Rec_to_List ## Otherwise, lists of cells (row, column) can be also defined. 
InZone = InZone_Rec_to_List(InZone) 

## Time = 0 can be selected to analyse before rainfall case while Time = RainfallTime can be used to analyse the end of rainflal. 
## Any time can be assigned within the rainfall duration as I implemented solution for cases with time < Time of Rainfall.
## If you change the formulation, it is also possible to use any time. (1 day: 86400 sec)
# TimeToAnalyse = 86400 ## unit:ses
TimeToAnalyse = np.array((0,))  ##np.array((0,43200,86400))

## Minimum number of cells inside the ellipsoidal sliding surface (Sub-discretize)
## If this number is not satisfied, the code halves the cells and increase the number of cells.
## Either “numpy.kron” or "scipy.interpolate.griddata" can be used, but manual change is required in the functions. 
## Default method is “numpy.kron”.
SubDisNum = 0  ## (Suggested: 100,200)

'''
#####################################################################################
## Part - 3
## Run the analysis
#####################################################################################
'''
## !!!  For reproducibility purposes, the example problems are provided. 
##For the validation problems and simplified case problem, small changes are required. 
## There are some small modifications to the code when the name assigned as one of the 
## {'Pr1', 'Pr2','Pr3S1Dry','Pr3S2Dry','Pr3S2Wet' 'SimpCase'}
ProblemName = 'Pr1' 
## !!!

#########
#########
## Multiprocessing in both MC phase and generation of the ellipsoidal sliding surfaces with FS calculations
## There are multiple run options 
## When zmax is variant, "C-MP-MP" is recommended.
## When zmax is invatiant, "S-MP-MP" is recommended.
#########
#########

## Import functions
from Functions_3DPLS_v1_0 import FSCalcEllipsoid_v1_0_SingleRrocess,FSCalcEllipsoid_v1_0_MutiProcess
from operator import itemgetter
from Functions_3DPLS_v1_0 import Ellipsoid_Generate_Main, Ellipsoid_Generate_Main_Multi,IndMC_Main, IndMC_Main_Multi
# Allocate processes for Monte Carlo simulations and calculations for ellipsoidal sliding surfaces individually
print("Total number of processors: %d"%(cpu_count()))
## Select the run option
## "C-XX-XX" is for Monte Carlo simulations conbined with generation of ellipsoidal sliding surfaces such that ellipsoids are generated for each simulations.This is suggested when zmax is variant. 
## "S-XX-XX" is for separate monte carlo simulations and ellipsoidal sliding surface generation such that ellipsoids are generated first and simulations are performed afterward.This is suggested when zmax is invariant.
## "C": combined, "S": separated, "SP": singe process, "MP": multi processes, "MT": multi threads
## "C-XX-YY": XX for Monte Carlo simulations, YY for generation of ellipsoidal sliding surfaces in each simulation.
## "S-XX-YY": XX for generation of ellipsoidal sliding surfaces, YY for Monte Carlo simulations.
Multiprocessing_Option_List = ["C-SP-SP","C-MP-SP","C-MP-MP","C-MP-MT",
                               "S-SP-SP","S-SP-MP","S-MP-SP","S-MP-MP"] 
Multiprocessing_Option = Multiprocessing_Option_List[0] ## Select 0-7

## Arrange the numbers of processors / threads for calculations
## For options "C-XX-XX"
TOTAL_PROCESSES_MC  = 2 ## Will be utilized if either "C-MP-SP" or "C-MP-MP" run option is selected
TOTAL_PROCESSES_ELL = 2 ## Will be utilized if "C-MP-MP" run option is selected
TOTAL_THREADS_ELL   = 2 ## Will be utilized if "C-MP-MT" run option is selected
## For options "S-XX-XX"
TOTAL_PROCESSES_IndMC  = 4 ## Will be utilized if either "S-SP-MP" or "S-MP-MP" run option is selected
TOTAL_PROCESSES_EllGen = 4 ## Will be utilized if either "S-MP-SP" or "S-MP-MP" run option is selected

# "FSCalcEllipsoid_v1_0_SingleRrocess" is normal procedure with single process unit. 
# "FSCalcEllipsoid_v1_0_MutiProcess" is developed to utilize multiple processors. 
# Parallelization was implemented at two level: Monte carlo simulations and generation of ellipsoidal sliding surfaces. 
# Select "MCRun_v1_0_SingleProcess", "MCRun_v1_0_MultiProcess", "MCRun_v1_0_MultiThread" in "FSCalcEllipsoid_v1_0_MutiProcess" function.
if __name__ == '__main__':
    
    ## Time before the main calculation part of the code
    t1 = time.time()   
    
    #####
    ## Combined Monte Carlo simulations and generation of ellipdoidal sliding surfaces. 
    #####
    
    ## Check Multiprocessing_option and run
    if (Multiprocessing_Option=="C-SP-SP"):
        print("Run option: (SP)", Multiprocessing_Option)
        
        FSCalcEllipsoid_v1_0_SingleRrocess(AnalysisType, FSCalType, RanFieldMethod, \
                                            InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                            nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                            Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                            Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                            ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                                            TimeToAnalyse, NoData,ProblemName)
        
        ## Time after the main calculation part of the code
        t2 = time.time()
        print("Time elapsed: ", t2-t1)
        
    ## Check Multiprocessing_option and run
    if (Multiprocessing_Option in ["C-MP-SP","C-MP-MP","C-MP-MT"]):
        print("Run option: (MP)", Multiprocessing_Option)

        FSCalcEllipsoid_v1_0_MutiProcess(Multiprocessing_Option, TOTAL_PROCESSES_MC ,TOTAL_PROCESSES_ELL,TOTAL_THREADS_ELL, \
                                              AnalysisType, FSCalType, RanFieldMethod, \
                                              InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                              nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                              Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                              Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                              ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                                              TimeToAnalyse, NoData, ProblemName)
        ## Time after the main calculation part of the code
        t2 = time.time()
        print("Time elapsed: ", t2-t1)

    #####
    ## Separated Monte Carlo simulations and generation of ellipdoidal sliding surfaces. 
    #####
    
    ## Check Multiprocessing_option and generate sliding surfaces
    if (Multiprocessing_Option in ["S-SP-SP","S-SP-MP"]):
        print("Run option: (Sliding surfaces: SP)", Multiprocessing_Option)
        
        ## Singleprocessing generation of ellipsoidal sliding surfaces
        AllInf = Ellipsoid_Generate_Main(InZone, SubDisNum, \
                                nrows, ncols, nel, cellsize, EllParam, \
                                SlopeInput,ZmaxInput, DEMInput,AspectInput,\
                                NoData,ProblemName)
        
        ## Save sliding surfaces' information
        AllInf  = AllInf[:]
        AllInf_sorted = sorted(AllInf,key=itemgetter(0))
        AllInf_sorted = np.asarray(AllInf_sorted)
        # from sys import getsizeof
        # round(getsizeof(AllInf_sorted) / 1024 / 1024,2) ## to get size in MB round to 2 digit
        os.chdir(Results_Directory)
        np.save("Elipsoidal_Sliding_Surfaces",AllInf_sorted)
        
        ## Time elapsed for generation of ellipsoidal sliding surfaces
        t2 = time.time()
        print("Time elapsed for generation of ellipsoidal sliding surfaces: ", t2-t1)

    ## Check Multiprocessing_option and generate sliding surfaces
    if (Multiprocessing_Option in ["S-MP-SP","S-MP-MP"]):
        print("Run option: (Sliding surfaces: MP)", Multiprocessing_Option)
        
        ## Multiprocessing generation of ellipsoidal sliding surfaces
        AllInf = Ellipsoid_Generate_Main_Multi(TOTAL_PROCESSES_EllGen, InZone, SubDisNum, \
                                    nrows, ncols, nel, cellsize, EllParam, \
                                    SlopeInput,ZmaxInput, DEMInput,AspectInput,\
                                    NoData,ProblemName)
        
        ## Save sliding surfaces' information
        AllInf  = AllInf[:]
        AllInf_sorted = sorted(AllInf,key=itemgetter(0))
        AllInf_sorted = np.asarray(AllInf_sorted)
        # from sys import getsizeof
        # round(getsizeof(AllInf_sorted) / 1024 / 1024,2) ## to get size in MB round to 2 digit
        os.chdir(Results_Directory)
        np.save("Elipsoidal_Sliding_Surfaces",AllInf_sorted)
        
        ## Time elapsed for generation of ellipsoidal sliding surfaces
        t2 = time.time()
        print("Time elapsed for generation of ellipsoidal sliding surfaces: ", t2-t1)


    ## If sliding surfaces have already been generated and saved, load the information directly and commented out th above two function.
    # os.chdir(Results_Directory) 
    # AllInf_sorted = np.load("Elipsoidal_Sliding_Surfaces.npy",allow_pickle=True)

    ## Check Multiprocessing_option and perfom Monte Carlo simulations
    if (Multiprocessing_Option in ["S-SP-SP","S-MP-SP"]):
        print("Run option: (MC simulations: SP)", Multiprocessing_Option)

        IndMC_Main(AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                                            InZone, Results_Directory, Maxrix_Directory, \
                                            nrows, ncols, cellsize, MCnumber, \
                                            Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                            Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                            SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                            TimeToAnalyse, NoData,ProblemName )
        
        ## Time elapsed for Monte Carlo simulations
        t3 = time.time()
        print("Time elapsed for Monte Carlo simulations: ", t3-t2)


    ## Check Multiprocessing_option and perfom Monte Carlo simulations
    if (Multiprocessing_Option in ["S-SP-MP","S-MP-MP"]):
        print("Run option: (MC simulations: MP)", Multiprocessing_Option)

        IndMC_Main_Multi(TOTAL_PROCESSES_IndMC, AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                                                InZone, Results_Directory, Maxrix_Directory, \
                                                nrows, ncols, cellsize, MCnumber, \
                                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                                SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                                TimeToAnalyse, NoData,ProblemName)    
        
        ## Time elapsed for Monte Carlo simulations
        t3 = time.time()
        print("Time elapsed for Monte Carlo simulations: ", t3-t2)


    
    '''
    #####################################################################################
    ## Part - 4
    ## Plot the resuts
    #####################################################################################
    '''
    
    ## Read the results 
    os.chdir(Results_Directory)  
    DataFiles = os.listdir()
    DataFiles = [i for i in DataFiles if i.endswith('FS_Values.npy')]
    Result_Files      = []  
    for i in range(np.size(DataFiles)):
        print(DataFiles[i])
        Temp = np.load(DataFiles[i])
        Temp2 = [Temp[m].flatten() for m in range(np.shape(Temp)[0])]
        Result_Files.append(Temp2)
    
    ## Rearrange a list for each time instances
    Results_Time = []
    for i in range(np.shape(Result_Files[0])[0]): ## over time
        Temp = [res[i] for res in Result_Files]
        Temp = np.asarray(Temp)
        Results_Time.append(Temp)
    
    # print(np.unique(Results_Time))
    
    # ## The plots should be drawn by the user. 
    # ## All results are stored in the result folder. 
    # ## Both statistical results and the results of each MC simulation can be calculated and corresponding plots can be drawn. 
    
    # ## There are 2 example plots below.
    
    # ###########################
    # ## Example 1
    # ## Mean factor of safety map of the study area
    # ###########################
    
    # os.chdir(Results_Directory)
    
    # ## Calculate the mean 
    # MeanFSData = []
    # for i in range(np.shape(Results_Time)[0]):
    #     Temp = np.mean(Results_Time[i],axis=0) ## Take the average over Monte Carlo simulations
    #     Temp = np.reshape(Temp, (nrows,ncols))
    #     MeanFSData.append(Temp) 
    
    # ## Select a time instance 
    # MeanFSData = MeanFSData[0]
    # MeanFSData[MeanFSData==0]=np.nan
    
    # ## Draw mean FS map 
    
    # ## Color map can be modified and newcmp can be used.
    # ## import lib for drawings 
    # # from matplotlib import cm
    # # from matplotlib.colors import ListedColormap  #, LinearSegmentedColormap
    # # viridis = cm.get_cmap('viridis', 256)
    # # newcolors = viridis(np.linspace(0, 1, 256))
    # # RedColor1 = np.array([255/256, 0, 0, 1])
    # # RedColor2 = np.array([241/256, 169/256, 160/256, 1])
    # # newcolors[int(256/10):int(256/5), :] = RedColor2
    # # newcolors[:int(256/10), :] = RedColor1
    # # newcmp = ListedColormap(newcolors)
    
    # font = {'family' : 'Times New Roman',
    #         'size'   : '14'}   #        'weight' : 'bold',
    # # font = {'family' : 'Times New Roman'}
    # plt.rc('font', **font)
    
    # ## Min and max values can be defiend
    # # minFS = 0
    # # maxFS = 5
    
    # ## Plot figure
    # fig, (ax1, cax) = plt.subplots(ncols=2,figsize=(4,6), gridspec_kw={"width_ratios":[1, 0.1]})
    # fig.subplots_adjust(wspace=0.2)
    # im  = ax1.imshow(MeanFSData)  
    # ## Axes label
    # # ax1.set_title('')
    # ax1.set_ylabel("i")
    # ax1.set_xlabel("j")
    
    # ## Color bar 
    # # fig.colorbar(im, cax=cax, ticks=np.array((1.0,1.1,1.2,1.4,1.6,1.8,2.0)))
    # fig.colorbar(im, cax=cax)
    
    # plt.show()
    
    ###########################
    ###########################
    
    
    ###########################
    ## Example 2
    ## Probability of failure map of the study area
    ###########################
    
    # os.chdir(Results_Directory)
    
    # ## Calculate the mean 
    # PfData = []
    # for i in range(np.shape(Results_Time)[0]):
    #     Temp = Results_Time[i]        ## FS data at one time instance
    #     Temp_MC = np.shape(Temp)[0]   ## MC number
    #     Index_zero = np.where(Temp[0] == 0) 
        
    #     Temp = np.where(Temp<1.0,1,0) ## FS<1.0
    #     Temp = np.sum(Temp, axis=0)   ## Number of failure 
    #     Temp = Temp / Temp_MC * 100  ## Probability of failure calculation
        
    #     Temp[Index_zero] = np.nan
    #     Temp = np.reshape(Temp, (nrows,ncols)) 
    #     PfData.append(Temp)
        
    
    # ## Select a time instance 
    # PfData = PfData[0]
    
    # font = {'family' : 'Times New Roman',
    #         'size'   : '14'}   #        'weight' : 'bold',
    # # font = {'family' : 'Times New Roman'}
    # plt.rc('font', **font)
    
    # ## Min and max values can be defiend
    # minPf = 0  
    # maxPf = 40   
        
    # ## Plot figure
    # fig, (ax1, cax) = plt.subplots(ncols=2,figsize=(4,6), gridspec_kw={"width_ratios":[1, 0.1]})
    
    # fig.subplots_adjust(wspace=0.2)
    # im  = ax1.imshow(PfData,vmin=minPf, vmax=maxPf)  
    # ## Axes label
    # # ax1.set_title('')
    # ax1.set_ylabel("i")
    # ax1.set_xlabel("j")
    
    # ## Color bar 
    # # fig.colorbar(im, cax=cax, ticks=np.array((1.0,1.1,1.2,1.4,1.6,1.8,2.0)))
    # # fig.colorbar(im, cax=cax)
    # fig.colorbar(im, cax=cax, label='$P_{f}$ (L) (%)')     
    # plt.show()
    
    # # Fig_Name = 'x.png' ## with .png
    # # plt.savefig(Fig_Name, dpi=dpisize)
    # # plt.savefig(Fig_Name) ## with .svg
    # plt.show()
    
    ###########################
    ###########################

