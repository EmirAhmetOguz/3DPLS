"""
The 3-Dimensional Probabilistic Landslide Susceptibility (3DPLS) Model:
The 3-Dimensional Probabilistic Landslide Susceptibility Model (3DPLS) is a Python code developed for landslide
susceptibility assessment. The 3DPLS model evaluates the landslide susceptibility on a local to a regional scale (i.e.
single slope to 10 km2) and allows for the effects of variability of model parameter on slope stability to be accounted
for.
The 3DPLS model couples the hydrological and the slope stability models. The hydrological model calculates the
transient pore pressure changes due to rainfall infiltration using Iverson’s linearized solution of the Richards equation
(Iverson 2000) assuming tension saturation. The slope stability model calculates the 𝐹𝑆 by utilizing the extension of
Bishop’s simplified method of slope stability analysis (Bishop 1955) to three dimensions, proposed in the study of
Hungr (1987). The 3DPLS model requires topographic data (e.g., DEM, slope, aspect, groundwater depth, depth to
bedrock, geological zones), hydrological parameters (e.g., steady background infiltration rate, permeability
coefficient, diffusivity), geotechnical parameters (e.g., soil unit weight, cohesion, friction angle), and rainfall data.

Developed by: Emir Ahmet Oguz, PhD Candidate at NTNU
The code was developed in 2020 April-December for research purposes. 
"""

## Libraries 
import os
import numpy as np  
import time
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  ##To remove the warnings (due to FS calculation iterations)

## Time before the analyses
t1 = time.time()

## Directories
os.path.dirname(os.path.realpath(__file__)) ## The folder is assigned as the current working directory first. 
Main_Directory      = os.path.dirname(os.getcwd()) ## The main folder
Code_Directory      = Main_Directory + '\\Codes' ## The folder including the code
GIS_Data_Directory  = Main_Directory + '\\InputData\\Validation\\Problem1'  ## The folder including the input data
## Create directories if they do not exist. 
if (os.path.exists(Main_Directory + '\\Codes\\Matrix') == False):
    os.makedirs(Main_Directory + '\\Codes\\Matrix')
if (os.path.exists(Main_Directory+ '\\Results') == False):
    os.makedirs(Main_Directory+ '\\Results')
if (os.path.exists(Main_Directory+ '\\InputData') == False):
    os.makedirs(Main_Directory+ '\\Results')
if (os.path.exists(Main_Directory+ '\\Results\\Results_3DPLS') == False):
    os.makedirs(Main_Directory+ '\\Results\\Results_3DPLS')
Results_Directory   = Main_Directory+ '\\Results' ## The folder to store results

os.chdir(Main_Directory) ## Change directory 

'''
#####################################################################################
## Part - 1 
## Get the data from GIS_Data_Directory (ASCII grid format files)
## Arrange the data
#####################################################################################
'''

## Import the function to read the ASCII grid format files
os.chdir(Code_Directory)
from Functions_3DPLS_v0_0 import ReadData 

## Dimensions of the problem domain 
## The user can define the dimensions or it can be read from the dem file. 
## Manual entry
# nrows, ncols         = 
# xllcorner, yllcorner = 
# cellsize             = 
# nel                  = nrows*ncols
## Entry from dem file (or any other file with ASCII format)
os.chdir(GIS_Data_Directory)
with open('dem.asc') as f:
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
        DirectionInput = NumberDataAndNames[i][1]   ## Direction of steepest slope (From TopoIndex or QGIS)
    elif (DataFiles[i] == 'rizero.asc'):        
        rizeroInput    =  NumberDataAndNames[i][1]  ## Steady, background infiltration 
    elif (DataFiles[i] == 'slope.asc'):
        SlopeInput    =  NumberDataAndNames[i][1]   ## Slope angle 
    elif (DataFiles[i] == 'zones.asc'):
        ZoneInput    =  NumberDataAndNames[i][1].astype(int)   ## Zones, converted to int
    elif (DataFiles[i] == 'aspect.asc'):
        AspectInput    =  NumberDataAndNames[i][1]  ## Aspect
    elif (DataFiles[i] == 'zmax.asc'):
        ZmaxInput    =  NumberDataAndNames[i][1]    ## Depth to bedrock
    elif (DataFiles[i] == 'source.asc'):
        SourceInput    =  NumberDataAndNames[i][1]  ## Initiation cells, source
    elif (DataFiles[i] == 'depthwt.asc'):
        HwInput        =  NumberDataAndNames[i][1]  ## Initiation cells, source
  
## Import the function to arrange the data (Note: Currently the function is for example Case study: Kvam Lansdlides)
## Depending on the problem and needs, the function can be modified.
# os.chdir(Code_Directory)
# from Functions_3DPLS_v0_0 import DataArrange
# ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput,ZmaxInput,HwInput = \
#     DataArrange(ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput, NoData)

## Rainfall input data (m/sec). 
riInp = np.array(([0.],[86400])) 

'''
#####################################################################################
## Part - 2
## Defining the parameters 
#####################################################################################
'''

## Discretize the domain
x = np.linspace(cellsize/2,(ncols-1)*cellsize+cellsize/2,num=ncols)
y = np.linspace(cellsize/2,(nrows-1)*cellsize+cellsize/2,num=nrows)
## Create a coordinate mesh
xm,ym=np.meshgrid(x,y)
## Reshape the coordinate meshes
xm, ym = np.reshape(xm,(nel,1)) , np.reshape(ym,(nel,1))

## Correlation length values 
CorrLenRangeX = np.array((0,)) ## Correlation length range in X direction 
CorrLenRangeY = np.array((0,)) ## Correlation length range in Y direction
CorrLenRange = CorrLenRangeX    ## Just for plots at the end (make equal correlation length in both direction) 

## The investigation zone for the ellipsoidal sliding surfaces. 
## Smaller rectangular zone should be defined inside the problem domain to prevent truncating the sliding surfaces by the edges.
## Row start-end and column start-end (including the ends) 
InZone = np.array(([int(nrows/2),int(nrows/2)],[int(ncols/2),int(ncols/2)])) ## Curently only 1 cell at the middle

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
## Save matrix to save time for the covariance matrix decomposition method, CMD 
SaveMat = 'YES' ## 'YES' - 'NO'

## Note: For now, the code is available to analyse 1-zone only.
if (AnalysisType == 'Undrained'): ## Soil properties for undrained case
    print()
    UwsInp  = 20.0  ## Unit weight of soil           
    MuSu    = 40.0  ## Mean undrained shear strength
    SoilPar = np.array((UwsInp, MuSu))
    
    # CoV range for analysis
    CoVRange   = np.array(('Low', 'Moderate', 'High')) ## Variability levels
    CoVRangeSu = np.array((0.1, 0.2, 0.3))             ## CoV values for undrained shear strength
    CoVRanges  = [CoVRangeSu]
     
elif (AnalysisType == 'Drained'): ## Soil properties for drained case
    print()
    UwsInp =   1.    ## Unit weight of soil 
    MuC    =   0.1     ## Mean cohesion
    MuPhi  =   0.0   ## Mean friction angle
    Ksat   =   1.0e-6 ## Saturated permeability
    Diff0   =  5.0e-6 ## Diffusivity
    SoilPar = np.array((UwsInp, MuC,MuPhi, Ksat, Diff0))

    ## CoV range for analysis
    CoVRange    = np.array(('Low',  )) ## Variability levels
    CoVRangeC   = np.array((0.,))             ## CoV values for cohesion
    CoVRangePhi = np.array((0.,))          ## CoV values for friction angle
    CoVRanges   = [CoVRangeC, CoVRangePhi]


## Ellipsoidal parameters
'''
## For the ellipsoidal parameters see Oguz et al. (2021)
## An average aspect within the current ellipdoid can be also implemented. 
## For now, EllAlpha is assigned by the user considerin the study area. 
## An average values of slope in a rectangular zone will be assigned as beta. 
## The dimensions of the ellipsoid are decided based on the landslide observations. 
'''
## The dimension of the ellipdoid in the direction of motion. 
Ella        = 1.
## The dimension of the ellipdoid perpendicular to the direction of motion. 
Ellb        = 1.
## The dimension of the ellipdoid perpendicular to the other two directions. 
Ellc        = 1.
## The aspect of the direction of motion. 
EllAlpha    = 90.   
##Offset of the ellipsoid. 
Ellz        = 0.5
EllParam = np.array((Ella, Ellb, Ellc, EllAlpha, Ellz)) ## All ellipsoidal parameters.

## Time = 0 can be selected to analyse before rainfall case while Time = RainfallTime can be used to analyse the end of rainflal. 
## Any time can be assigned within the rainfall duration as I implemented solution for cases with time < Time of Rainfall.
## If you change the formulation, it is also possible to use any time. (1 day: 86400 sec)
TimeToAnalyse = 0 ## unit:sec

## Minimum number of cells inside the ellipsoidal sliding surface (Sub-discretize)
## If this number is not satisfied, the code halves the cells and increase the number of cells. 
SubDisNum = 0  ## (Suggested: 100,200)

'''
#####################################################################################
## Part - 3
## Run the analysis
## Each CoV with a correlation length will be analysed using Monte Carlo method. 
#####################################################################################
'''

## !!!  For reproducibility purposes, the example problems are provided. 
##For the validation problems and simplified case problem, small changes are required. 
## There are some small modifications to the code when the name assigned as one of the 
## {'Pr1', 'Pr2','Pr3S1Dry','Pr3S2Dry','Pr3S2Wet' 'SimpCase'}
ProblemName = 'Pr1'
## !!!


## Importing FSCalcEllipsoid function
from Functions_3DPLS_v0_0 import FSCalcEllipsoid

## Allacate for average statistics includes np.mean, hmean, np.min, np.max, np.std 
StatisticsAverage = []

# CoV = 'Low'  ##'Low', 'Moderate', 'High', 'Kvam' 
for CoV in CoVRange: ## Analyses start for each CoV level
    print("Analysis type: " + AnalysisType)   ## Print analysis type 
    print("CoV Range Name: " + CoV )          ## Print CoV level 
    # print(np.where(CoVRange == CoV)[0][0])
    
    ## For rach variability level, mean values for each correlation length exist. 
    StatisticsAverage = FSCalcEllipsoid(AnalysisType, FSCalType, RanFieldMethod, InZone, SubDisNum, Results_Directory, Code_Directory, \
                                                nrows, ncols, nel, cellsize, \
                                                SoilPar, CoV, CoVRange, CoVRanges, EllParam, \
                                                StatisticsAverage, MCnumber, CorrLenRange, CorrLenRangeX, CorrLenRangeY, SaveMat, \
                                                SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, TimeToAnalyse, NoData,ProblemName)

t2 = time.time() ## Time after the analyses
print("All analyses terminate in %f. sec " %(t2-t1)) ## Printing the total time passed. 

## To save the average statistics: StatisticsAverage
os.chdir(Results_Directory + '\\Results_3DPLS')    
with open('StatisticsAverage.npy', 'wb') as fp:
    pickle.dump(StatisticsAverage, fp)


'''
#####################################################################################
## Part - 4
## Plot the resuts
#####################################################################################
'''

## The plots should be drawn by the user. 
## All results are stored in the result folder. 
## Both statistical results and the results of each MC simulation can be found and corresponding plots can be drawn. 

## There are 2 example plots below.

###########################
## Example 1
## Mean factor of safety map of the study area for a given CoV level and correlation length. 
###########################

# os.chdir(Results_Directory + '\\Results_3DPLS')
# ## Define the CoV level and correlation length. 
# CoVRange_Plot     = CoVRange[0]
# CorrLenRange_Plot = CorrLenRange[0]

# ## Name of the result file
# NameResFile = 'CoV_%s_Theta_%.4d_%.4d_FSInitiation.npy'%(CoVRange_Plot,CorrLenRange_Plot, CorrLenRange_Plot)
# FSData = np.load(NameResFile)  ## Load the data

# MeanFSData = np.mean(FSData,axis=0) ## Take the average 

# ## Reshape the mean FS data aaccording to the size of the interested zone
# ZoneRow    = InZone[0][1]-InZone[0][0]+1
# Zonecolumn = InZone[1][1]-InZone[1][0]+1
# MeanFSData = np.reshape(MeanFSData,(ZoneRow,Zonecolumn))


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
# # minFS = 1
# # maxFS = 2

# fig, (ax1, cax) = plt.subplots(ncols=2,figsize=(4,6), gridspec_kw={"width_ratios":[1, 0.1]})
# fig.subplots_adjust(wspace=0.2)

# ## Plot figure
# # im  = ax1.imshow(MeanFSData,vmin=minFS, vmax=maxFS,cmap=newcmp)  #,cmap='jet,cmap='jet_r'
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


# ###########################
# ## Example 2
# ## Effect of correlation length and variability level on the mean global factor of safety
# ## Mean global factor of safety is the average of minimim FS over the study area for a given number of MC simulations. 
# ###########################

# os.chdir(Results_Directory + '\\Results_3DPLS')
# StatisticsAverage = np.load('StatisticsAverage.npy', allow_pickle=True)
# ## StatisticsAverage[0][0][:,2] ##Low variability level - Min Value - Correlation lengths
# ## StatisticsAverage[1][0][:,2] ##Moderate variability level - Min Value -Correlation lengths
# ## StatisticsAverage[2][0][:,2] ##High variability level - Min Value - Correlation lengths

# ## Arrangements for the figure
# dotsize = 5    
# dashscale =(5, 8)  #linestyle ='--', dashes=dashscale
# markeredgewidth = 0.5
# plotlinewidth = 1. 
# legendfontsize = 10
# axislegendfontsize = 14
# markers = np.array(('^','s','o','|','x','+'))
# text_style   = dict(horizontalalignment='right', verticalalignment='center', fontsize=12, fontfamily='Times New Roman')
# marker_style = dict( markersize=dotsize, markeredgecolor="black")   #,markerfacecolor="tab:blue", color='0.8',linestyle='-',

# ## Create figure
# fig, axs = plt.subplots(1, 1, figsize=(4,4))
# # fig.suptitle('')

# ## Axis 1 
# ax1 = plt.subplot(111)

# ## Plot the results
# ax1.plot(CorrLenRange,StatisticsAverage[0][0][:,2],marker=markers[3],label='3DPLS - Low', linestyle =':', linewidth=plotlinewidth, c='g',**marker_style) 
# ax1.plot(CorrLenRange,StatisticsAverage[1][0][:,2],marker=markers[4],label='3DPLS - Moderate', linestyle =':', linewidth=plotlinewidth, c='b',**marker_style) 
# ax1.plot(CorrLenRange,StatisticsAverage[2][0][:,2],marker=markers[5],label='3DPLS - High', linestyle =':', linewidth=plotlinewidth, c='r',**marker_style) 

# ## Arrange label and sub title
# # ax1.set_title('')
# ax1.set_xlabel('Correlation length')
# ax1.set_ylabel('$\u03BC_{g}$',fontsize = axislegendfontsize)  #'$\u03BC_{FS^{g}}$'  #'Mean of $FS_{mean}$'  '$\u03BC_{FS}$'

# ## Arrange limit and ticks
# ## For dained
# # ax1.set_yticks(np.array((1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40)))
# # ax1.set_ylim((1.05,1.40))
# ## For undrained case
# ax1.set_yticks(np.array((1.6,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7)))
# ax1.set_ylim((1.7,2.8))

# ax1.set_xticks(np.array((0,100,200,300,400,500,600,700,800,900,1000)))
# ax1.set_xlim((-50,1050))

# ## Arrange legend
# ax1.legend(loc=4,fontsize =legendfontsize)
# ax1.text(0,1.33, '(a)', fontsize=12,horizontalalignment='left')  #, verticalalignment='center'

# ## Arrange and save
# plt.tight_layout()
# # plt.subplots_adjust(top=0.85)
# # plt.savefig(".png", dpi=1200)
# # plt.savefig(".svg")
# plt.show()

###########################
###########################

