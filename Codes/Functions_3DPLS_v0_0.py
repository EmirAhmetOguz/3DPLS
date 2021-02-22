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

# Libraries 
import numpy as np
import os
import matplotlib.pyplot as plt
# import rasterio
# from rasterio.plot import show
# import geopandas as gpd
# from rasterstats import zonal_stats
# import matplotlib.colors
import scipy.linalg as sl
from scipy.stats.mstats import hmean  
# import matplotlib.pyplot as plt
from scipy.optimize import root
import time
from scipy import special

#---------------------------------------------------------------------------------
def ReadData(FileNameList,nNumber = 6):
    '''
    ## The aim is to to read the any file in  ASCII grid format.
    ## You should provide file name and starting row number for the Data.
    ## The function returns all information in the file --> data with max/min
    ## * Originally from 'DataFunction_v2_4.py'
    
    Parameters
    ----------
    FileNameList : List
        The list of the files to read.
    nNumber : int
        The number of lines to skip. The default is 6.
    Returns
    -------
    The data in the file in  ASCII grid format.
    
    '''
    
    ## Allocate
    ReturnData=[]

    for FileName in FileNameList:
        ## Allocate
        data=[]
        data2=[]
        
        ## Read the file and store the data
        with open(FileName) as f:
            for line in f:
                data.append(line.split())
        f.close()
        
        ## Read the numbers below explanations
        for i in range(nNumber,len(data)+(-1 if data[len(data)-1]==[] else 0)):
            data2.append([float(n) for n in data[i]])
        NumberData=np.matrix(data2) ## Convert to matrix
        ReturnData.append(NumberData)
        ## Max and Min of the numbers
        #MaxData=np.matrix.max(NumberData)
        #MinData=np.matrix.min(NumberData)

    return(ReturnData)

#---------------------------------------------------------------------------------
def DataArrange(ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput, NoData):
    '''
    ## Here, the aim is to arrange the data upon the desire. 
    ## Any arangement can be done here.
    ## Currently the arrangement is for example Case Study: Kvam Landslides. 

    Parameters
    ----------
    ZoneInput : Array
         Zones.
    SlopeInput : Array
        Slope angle.
    DirectionInput : Array
        Direction of steepest slope (From TopoIndex or QGIS).
    DEMInput : Array
        Digital elevation map data.
    AspectInput : Array
        Aspect.
    rizeroInput : Array
        Steady, background infiltration.
    NoData : int
        The value of No Data. 

    Returns
    -------
    Arranged data with the depth of ground water table.

    '''
    
    ## Arrange zone data
    ## Get unique values if needeed. Then, you can change them.
    #ZoneUnique = np.unique(ZoneInput.flatten(),axis=1)[0] 
    #print(ZoneUnique)
    
    ## Some arrangements upon the need
    ZoneInput[ZoneInput == 21 ] = 1     # Zone 1: Moraine
    ZoneInput[ZoneInput == 17 ] = 2     # Zone 2: Shallow moraine
    ZoneInput[ZoneInput == 5  ] = NoData
    ZoneInput[ZoneInput == 9  ] = NoData
    ZoneInput[ZoneInput == 22 ] = NoData
            
    ## Arrange the rest of input according to NoData, -9999
    SlopeInput[ZoneInput == NoData ]     = NoData
    DirectionInput[ZoneInput == NoData ] = NoData
    DEMInput[ZoneInput == NoData ]       = NoData
    AspectInput[ZoneInput == NoData ]    = NoData
    rizeroInput[ZoneInput == NoData ]    = NoData
    
    ## Instead of using the Zmax from GIS, formulation was utilized. 
    ## Calculate Zmax with the following formulation: mean_zmax = -2.578*tan(slope) + 2.612
    ZmaxInput =  -2.578 * np.tan(np.radians(SlopeInput)) + 2.612
    ZmaxInput[ZmaxInput < 0.4 ] = 0.4  # Min depth is 0.4 m.   
    # ZmaxInput[ZoneInput == 2. ] = 0.4  # Depth of shallow moraine zone might be assigned as 0.4 m. 
    ## Arrange according to NoData
    ZmaxInput[ZoneInput == NoData ] = NoData
    #print(np.min(ZmaxInput[ZmaxInput != NoData]),np.max(ZmaxInput[ZmaxInput != NoData]))
    ZoneInput = np.array(ZoneInput)
    
    ## Water depth (Half of the depth to bedrock according to the saturation ratio of the study area)
    HwInput = ZmaxInput / 2
    HwInput[ZoneInput == NoData ] = NoData
    #print(np.min(HwInput[HwInput != NoData]),np.max(HwInput[HwInput != NoData]))
    
    return(ZoneInput,SlopeInput,DirectionInput,DEMInput,AspectInput,rizeroInput,ZmaxInput,HwInput)

#---------------------------------------------------------------------------------
def FSCalcEllipsoid(AnalysisType, FSCalType, RanFieldMethod, InZone, SubDisNum, Results_Directory, Code_Directory, \
                            nrows, ncols, nel, cellsize, \
                            SoilPar, CoV, CoVRange, CoVRanges, EllParam, \
                            StatisticsAverage, MCnumber, CorrLenRange, CorrLenRangeX, CorrLenRangeY, SaveMat, \
                            SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, TimeToAnalyse, NoData,ProblemName):
    '''
    ## This is the main function in which the calculations are done.
    ## Different CoV values with different values of correlation length are analysed here. 
    ## Many other functions are called and used inside this function. 
    
    Parameters
    ----------
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    InZone: array of int 
        The interested zone for the analysis. 
    SubDisNum : int
         Minimum number of cells desired inside the ellipsoidal sliding surface.
    Results_Directory : str
        The folder to store data.
    Code_Directory : str
        The folder including the code.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    SoilPar : array of floats
        The array of soil properties depending on the AnalysisType .
    CoV : str
        CoV level name for analysis.
    CoVRange : arrau of str
        CoV range for analysis.
    CoVRanges : list
        CoV value of parameters for each CoV level.
    EllParam : array of float
        Ellipsoidal parameters.
    StatisticsAverage : list
        The list to store average statistics for each level of CoV.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    CorrLenRange : array of int
        Correlation length values.
    CorrLenRangeX : array of int
        Correlation length values in X direction.
    CorrLenRangeY : array of int
        Correlation length values in Y direction.
    SlopeInput : Array
        Slope angle.
    ZmaxInput : array
        Depth to bedrock.
    DEMInput : Array
        Digital elevation map data.
    HwInput : array
        Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
        the rainfall data.
    TimeToAnalyse : int
        The time of analysis (in the range of rainfall).
    NoData : int
        The value of No Data. 

    Returns
    -------
    StatisticsAverage
         Average statistics

    '''
    
    ## Get the parameters 
    if (AnalysisType == 'Undrained'): 
        UwsInp = SoilPar[0]
        MuSu   = SoilPar[1]  
        print('Variation level is ' + CoV)  ## Print the variability level of the current run    
        CoVSu = CoVRanges[0][np.where(CoVRange == CoV)[0][0]]
        print('Coefficient of Variation is %.1f for Su. '%(CoVSu)) 
              
    elif (AnalysisType == 'Drained'):        
        UwsInp = SoilPar[0]
        MuC    = SoilPar[1]
        MuPhi  = SoilPar[2]      
        Ksat   = SoilPar[3] ## For Iverson infiltration
        Diff0  = SoilPar[4] ## For Iverson infiltration  
        print('Variation level is ' + CoV) ## Print the variability level of the current run       
        CoVC    = CoVRanges[0][np.where(CoVRange == CoV)[0][0]]
        CoVPhi  = CoVRanges[1][np.where(CoVRange == CoV)[0][0]]       
        print('Coefficient of Variation is %.3f for cohesion. '%(CoVC))
        print('Coefficient of Variation is %.3f for friction angle. '%(CoVPhi))    
              
    StatAllInit=[] ## Allocate   

    ## In case of different corr. length, you can use the other for loop. 
    ## Here, correlation lengths in X and Y directions are assumed equal. 
    # thetax = 0
    for thetax in (CorrLenRangeX):
        # for thetay in (CorrLenRangeY):
        #     CorrLengthX, CorrLengthY = thetax, thetay
            CorrLengthX, CorrLengthY = thetax, thetax           
            print('Correlation length = %d, %d' %(CorrLengthX,CorrLengthY))
       
            ## Allocate 
            FSAll=[]
            # FSAllInit=[] ## For some special regions (ex. initiation zones) 
            StatInit=[]
            
            # count = 1
            for count in range(MCnumber): # One MC analysis in each for loop                
                ## Print current MC number
                print("VarLev:%s \ MC:%d \ Corrlength:%d_%d "%(CoV, count, CorrLengthX,CorrLengthY))     
                ## Allocate 
                SoilParField =[] ## This will contatin random field data. 
                                
                ## Generate random fields of the parameters depending on the method selection. 
                if (AnalysisType == 'Undrained'):                
                    if (RanFieldMethod == 'SCMD'):          
                        SuInp = StepwiseRFR(nrows, ncols, cellsize, CorrLengthX, CorrLengthY , MuSu, CoVSu, 'LN') 
                    elif (RanFieldMethod == 'CMD'):                                               
                        SuInp    =  RFR(nrows, ncols, cellsize,Code_Directory, CorrLengthX,CorrLengthY, \
                                        MuSu, CoVSu,'LN', SaveMat )
                    SoilParField.append(SuInp) ## Append Su data 

                elif (AnalysisType == 'Drained'):  
                    if (RanFieldMethod == 'SCMD'):
                        CInp   = StepwiseRFR(nrows, ncols, cellsize, CorrLengthX, CorrLengthY , MuC, CoVC, 'LN')
                        PhiInp = StepwiseRFR(nrows, ncols, cellsize, CorrLengthX, CorrLengthY , MuPhi, CoVPhi, 'N')   
                    elif (RanFieldMethod == 'CMD'): 
                        CInp   =  RFR(nrows, ncols, cellsize,Code_Directory, CorrLengthX,CorrLengthY, \
                                        MuC, CoVC, 'LN',  SaveMat)
                        PhiInp =  RFR(nrows, ncols, cellsize,Code_Directory, CorrLengthX,CorrLengthY, \
                                        MuPhi, CoVPhi, 'N', SaveMat ) 
                    SoilParField.append(CInp)   ## Append C data 
                    SoilParField.append(PhiInp) ## Append Phi data 
                
                # plt.figure()
                # plt.imshow(SuInp) 
                # plt.colorbar() 
                # plt.show()
                # # To see the histogram og two diferent distribution
                # plt.hist(SuInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
                # plt.show()
                # np.sum(np.where(SuInp<0,1,0))   
                # print (np.mean(SuInp), np.std(SuInp)/np.mean(SuInp))   
     
                # plt.figure()
                # plt.imshow(CInp) 
                # plt.colorbar() 
                # plt.show()
                # # To see the histogram og two diferent distribution
                # plt.hist(CInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
                # plt.show()
                # np.sum(np.where(CInp<0,1,0))   
                # print (np.mean(CInp), np.std(CInp)/np.mean(CInp))                
        
                # plt.figure()
                # plt.imshow(PhiInp) 
                # plt.colorbar() 
                # plt.show()
                # # To see the histogram og two diferent distribution
                # plt.hist(PhiInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
                # plt.show()
                # np.sum(np.where(PhiInp<0,1,0))   
                # print (np.mean(PhiInp), np.std(PhiInp)/np.mean(PhiInp))         
       
                t1 = time.time() ## Time before analysis for the current MC analysis number    
                ## Allocate FS 
                TempLst = {i:[] for i in list(range(0,nrows*ncols))}
   
                '''
                ## It is possible to generate an ellipsoidal sliding surface at any cell given. 
                ## Note: If an ellipsoidal sliding surface is truncated by the boundary of the problem domain, the results will be misleading. 
                ## Therefore, it is advised to extend the problem domain and generate sliding surfaces for the area of interest.
                '''                
                ## Elllipsoidal sliding surface is generated for a given row and column number.
                ## To test: i,j = 19,19 or 60,40  or 10,39 or i,j = 255,224 
                for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest    (example:range(10,108), np.arange(11,108,2)) 
                    for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest 
  
                        ## Current Row and Column for the ellipsoid center
                        EllRow    = i
                        EllColumn = j
                        # print(i,j)
                        
                        # t_start = time.time() ## Time before FS calculation for 1 ellipdoidal sliding surface
                                          
                        ## Calculate the FS for the ellipsoid
                        indexes, FS = EllipsoidFSWithSubDis(AnalysisType, FSCalType, SubDisNum, \
                                                                    nrows, ncols, nel, cellsize, \
                                                                    SoilPar, SoilParField, \
                                                                    EllParam, EllRow, EllColumn, \
                                                                    SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, TimeToAnalyse, NoData,ProblemName)
                 
                        print(FS)                    
                        # t_finish = time.time() ## Time after FS calculation for 1 ellipdoidalsliding surface
                        # print(t_finish-t_start) ## Time required to analyse sliding surface
                        
                        ## Assign the values to the correspoinding cells 
                        for ind in indexes:
                            # print(ind)
                            TempLst[ind].append(FS)
                            # templst[ind].append(np.round(FS,8))
                              
                t2 = time.time()  ## Time after analysis for the current MC analysis number  
                print("Terminates in %f. min " %((t2-t1)/60))
    
                '''
                ## Just to test the remaining part, the saved file can be used.
                ## Save the data to test the remaining part if needed
                # np.save('TempList',TempLst)
                ## Load the data to test the remaining part if needed
                # os.chdir(Code_Directory)
                # TempLst = np.load('TempList.npy', allow_pickle=True)
                # TempLst = TempLst.item()
                '''
                
                ## Allocate FSResults such that the cell has min FS value among all FS values.
                FSResult =  np.zeros((np.shape(range (InZone[0,0],InZone[0,1]+1))[0],np.shape(range (InZone[1,0],InZone[1,1]+1))[0]))
                for i in range((np.shape(FSResult)[0])):
                    for j in  range((np.shape(FSResult)[1])):
                        temind = InZone[0,0]*ncols + InZone[1,0] +  j + i * (ncols) ## The index of first cell analyzed according to the whole problem domain. 
                        # print(i, j,  temind)
                        FSResult[i,j] = np.min(TempLst[temind]) 
                FSResult = FSResult.flatten() 
                
                # See = np.reshape(FSResult,(np.shape(range (10,108))[0],np.shape(range (20,78))[0]))
                # fig, axs = plt.subplots(1, 1)
                # plt.title('FS')
                # plt.imshow(See)
                # plt.colorbar() 
                # plt.show()
      
                ## Combine results of FS 
                FSAll.append(FSResult)      ## For all 
                # FSAllInit.append(FSResult)  ## For some special regions (ex. initiation zones) 
                StatInit.append([np.mean(FSResult)] + [hmean(FSResult)]+ [np.min(FSResult)] \
                                  +[np.max(FSResult)] +[np.std(FSResult)] )
        
                ## Write Initiation zone FS.
                ToWrite = np.asarray(FSAll)
                os.chdir(Results_Directory + '\\Results_3DPLS')
                NameResFile = 'CoV_%s_Theta_%.4d_%.4d_FSInitiation'%(CoV,CorrLengthX, CorrLengthY)
                np.save(NameResFile, ToWrite) #Save as .npy
                          
            ## Combine the statistics mean, min, max, std. dev. for initiation and surrounding zone
            StatAllInit.append(np.asarray(StatInit))
    
    ## Save statistics
    os.chdir(Results_Directory + '\\Results_3DPLS')
    np.save('Statistics_CoV_%s_Initiation.npy'%(CoV), StatAllInit)
    
    ##########################################################
    ## To plot the results for each CoV value. For some special regions (ex. initiation zones) and all zone. 
    ##########################################################   
    ## if needed to work on plots etc.
    #StatAllInit = np.load('Statistics_CoV_XX_Initiation.npy')
        
    ## This part is for the FS data and plots     
    ## Mean statistics
    MeanMCInit = []
    for i in range(np.shape(StatAllInit)[0]):
        MeanMCInit.append([np.mean(StatAllInit[i][:,0])] + [np.mean(StatAllInit[i][:,1])] + [np.mean(StatAllInit[i][:,2])] \
                            +[np.mean(StatAllInit[i][:,3])] + [np.mean(StatAllInit[i][:,4])])
    MeanMCInit = np.asarray(MeanMCInit)   
    
    ## Plot figure  
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Initiation Zone CoV %s' %(CoV))
    
    ## Effect of Corr.Length on Initiation zone FS_mean
    axs[0, 0].scatter(CorrLenRange,MeanMCInit[:,0], c='r', marker="^", label=('AritMean of Aritmetic means'))
    axs[0, 0].scatter(CorrLenRange,MeanMCInit[:,1], c='b', marker="^", label=('AritMean of Harmonic means'))
    # axs[0, 0].scatter(CorrLenRange,MeanMCInit[:,5], c='k', marker="s", label=('HarmMean of Aritmetic means'))
    # axs[0, 0].scatter(CorrLenRange,MeanMCInit[:,6], c='k', marker="s", label=('HarmMean of Harmonic means'))
    axs[0, 0].set_title('Effect of Corr.Length on mean FS')
    axs[0, 0].set_xlabel('Correlation length (m)')
    axs[0, 0].set_ylabel('Mean of $FS_{mean}$')
    axs[0, 0].legend()
    
    ## Effect of Corr.Length on Initiation zone FS_min
    axs[0, 1].scatter(CorrLenRange,MeanMCInit[:,2], c='r', marker="^")
    axs[0, 1].set_title('Effect of Corr.Length on min FS')
    axs[0, 1].set_xlabel('Correlation length (m)')
    axs[0, 1].set_ylabel('Min of $FS_{mean}$')
    
    ## Effect of Corr.Length on Initiation zone FS_max
    axs[1, 0].scatter(CorrLenRange,MeanMCInit[:,3], c='r', marker="^")
    axs[1, 0].set_title('Effect of Corr.Length on max FS')
    axs[1, 0].set_xlabel('Correlation length (m)')
    axs[1, 0].set_ylabel('Max of $FS_{mean}$')
    
    ## Effect of Corr.Length on Initiation zone FS_std. dev.
    axs[1, 1].scatter(CorrLenRange,MeanMCInit[:,4], c='r', marker="^")
    axs[1, 1].set_title('Effect of Corr.Length on std. dev. FS')
    axs[1, 1].set_xlabel('Correlation length (m)')
    axs[1, 1].set_ylabel('Std. dev. of $FS_{mean}$')
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.savefig(Results_Directory + '\\Results_3DPLS'+'\\Plot_CoV_%s_InitiationZone.png'%(CoV),dpi=600)
    plt.show()

    ## Average statistics
    StatisticsAverage.append ([MeanMCInit])
    
    return(StatisticsAverage)

#---------------------------------------------------------------------------------
def EllipsoidFSWithSubDis(AnalysisType, FSCalType, SubDisNum, \
                                nrows, ncols, nel, cellsize, \
                                SoilPar, SoilParField, \
                                EllParam, EllRow, EllColumn, \
                                SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, TimeToAnalyse, NoData,ProblemName):
    '''
    ## This function calculates the factor of safety by assuming an ellipsoidal shape.
    ## If the number of cells inside the ellipsoid is less than the desider, the cellsize is halved.
    ## Sub-discretization minimum column number should be enough to have a stable result. 
    ## To decrease the computational demand and required time, lower values can be used. 
    ## One of the three methods, 'Normal3D' - 'Bishop3D' - 'Janbu3D', for factor of safety calculation can be used. 
    
    Parameters
    ----------
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.    
    SubDisNum : int
         Minimum number of cells desired inside the ellipsoidal sliding surface.      
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    SoilPar : array of floats
        The array of soil properties depending on the AnalysisType . 
    SoilParField : list
        The list of random fields of soil parameters.     
    EllParam : array of float
        Ellipsoidal parameters.     
    EllRow : int
        Current row for the ellipsoid center.
    EllColumn : int
        Current column for the ellipsoid center.      
    SlopeInput : Array
        Slope angle.
    ZmaxInput : array
        Depth to bedrock.
    DEMInput : Array
        Digital elevation map data.
    HwInput : array
        Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
        the rainfall data.
    TimeToAnalyse : int
        The time of analysis (in the range of rainfall).
    NoData : int
        The value of No Data. 
        
    Returns
    -------
    indexesOriginal
        The indexes of the cells inside the current analysed sliding zone
    FS3D
        The calculated FS value

    '''

    ## Cell center coordinate according to the left bottom 
    ## Option-1 (Global)
    CoorG =[] ## Allocate
    for i in range(nrows):
        for j in range(ncols):
            CoorG.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
    CoorG = np.asarray(CoorG) # (row #, column #, x coordinate, y coordinate) 
    ## Option-2
    #Gx = np.linspace(cellsize/2,(ncols-1)*cellsize+cellsize/2,num=ncols)
    #Gy = np.linspace((nrows-1)*cellsize+cellsize/2,cellsize/2,num=nrows)
    ## Create a coordinate mesh
    #CoorGX,CoorGY=np.meshgrid(Gx,Gy)

    ## Ellipsoidal parameters
    ## Global coordinates based on EllRow, EllColumn
    temp = CoorG[np.ix_(CoorG[:,0] == np.array((EllRow)), np.array([False, True, True, True]))]
    temp = temp[np.ix_(temp[:,0] == np.array((EllColumn)), np.array([False, True, True]))]
    ## Coordinates accotding to the global
    EllCenterX  = temp[0][0]                    
    EllCenterY  = temp[0][1] 
    ## Ellipsoidal center DEM data           
    EllDEMCenter = DEMInput[EllRow, EllColumn] 
    ## Ellipsoid dimensions and orientation
    Ella        = EllParam[0]
    Ellb        = EllParam[1] 
    Ellc        = EllParam[2]
    EllAlpha    = EllParam[3]
    Ellz        = EllParam[4]  ## Offset of the ellipsoid
    

    ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
    ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
    ## Coordinates according to the center of allipsoid, e
    CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
    x1 = CoorEll[:,2] * np.cos(np.radians(EllAlpha)) -  CoorEll[:,3] * np.sin(np.radians(EllAlpha))
    y1 = CoorEll[:,3] * np.cos(np.radians(EllAlpha)) +  CoorEll[:,2] * np.sin(np.radians(EllAlpha))
    ## Coordinates according to the ellipsoid coordinate system, e'(e rotated by EllAlpha)
    CoorEll1 = np.concatenate((CoorEll[:,[0, 1]],np.reshape(x1,(np.shape(CoorG)[0],1)),np.reshape(y1,(np.shape(CoorG)[0],1))),axis=1)    
    ## Cells inside the zone of rectangle with the dimensions of (2*Ella – 2* Ellb) 
    CellsInsideRect = CoorEll1[ (np.abs(CoorEll1[:,2])<= Ella) & (np.abs(CoorEll1[:,3])<= Ellb)]
    ## Slope of the ellipsoid is the average slope around the ellipsoid (a rectangular area)
    SlopeRect = SlopeInput[ CellsInsideRect[:,0].astype(int),CellsInsideRect[:,1].astype(int)]
    SlopeRect = SlopeRect[SlopeRect!=NoData] #Remove the value of no data, NoData
    EllBeta   = np.mean(SlopeRect) 

    
    ##!!!
    ##This part is for validation problem 3
    if ( ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry' or ProblemName == 'Pr3S2Wet'):  
        EllBeta   = np.degrees(np.arctan(0.5))
    ##!!!


    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInsideRect)[0]):
    #     See[int(CellsInsideRect[i][0]),int(CellsInsideRect[i][1])] = CellsInsideRect[i][1]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()

    '''
    There is no nan inside the working zone. 
    # np.sum(np.where(SlopeInput[InZone[0,0]:InZone[0,1]+1,InZone[1,0]:InZone[1,1]+1]==NoData,1,0))  
    # np.sum(np.where(DEMInput[InZone[0,0]:InZone[0,1]+1,InZone[1,0]:InZone[1,1]+1]==NoData,1,0))  
    # np.sum(np.where(ZmaxInput[InZone[0,0]:InZone[0,1]+1,InZone[1,0]:InZone[1,1]+1]==NoData,1,0)) 
    '''

    # Assign the parameters
    if (AnalysisType == 'Undrained'):
        #SoilPar = np.array(UwsInp, MuSu)
        Gamma = SoilPar[0]      ## Unit weight 
        Su   = SoilParField[0]  ## Undrained shear strength 
    elif (AnalysisType == 'Drained'):
        #SoilPar = np.array(UwsInp, MuC,MuPhi)
        Gamma = SoilPar[0]      ## Unit weight 
        c    = SoilParField[0]  ## Cohesion
        phi  = SoilParField[1]  ## Friction angle 
        Ksat   = SoilPar[3] ## For Iverson infiltration
        Diff0  = SoilPar[4] ## For Iverson infiltration  
    
    #########################################################################
    #########################################################################
    
    ## Determine the cell inside the ellipsoid 
    ## Returned CellsInside variable --> [ row(from top), column(from left), xe', ye' ]
    ## Two function was considered CellsInsideEll and CellsInsideEllV2 considering 2D, and 3D approachs respectively. 
    ## CellsInsideEllV2 is better in identifying the cells inside. 
    # indexes, CellsInside = CellsInsideEll(CoorG,EllCenterX,EllCenterY,Ella,Ellb,Ellc, Ellz, EllAlpha,EllBeta)
    indexes, CellsInside = CellsInsideEllV2(nrows,ncols,cellsize,DEMInput, EllCenterX,EllCenterY,EllDEMCenter, EllAlpha,EllBeta, Ella,Ellb,Ellc,Ellz)
    CellsInside[np.abs(CellsInside)<1e-10] = 0 ## Correction to the very low values 
 
    ## Calculate the depth at a given cell
    ## If the DEM of the cell is lower than the DEM of the sliding surface, it is removed.
    Depths = EllDepth (CellsInside[:,2],CellsInside[:,3],EllCenterX,EllCenterY,Ella,Ellb,Ellc,Ellz, EllAlpha,EllBeta)
    Depths = np.reshape(Depths,(np.shape(CellsInside)[0],1))
    Depths[np.isnan(Depths)] = 0
    ## DEM of the sliding surface
    DepthDEM = EllDEMCenter - ( np.reshape(CellsInside[:,2],(np.shape(CellsInside)[0],1)) * np.tan(np.radians(EllBeta)) + Depths )
 
    
    ##!!!
    ##This part is for validation problem 2
    if ( ProblemName == 'Pr2'):  
        SlopeInput2 = np.zeros((nrows))
        SlopeInput2[:int(0.58/cellsize)+1] = SlopeInput[1,1]
        DEMInput2 = np.zeros((nrows))
        for i in range(nrows-1):
            DEMInput2[i+1] = DEMInput2[i] + + ( cellsize/2 * np.tan(np.radians(SlopeInput2[i])) + cellsize/2 * np.tan(np.radians(SlopeInput2[i+1])) )
        DEMInput2 = DEMInput2 + cellsize/2*np.tan(np.radians(SlopeInput[1,1]))
        DEMInput2 = DEMInput2[::-1]
        DEMInput2 = np.transpose([DEMInput2] * ncols)
        
        DEMInput = DEMInput2
    ##!!!
    
    DEMdiff = np.reshape( DEMInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] , (np.shape(CellsInside)[0],1)) - DepthDEM
    ## Removing condition to eliminate the cells lower than the DEM of the sliding surface
    RemoveCond  = np.where(DEMdiff< 0)[0] 
    CellsInside = np.delete(CellsInside, RemoveCond, 0)
    Depths      = np.delete(Depths, RemoveCond, 0)
    DepthDEM    = np.delete(DepthDEM, RemoveCond, 0 )
    indexes     = np.delete(indexes, RemoveCond, 0)
    
    ## Save the indexes acccording to the original indexing
    indexesOriginal = indexes
        
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = indexes[i]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
    
    ## Allocate temporary index
    IndexTemp = np.arange(0,nrows*ncols,dtype=int)
    IndexTemp = np.reshape(IndexTemp, (nrows,ncols))
    
    ## Subdiscretisize if the number of cells are low (by halving the cell size and arranging the data). 
    ## SubDisNum is the minimum number of cells defined by the user in the main code.         
    while (np.shape(CellsInside)[0] < SubDisNum):
        
        ## Halve the cellsize for this part only
        cellsize = cellsize / 2
        nrows    = nrows    * 2  ## Double the number of rows
        ncols    = ncols    * 2  ## Double the number of columns
        
        ## Cell center coordinate according to the left bottom option-1 (Global)
        CoorG =[]
        for i in range(nrows):
            for j in range(ncols):
                CoorG.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
        CoorG = np.asarray(CoorG) # (row #, column #, x coordinate, y coordinate)     
        
        ## Arrangement of the soil strength parameters
        if (AnalysisType == 'Undrained'):
            Su   = np.kron(Su, np.ones((2,2)))      
        elif (AnalysisType == 'Drained'):
            c    = np.kron(c, np.ones((2,2)))
            phi  = np.kron(phi, np.ones((2,2)))
        
        ## Arrangments with new cellsize 
        SlopeInput = np.kron(SlopeInput, np.ones((2,2)))
        ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))
        
        ### !!!
        ##This part is for simplified case
        if ( ProblemName == 'SimpCase'):  
            ## np.kron does not work for the simplified case. 
            ## For simplified case, DEM should be calculated again. 
            # Allocate DEMs
            DEMInput = np.zeros((nrows,ncols))    
            for i in range(ncols-1,-1,-1):
                for j in range(nrows-2,-1,-1):
                    DEMInput[j,i] = DEMInput[j+1,i] + ( cellsize/2 * np.tan(np.radians(SlopeInput[j+1,i])) + cellsize/2 * np.tan(np.radians(SlopeInput[j,i])) )
            DEMInput = DEMInput + 50 #I added but no need. 
        else:
            DEMInput   = np.kron(DEMInput, np.ones((2,2)))
        ### !!!
        
        IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
        HwInput   = np.kron(HwInput, np.ones((2,2)))  
        rizeroInput  = np.kron(rizeroInput, np.ones((2,2)))  
        
        ## Determine the cell inside the ellipsoid 
        ## Returned CellsInside variable --> [ row(from top), column(from left), xe', ye' ]
        ## Two function was considered CellsInsideEll and CellsInsideEllV2 considering 2D, and 3D approachs respectively. 
        ## CellsInsideEllV2 is better in identifying the cells inside. 
        # indexes, CellsInside = CellsInsideEll(CoorG,EllCenterX,EllCenterY,Ella,Ellb,Ellc, Ellz, EllAlpha,EllBeta)
        indexes, CellsInside = CellsInsideEllV2(nrows,ncols,cellsize,DEMInput, EllCenterX,EllCenterY,EllDEMCenter, EllAlpha,EllBeta, Ella,Ellb,Ellc,Ellz)
        CellsInside[np.abs(CellsInside)<1e-10] = 0 ## correction to the very low values 
    
        ## Calculate the depth at a given cell
        ## If the DEM of the cell is lower than the DEM of the sliding surface, it is removed.
        Depths = EllDepth (CellsInside[:,2],CellsInside[:,3],EllCenterX,EllCenterY,Ella,Ellb,Ellc,Ellz, EllAlpha,EllBeta)
        Depths = np.reshape(Depths,(np.shape(CellsInside)[0],1))
        Depths[np.isnan(Depths)] = 0
        ## DEM of the sliding surface
        DepthDEM = EllDEMCenter - ( np.reshape(CellsInside[:,2],(np.shape(CellsInside)[0],1)) * np.tan(np.radians(EllBeta)) + Depths )
        # DEMdiff = (DEMInput[(int(CellsInside[i,0]),int(CellsInside[i,1]))] - DepthDEM[i,0]) 
        DEMdiff = np.reshape( DEMInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] , (np.shape(CellsInside)[0],1)) - DepthDEM
        ## Removing condition to eliminate the cells lower than the DEM of the sliding surface
        RemoveCond  = np.where(DEMdiff< 0)[0] 
        CellsInside = np.delete(CellsInside, RemoveCond, 0)
        Depths      = np.delete(Depths, RemoveCond, 0)
        DepthDEM    = np.delete(DepthDEM, RemoveCond, 0 )
        indexes     = np.delete(indexes, RemoveCond, 0)
        
        ## Save the indexes acccording to the original indexing
        indexesOriginal = IndexTemp.flatten()[indexes]
        indexesOriginal = np.unique(indexesOriginal)

    ## Calculation of the weight and thickness
    Weight = np.zeros((np.shape(CellsInside)[0],1))    ## Allocate
    Thickness = np.zeros((np.shape(CellsInside)[0],1)) ## Allocate   
    for i in range(np.shape(CellsInside)[0]):
        ## Depth of the sliding surface at each cell
        DEMdiff2 = (DEMInput[(int(CellsInside[i,0]),int(CellsInside[i,1]))] - DepthDEM[i,0])  
        ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
        Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
        Weight[i]    = Thickness[i] * cellsize**2 * Gamma   ## Weight
     
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = Thickness[i]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
          
    '''
    ## The part of vectos, gradients, normal vectors for the calculation of the slope angles. 
    ## Some lines are commented out. They actually work and can be used but not needed at the current version. 
    '''
    
    ## Coordinate according to the e'' coordinate system (center of ellipsoid)
    xvalues = np.reshape( CellsInside[:,2] / np.cos(np.radians(EllBeta)) + Depths[:,0] * np.sin(np.radians(EllBeta))   , (np.shape(CellsInside)[0],1) )
    yvalues = np.reshape( CellsInside[:,3] , (np.shape(CellsInside)[0],1) )
    zvalues = - np.reshape(calz( xvalues[:,0] , yvalues[:,0] ,Ella,Ellb,Ellc) , (np.shape(CellsInside)[0],1) )
    ## zvalues is qeual to zvalues2. Which shows that depths are correct. 
    # zvalues2 = np.reshape( Depths[:,0]  * np.cos(np.radians(EllBeta)) + Ellz  , (np.shape(CellsInside)[0],1) )
    
    ## Cells inside according to e'' coordinate system 
    CellsInsideE11 = np.concatenate( (CellsInside[:,0:2],xvalues,yvalues,zvalues),axis=1)
    
    ## Gradient According to e'' 
    GradientsE11 = [] ## Allocate
    for i in range(np.shape(CellsInsideE11)[0]):
        GradientsE11.append( (2 * (CellsInsideE11[i,2]) / Ella**2, 2 * (CellsInsideE11[i,3]) / Ellb**2 , 2 * (CellsInsideE11[i,4]) / Ellc**2) )
    GradientsE11 = np.asarray(GradientsE11)
    
    """
    ## On XZ'' plane 
    """
    
    ## Angle between normal vector and x'' axis, on XZ'' plane
    ## Normal vector
    n = np.array((GradientsE11[:,0],GradientsE11[:,2])).T     
    vx11= np.array([1.,0.])
    uvx11 = vx11 / np.linalg.norm(vx11)
    
    ## Angle of the normal vector on XZ'' according to e'' coordinate system (by dot product)
    AngleNormalXZE11 =[]  ## Allocate
    for i in range(np.shape(n)[0]):
        un = n[i] / np.linalg.norm(n[i])
        AngleNormalXZE11.append( np.degrees(np.arccos(np.dot(un, uvx11)) ) )
    AngleNormalXZE11 = np.asarray(AngleNormalXZE11)
    AngleNormalXZE11[np.isnan(AngleNormalXZE11)] = 90
    
    ## Angle between tangent line and X'' on XZ'' plane
    AngleTangentXZE11 = AngleNormalXZE11 - 90 
    
    ## Angle between tangent and x' axis (angle - EllBeta) on XZ' plane
    AngleTangentXZE1 = AngleTangentXZE11 + EllBeta
    
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = AngleTangentXZE1[i]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
    
    """
    ## On YZ'' plane 
    """ 
    
    # ## Angle between normal vector and Y'' axis on YZ'' plane
    # n = np.array((GradientsE11[:,1],GradientsE11[:,2])).T
    # vx11= np.array([1.,0.])
    # uvx11 = vx11 / np.linalg.norm(vx11)
    # AngleNormalYZE11 =[]
    # for i in range(np.shape(n)[0]):
    #     un = n[i] / np.linalg.norm(n[i])
    #     AngleNormalYZE11.append( np.degrees(np.arccos(np.dot(un, uvx11)) ) )
    # AngleNormalYZE11 = np.asarray(AngleNormalYZE11)
    # #AngleNormalYZE11[np.isnan(AngleNormalYZE11)] = 90
    
    # ## Angle between tangent line and Y''  on YZ'' plane
    # AngleTangentYZE11 = AngleNormalYZE11 - 90 
    
    # ## Angle between tangent and Y' axis (angle - EllBeta) on YZ' plane
    # AngleTangentYZE1 = AngleTangentYZE11
    
    '''
    ## It is also possible to transfer the gradients according to the e' and then calculate the tangents.
    ## Below you see the code doing this.
    '''
    # Transform the vector to the e' coordinate system 
    GradientsE1 = np.zeros (( (np.shape(CellsInside)[0],3)))
    GradientsE1[:,0] = GradientsE11[:,0] * np.cos(np.radians(EllBeta)) +  GradientsE11[:,2]  * np.sin(np.radians(EllBeta))
    GradientsE1[:,1] = GradientsE11[:,1]
    GradientsE1[:,2] = -GradientsE11[:,0] * np.sin(np.radians(EllBeta)) +  GradientsE11[:,2]  * np.cos(np.radians(EllBeta))
    
    # ## Angle between normal vector and x' axis on XZ' plane
    # n = np.array((GradientsE1[:,0],GradientsE1[:,2])).T
    # vx11= np.array([1.,0.])
    # uvx11 = vx11 / np.linalg.norm(vx11)
    # AngleNormalXZE11try2 =[]
    # for i in range(np.shape(n)[0]):
    #     un = n[i] / np.linalg.norm(n[i])
    #     AngleNormalXZE11try2.append( np.degrees(np.arccos(np.dot(un, uvx11)) ) )
    # AngleNormalXZE11try2 = np.asarray(AngleNormalXZE11try2)
    # # AngleNormalXZE11[np.isnan(AngleNormalXZE11)] = 90
    # ## Angle between tangent line and X'  on XZ' plane
    # AngleTangentXZE11try2 = AngleNormalXZE11try2 - 90 
    
    ## Calculation of Asp (Aspect)
    ## Angle between normal vector and x' axis on XY' plane
    GradientsE1Minus = - GradientsE1
    n = np.array((GradientsE1Minus[:,0],GradientsE1Minus[:,1])).T
    vx11= np.array([1.,0.])
    uvx11 = vx11 / np.linalg.norm(vx11)
    AngleNormalXYE1 =[] ## Allocate. It will be assigned to Asp below. 
    for i in range(np.shape(n)[0]):
        un = n[i] / np.linalg.norm(n[i])
        AngleNormalXYE1.append( np.degrees(np.arccos(np.dot(un, uvx11)) ) )
    AngleNormalXYE1 = np.asarray(AngleNormalXYE1)
    #AngleNormalXZE11[np.isnan(AngleNormalXZE11)] = 90  
    
    ## Calculation of Theta (Dip angle)
    ## Angle between 3D normal vector and Z' axis.
    n = np.array((GradientsE1Minus[:,0],GradientsE1Minus[:,1],GradientsE1Minus[:,2])).T
    vx11= np.array([0.,0.,1.])
    uvx11 = vx11 / np.linalg.norm(vx11)
    AngleNormal3DWE1 =[]  ## Allocate. It will be assigned to Theta below. 
    for i in range(np.shape(n)[0]):
        un = n[i] / np.linalg.norm(n[i])
        AngleNormal3DWE1.append( np.degrees(np.arccos(np.dot(un, uvx11)) ) )
    AngleNormal3DWE1 = np.asarray(AngleNormal3DWE1)
    
    ## From now on, the symbols are revised for the FS calculations. 
    Theta =  AngleNormal3DWE1   ## Dip angle
    ## In this model, (Xe',Ye') coordinate system is oriented towards main incilination direction. 
    ## Therefore, min inclination direction of the landslide (AvrAsp) becomes zero with respect to Xe'. 
    AvrAsp = 0                   
    Asp    = AngleNormalXYE1    ## Aspect of the cells
        
    ## The apparent dip of the main direction of inclination of the sliding surface (ThetaAvr) is calculated.
    ## The details can be seen in Xie et al. (2003) (Doi:10.1061/(ASCE)1090-0241(2003)129:12(1109))
    TanThetaAvr = np.tan(np.radians(Theta)) * abs(np.cos(np.radians(Asp-AvrAsp)))
    ThetaAvr = np.degrees(np.arctan(TanThetaAvr))
       
    ## This formulation also gives the tangents !! 
    ## This can be checked to see that the above formulation and the previous calculations are resulting in the same values.
    Tanxz = np.degrees(np.arctan(np.tan(np.radians(Theta))*np.cos(np.radians(Asp))))    
    # Tanxz = AngleTangentXZE1  
    Tanyz = np.degrees(np.arctan(np.tan(np.radians(Theta))*np.sin(np.radians(Asp))))
    ## Limit the slopes and areas (by trial and error). 
    Tanxz[Tanxz>85.] = 85.
    Tanyz[Tanyz>85.] = 85.
    
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = ThetaAvr[i]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
    # See = See[67:70,68:87]
         
    ## Calculation of the area by using the formulation given in Xie's or Hungr's papers. 
    A=[] ## Allocate
    for i in range(np.shape(CellsInside)[0]):
        A.append((cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(Tanyz[i])))**2) *((np.sin(np.radians((Tanxz[i]))))**2) )) )/ \
                        (((np.cos(np.radians((Tanyz[i])))))*((np.cos(np.radians((Tanxz[i])))))))
    A = np.asarray(A)
    A = np.reshape(A, (np.shape(CellsInside)[0],1))
    
    ## Remove very high area values. When the bishop3D method is used, this does not change the results significanlty.         
    LimitValue = 1 + np.max(Thickness)/cellsize  ## By trials 
    A[A>LimitValue*cellsize**2] = LimitValue*cellsize**2 
         
    '''
    ## This part is the truncation part according to the ZmaxInput.
    ## If maximum depth (ZmaxInput) is less than the thickness to the depth of the sliding surface (Thickness),
    ## the cell is truncated and the parameters are reassigned/calculated. 
    ## Modified parameters: A, thickness, weight, ThetaAvr and Theta change. 
    '''
    
    ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))  ## Reshape 
    condzmax = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
    
    ## Arrangement of the parameters using truncation condition above
    ## Thickness becomes ZmaxInput for the truncated cells.
    Thickness[condzmax] = ZmaxInput[indexes][condzmax]
    
    ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
    ## Area is recalculated. 
    A[condzmax] = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                        (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
    
    ## Weight is recalculated.                     
    Weight[condzmax] =  ZmaxInput[indexes][condzmax] * (cellsize**2) * Gamma 
    
    # Reshape and truncate to the slope of the current cell
    ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
    Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
    AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
    
    ThetaAvr[condzmax] = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
    Theta[condzmax] = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]
    AngleTangentXZE1[condzmax] = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
     
        
    ## !!!
    ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
    if ( ProblemName == 'Pr3S2Dry' or ProblemName == 'Pr3S2Wet'): 
        ## Correction for truncation angle
        ## Normally slope of the cell is assigned. Here, zero is assigned.
        tempvalues = np.ones((np.shape(condzmax)[0],1)) * (cellsize**2)
        A[condzmax] = tempvalues 
        ThetaAvr[condzmax] =  np.zeros((np.shape(condzmax)[0],1))
        Theta[condzmax] = np.zeros((np.shape(condzmax)[0],1))
        AngleTangentXZE1[condzmax] = np.zeros((np.shape(condzmax)[0],1))
        
        ## Assign weak layer properties for the truncated cells.
        ## Cohesion
        c = np.reshape(c.flatten(), (nrows*ncols,1))
        tempvalues =  c[indexes][:]
        tempvalues[condzmax] = np.zeros((np.shape(condzmax)[0],1))
        c[indexes] = tempvalues
        c = np.reshape(c, (nrows,ncols))
        
        ## Friction angle
        phi = np.reshape(phi.flatten(), (nrows*ncols,1))
        # phi[indexes][condzmax] = np.zeros((9520, 1))
        tempvalues =  phi[indexes][:]
        tempvalues[condzmax] = np.ones((np.shape(condzmax)[0],1)) * 10.
        phi[indexes] = tempvalues
        phi =np.reshape(phi, (nrows,ncols))


        # See = np.zeros((nrows,ncols))
        # for i in range(np.shape(CellsInside)[0]):
        #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = c[int(CellsInside[i][0]),int(CellsInside[i][1])]
        # fig, axs = plt.subplots(1, 1)
        # #plt.title('')
        # plt.imshow(See)
        # plt.colorbar() 
        # plt.show()
        
    ## !!!
    
    
    ## Remove very high area values again. 
    LimitValue = 1 + np.max(Thickness)/cellsize  ## By trials
    A[A>LimitValue*cellsize**2] = LimitValue*cellsize**2 
    
    ####################################
    ## Iverson (2000), infiltraion model 
    ####################################
    ## The calculation of the pore water forces using Iverson's solution of infiltration (Iverson, 2000)
    ## Slope parallel flow is assummed.    
    ## If the analysis is undrained, pore water forces are zero. If it is drained, calculations are performed using Iverson's solution-.
    if (AnalysisType == 'Undrained'):
        PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
    elif (AnalysisType == 'Drained'):
        PoreWaterForce = HydrologyModel(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
    
    ## !!!
    ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
    if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):  
        ## Correction 
        PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1)) 
        
    if (ProblemName =='Pr3S2Wet'):  
        ## Correction 
        PoreWaterForce[PoreWaterForce<0] = 0.
    ## !!!
    
    
    
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = PoreWaterForce[i] 
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
  
    ## Factor of safety calculation for 3D ellipsoidal shape. 
    if (AnalysisType == 'Drained'):
        
        ## Arrange the cohesion and friction angle with corresponding indexes 
        c = c.flatten()[indexes]
        c = np.reshape( c , (np.shape(CellsInside)[0],1) )
        phi = phi.flatten()[indexes]
        phi = np.reshape( phi , (np.shape(CellsInside)[0],1) )

        ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'        
        if (FSCalType=='Normal3D'):           
            FS3D = FSNormal3D(c, phi, A, Weight, PoreWaterForce, Theta, ThetaAvr, AngleTangentXZE1 )
            # print( "Normal 3D FS is %.5f"%FS3D)           
        elif (FSCalType=='Bishop3D'):            
            FS3D = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1))
            FS3D = FS3D.x[0]
            # print( "Bishop 3D FS is %.5f"%FS3D)                           
        elif (FSCalType=='Janbu3D'):            
            FS3D = root(FSJanbu3D, 1.,args=(c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1))
            FS3D = FS3D.x[0]
            # print( "Janbu 3D FS is %.5f"%FS3D)               
        
    elif (AnalysisType == 'Undrained'):
  
        ## Arrange the undrained shear strength and friction angle with corresponding indexes 
        Su = Su.flatten()[indexes]
        Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
        phi = np.zeros((np.shape(CellsInside)[0],1))
        
        ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
        if (FSCalType=='Normal3D'):   
            FS3D = FSNormal3D(Su, phi, A, Weight, PoreWaterForce, Theta, ThetaAvr, AngleTangentXZE1 )
            # print( "Normal 3D FS is %.5f"%FS3D)                
        elif (FSCalType=='Bishop3D'):            
            FS3D = root(FSBishop3D, 1.,args=(Su, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1))
            FS3D = FS3D.x[0]
            # print( "Bishop 3D FS is %.5f"%FS3D)                           
        elif (FSCalType=='Janbu3D'):            
            FS3D = root(FSJanbu3D, 1.,args=(Su, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1))
            FS3D = FS3D.x[0]
            # print( "Janbu 3D FS is %.5f"%FS3D)             
    
    return(indexesOriginal,FS3D)

#---------------------------------------------------------------------------------
def RFR(nrows, ncols, cellsize,Code_Directory, CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N', SaveMat = 'NO'):
    '''
    ## Random fields are created.
    
    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    cellsize : int
        Cell size.
    Code_Directory: str
        The folder including the code to save the Matrix.
    CorrLengthX : int/float
        Correlation length range in X direction.
    CorrLengthY : int/float
        Correlation length range in Y direction.
    ParMean : int/float
        Mean of the parameter.
    ParCoV : float
        Coefficient of variation of the parameter.
    DistType : str
        Distribution type, 'N' or 'LN' . The default is 'N'.
    SaveMat: str
        Whether the mayrix is saved or not. The default is 'NO'.

    Returns
    -------
    ParInp: 2D Array
        Random field data of the parameter.
    '''

    nel = nrows*ncols    
    ## Discretize the domain
    x = np.linspace(cellsize/2,(ncols-1)*cellsize+cellsize/2,num=ncols)
    y = np.linspace(cellsize/2,(nrows-1)*cellsize+cellsize/2,num=nrows)    
    ## Create a coordinate mesh
    xm,ym=np.meshgrid(x,y)
    ## Reshape the coordinate meshes
    xm, ym = np.reshape(xm,(nel,1)) , np.reshape(ym,(nel,1))

    ## Allocate correlation matrix. 
    if (CorrLengthX==0 or CorrLengthY==0):
        ## If the correlation length in either X or Y is zero, then random field with zero correlation length in both direction is created.
        ## It check the Matrix folder for the corr. matrix. If there is no, it creates.                 
        Name = 'Cell_%d_%d_Size_%d' %(nrows,ncols,cellsize)+'_Theta_%d_%d.npy'%(CorrLengthX, CorrLengthY)        
        os.chdir(Code_Directory + '\\Matrix')
        if os.path.isfile(Name):
            CorrMat = np.load(Name)
        else:
            CorrMat = np.identity(nel)

    else:
        ## It check the Matrix folder for the corr. matrix. If there is no, it creates. 
        Name = 'Cell_%d_%d_Size_%d' %(nrows,ncols,cellsize)+'_Theta_%d_%d.npy'%(CorrLengthX, CorrLengthY)
        os.chdir(Code_Directory + '\\Matrix')
        if os.path.isfile(Name):
            CorrMat = np.load(Name)
        else:
            CorrMat = np.zeros((nrows*ncols,nrows*ncols))
            for i in range (nel):
                for j in range (i,nel):
                    CorrMat[i,j] = np.exp(-2.0*np.sqrt((xm[i,0]-xm[j,0])**2/CorrLengthX**2+(ym[i,0]-ym[j,0])**2/CorrLengthY**2))
                    CorrMat[j,i] = CorrMat[i,j]  
    
    ## Save the matrix for further use.
    if (SaveMat == 'YES'):
        if (os.path.isfile(Name) == False):
            np.save(Name,CorrMat)
            print("Correlation matrix is generated for %d_%d " %(CorrLengthX, CorrLengthY)) 
              
    ## Cholesky decomposition
    # A1 = sl.cholesky(CorrMat,lower=True)
    A1  = np.linalg.cholesky(CorrMat)   
    ## Random number
    U = np.random.normal(0,1,nel)

    if  (DistType == 'N'): ## For normal distribution
        ParInp = ParMean + (ParCoV*ParMean)*np.reshape(np.dot(A1,U),(nrows,ncols))
          
    elif (DistType == 'LN'): ## For lognormal distribution
        ## Parameters of the underlying normal distribution        
        SigLnPar = np.sqrt(np.log(1+ParCoV**2))
        MuLnPar  = np.log(ParMean)-0.5*SigLnPar**2
        ParInp = np.exp(MuLnPar + SigLnPar*np.reshape(np.dot(A1,U),(nrows,ncols)))
          
    ## Plotting the fields. 
    #plt.figure(figsize = (3,3))
    # plt.figure()
    # plt.imshow(ParInp)
    # #plt.clim( , )
    # plt.colorbar() 
    # #plt.title('')
    # #plt.savefig(r'',dpi=600)
    # plt.show()
     
    return(ParInp)

# ParInp = RFR(30, 30, 5,Code_Directory, 30, 30, 5, 0.1,'N', 'NO')
# ## Check the samples 
# ## To see the histogram og two diferent distribution
# plt.hist(ParInp, bins=30,density=True, facecolor='g', histtype= 'barstacked')
# plt.show()
# np.sum(np.where(ParInp<0,1,0))
# print (np.mean(ParInp), np.std(ParInp)/np.mean(ParInp))

#---------------------------------------------------------------------------------
def StepwiseRFR(nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N'):
    '''
     ## Stepwise random field realization are created.
     ## This decreases the time significantly while it has one drawback of assuming separable correlation coefficient function.
     
    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    cellsize : int
        Cell size.
    CorrLengthX : int/float
        Correlation length range in X direction.
    CorrLengthY : int/float
        Correlation length range in Y direction.
    ParMean : int/float
        Mean of the parameter.
    ParCoV : float
        Coefficient of variation of the parameter.
    DistType : str
        Distribution type, 'N' or 'LN' . The default is 'N'.

    Returns
    -------
    ParInp: 2D Array
        Random field data of the parameter.
    '''

    nel = nrows*ncols
    ## Discretize the domain
    x = np.linspace(cellsize/2,(ncols-1)*cellsize+cellsize/2,num=ncols)
    y = np.linspace(cellsize/2,(nrows-1)*cellsize+cellsize/2,num=nrows)
    ## Create a coordinate mesh
    xm,ym=np.meshgrid(x,y)
    ## Reshape the coordinate meshes
    xm, ym = np.reshape(xm,(nel,1)) , np.reshape(ym,(nel,1))

    ## Allocate
    CorrMatX = np.zeros((ncols,ncols))
    CorrMatY = np.zeros((nrows,nrows))



    # t1 = time.time() ## Time (if needed)   
    
    if (CorrLengthX == 0 or  CorrLengthY == 0):        
        CorrMatX = np.identity(ncols)
        CorrMatY = np.identity(nrows)       
    else:  
        ## Correlation matrix for X and Y
        for i in range(ncols):
            for j in range(i,ncols):
                CorrMatX[i,j] = np.exp(-2.0*np.sqrt((x[i]-x[j])**2/CorrLengthX**2))
                CorrMatX[j,i] = CorrMatX[i,j]        
        for i in range(nrows):
            for j in range(i,nrows):
                CorrMatY[i,j] = np.exp(-2.0*np.sqrt((y[i]-y[j])**2/CorrLengthY**2))
                CorrMatY[j,i] = CorrMatY[i,j]
                
    ## Cholesky decomposition and generation of random field
    U       = np.random.normal(0,1,(ncols,nrows))
    Ax      = sl.cholesky(CorrMatX,lower=True)
    Ay      = sl.cholesky(CorrMatY,lower=True)    
    
    ## U vector can be defined in two ways: (ncols, nrows) or (nrows, ncols)
    # cInp =(MuC + (COVC*MuC)*np.matmul(np.matmul(Ay,U.T),np.transpose(Ax)))    
    # cInp =(MuC + (COVC*MuC)*np.matmul(np.matmul(Ax,U),np.transpose(Ay))).T

    if  (DistType == 'N'):
        
        ParInp =(ParMean + (ParCoV*ParMean)*np.matmul(np.matmul(Ax,U),np.transpose(Ay))).T   #For normal distribution
    
    elif (DistType == 'LN'):
        
        # Parameters of the underlying normal distribution        
        SigLnPar = np.sqrt(np.log(1+ParCoV**2))
        MuLnPar  = np.log(ParMean)-0.5*SigLnPar**2

        ParInp      = (np.exp(MuLnPar + SigLnPar* np.matmul(np.matmul(Ax,U),np.transpose(Ay)))).T #For lognormal distribution
        
    # t2 = time.time() ## Time (if needed)   
    # print('Stepwise method time: %f ' %(t2-t1)) ## Time required to create random field
     
    ## Plotting the fields. 
    # #plt.figure(figsize = (3,3))
    # plt.figure()
    # plt.imshow(ParInp)
    # #plt.clim(3000 ,8000)
    # plt.colorbar() 
    # #plt.title('Cohesion')
    # #plt.savefig(r'',dpi=600)
    # plt.show()
 
    return(ParInp)

# ParInp = StepwiseRFR(135, 91, 10,0,0, 6, 0.3, 'LN')   #(nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N')
# # Check the samples 
# # To see the histogram og two diferent distribution
# plt.hist(ParInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
# plt.show()
# np.sum(np.where(ParInp<0,1,0))

#---------------------------------------------------------------------------------
def CellsInsideEllV2(nrows,ncols,cellsize,DEMInput, EllCenterX,EllCenterY,EllDEMCenter, EllAlpha,EllBeta, Ella,Ellb,Ellc,Ellz):
    '''
    ## This function returns the cells inside the ellipsoid.
    ## This is 3D version of the previous function, CellsInsideEll
    ## Returned CellsInside variable --> [ row(from top), column(from left), xe', ye' ]
    
    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size..
    DEMInput : Array
        Digital elevation map data.        
    EllCenterX : float
        X coordinate accotding to the global.
    EllCenterY : float
        Y coordinate accotding to the global.
    EllDEMCenter : list
        Ellipsoidal center DEM data .       
    EllAlpha : float
        The aspect of the direction of motion..
    EllBeta : float
        Slope of the ellipsoid in the direction of motion.    
    Ella : float
        The dimension of the ellipdoid in the direction of motion..
    Ellb : float
        The dimension of the ellipdoid perpendicular to the direction of motion..
    Ellc : float
        The dimension of the ellipdoid perpendicular to the other two directions.
    Ellz : float
        Offset of the ellipsoid.

    Returns
    -------
    indexes
        Indexes of the cells inside the sliding zone.
    CellsInside 
        CellsInside variable --> [ row(from top), column(from left), xe', ye' ]                                            

    '''
             
    ## Cell center coordinate according to the left bottom option-1
    CoorG2 =[]
    for i in range(nrows):
        for j in range(ncols):
            CoorG2.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize, DEMInput[i,j] ])
    CoorG2 = np.asarray(CoorG2) ## (row #, column #, x coordinate, y coordinate)         
    ## Coordinates according to the center of allipsoid, e
    CoorEll = CoorG2[:]-np.array((0,0,EllCenterX,EllCenterY,EllDEMCenter)) 
    
    ## Coordinates according to the center of allipsoid with a rotation alpha, el
    x1 = CoorEll[:,2] * np.cos(np.radians(EllAlpha)) -  CoorEll[:,3] * np.sin(np.radians(EllAlpha))
    y1 = CoorEll[:,3] * np.cos(np.radians(EllAlpha)) +  CoorEll[:,2] * np.sin(np.radians(EllAlpha))
    z1 = CoorEll[:,4]
    
    ## Coordinates according to the center of allipsoid with a rotation alpha, elll
    x111 = x1 - np.array((Ellz*np.sin(np.radians(EllBeta))))
    y111 = y1
    z111 = z1 - np.array((Ellz*np.cos(np.radians(EllBeta))))
    
    ## Coordinates according to the center of allipsoid with a rotation beta, ell
    x11 = x111 * np.cos(np.radians(EllBeta)) -  z111 * np.sin(np.radians(EllBeta))
    y11 = y111
    z11 = z111 * np.cos(np.radians(EllBeta)) +  x111 * np.sin(np.radians(EllBeta))
    
    ## Coordinates (e'')
    CoorEll11 = np.concatenate((CoorEll[:,[0, 1]],np.reshape(x11,(np.shape(CoorG2)[0],1)),np.reshape(y11,(np.shape(CoorG2)[0],1)),np.reshape(z11,(np.shape(CoorG2)[0],1))),axis=1)
    
    ## Find the cells inside the ellipsoidal sliding surface
    CellsInside =[]    
    indexes =[]
    for i in range(np.shape(CoorEll11)[0]):
        cond = (( CoorEll11[i][2]**2/Ella**2 + CoorEll11[i][3]**2/Ellb**2 + CoorEll11[i][4]**2/Ellc**2 ) <= 1)
        if cond:
            CellsInside.append([CoorEll[i,0],CoorEll[i,1],x1[i], y1[i], z1[i]] )
            indexes.append((i))
    CellsInside = np.asarray(CellsInside) 
    indexes = np.asarray(indexes) 
        
    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(CellsInside)[0]):
    #     See[int(CellsInside[i][0]),int(CellsInside[i][1])] = indexes[i]
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
    
    return(indexes,CellsInside)  ## Returns the indexes and infromation of the Cells 

#---------------------------------------------------------------------------------
def EllDepth (x,y,EllCenterX,EllCenterY,Ella,Ellb,Ellc,Ellz,EllAlpha,EllBeta):
    '''
    ## This function calculates the vertical depth at a given x,y accorging to e' coordinate system
    ## (x**2)/(a**2)+(y**2)/(b**2)+(z**2)/(c**2) = 1 : ellipsoid formulation
    ## The formulation is derived using the geometrical relationships. 
    
    Parameters
    ----------
    x : float
        x coordinate of the point accorging to e' coordinate system.
    y : float
        y coordinate of the point accorging to e' coordinate system.
    EllCenterX : TYPE
         x coordinate of the ellipsoid center accorging to the global coordinate system.
    EllCenterY : TYPE
        y coordinate of the ellipsoid center accorging to the global coordinate system.
    Ella : float
        The dimension of the ellipdoid in the direction of motion..
    Ellb : float
        The dimension of the ellipdoid perpendicular to the direction of motion..
    Ellc : float
        The dimension of the ellipdoid perpendicular to the other two directions.
    Ellz : float
        Offset of the ellipsoid.
    EllAlpha : float
        The aspect of the direction of motion..
    EllBeta : float
        Slope of the ellipsoid in the direction of motion.

    Returns
    -------
    Depth
        Depth values. 

    ''' 
    
    ## Adjustment to the x due to Ellz offset
    x = x - Ellz * np.sin(np.radians(EllBeta))
    
    Ella2 = Ella * np.sqrt( 1-(y**2)/(Ellb**2) )
    Ellc2 = Ellc * np.sqrt( 1-(y**2)/(Ellb**2) )
    
    c1 = (Ellc2**2) * ((np.tan(np.radians(EllBeta)))**2) 
    c2 = (Ellc2**2) * ((np.tan(np.radians(EllBeta)))**2) / (Ella2**2)
    c3 = abs(x) / (np.cos(np.radians(EllBeta)))

    A = 1 + c2
    B = np.sign(x) * 2 * c2 * c3
    C = c2 * (c3**2) - c1
    
    k1 = (-B + np.sqrt( (B**2) - 4 * A * C ) ) / (2*A)

    Depth = k1 / np.sin(np.radians(EllBeta))   ## Towards bottom
    # Depth2 = k2 / np.sin(np.radians(EllBeta)) ## Towards up 
    
    ## Adjust the depth according to the Ellz value.
    Dz = Ellz / np.cos(np.radians(EllBeta))
    Depth = Depth - Dz

    return (Depth)

#--------------------------------------------------------------------------------- 
def calz(x,y,Ella,Ellb,Ellc):
    '''
    ## z value according to the e'' coordinate system. x, y should be according to e'' 
    
    Parameters
    ----------
    x : float
        x coordinate of the point accorging to e' coordinate system.
    y : float
        y coordinate of the point accorging to e' coordinate system.
    Ella : float
        The dimension of the ellipdoid in the direction of motion..
    Ellb : float
        The dimension of the ellipdoid perpendicular to the direction of motion..
    Ellc : float
        The dimension of the ellipdoid perpendicular to the other two directions.
   
    Returns
    -------
    "Ellc * np.sqrt( 1 - (x**2)/(Ella**2) - (y**2)/(Ellb**2))"
        z value according to the e'' coordinate system. x, y should be according to e'' . 

    ''' 
       
    return(Ellc * np.sqrt( 1 - (x**2)/(Ella**2) - (y**2)/(Ellb**2)))

#---------------------------------------------------------------------------------  
def HydrologyModel(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput):
    '''
    ####################################
    ## Iverson (2000), infiltraion model 
    ####################################
    ## When calculating the pore water pressure, "Thickness" is the depth of sliding. 
    ## The depth of ground water depth and thickness will be used to calculate the pressure by assuming slope parallel water flow. 
    ## For now, the time variable for the pore water pressure values, TimeToAnalyse, should be between the start and end of the rainfall. 
    ## The formulation can be simply modified for the times greater than the rainfall duration. 

    Parameters
    ----------
    TimeToAnalyse : int
        The time of analysis (in the range of rainfall).
    CellsInside : list
        The cells inside the ellipsoidal sliding zone. 
    A : array of floats
        Sliding base area.
    HwInput : array
        Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
        the rainfall data.
    Ksat : float
        Saturated permeability.
    Diff0 : float
        Diffusivity.
    Thickness : array of floats
        Thickness of the cells inside the sliding zone.
    SlopeInput : Array
        Slope angle.

    Returns
    -------
    PoreWaterForce
         Pore water pressure for each cell base. 

    '''
           
    ## Depth of ground water table for the cells inside sliding zone
    HwValuesInside = HwInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))]
    HwValuesInside = np.reshape(HwValuesInside, (np.shape(CellsInside)[0],1))
    
    ## Steady, background infiltration for the cells inside sliding zone
    rizeroInside =  rizeroInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))]
    rizeroInside = np.reshape(rizeroInside, (np.shape(CellsInside)[0],1))
    
    ## Arrange saturated permeability and diffusivity parameters for the calculations
    KsatInside  = np.ones((np.shape(CellsInside)[0],1)) * Ksat
    Diff0inside = np.ones((np.shape(CellsInside)[0],1)) * Diff0
    
    ## Slope angles for the cells inside sliding zone
    SlopeInside = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))]
    SlopeInside = np.reshape(SlopeInside, (np.shape(CellsInside)[0],1))
    
    ## Arrange the rainfall input data (m/sec)
    riInside    = np.ones((np.shape(CellsInside)[0],1)) * riInp[0]
     
    ## The parameter Beta in Iverson's formulation (See Iverson (2000))
    BetaIverson = np.square(np.cos(np.radians(SlopeInside[:]))) - rizeroInside / KsatInside
    
    ## Currently the formulation is for time lower or equal the time of the rainfall. 
    ## It can be simply modified to analyse times after rainfall. 
    if (TimeToAnalyse > riInp[1][0] or TimeToAnalyse < 0 ):
        print('Check time to analse! If you want to assign time greater than the storm, modify the equations!')
        print('Now, the time is assigned as zero')
        TimeToAnalyse = 0
        
    if (TimeToAnalyse == 0):
        SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
        TransientP = np.zeros((np.shape(CellsInside)[0],1))                 ## unit is m
        
    else:
        SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
        DiffIverson = np.multiply( (4 * Diff0inside), np.square(np.cos(np.radians(SlopeInside[:]))))
        t_star = (TimeToAnalyse * DiffIverson) / np.square(Thickness)
        Response = np.multiply(np.sqrt(t_star/np.pi), np.exp(-1 / t_star)) - special.erfc(1/np.sqrt(t_star))
        TransientP = np.multiply( (np.multiply((riInside / KsatInside), Thickness)), Response)  ## unit is m
     
    ## Pore water pressure is the sum of steady and transient pressure heads     
    PoreWaterPressure =  (SteadyP + TransientP) * 10 ## 10 is the unit weight of water
     
    ## The values are limited by values assuming saturated soil with slope parallel flow.
    MaxWaterPressure =  np.multiply(Thickness, BetaIverson) * 10 ## 10 is the unit weight of water
    ## Limit the pore pressures values by Z*BetaIverson
    # for i in range(np.shape(CellsInside)[0]):
    #     if (PoreWaterPressure[i] > MaxWaterPressure[i]):
    #          PoreWaterPressure[i] = MaxWaterPressure[i]
    PoreWaterPressure = np.where(PoreWaterPressure>MaxWaterPressure,MaxWaterPressure,PoreWaterPressure)    
        
    ## Calculate pore water pressure
    PoreWaterForce = np.multiply(PoreWaterPressure, A)
        
    return(PoreWaterForce)

#---------------------------------------------------------------------------------
def FSNormal3D(c, phi, A, Weight, PoreWaterForce, Theta, ThetaAvr, AngleTangentXZE1 ):
    '''
    ## This function calculates the factor of safety by normal formulation.

    Parameters
    ----------
    c : array of float
        Random field data of cohesion / undrained shear strength.
    phi : array of float.
        Random field data of friction angle. 
    A : array of floats
        Sliding base area.
    Weight : array of floats
        Weight of the soil columns inside the sliding zone.
    PoreWaterForce : TYarray of floatsPE
        Pore water pressure for each cell base..
    Theta : array of floats
        Angle between 3D normal vector and Z' axis: Dip angle (AngleNormal3DWE1).
    ThetaAvr : array of floats
        he apparent dip of the main direction of inclination of the sliding surface: ThetaAvr.
    AngleTangentXZE1 : array of floats
        Angle between tangent and x' axis (angle - EllBeta) on XZ' plane.
        Note: ThetaAvr is absolute of AngleTangentXZE1. To consider the counter weigts sign is needed.

    Returns
    -------
    FS3D
        3D factor of safety of the ellipsoidal sliding surface.

    '''

    ## Allocate
    ResForce = []
    DriForce = []    
    
    ## Calculation of resistance and driving forces 
    for i in range(np.shape(A)[0]):
        ResForce.append((c[i]*A[i]+(Weight[i]*np.cos(np.radians(Theta[i]))-PoreWaterForce[i])*np.tan(np.radians(phi[i])))* np.cos(np.radians(ThetaAvr[i])))
        DriForce.append( np.sign(AngleTangentXZE1[i]) *  Weight[i] * np.sin(np.radians(ThetaAvr[i])) * np.cos(np.radians(ThetaAvr[i]))) 
    
    ## Adjust resistence and driving forces
    ResForce = np.asarray(ResForce)
    ResForce = abs(ResForce)
    DriForce = np.asarray(DriForce)
    ## Calculate FS
    FS3D = np.sum(ResForce) / np.sum(DriForce) 
    #print(FS3D)
    
    return (FS3D)

#---------------------------------------------------------------------------------
def FSBishop3D(x, *DataFunction): 
    '''
    ## This function calculates the factor of safety by utilizing Bishop 3D mehod. 
    ## Bishop 3D
    
    Parameters
    ----------
    x : float
        Initial guess.
    *DataFunction : list of parameters
        args=(c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1).

    Returns
    -------
    Difference : float
        The difference in the calculated FS and previous guessed FS.

    '''
    
    ## Arrange the parameters
    c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1 = DataFunction

    ## Arrange the values
    ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
    Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
    A = np.reshape(A, (np.shape(CellsInside)[0],1))
    AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))

    ## Allocate
    NValues = np.zeros((np.shape(CellsInside)[0],1))
    part1 =  np.zeros((np.shape(CellsInside)[0],1))
    part2 =  np.zeros((np.shape(CellsInside)[0],1))
    
    ## Initial guess
    FSGuess = x
    
    ## Calculations for Bishop 3D method
    for i in range(np.shape(CellsInside)[0]):        
        NValues[i] = ( Weight[i] + (1/FSGuess)*PoreWaterForce[i]*np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])) - (1/FSGuess)*c[i]*A[i]* np.sin(np.radians(ThetaAvr[i])) ) / ( np.cos(np.radians(Theta[i])) +  (1/FSGuess)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))
        # NValues[i] = (0 if NValues[i] < 0 else NValues[i] )
        part1[i] = np.sign(AngleTangentXZE1[i]) * Weight[i] * np.sin(np.radians(ThetaAvr[i]))
        # part1[i] = (NValues[i] - PoreWaterForce[i]) *  np.tan(np.radians(phi)) * (1/FSGuess) + c*A[i]* (1/FSGuess) 
        part2[i] = ((Weight[i]- PoreWaterForce[i]*np.cos(np.radians(Theta[i])))*np.tan(np.radians(phi[i])) + c[i]*A[i]* np.cos(np.radians(Theta[i]))) / ( np.cos(np.radians(Theta[i])) +  (1/FSGuess)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))
    
    ## Calculate FC
    FSCalc = ((np.sum(part1))**(-1)) * np.sum(part2)
    
    ## Calculate the difference 
    Difference = FSCalc - FSGuess

    ## To see the part 1 and part 2 if needed
    # NValuesBishop =  np.zeros((np.shape(CellsInside)[0],1))
    # part1Bishop   =  np.zeros((np.shape(CellsInside)[0],1))
    # part2Bishop   =  np.zeros((np.shape(CellsInside)[0],1))
    # for i in range(np.shape(CellsInside)[0]): 
    #     NValuesBishop[i] = ( Weight[i] + (1/FS3D)*PoreWaterForce[i]*np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])) - (1/FS3D)*c[i]*A[i]* np.sin(np.radians(ThetaAvr[i])) ) / ( np.cos(np.radians(Theta[i])) +  (1/FS3D)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))
    #     part1Bishop[i] = np.sign(AngleTangentXZE1[i]) * Weight[i] * np.sin(np.radians(ThetaAvr[i]))        
    #     part2Bishop[i] = ((Weight[i]- PoreWaterForce[i]*np.cos(np.radians(Theta[i])))*np.tan(np.radians(phi[i])) + c[i]*A[i]* np.cos(np.radians(Theta[i]))) / ( np.cos(np.radians(Theta[i])) +  (1/FS3D)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))  
    # #FSBishopData.append((cellsize,np.shape(CellsInside)[0], FS3D, np.sum(part2Bishop) ,np.sum(part1Bishop) ))
    # #FSBishopDataRes.append(( CellsInside[:,0], CellsInside[:,2], part2Bishop))
    # #FSBishopDataDri.append(( CellsInside[:,0], CellsInside[:,1], part1Bishop))
    # #FSBishopData   = np.asarray(FSBishopData) 
    
    return (Difference)

#---------------------------------------------------------------------------------

def FSJanbu3D(x, *DataFunction): 
    '''
    ## This function calculates the factor of safety by utilizing Janbu 3D mehod. 
    ## With new formulation (Janbu old paper)

    Parameters
    ----------
    x : float
        Initial guess.
    *DataFunction : list of parameters
        args=(c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1).

    Returns
    -------
    Difference : float
        The difference in the calculated FS and previous guessed FS.

    '''
    
    ## Arrange the parameters
    c, phi, CellsInside, Weight, PoreWaterForce,ThetaAvr, Theta, A,AngleTangentXZE1 = DataFunction

    ## Arrange the values
    ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
    Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
    A = np.reshape(A, (np.shape(CellsInside)[0],1))
    #Asp = np.reshape(Asp, (np.shape(CellsInside)[0],1))
    AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))

    ## Allocate
    NValues = np.zeros((np.shape(CellsInside)[0],1))
    ResCal  = np.zeros((np.shape(CellsInside)[0],1))
    Drical  = np.zeros((np.shape(CellsInside)[0],1))
    
    ## Initial guess
    FSGuess = x
    
    ## Calculations for Janbu 3D method
    for i in range(np.shape(CellsInside)[0]):        
        NValues[i] = ( Weight[i] + (1/FSGuess)*PoreWaterForce[i]*np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])) - (1/FSGuess)*c[i]*A[i]* np.sin(np.radians(ThetaAvr[i])) ) / ( np.cos(np.radians(Theta[i])) +  (1/FSGuess)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))
        # NValues[i] = (0 if NValues[i] < 0 else NValues[i] )
        ResCal[i] = (c[i]*A[i]+ (NValues[i] - PoreWaterForce[i] ) *  np.tan(np.radians(phi[i]))) *  np.cos(np.radians(ThetaAvr[i])) 
        Drical[i] = (np.sign(AngleTangentXZE1[i]) * (NValues[i]*np.cos(np.radians(Theta[i]))) * np.tan(np.radians(ThetaAvr[i])))
        # Drical[i] = ((NValues[i]*np.cos(np.radians(Theta[i]))) * np.tan(np.radians(ThetaAvr[i])))

    ## Calculate FC
    FSCalc = np.sum(ResCal) / np.sum(Drical)

    ## Calculate the difference 
    Difference = FSCalc - FSGuess 
    
    ## To see the Drical and ResCal if needed
    # #FSJanbuData   = []
    # NValuesJanbu = np.zeros((np.shape(CellsInside)[0],1))
    # ResCalJanbu  = np.zeros((np.shape(CellsInside)[0],1))
    # DricalJanbu  = np.zeros((np.shape(CellsInside)[0],1))
    # for i in range(np.shape(CellsInside)[0]): 
    #     NValuesJanbu[i] = ( Weight[i] + (1/FS3D)*PoreWaterForce[i]*np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])) - (1/FS3D)*c[i]*A[i]* np.sin(np.radians(ThetaAvr[i])) ) / ( np.cos(np.radians(Theta[i])) +  (1/FS3D)* np.tan(np.radians(phi[i])) * np.sin(np.radians(ThetaAvr[i])))
    #     ResCalJanbu[i] = (c[i]*A[i]+ (NValuesJanbu[i] - PoreWaterForce[i] ) *  np.tan(np.radians(phi[i]))) *  np.cos(np.radians(ThetaAvr[i])) 
    #     DricalJanbu[i] = (np.sign(AngleTangentXZE1[i]) * (NValuesJanbu[i]*np.cos(np.radians(Theta[i]))) * np.tan(np.radians(ThetaAvr[i])))    
    # #FSJanbuData.append((cellsize,np.shape(CellsInside)[0], FS3D, np.sum(ResCalJanbu) ,np.sum(DricalJanbu) ))
    # #FSJanbuDataRes.append(( CellsInside[:,0], CellsInside[:,2], ResCalJanbu))
    # #FSJanbuDataDri.append(( CellsInside[:,0], CellsInside[:,1], DricalJanbu))
    # #FSJanbuData    = np.asarray(FSJanbuData)
       
    return (Difference)

