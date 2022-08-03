"""
#####
## 3DPLS version 1.0
## Functions code
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

# Libraries
import numpy as np
import os
import scipy.linalg as sl
from scipy.optimize import root
from scipy import special
from multiprocessing import Process, Queue, Array, Manager    ## For processes
from queue import Queue as Queue_Threads ## For threads
from threading import Thread,Lock        ## For threads
import time
import matplotlib.pyplot as plt
from scipy.stats.mstats import hmean
from scipy.interpolate import griddata
# import rasterio
# from rasterio.plot import show
# import geopandas as gpd
# from rasterstats import zonal_stats
# import matplotlib.colors


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
    ## Currently the arrangement is for example Case Study: Kvam Landslides (Oguz et al. 2022). 

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
    ZoneInput[ZoneInput == 17 ] = 1     # Zone 2: Shallow moraine (Still accepted Moraine in this case)
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
def InZone_Rec_to_List(InZone):
    """
    This function returns the lists of cells (row, column) inside a zone of interest.
    Parameters
    ----------
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
    
    Returns
    -------
    InZone_List: List of lists
        List of cells (row,column)
    """
    ## Convert the zone to lists of cells (row, column)
    InZone_List = []
    for i in range (InZone[0,0],InZone[0,1]+1):      
        for j in range (InZone[1,0],InZone[1,1]+1):
            InZone_List.append([i,j])
    return(InZone_List)

#---------------------------------------------------------------------------------
def RFR(nrows, ncols, cellsize,Maxrix_Directory, CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N', SaveMat = 'NO'):
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
        os.chdir(Maxrix_Directory)
        if os.path.isfile(Name):
            CorrMat = np.load(Name)
        else:
            CorrMat = np.identity(nel)

    else:
        ## It check the Matrix folder for the corr. matrix. If there is no, it creates. 
        Name = 'Cell_%d_%d_Size_%d' %(nrows,ncols,cellsize)+'_Theta_%d_%d.npy'%(CorrLengthX, CorrLengthY)
        os.chdir(Maxrix_Directory)
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
     
    # ParInp = RFR(30, 30, 5,Code_Directory, 30, 30, 5, 0.1,'N', 'NO')
    # ## Check the samples 
    # ## To see the histogram og two diferent distribution
    # plt.hist(ParInp, bins=30,density=True, facecolor='g', histtype= 'barstacked')
    # plt.show()
    # np.sum(np.where(ParInp<0,1,0))
    # print (np.mean(ParInp), np.std(ParInp)/np.mean(ParInp))
    
    return(ParInp)

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
    # U       = np.random.normal(0,1,(ncols,nrows)) ## If you generate U in this way, you cannot compare with normal RFR.
    U       = np.random.normal(0,1,(nrows,ncols)).T 
    
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

    # ParInp = StepwiseRFR(135, 91, 10,0,0, 6, 0.3, 'LN')   #(nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N')
    # # Check the samples 
    # # To see the histogram og two diferent distribution
    # plt.hist(ParInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
    # plt.show()
    # np.sum(np.where(ParInp<0,1,0))

    return(ParInp)

#---------------------------------------------------------------------------------
def StepwiseRFRv2(nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV,Maxrix_Directory, DistType = 'N',SaveMat = 'NO',*args): ## OK 
    '''
     ## Stepwise random field realization are created.
     ## This decreases the time significantly while it has one drawback of assuming separable correlation coefficient function.
     ## In v2, the code checks for the matrix data from a given folder. 
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
    Maxrix_Directory = str
        The folder for the matrix
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

    ## Allocate
    CorrMatX = np.zeros((ncols,ncols))
    CorrMatY = np.zeros((nrows,nrows))

    NameX = 'Cell_%d_%d_Size_%d' %(nrows,ncols,cellsize)+'_Theta_%d_%d_X_Dir.npy'%(CorrLengthX, CorrLengthY)         
    NameY = 'Cell_%d_%d_Size_%d' %(nrows,ncols,cellsize)+'_Theta_%d_%d_Y_Dir.npy'%(CorrLengthX, CorrLengthY)         
       
    # t1 = time.time() ## Time (if needed)   
    
    if (CorrLengthX == 0 or  CorrLengthY == 0):
        os.chdir(Maxrix_Directory)
        if (os.path.isfile(NameX) and os.path.isfile(NameY)):
            CorrMatX = np.load(NameX)
            CorrMatY = np.load(NameY)
        else:
            CorrMatX = np.identity(ncols)
            CorrMatY = np.identity(nrows)     
        
    else:  
        ## Correlation matrix for X and Y
        os.chdir(Maxrix_Directory)
        if (os.path.isfile(NameX) and os.path.isfile(NameY)):
            CorrMatX = np.load(NameX)
            CorrMatY = np.load(NameY)
        else:   
            for i in range(ncols):
                for j in range(i,ncols):
                    CorrMatX[i,j] = np.exp(-2.0*np.sqrt((x[i]-x[j])**2/CorrLengthX**2))
                    CorrMatX[j,i] = CorrMatX[i,j]        
            for i in range(nrows):
                for j in range(i,nrows):
                    CorrMatY[i,j] = np.exp(-2.0*np.sqrt((y[i]-y[j])**2/CorrLengthY**2))
                    CorrMatY[j,i] = CorrMatY[i,j]
     
    ## Save the matrix for further use.
    if (SaveMat == 'YES'):
        os.chdir(Maxrix_Directory)
        if (os.path.isfile(NameX) == False and os.path.isfile(NameY) == False):
            np.save(NameX,CorrMatX)
            np.save(NameY,CorrMatY)
            # print("Correlation matrix is generated for %d_%d for both X and Y directions" %(CorrLengthX, CorrLengthY)) 
    
    # t2 = time.time() 
     
    ## Cholesky decomposition and generation of random field
    # U       = np.random.normal(0,1,(ncols,nrows))
    # np.random.seed(args[0][0]) ### For testing (Shared array change)
    U       = np.random.normal(0,1,(nrows,ncols)).T ## This is correct
    Ax      = sl.cholesky(CorrMatX,lower=True)
    Ay      = sl.cholesky(CorrMatY,lower=True)    
    
    ## U vector can be defined in two ways: (ncols, nrows) or (nrows, ncols)
    # cInp =(MuC + (COVC*MuC)*np.matmul(np.matmul(Ay,U.T),np.transpose(Ax)))    
    # cInp =(MuC + (COVC*MuC)*np.matmul(np.matmul(Ax,U),np.transpose(Ay))).T

    # t3 = time.time() 

    # print('Correlation matrix: %f'%(t2-t1))
    # print('Decomposition: %f'%(t3-t2))
    
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

    # nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 1592, 2545, 10,5000,5000, 6, 0.3, 'LN'
    # SaveMat = 'YES'
    ##           (nrows, ncols, cellsize,CorrLengthX,CorrLengthY, ParMean, ParCoV, DistType = 'N')
    # ParInp = StepwiseRFRv2(1592, 2545, 10,5000,5000, 8, 0.3, 'N', 'YES')   
    ## Check the samples 
    # To see the histogram og two diferent distribution
    # plt.hist(ParInp, bins=50,density=True, facecolor='g', histtype= 'barstacked')
    # plt.show()
    # np.sum(np.where(ParInp<0,1,0))

    # t2 = time.time() ## Time after the analyses
    # print("%f. sec " %(t2-t1)) ## Printing the total time passed. 
    
    return(ParInp)

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
def EllDepth(x,y,EllCenterX,EllCenterY,Ella,Ellb,Ellc,Ellz,EllAlpha,EllBeta):
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
    ## Calculate z value according to the e'' coordinate system. x, y should be according to e'' 
    
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
    
    return(Ellc * np.sqrt( 1 - (x**2)/(Ella**2) - (y**2)/(Ellb**2))) ## directly calculated

#---------------------------------------------------------------------------------  
def HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput):
    '''
    In v1_0, Ksat, Diff0 can be variable. I changed only the KsatInside and Diff0inside.
    It provides results at given time instances. 
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
    # KsatInside  = np.ones((np.shape(CellsInside)[0],1)) * Ksat
    KsatInside = Ksat[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] ## v1_0
    KsatInside = np.reshape(KsatInside, (np.shape(CellsInside)[0],1))              ## v1_0
    # Diff0inside = np.ones((np.shape(CellsInside)[0],1)) * Diff0
    Diff0inside = Diff0[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] ## v1_0
    Diff0inside = np.reshape(Diff0inside, (np.shape(CellsInside)[0],1))              ## v1_0
    
    ## Slope angles for the cells inside sliding zone
    SlopeInside = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))]
    SlopeInside = np.reshape(SlopeInside, (np.shape(CellsInside)[0],1))
    
    ## Arrange the rainfall input data (m/sec)
    riInside    = np.ones((np.shape(CellsInside)[0],1)) * riInp[0]
     
    ## The parameter Beta in Iverson's formulation (See Iverson (2000))
    BetaIverson = np.square(np.cos(np.radians(SlopeInside[:]))) - rizeroInside / KsatInside
    
    '''
    ## Old version for single time instance
    '''
    
    # ## Currently the formulation is for time lower or equal the time of the rainfall. 
    # ## It can be simply modified to analyse times after rainfall. 
    # if (TimeToAnalyse > riInp[1][0] or TimeToAnalyse < 0 ):
    #     print('Check time to analse! If you want to assign time greater than the storm, modify the equations!')
    #     print('Now, the time is assigned as zero')
    #     TimeToAnalyse = 0
        
    # if (TimeToAnalyse == 0):
    #     SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
    #     TransientP = np.zeros((np.shape(CellsInside)[0],1))                 ## unit is m
        
    # else:
    #     SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
    #     DiffIverson = np.multiply( (4 * Diff0inside), np.square(np.cos(np.radians(SlopeInside[:]))))
    #     t_star = (TimeToAnalyse * DiffIverson) / np.square(Thickness)
    #     Response = np.multiply(np.sqrt(t_star/np.pi), np.exp(-1 / t_star)) - special.erfc(1/np.sqrt(t_star))
    #     TransientP = np.multiply( (np.multiply((riInside / KsatInside), Thickness)), Response)  ## unit is m
     
    # ## Pore water pressure is the sum of steady and transient pressure heads     
    # PoreWaterPressure =  (SteadyP + TransientP) * 10 ## 10 is the unit weight of water
     
    # ## The values are limited by values assuming saturated soil with slope parallel flow.
    # MaxWaterPressure =  np.multiply(Thickness, BetaIverson) * 10 ## 10 is the unit weight of water
    # ## Limit the pore pressures values by Z*BetaIverson
    # # for i in range(np.shape(CellsInside)[0]):
    # #     if (PoreWaterPressure[i] > MaxWaterPressure[i]):
    # #          PoreWaterPressure[i] = MaxWaterPressure[i]
    # PoreWaterPressure = np.where(PoreWaterPressure>MaxWaterPressure,MaxWaterPressure,PoreWaterPressure)    
        
    # ## Calculate pore water pressure
    # PoreWaterForce = np.multiply(PoreWaterPressure, A)
    
    
    '''
    ## New version for multiple time instances
    '''
    
    PoreWaterForce = []
    for TimeCurrent in TimeToAnalyse:
        # print(TimeCurrent)
        ## Currently the formulation is for time lower or equal the time of the rainfall. 
        ## It can be simply modified to analyse times after rainfall. 
        if (TimeCurrent > riInp[1][0] or TimeCurrent < 0 ):
            print('Check time to analse! If you want to assign time greater than the storm, modify the equations!')
            print('Now, the time is assigned as zero')
            # TimeToAnalyse = 0
            
        if (TimeCurrent == 0):
            SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
            TransientP = np.zeros((np.shape(CellsInside)[0],1))                 ## unit is m
            
        else:
            SteadyP =  np.multiply((Thickness - HwValuesInside), BetaIverson)   ## unit is m
            DiffIverson = np.multiply( (4 * Diff0inside), np.square(np.cos(np.radians(SlopeInside[:]))))
            t_star = (TimeCurrent * DiffIverson) / np.square(Thickness)
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
        PoreWaterForce.append(np.multiply(PoreWaterPressure, A))
    
    return(PoreWaterForce)


#---------------------------------------------------------------------------------  
def HydrologyModel_v1_0_SingleTime(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput):
    '''
    In v1_0, Ksat, Diff0 can be variable. I changed only the KsatInside and Diff0inside.
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
    # KsatInside  = np.ones((np.shape(CellsInside)[0],1)) * Ksat
    KsatInside = Ksat[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] ## v1_0
    KsatInside = np.reshape(KsatInside, (np.shape(CellsInside)[0],1))              ## v1_0
    # Diff0inside = np.ones((np.shape(CellsInside)[0],1)) * Diff0
    Diff0inside = Diff0[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))] ## v1_0
    Diff0inside = np.reshape(Diff0inside, (np.shape(CellsInside)[0],1))              ## v1_0
    
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
def FSNormal3D(c, phi, A, Weight, PoreWaterForce, Theta, ThetaAvr, AngleTangentXZE1):
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
    
    '''
    ## Old version for single time instance
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
    
    '''
    ## New version for multiple time instances
    '''
    
    # FS3D = []
    # for PoreWaterForceCurrent in PoreWaterForce:
        
    #     ## Allocate
    #     ResForce = []
    #     DriForce = []    
        
    #     ## Calculation of resistance and driving forces 
    #     for i in range(np.shape(A)[0]):
    #         ResForce.append((c[i]*A[i]+(Weight[i]*np.cos(np.radians(Theta[i]))-PoreWaterForce[i])*np.tan(np.radians(phi[i])))* np.cos(np.radians(ThetaAvr[i])))
    #         DriForce.append( np.sign(AngleTangentXZE1[i]) *  Weight[i] * np.sin(np.radians(ThetaAvr[i])) * np.cos(np.radians(ThetaAvr[i]))) 
        
    #     ## Adjust resistence and driving forces
    #     ResForce = np.asarray(ResForce)
    #     ResForce = abs(ResForce)
    #     DriForce = np.asarray(DriForce)
    #     ## Calculate FS
    #     FS3D_current = np.sum(ResForce) / np.sum(DriForce) 
    #     #print(FS3D_current)
    #     FS3D.append(FS3D_current)
    
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

#---------------------------------------------------------------------------------
def Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                               Parameter_CorrLenX, Parameter_CorrLenY, \
                               nrows, ncols, cellsize, \
                               RanFieldMethod, ZoneInput, Maxrix_Directory, NoData = -9999, SaveMat = 'NO', *args):
    """
    This function generated random fields for given parameters for multiple soil types. 

    Parameters
    ----------
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    cellsize : int
        Cell size.    
    RanFieldMethod : str
        The method for the random field generation, 'CMD' - 'SCMD'.
    ZoneInput : Array
         Zones.
    Maxrix_Directory = str
        The folder for the matrix
    NoData : int, optional
        The value of No Data. The default is -9999.
    SaveMat : str, optional
        Whether the mayrix is saved or not. The default is 'NO'.
    *args : list of number (not needed)
        Used to generate same random fields by aranging seed number.

    Returns
    -------
    Parameter_Fields: Array
        Ranfom fields over study area for given model parameters.
    
    """
    
    ## Zone numbers over the study area
    ZoneNumber = list(set(np.asarray(ZoneInput).ravel()))
    if any(elem==NoData for elem in ZoneNumber):
        ZoneNumber.remove(NoData)
    ZoneNumber = np.asarray(ZoneNumber)
    
    #################################
    ## Generate parameter fields
    ## In the current version, the random field is generated for the whole study area for each zone and then clipped.
    #################################
    
    ## Allocate
    Parameter_Fields = np.ones((np.shape(Parameter_CoVs)[0],nrows,ncols)) * NoData
    
    ## Generate ranfom field for each parameter individually
    ## {c, phi, uws, ksat, diffus} for drained analysis
    ## {Su, uws} for drained analysis
    for i in range(np.shape(Parameter_CoVs)[0]): 
    
        ## For the current parameter, define which zone is "NotVariable", "HomVariable", or "SpaVariable"
        ## "NotVariable": CoV is zero
        ## "HomVariable": CoV is not zero and correlation length is infinite
        ## "SpaVariable": CoV is not zoro and correlation length is not infinite
        ZoneInd_NotVariable = np.where(Parameter_CoVs[i]==0)[1]
        ZoneInd_HomVariable = np.where( ( (Parameter_CoVs[i]!=0) & ((Parameter_CorrLenX[i]=='inf') | (Parameter_CorrLenY[i]=='inf'))))[1]
        ZoneInd_SpaVariable = np.where( ( (Parameter_CoVs[i]!=0) & ((Parameter_CorrLenX[i]!='inf') & (Parameter_CorrLenY[i]!='inf'))))[1]
        
        ## If there is inconsistency in the zone count, print error! 
        if (np.shape(ZoneInd_NotVariable)[0]+np.shape(ZoneInd_HomVariable)[0]+np.shape(ZoneInd_SpaVariable)[0] != np.shape(ZoneNumber)[0]):
            print("Problem in Par_Fields function!") 
        
        ## Assign parmeter values for "NotVariable" zones
        for CurrentZoneIndex in ZoneInd_NotVariable:
            # print(CurrentZoneIndex)
            CurrentZoneNumber = ZoneNumber[CurrentZoneIndex]
            ## Assign constant parameter value to the corresponding cells 
            Parameter_Fields[i][ZoneInput == CurrentZoneNumber] = Parameter_Means[i,0,CurrentZoneIndex]
        
        ## Assign parmeter values for "HomVariable" zones
        for CurrentZoneIndex in ZoneInd_HomVariable:
            # print(CurrentZoneIndex)
            CurrentZoneNumber = ZoneNumber[CurrentZoneIndex]
            
            if (Parameter_Dist[i,0,CurrentZoneIndex] == 'N'):        
                ## Draw samples from a normal distribution.
                # np.random.seed(2021+args[0]) ### For testing (Shared array change)
                RandomValue = np.random.normal( Parameter_Means[i,0,CurrentZoneIndex] ,  Parameter_Means[i,0,CurrentZoneIndex] *  Parameter_CoVs[i,0,CurrentZoneIndex] )
                                    
            elif (Parameter_Dist[i,0,CurrentZoneIndex] == 'LN'):
                # Parameters of the underlying normal distribution        
                SigLnPar = np.sqrt( np.log( 1+ Parameter_CoVs[i,0,CurrentZoneIndex]**2 ) )
                MuLnPar  = np.log( Parameter_Means[i,0,CurrentZoneIndex] ) - 0.5*SigLnPar**2
                ## Draw samples from a log-normal distribution.
                # np.random.seed(2021+args[0]) ### For testing (Shared array change)
                RandomValue  = np.random.lognormal( MuLnPar ,  SigLnPar)
            
            ## Assign variable (but homogeneous) parameter value to the corresponding cells 
            Parameter_Fields[i][ZoneInput == CurrentZoneNumber] = RandomValue
            
            # ## To see the histogram and statistical moments
            # ## Normal dist
            # RandomValue  = np.random.normal( Parameter_Means[i,0,CurrentZoneIndex] ,  Parameter_Means[i,0,CurrentZoneIndex] *  Parameter_CoVs[i,0,CurrentZoneIndex]  ,2000)
            # ## Lognormal Dist
            # RandomValue  = np.random.lognormal( np.log( Parameter_Means[i,0,CurrentZoneIndex] ) - 0.5*(np.sqrt( np.log( 1+ Parameter_CoVs[i,0,CurrentZoneIndex]**2 ) ))**2 ,  np.sqrt( np.log( 1+ Parameter_CoVs[i,0,CurrentZoneIndex]**2 ) ) , 10000 )
            # ## To see the histogram og two diferent distribution
            # plt.hist(RandomValue, bins=50,density=True, facecolor='g', histtype= 'barstacked')
            # plt.show()
            # print('Mean and CoV are %.3f and %.3f'%(np.mean(RandomValue), np.std(RandomValue) / np.mean(RandomValue) ))
        
        ## Assign parmeter values for "SpaVariable" zones
        for CurrentZoneIndex in ZoneInd_SpaVariable:
            # print(CurrentZoneIndex)
            CurrentZoneNumber = ZoneNumber[CurrentZoneIndex]
            
            if (RanFieldMethod == 'SCMD'):
                FieldData = StepwiseRFRv2(nrows, ncols, cellsize,\
                                          int(Parameter_CorrLenX[i,0,CurrentZoneIndex]),int(Parameter_CorrLenY[i,0,CurrentZoneIndex]), \
                                          Parameter_Means[i,0,CurrentZoneIndex], Parameter_CoVs[i,0,CurrentZoneIndex], \
                                          Maxrix_Directory, Parameter_Dist[i,0,CurrentZoneIndex], SaveMat,args) 
            elif (RanFieldMethod == 'CMD'):
                FieldData = RFR(nrows, ncols, cellsize, Maxrix_Directory, \
                                int(Parameter_CorrLenX[i,0,CurrentZoneIndex]),int(Parameter_CorrLenY[i,0,CurrentZoneIndex]), \
                                    Parameter_Means[i,0,CurrentZoneIndex], Parameter_CoVs[i,0,CurrentZoneIndex], \
                                    Parameter_Dist[i,0,CurrentZoneIndex], SaveMat)
                
            ## Assign spatially variable parameter values to the corresponding cells 
            Parameter_Fields[i][ZoneInput == CurrentZoneNumber] = FieldData[ZoneInput == CurrentZoneNumber]
    
        # ## See the plot
        # # os.chdir(Results_Directory)
        # Field=Parameter_Fields[i]
        # Field[Field==NoData]=np.nan
        # # Plotting the fields. 
        # #plt.figure(figsize = (3,3))
        # plt.figure()
        # plt.imshow(Field)
        # #plt.clim(3000 ,8000)
        # plt.colorbar() 
        # #plt.title('')
        # # plt.savefig(".png", dpi=500)
        # plt.show()

    return(Parameter_Fields) 

#---------------------------------------------------------------------------------
def Zmax_Variable(SlopeInput,CoV_Zmax,MinZmax, nrows, ncols, cellsize, NoData):
    """
    This function is used to generate a new max depth map of the study area.
    The mean is calculated by using an empirical funciton and the an assigned CoV is employed for the variability. 
    Normal distribution is assummed.
    Parameters
    ----------
    SlopeInput : Array
        Slope angle.
    CoV_Zmax : float
        Coefficient of variation of the parameter.
    MinZmax : float
        Min value of the max depth to bedrock.
    GIS_Data_Directory : str
        The folder including the input data.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    xllcorner : float
        X coordinate of the lower left corner.
    yllcorner : float
        Y coordinate of the lower left corner..
    cellsize : int
        Cell size.        
    NoData : int
        The value of No Data.

    Returns
    -------
    ZmaxInput: array
        Map of depth to bedrock over the study area. 

    """
    
    ## Equation is for Kvam case study: see Oguz et al. (2022)
    ZmaxInput_mean   =  -2.578 * np.tan(np.radians(SlopeInput)) + 2.612
    ZmaxInput_StdDev =  CoV_Zmax * ZmaxInput_mean
    
    ## Another common Zmax calculation using the formulation from “DeRose (1996), Salciarini et al. (2006) and Baum et al. (2010)”
    ## zmax = 5.0*exp(-0.04*slope)  
    # ZmaxInput_mean   = 5.0 * np.exp(-0.04*SlopeInput) 
    # ZmaxInput_StdDev = CoV_Zmax * ZmaxInput_mean
    
    ZmaxInput = ZmaxInput_mean + ZmaxInput_StdDev * np.random.normal(0,1) 
    ZmaxInput[ZmaxInput<MinZmax] = MinZmax
    ZmaxInput[SlopeInput == NoData ] = NoData
       
    return(ZmaxInput)

#---------------------------------------------------------------------------------
def FSCalcEllipsoid_v1_0_SingleRrocess(AnalysisType, FSCalType, RanFieldMethod, \
                                    InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                    nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                    ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                                    TimeToAnalyse, NoData,ProblemName = ''):
    '''
    ## This is the main function in which the calculations are done with single processor.
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
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    EllParam : array of float
        Ellipsoidal parameters.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.      
    SaveMat: str
        Whether the mayrix is saved or not.
     ZoneInput : array
         Zones.
     SlopeInput : array
         Slope angle.
     ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar, CoV_Zmax, MinZmax)
     ZmaxInput : array
         Depth to bedrock.
     DEMInput : array
         Digital elevation map data.
     HwInput : array
         Ground water table depth.
     rizeroInput : array
         Steady, background infiltration.
     riInp : array
         the rainfall data.
     AspectInput : array
         Values of aspect.
     TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
     NoData : int
         The value of No Data. 
     ProblemName : str, optional
         Name of the problem to reproduce results. 

    Returns
    -------
    '''

    ## Allocate
    # FS_All_MC = [0]*(MCnumber * nrows * ncols)
    # FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * MCnumber * nrows * ncols)
    
    ########### (Change aspect)
    ## Change aspect of NoData to np.nan 
    AspectInput[AspectInput == NoData] = np.nan 
    ###########
    
    # MC_current = 0
    for MC_current in range(MCnumber): # One MC analysis in each for loop                
        ## Print current MC number
        print("MC:%d"%(MC_current))     


        ## Generate parameter fields
        Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                       Parameter_CorrLenX, Parameter_CorrLenY, \
                                       nrows, ncols, cellsize, \
                                       RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
        
        ## In case of having variable Zmax in the probabilistic analysis
        if (ZmaxArg[0]=='YES'): ## ZmaxArg = (ZmaxVar, CoV_Zmax, MinZmax)
            ZmaxInput = Zmax_Variable(SlopeInput,ZmaxArg[1],ZmaxArg[2], nrows, ncols, cellsize, NoData)

    
        # ## Allocate FS 
        # TempLst = {i:[] for i in list(range(0,nrows*ncols))}
        FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols) ### (Shared array change)
        
        '''
        ## It is possible to generate an ellipsoidal sliding surface at any cell given. 
        ## Note: If an ellipsoidal sliding surface is truncated by the boundary of the problem domain, the results will be misleading. 
        ## Therefore, it is advised to extend the problem domain and generate sliding surfaces for the area of interest.
        '''                
        ## Elllipsoidal sliding surface is generated for a given row and column number.
        # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest    (example:range(10,108), np.arange(11,108,2)) 
        #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest 
        #         ## Current Row and Column for the ellipsoid center
        #         EllRow    = i
        #         EllColumn = j
        #         print(EllRow,EllColumn)
                
        for i in range(np.shape(InZone)[0]): ## (InZone change)
                ## Current Row and Column for the ellipsoid center
                EllRow, EllColumn = InZone[i]
                # print(EllRow,EllColumn)

                # t_start = time.time() ## Time before FS calculation for 1 ellipdoidal sliding surface         
                ## Calculate the FS for the ellipsoid
                indexes, FS, FS_All_MC = EllipsoidFSWithSubDis_v1_0_SingleProcess(FS_All_MC,MC_current, MCnumber, AnalysisType, FSCalType, SubDisNum, \
                                                            nrows, ncols, nel, cellsize, \
                                                            Parameter_Fields, \
                                                            EllParam, EllRow, EllColumn, \
                                                            SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                                                            TimeToAnalyse, NoData,ProblemName)
         
                # print(FS)                    
                # t_finish = time.time() ## Time after FS calculation for 1 ellipdoidalsliding surface
                # print(t_finish-t_start) ## Time required to analyse sliding surface
                
                # ## Assign the values to the correspoinding cells 
                # for ind in indexes:
                #     # print(ind)
                #     TempLst[ind].append(FS)
                    # templst[ind].append(np.round(FS,8))
                      
        # t2 = time.time()  ## Time after analysis for the current MC analysis number  
        # print("Terminates in %f. min " %((t2-t1)/60))

        FS_All_MC_InZone = np.asarray(FS_All_MC)
        # FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],MCnumber,nrows,ncols)) ### (Shared array change)
        FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) ### (Shared array change)

        ## Write FS for the current MC simulation
        os.chdir(Results_Directory)
        NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
        
    return()

#---------------------------------------------------------------------------------
def FSCalcEllipsoid_v1_0_MutiProcess(Multiprocessing_Option,TOTAL_PROCESSES_MC, TOTAL_PROCESSES_ELL, TOTAL_THREADS_ELL, \
                                     AnalysisType, FSCalType, RanFieldMethod, \
                                     InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                     nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                     Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                     Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                     ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                                     TimeToAnalyse, NoData,ProblemName = ''):
    '''
    ## This is the main function in which the calculations are done with multiple processor.
    ## Many other functions are called and used inside this function. 
    
    Parameters
    ----------
    Multiprocessing_Option : str
       Run optiion for the analyses: one of {"C-SP-SP","C-MP-SP","C-MP-MP","C-MP-MT","S-SP-SP","S-SP-MP","S-MP-SP","S-MP-MP"}.
    TOTAL_PROCESSES_MC: int
        Numbers of processors for Monte Carlo Simulations.
    TOTAL_PROCESSES_ELL: int
        Numbers of processors for generation of ellipsoidal sliding surfaces.
    TOTAL_THREADS_ELL: int
        Numbers of threads for generation of ellipsoidal sliding surfaces.              
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
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    EllParam : array of float
        Ellipsoidal parameters.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.      
    SaveMat: str
        Whether the mayrix is saved or not.
    ZoneInput : array
         Zones.
    SlopeInput : array
         Slope angle.
    ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar, CoV_Zmax, MinZmax)
    ZmaxInput : array
         Depth to bedrock.
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : array
         Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str, optional
         Name of the problem to reproduce results. 

    Returns
    -------

    '''

    ## Allocate array for multiprocessing 
    ## Lock is necessary because multiple processors need to access the same cell to asssign the min. 
    # FS_All_MC = Array('d', [0]*(MCnumber * nrows * ncols), lock=True)  
    # FS_All_MC = Array('d', [0]*(np.shape(TimeToAnalyse)[0] * MCnumber * nrows * ncols), lock=True)   ### (Shared array change)
    FS_All_MC =[]   ### (Shared array change)
    
    ########### (Change aspect)
    ## Change aspect of NoData to np.nan 
    AspectInput[AspectInput == NoData] = np.nan 
    ###########
    
    # manager = Manager()
    # shared_list_FS = manager.list()
    
    if __name__ == 'Functions_3DPLS_v1_0':
        queue_mc = Queue(maxsize=1000)
    
        processes_mc = []
        for i in range(TOTAL_PROCESSES_MC):
            
            ## "MCRun_v1_0_SingleProcess" performs a MC with single processor in the generation of ellipsoidal sliding surfaces. 
            ## "MCRun_v1_0_MultiProcess" utilize multiple processors in the generation of ellipsoidal sliding surfaces.
            ## "MCRun_v1_0_MultiThread" utilize multiple threads in the generation of ellipsoidal sliding surfaces.
            ## Note: "MCRun_v1_0_MultiThread" is not working well. There is a conflict in the assignment of the FS values. 
            
            ## Check Multiprocessing_option and run
            if (Multiprocessing_Option=="C-MP-SP"):
                print("Run option:", Multiprocessing_Option)
                p_mc = Process(target=MCRun_v1_0_SingleProcess, args=(queue_mc, FS_All_MC, \
                                AnalysisType, FSCalType, RanFieldMethod, \
                                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                                TimeToAnalyse, NoData,ProblemName))
            
            ## Check Multiprocessing_option and run
            if (Multiprocessing_Option=="C-MP-MP"):
                print("Run option:", Multiprocessing_Option)
                p_mc = Process(target=MCRun_v1_0_MultiProcess, args=(queue_mc, FS_All_MC,TOTAL_PROCESSES_ELL, \
                                AnalysisType, FSCalType, RanFieldMethod, \
                                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                                TimeToAnalyse, NoData,ProblemName ))
            ## Check Multiprocessing_option and run
            if (Multiprocessing_Option=="C-MP-MT"):
                print("Run option:", Multiprocessing_Option)       
                p_mc = Process(target=MCRun_v1_0_MultiThread, args=(queue_mc, FS_All_MC,TOTAL_THREADS_ELL, \
                                AnalysisType, FSCalType, RanFieldMethod, \
                                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                                TimeToAnalyse, NoData,ProblemName ))
            
            processes_mc.append(p_mc)
            p_mc.start()  
    
        for i in range(MCnumber):      
            queue_mc.put(i)
            print("queue_mc.put([,]): MC ",i)       

        for _ in range(TOTAL_PROCESSES_MC): queue_mc.put(None)
        for p_mc in processes_mc: p_mc.join()

        # return(FS_All_MC)
        return()


#---------------------------------------------------------------------------------
def MCRun_v1_0_SingleProcess(queue_mc, FS_All_MC, \
                AnalysisType, FSCalType, RanFieldMethod, \
                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                TimeToAnalyse, NoData,ProblemName = '' ):
    """
    Monte Carlo simulations with single processor. 
    This is a main function before actual function in which the calculations are performed.

    Parameters
    ----------
    queue_mc : -
        Queue of the tasks for multiprocessing.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
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
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    EllParam : array of float
        Ellipsoidal parameters.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.      
    SaveMat: str
        Whether the mayrix is saved or not.
    ZoneInput : array
         Zones.
    SlopeInput : array
         Slope angle.
    ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar
    ZmaxInput : array
         Depth to bedrock.
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str, optional
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """
    
    ## Keep the original parameters
    arg_original_mc = (AnalysisType, FSCalType, RanFieldMethod, \
                    InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                    nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                    ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                    TimeToAnalyse, NoData)
    
    queue_data_mc = queue_mc.get() 
        
    
    while queue_data_mc is not None:
        
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, RanFieldMethod, \
                        InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                        nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                        Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                        Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                        ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                        TimeToAnalyse, NoData) = arg_original_mc[:]


        MC_current = int(queue_data_mc)
        print("---->------>----->>>>>",int(MC_current))



        Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                       Parameter_CorrLenX, Parameter_CorrLenY, \
                                       nrows, ncols, cellsize, \
                                       RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
                                   
        ## In case of having variable Zmax in the probabilistic analysis
        if (ZmaxArg[0]=='YES'): ## ZmaxArg = (ZmaxVar, CoV_Zmax, MinZmax)
            ZmaxInput = Zmax_Variable(SlopeInput,ZmaxArg[1],ZmaxArg[2], nrows, ncols, cellsize, NoData)
        
        
        ## Allocate FS 
        # TempLst = {i:[] for i in list(range(0,nrows*ncols))}
        FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols) ### (Shared array change)
        
        '''
        ## It is possible to generate an ellipsoidal sliding surface at any cell given. 
        ## Note: If an ellipsoidal sliding surface is truncated by the boundary of the problem domain, the results will be misleading. 
        ## Therefore, it is advised to extend the problem domain and generate sliding surfaces for the area of interest.
        '''                
        ## Elllipsoidal sliding surface is generated for a given row and column number.
        # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest    (example:range(10,108), np.arange(11,108,2)) 
        #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest 
        #         ## Current Row and Column for the ellipsoid center
        #         EllRow    = i
        #         EllColumn = j
        #         # print(EllRow,EllColumn)
    
        for i in range(np.shape(InZone)[0]): ## (InZone change)
                ## Current Row and Column for the ellipsoid center
                EllRow, EllColumn = InZone[i]
                # print(EllRow,EllColumn)
                # t_start = time.time() ## Time before FS calculation for 1 ellipdoidal sliding surface
                
                ## Calculate the FS for the ellipsoid
                indexes, FS, FS_All_MC = EllipsoidFSWithSubDis_v1_0_SingleProcess(FS_All_MC,MC_current,MCnumber, AnalysisType, FSCalType, SubDisNum, \
                                                            nrows, ncols, nel, cellsize, \
                                                            Parameter_Fields, \
                                                            EllParam, EllRow, EllColumn, \
                                                            SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                                                            TimeToAnalyse, NoData,ProblemName)

         
                # print(FS)                    
                # t_finish = time.time() ## Time after FS calculation for 1 ellipdoidalsliding surface
                # print(t_finish-t_start) ## Time required to analyse sliding surface
                                      
        # t2 = time.time()  ## Time after analysis for the current MC analysis number  
        # print("Terminates in %f. min " %((t2-t1)/60))


        FS_All_MC_InZone = np.asarray(FS_All_MC)
        FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) ### (Shared array change)
       
        ## Write FS for the current MC simulation
        os.chdir(Results_Directory)
        NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
           
        ## Take another task from the queue
        queue_data_mc = queue_mc.get()                  
    # return()

#---------------------------------------------------------------------------------
def MCRun_v1_0_MultiProcess(queue_mc, FS_All_MC,TOTAL_PROCESSES_ELL, \
                AnalysisType, FSCalType, RanFieldMethod, \
                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                TimeToAnalyse, NoData,ProblemName = '' ):
    """
    Monte Carlo simulations with multiple processor. 
    This is a main function before actual function in which the calculations are performed.

    Parameters
    ----------

    queue_mc : -
        Queue of the tasks for multiprocessing.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    TOTAL_PROCESSES_ELL: int
        Numbers of processors for generation of ellipsoidal sliding surfaces.
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
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    EllParam : array of float
        Ellipsoidal parameters.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.      
    SaveMat: str
        Whether the mayrix is saved or not.
    ZoneInput : array
         Zones.
    SlopeInput : array
         Slope angle.
    ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar
    ZmaxInput : array
         Depth to bedrock.
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """


    ## Keep the original parameters
    arg_original_mc = (AnalysisType, FSCalType, RanFieldMethod, \
                    InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                    nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                    ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                    TimeToAnalyse, NoData)
    
    queue_data_mc = queue_mc.get() 
        
    
    while queue_data_mc is not None:
        
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, RanFieldMethod, \
                        InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                        nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                        Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                        Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                        ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                        TimeToAnalyse, NoData) = arg_original_mc[:]


        MC_current = int(queue_data_mc)
        print("---->------>----->>>>>",int(MC_current))
        
        ### To control the seed number for testing (Shared array change)
        # args = [2022] ##[MC_current]
        Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                       Parameter_CorrLenX, Parameter_CorrLenY, \
                                       nrows, ncols, cellsize, \
                                       RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
        
        ## In case of having variable Zmax in the probabilistic analysis
        if (ZmaxArg[0]=='YES'): ## ZmaxArg = (ZmaxVar, CoV_Zmax, MinZmax)
            ZmaxInput = Zmax_Variable(SlopeInput,ZmaxArg[1],ZmaxArg[2], nrows, ncols, cellsize, NoData)
            
        ## Allocate FS 
        # TempLst = {i:[] for i in list(range(0,nrows*ncols))}
        FS_All_MC = Array('d', [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols), lock=True)   ### (Shared array change)

        '''
        ## It is possible to generate an ellipsoidal sliding surface at any cell given. 
        ## Note: If an ellipsoidal sliding surface is truncated by the boundary of the problem domain, the results will be misleading. 
        ## Therefore, it is advised to extend the problem domain and generate sliding surfaces for the area of interest.
        '''  

        if __name__ == 'Functions_3DPLS_v1_0':
            
            queue_ell = Queue(maxsize=1000)
            # print(__name__)
            
            processes_ell = [] ## For processes
      
            for i in range(TOTAL_PROCESSES_ELL): ## For Processes
                p_ell = Process(target=EllipsoidFSWithSubDis_v1_0_MultiProcess, args=(queue_ell,FS_All_MC, \
                                                            AnalysisType,MCnumber, FSCalType, SubDisNum, \
                                                            nrows, ncols, nel, cellsize, \
                                                            Parameter_Fields, \
                                                            EllParam, \
                                                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                                                            rizeroInput, riInp, AspectInput, \
                                                            TimeToAnalyse, NoData,ProblemName))
                processes_ell.append(p_ell)
                p_ell.start()


            # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest
            #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest                                                       
            #         EllRow, EllColumn = i,j
            for i in range(np.shape(InZone)[0]): ## (InZone change)
                    ## Current Row and Column for the ellipsoid center
                    EllRow, EllColumn = InZone[i]
                    # print(EllRow,EllColumn)
                    queue_ell.put([EllRow,EllColumn,nrows,ncols,MC_current])
                    print("queue_ell.put([,]):",EllRow,",",EllColumn)
                    
            for _ in range(TOTAL_PROCESSES_ELL): queue_ell.put(None)
            for p_ell in processes_ell: p_ell.join() ## For Processes                           

        

        
        FS_All_MC_InZone = np.asarray(FS_All_MC)
        FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) ### (Shared array change)
       
        ## Write FS for the current MC simulation
        os.chdir(Results_Directory)
        NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
       
        ## Take another task from the queue
        queue_data_mc = queue_mc.get()                  
    # return()

#---------------------------------------------------------------------------------
def MCRun_v1_0_MultiThread(queue_mc, FS_All_MC,TOTAL_THREADS_ELL, \
                AnalysisType, FSCalType, RanFieldMethod, \
                InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                ZoneInput, SlopeInput, ZmaxArg, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                TimeToAnalyse, NoData,ProblemName = '' ):
    """
    Monte Carlo simulations with multiple threads. 
    This is a main function before actual function in which the calculations are performed.

    Parameters
    ----------
    queue_mc : -
        Queue of the tasks for multiprocessing.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    TOTAL_THREADS_ELL: int
        Numbers of threads for generation of ellipsoidal sliding surfaces.              
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
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    nel : int 
        Total number of cells.
    cellsize : int
        Cell size.
    EllParam : array of float
        Ellipsoidal parameters.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.      
    SaveMat: str
        Whether the mayrix is saved or not.
    ZoneInput : array
         Zones.
    SlopeInput : array
         Slope angle.
    ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar
    ZmaxInput : array
         Depth to bedrock.
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """
    
    ## Keep the original parameters
    arg_original_mc = (AnalysisType, FSCalType, RanFieldMethod, \
                    InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                    nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                    ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput, \
                    TimeToAnalyse, NoData)
    
    queue_data_mc = queue_mc.get() 
        
    
    while queue_data_mc is not None:
        
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, RanFieldMethod, \
                        InZone, SubDisNum, Results_Directory, Code_Directory, Maxrix_Directory, \
                        nrows, ncols, nel, cellsize, EllParam, MCnumber, \
                        Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                        Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                        ZoneInput, SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp,AspectInput, \
                        TimeToAnalyse, NoData) = arg_original_mc[:]


        MC_current = int(queue_data_mc)
        print("---->------>----->>>>>",int(MC_current))


        Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                       Parameter_CorrLenX, Parameter_CorrLenY, \
                                       nrows, ncols, cellsize, \
                                       RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
        
        ## In case of having variable Zmax in the probabilistic analysis
        if (ZmaxArg[0]=='YES'): ## ZmaxArg = (ZmaxVar, CoV_Zmax, MinZmax)
            ZmaxInput = Zmax_Variable(SlopeInput,ZmaxArg[1],ZmaxArg[2], nrows, ncols, cellsize, NoData)
        
        ## Allocate FS  ### (Shared array change)
        # FS_All_MC = Array('d', [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols), lock=True)   ### (Shared array change)
        FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols) ### (Shared array change)
        '''
        ## It is possible to generate an ellipsoidal sliding surface at any cell given. 
        ## Note: If an ellipsoidal sliding surface is truncated by the boundary of the problem domain, the results will be misleading. 
        ## Therefore, it is advised to extend the problem domain and generate sliding surfaces for the area of interest.
        '''  
                
        if __name__ == 'Functions_3DPLS_v1_0':
            
            queue_ell = Queue_Threads(maxsize=1000)
            # print(__name__)
            
            threads_ell = [] ## For processes
      
            for i in range(TOTAL_THREADS_ELL): ## For Processes
                t_ell = Thread(target=EllipsoidFSWithSubDis_v1_0_MultiThread, args=(queue_ell,FS_All_MC, \
                                                            AnalysisType,MCnumber, FSCalType, SubDisNum, \
                                                            nrows, ncols, nel, cellsize, \
                                                            Parameter_Fields, \
                                                            EllParam, \
                                                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                                                            rizeroInput, riInp,AspectInput,
                                                            TimeToAnalyse, NoData,ProblemName))
                threads_ell.append(t_ell)
                t_ell.start()


            # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest
            #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest                                                         
            #         EllRow, EllColumn = i,j
            for i in range(np.shape(InZone)[0]): ## (InZone change)
                    ## Current Row and Column for the ellipsoid center
                    EllRow, EllColumn = InZone[i]
                    # print(EllRow,EllColumn)
                    queue_ell.put([EllRow,EllColumn,nrows,ncols,MC_current])
                    print("queue_ell.put([,]):",EllRow,",",EllColumn)
                    
            for _ in range(TOTAL_THREADS_ELL): queue_ell.put(None)
            for t_ell in threads_ell: t_ell.join() ## For threads                           

        
        FS_All_MC_InZone = np.asarray(FS_All_MC)
        FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) ### (Shared array change)

        ## Write FS for the current MC simulation
        os.chdir(Results_Directory)
        NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
       
        ## Take another task from the queue
        queue_data_mc = queue_mc.get()                  
    # return()

#---------------------------------------------------------------------------------
def EllipsoidFSWithSubDis_v1_0_SingleProcess(FS_All_MC,MC_current,MCnumber,AnalysisType, FSCalType, SubDisNum, \
                        nrows, ncols, nel, cellsize, \
                        Parameter_Fields, \
                        EllParam, EllRow, EllColumn, \
                        SlopeInput, ZmaxInput, DEMInput, HwInput, rizeroInput, riInp, AspectInput,\
                        TimeToAnalyse, NoData,ProblemName = ''):
    """
    This function calculates the factor of safety by assuming an ellipsoidal shape, and used in single processor function.
   
    Parameters
    ----------
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    MC_current : int
        Current number of Monte Carlo simulation.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
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
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    EllParam : array of float
        Ellipsoidal parameters.
    EllRow : int
        Current row for the ellipsoid center.
    EllColumn : int
        Current column for the ellipsoid center.  
    SlopeInput : array
         Slope angle.
    ZmaxArg: array
         Values regarding the variability of Zmax(ZmaxVar
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    indexesOriginal: array 
        The indexes of the cells inside the current analysed sliding zone
    np.round(FS3D,5) : float
        The calculated FS value
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    
    """

    ## Original numbers of rows and cols 
    nrows_org, ncols_org = nrows, ncols
    
    # Assign the parameters
    if (AnalysisType == 'Drained'):   
        ## {c, phi, uws, ksat, diffus} for drained analysis
        c      = Parameter_Fields[0]  ## Cohesion
        phi    = Parameter_Fields[1]  ## Friction angle 
        Gamma  = Parameter_Fields[2]  ## Unit weight 
        Ksat   = Parameter_Fields[3]  ## Hydraulic conductivity
        Diff0  = Parameter_Fields[4]  ## Diffusivity
    elif (AnalysisType == 'Undrained'):
        ## {Su, uws} for drained analysis
        Su     = Parameter_Fields[0]  ## Undrained shear strength 
        Gamma  = Parameter_Fields[1]  ## Unit weight 
 

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
    Ellz        = EllParam[4]  ## Offset of the ellipsoid
    
    ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
    ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
    ## Coordinates according to the center of allipsoid, e
    CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
    
    ## Check if EllAlpha should be calculated or assigned. 
    EllAlpha_Calc = EllParam[5]
    if (EllAlpha_Calc == "Yes"):        
        ########### (Change aspect)
        ## Calculate the distance (over xy) to the center of the ellipoid
        Distance_to_Center = np.sqrt(CoorEll[:,2]**2 + CoorEll[:,3]**2)
        Cell_Inside_Circle = CoorEll[(Distance_to_Center<= Ella)]
        EllAlpha = np.nanmean(AspectInput[ Cell_Inside_Circle[:,0].astype(int), Cell_Inside_Circle[:,1].astype(int)])
        EllAlpha = EllAlpha - 90 ## 90 is the different between the values 
        ###########
    elif (EllAlpha_Calc == "No"):
        EllAlpha    = EllParam[3]
    
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
    
    # # See the current sliding suface
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
        
        ## Save previous Coordinates as old coordinates. 
        if (nrows==nrows_org):
            CoorG_old = CoorG
        else:
            CoorG_old = CoorG_new ## Ignore undefined name error here. It will not require it in the first loop.
            
        ## Halve the cellsize for this part only
        cellsize = cellsize / 2
        nrows    = nrows    * 2  ## Double the number of rows
        ncols    = ncols    * 2  ## Double the number of columns
                
        ## Cell center coordinate according to the left bottom option-1 (Global)
        ## New coordinates in the discretization
        CoorG_new =[]
        for i in range(nrows):
            for j in range(ncols):
                CoorG_new.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
        CoorG_new = np.asarray(CoorG_new) # (row #, column #, x coordinate, y coordinate)     
        
        ## Follow the index numbers
        IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
        
        ## Arrangement of the soil strength parameters according to new cellsize
        if (AnalysisType == 'Drained'):
            ## {c, phi, uws, ksat, diffus} for drained analysis
            c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
            phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
            Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
            Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
            Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
        elif (AnalysisType == 'Undrained'):
            ## {Su, uws} for drained analysis
            Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
            Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
            
        ## Arrangments with new cellsize 
        ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
        ## "numpy.kron"
        SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
        ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
        DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
        HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
        rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input
                
        ## Linear interpolation
        # ## Slope input
        # SlopeInput[SlopeInput==NoData]   = np.nan
        # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
        # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
        # ## Zmax input
        # ZmaxInput[ZmaxInput==NoData]     = np.nan
        # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
        # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
        # ## DEM input
        # DEMInput[DEMInput==NoData]       = np.nan
        # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # DEMInput = np.reshape(DEMInput, (nrows,ncols))
        # np.nan_to_num(DEMInput,copy=False, nan=NoData)
        # ## Ground water table input
        # HwInput[HwInput==NoData]         = np.nan
        # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # HwInput = np.reshape(HwInput, (nrows,ncols))
        # np.nan_to_num(HwInput,copy=False, nan=NoData)
        # ## Background infiltration rate input
        # rizeroInput[rizeroInput==NoData] = np.nan
        # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
        # np.nan_to_num(rizeroInput,copy=False, nan=NoData)
        
        
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
        ### !!!

        
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
        Cell_row, Cell_column = int(CellsInside[i,0]),int(CellsInside[i,1])
        ## Depth of the sliding surface at each cell
        DEMdiff2 = (DEMInput[Cell_row, Cell_column] - DepthDEM[i,0])  
        ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
        Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
        Weight[i]    = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
        
        
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
    
    ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))   ## Reshape 
    condzmax  = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
    
    ## Arrangement of the parameters using truncation condition above
    ## Thickness becomes ZmaxInput for the truncated cells.
    Thickness[condzmax] = ZmaxInput[indexes][condzmax]
    
    ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
    ## Area is recalculated. 
    TempA = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                        (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
    A[condzmax] = np.reshape(TempA, (np.shape(TempA)[0],1))
    
    ## Weight is recalculated.                     
    for i in condzmax:
        # print(i)
        Weight[i]    = ZmaxInput[indexes][i]  * cellsize**2 * Gamma.flatten()[indexes][i]   ## Weight
        

    # Reshape and truncate to the slope of the current cell
    ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
    Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
    AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
    
    TempSlope = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
    ThetaAvr[condzmax]         = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
    Theta[condzmax]            = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
    AngleTangentXZE1[condzmax] = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
            
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
    if (AnalysisType == 'Drained'):
        PoreWaterForce = HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
    elif (AnalysisType == 'Undrained'):
        # PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
        PoreWaterForce = [np.zeros((np.shape(CellsInside)[0],1))]*np.shape(TimeToAnalyse)[0]
    
    ## !!!
    ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
    if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):   
        for corr_n in range(np.shape(PoreWaterForce)[0]):
            ## Correction 
            PoreWaterForce[corr_n] = np.zeros((np.shape(CellsInside)[0],1)) 
        
    if (ProblemName =='Pr3S2Wet'):      
        for corr_n in range(np.shape(PoreWaterForce)[0]):
            ## Correction
            Temp = PoreWaterForce[corr_n]
            Temp[Temp<0] = 0.
            PoreWaterForce[corr_n] = Temp
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
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce:
                FS3D_current = FSNormal3D(c, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                # print( "Normal 3D FS is %.5f"%FS3D_current) 
                FS3D.append(FS3D_current)
                                 
        elif (FSCalType=='Bishop3D'):            
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce:
                FS3D_current = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                FS3D_current = FS3D_current.x[0]
                # print( "Bishop 3D FS is %.5f"%FS3D_current)  
                FS3D.append(FS3D_current)
        
        elif (FSCalType=='Janbu3D'):            
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce:
                FS3D_current = root(FSJanbu3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                FS3D_current = FS3D_current.x[0]
                # print( "Janbu 3D FS is %.5f"%FS3D_current)   
                FS3D.append(FS3D_current)
                
        
    elif (AnalysisType == 'Undrained'):
  
        ## Arrange the undrained shear strength and friction angle with corresponding indexes 
        Su = Su.flatten()[indexes]
        Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
        phi = np.zeros((np.shape(CellsInside)[0],1))
        
        ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
        if (FSCalType=='Normal3D'):  
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce:
                FS3D_current = FSNormal3D(Su, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                # print( "Normal 3D FS is %.5f"%FS3D_current)            
                FS3D.append(FS3D_current)
            
        elif (FSCalType=='Bishop3D'):     
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce: 
                FS3D_current = root(FSBishop3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                FS3D_current = FS3D_current.x[0]
                # print( "Bishop 3D FS is %.5f"%FS3D_current)   
                FS3D.append(FS3D_current)
        
        elif (FSCalType=='Janbu3D'): 
            
            FS3D = []
            for PoreWaterForceCurrent in PoreWaterForce:
                FS3D_current = root(FSJanbu3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                FS3D_current = FS3D_current.x[0]
                # print( "Janbu 3D FS is %.5f"%FS3D_current)
                FS3D.append(FS3D_current)
    
    ## Changing these values are only related to the indexing. 
    ## I changed overall shared array to the sub-shared arrays. Therefore, indexing needs this adjustment.
    MCnumber, MC_current = 1,0 ### (Shared array change) 
    
    ## Original indexes of the cells inside the sliding surface
    indexesOriginal = np.asarray(indexesOriginal,dtype=int)
    
    ## For each time instance, assign the min FS to the cells 
    for TimeInd in range(np.shape(TimeToAnalyse)[0]):
        # print(TimeInd)
        ## FS3D value at the current time instance
        FS3D_current = np.round(FS3D[TimeInd],5)

        ## Global indexes
        GlobalIndexFS   = TimeInd * nrows_org * ncols_org * MCnumber + \
                                    nrows_org * ncols_org * MC_current + \
                                    (indexesOriginal//ncols_org) * ncols_org + (indexesOriginal%ncols_org)
        GlobalIndexFS = np.asarray(GlobalIndexFS,dtype=int)
        
        ## Assign the min FS to the cells inside the sliding surface
        for i in GlobalIndexFS:
            FS_All_MC[i] = FS3D_current if ((FS_All_MC[i]==0) or (FS_All_MC[i] > FS3D_current)) else FS_All_MC[i]
            
        ## If you want to check
        # B = np.asarray(FS_All_MC)
        # C = np.reshape(B, (np.shape(TimeToAnalyse)[0],MCnumber,nrows_org,ncols_org))   
        # C0 = C[0][0]
        # C1 = C[1][0]
        # C2 = C[2][0]

    return(indexesOriginal,np.round(FS3D,5),FS_All_MC)

#---------------------------------------------------------------------------------
def EllipsoidFSWithSubDis_v1_0_MultiProcess(queue,FS_All_MC, \
                                            AnalysisType, MCnumber, FSCalType, SubDisNum, \
                                            nrows, ncols, nel, cellsize, \
                                            Parameter_Fields, \
                                            EllParam, \
                                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                                            rizeroInput, riInp, AspectInput, \
                                            TimeToAnalyse, NoData,ProblemName = ''):
    """
    This function calculates the factor of safety by assuming an ellipsoidal shape, and used in multiple processor function.

    Parameters
    ----------
    queue_mc : -
        Queue of the tasks for multiprocessing.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
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
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    EllParam : array of float
        Ellipsoidal parameters.
    SlopeInput : array
         Slope angle.
    ZmaxInput: array
        Map of depth to bedrock over the study area. 
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """

    ## Keep the original parameters
    arg_original_ell = (AnalysisType, FSCalType, SubDisNum, \
                            nrows, ncols, nel, cellsize, \
                            Parameter_Fields, \
                            EllParam, \
                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                            rizeroInput, riInp, AspectInput, \
                            TimeToAnalyse, NoData,ProblemName)
    
    queue_data = queue.get()
    
    while queue_data is not None:
        
    
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, SubDisNum, \
            nrows, ncols, nel, cellsize, \
            Parameter_Fields, \
            EllParam, \
            SlopeInput, ZmaxInput, DEMInput, HwInput, \
            rizeroInput, riInp, AspectInput, \
            TimeToAnalyse, NoData,ProblemName) = arg_original_ell[:]
        
        ## Extract the prameters from the queue
        EllRow,EllColumn = int(queue_data[0]), int(queue_data[1])
        nrows_org, ncols_org =  int(queue_data[2]), int(queue_data[3])
        MC_current = int(queue_data[4])
    
    
        
        
        # Assign the parameters
        if (AnalysisType == 'Drained'):   
            ## {c, phi, uws, ksat, diffus} for drained analysis
            c      = Parameter_Fields[0]  ## Cohesion
            phi    = Parameter_Fields[1]  ## Friction angle 
            Gamma  = Parameter_Fields[2]  ## Unit weight 
            Ksat   = Parameter_Fields[3]  ## Hydraulic conductivity
            Diff0  = Parameter_Fields[4]  ## Diffusivity
        elif (AnalysisType == 'Undrained'):
            ## {Su, uws} for drained analysis
            Su     = Parameter_Fields[0]  ## Undrained shear strength 
            Gamma  = Parameter_Fields[1]  ## Unit weight 
     
    
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
        Ellz        = EllParam[4]  ## Offset of the ellipsoid
        
        ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
        ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
        ## Coordinates according to the center of allipsoid, e
        CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
        
        ## Check if EllAlpha should be calculated or assigned. 
        EllAlpha_Calc = EllParam[5]
        if (EllAlpha_Calc == "Yes"):        
            ########### (Change aspect)
            ## Calculate the distance (over xy) to the center of the ellipoid
            Distance_to_Center = np.sqrt(CoorEll[:,2]**2 + CoorEll[:,3]**2)
            Cell_Inside_Circle = CoorEll[(Distance_to_Center<= Ella)]
            EllAlpha = np.nanmean(AspectInput[ Cell_Inside_Circle[:,0].astype(int), Cell_Inside_Circle[:,1].astype(int)])
            EllAlpha = EllAlpha - 90 ## 90 is the different between the values 
            ###########
        elif (EllAlpha_Calc == "No"):
            EllAlpha    = EllParam[3]
        
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
        
        # # See the current sliding suface
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

            ## Save previous Coordinates as old coordinates. 
            if (nrows==nrows_org):
                CoorG_old = CoorG
            else:
                CoorG_old = CoorG_new ## Ignore undefined name error here. It will not require it in the first loop.
                
            ## Halve the cellsize for this part only
            cellsize = cellsize / 2
            nrows    = nrows    * 2  ## Double the number of rows
            ncols    = ncols    * 2  ## Double the number of columns
            
            ## Cell center coordinate according to the left bottom option-1 (Global)
            CoorG_new =[]
            for i in range(nrows):
                for j in range(ncols):
                    CoorG_new.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
            CoorG_new = np.asarray(CoorG_new) # (row #, column #, x coordinate, y coordinate)     
            
            ## Follow the index numbers
            IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
            
            ## Arrangement of the soil strength parameters according to new cellsize
            if (AnalysisType == 'Drained'):
                ## {c, phi, uws, ksat, diffus} for drained analysis
                c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
                phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
                Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
            elif (AnalysisType == 'Undrained'):
                ## {Su, uws} for drained analysis
                Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                
            ## Arrangments with new cellsize 
            ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
            ## "numpy.kron"
            SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
            ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
            DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
            HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
            rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input

            # ## Linear interpolation
            # ## Slope input
            # SlopeInput[SlopeInput==NoData]   = np.nan
            # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
            # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
            # ## Zmax input
            # ZmaxInput[ZmaxInput==NoData]     = np.nan
            # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
            # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
            # ## DEM input
            # DEMInput[DEMInput==NoData]       = np.nan
            # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # DEMInput = np.reshape(DEMInput, (nrows,ncols))
            # np.nan_to_num(DEMInput,copy=False, nan=NoData)
            # ## Ground water table input
            # HwInput[HwInput==NoData]         = np.nan
            # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # HwInput = np.reshape(HwInput, (nrows,ncols))
            # np.nan_to_num(HwInput,copy=False, nan=NoData)
            # ## Background infiltration rate input
            # rizeroInput[rizeroInput==NoData] = np.nan
            # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
            # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          

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
            ### !!!
   
    
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
            Cell_row, Cell_column = int(CellsInside[i,0]),int(CellsInside[i,1])
            ## Depth of the sliding surface at each cell
            DEMdiff2 = (DEMInput[Cell_row, Cell_column] - DepthDEM[i,0])  
            ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
            Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
            Weight[i]    = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
            
            
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
        
        ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))   ## Reshape 
        condzmax  = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
        
        ## Arrangement of the parameters using truncation condition above
        ## Thickness becomes ZmaxInput for the truncated cells.
        Thickness[condzmax] = ZmaxInput[indexes][condzmax]
        
        ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
        ## Area is recalculated. 
        
        TempA = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                            (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
        A[condzmax] = np.reshape(TempA, (np.shape(TempA)[0],1))
        
        ## Weight is recalculated.                     
        for i in condzmax:
            # print(i)
            Weight[i]    = ZmaxInput[indexes][i]  * cellsize**2 * Gamma.flatten()[indexes][i]   ## Weight
            
    
        # Reshape and truncate to the slope of the current cell
        ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
        Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
        AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
        
        TempSlope = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
        ThetaAvr[condzmax]         = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        Theta[condzmax]            = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        AngleTangentXZE1[condzmax] = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        
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
        if (AnalysisType == 'Drained'):
            PoreWaterForce = HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
        elif (AnalysisType == 'Undrained'):
            # PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
            PoreWaterForce = [np.zeros((np.shape(CellsInside)[0],1))]*np.shape(TimeToAnalyse)[0]
        
        ## !!!
        ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry       
        if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):   
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction 
                PoreWaterForce[corr_n] = np.zeros((np.shape(CellsInside)[0],1)) 
            
        if (ProblemName =='Pr3S2Wet'):      
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction
                Temp = PoreWaterForce[corr_n]
                Temp[Temp<0] = 0.
                PoreWaterForce[corr_n] = Temp
                
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
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(c, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current) 
                    FS3D.append(FS3D_current)
                                     
            elif (FSCalType=='Bishop3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)  
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)
            
         
        elif (AnalysisType == 'Undrained'):
      
            ## Arrange the undrained shear strength and friction angle with corresponding indexes 
            Su = Su.flatten()[indexes]
            Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
            phi = np.zeros((np.shape(CellsInside)[0],1))
            
            ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
            if (FSCalType=='Normal3D'):  
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(Su, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current)            
                    FS3D.append(FS3D_current)
                
            elif (FSCalType=='Bishop3D'):     
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce: 
                    FS3D_current = root(FSBishop3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'): 
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)
                    FS3D.append(FS3D_current)
            
        ## Changing these values are only related to the indexing. 
        ## I changed overall shared array to the sub-shared arrays. Therefore, indexing needs this adjustment.
        MCnumber, MC_current = 1,0 ### (Shared array change) 
        
        ## Original indexes of the cells inside the sliding surface
        indexesOriginal = np.asarray(indexesOriginal,dtype=int)
        
        ## For each time instance, assign the min FS to the cells 
        for TimeInd in range(np.shape(TimeToAnalyse)[0]):
            # print(TimeInd)
            ## FS3D value at the current time instance
            FS3D_current = np.round(FS3D[TimeInd],5)
        
            ## Global indexes
            GlobalIndexFS   = TimeInd * nrows_org * ncols_org * MCnumber + \
                                        nrows_org * ncols_org * MC_current + \
                                        (indexesOriginal//ncols_org) * ncols_org + (indexesOriginal%ncols_org)
            GlobalIndexFS = np.asarray(GlobalIndexFS,dtype=int)
            
            
            ## Assign the min FS to the cells inside the sliding surface
            for i in GlobalIndexFS:
                FS_All_MC[i] = FS3D_current if ((FS_All_MC[i]==0) or (FS_All_MC[i] > FS3D_current)) else FS_All_MC[i]
        
        
        print("************************************************************ MC:", int(queue_data[4]),  
              "  Ellipsoid center location:", nrows_org * ncols_org * MC_current + EllRow * ncols_org + EllColumn , 
              "  FS: ", np.round(FS3D,5))
        
        queue_data = queue.get()
        # return()

#---------------------------------------------------------------------------------
def EllipsoidFSWithSubDis_v1_0_MultiThread(queue,FS_All_MC, \
                                            AnalysisType,MCnumber, FSCalType, SubDisNum, \
                                            nrows, ncols, nel, cellsize, \
                                            Parameter_Fields, \
                                            EllParam, \
                                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                                            rizeroInput, riInp, AspectInput, \
                                            TimeToAnalyse, NoData,ProblemName = ''):
    """
    This function calculates the factor of safety by assuming an ellipsoidal shape, and used in multiple thread function.

    Parameters
    ----------
    queue_mc : -
        Queue of the tasks for multiprocessing.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
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
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    EllParam : array of float
        Ellipsoidal parameters.
    SlopeInput : array
         Slope angle.
    ZmaxInput: array
        Map of depth to bedrock over the study area. 
    DEMInput : array
         Digital elevation map data.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    AspectInput : array
         Values of aspect.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """
    
    ## Keep the original parameters
    arg_original_ell = (AnalysisType, FSCalType, SubDisNum, \
                            nrows, ncols, nel, cellsize, \
                            Parameter_Fields, \
                            EllParam, \
                            SlopeInput, ZmaxInput, DEMInput, HwInput, \
                            rizeroInput, riInp, AspectInput,\
                            TimeToAnalyse, NoData,ProblemName)
    
    queue_data = queue.get()
    
    while queue_data is not None:
        
    
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, SubDisNum, \
            nrows, ncols, nel, cellsize, \
            Parameter_Fields, \
            EllParam, \
            SlopeInput, ZmaxInput, DEMInput, HwInput, \
            rizeroInput, riInp,AspectInput,\
            TimeToAnalyse, NoData,ProblemName) = arg_original_ell[:]
        
        ## Extract the prameters from the queue
        EllRow,EllColumn = int(queue_data[0]), int(queue_data[1])
        nrows_org, ncols_org =  int(queue_data[2]), int(queue_data[3])
        MC_current = int(queue_data[4])
        
        
        # Assign the parameters
        if (AnalysisType == 'Drained'):   
            ## {c, phi, uws, ksat, diffus} for drained analysis
            c      = Parameter_Fields[0]  ## Cohesion
            phi    = Parameter_Fields[1]  ## Friction angle 
            Gamma  = Parameter_Fields[2]  ## Unit weight 
            Ksat   = Parameter_Fields[3]  ## Hydraulic conductivity
            Diff0  = Parameter_Fields[4]  ## Diffusivity
        elif (AnalysisType == 'Undrained'):
            ## {Su, uws} for drained analysis
            Su     = Parameter_Fields[0]  ## Undrained shear strength 
            Gamma  = Parameter_Fields[1]  ## Unit weight 
     
    
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
        Ellz        = EllParam[4]  ## Offset of the ellipsoid
        
        ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
        ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
        ## Coordinates according to the center of allipsoid, e
        CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
        
        ## Check if EllAlpha should be calculated or assigned. 
        EllAlpha_Calc = EllParam[5]
        if (EllAlpha_Calc == "Yes"):        
            ########### (Change aspect)
            ## Calculate the distance (over xy) to the center of the ellipoid
            Distance_to_Center = np.sqrt(CoorEll[:,2]**2 + CoorEll[:,3]**2)
            Cell_Inside_Circle = CoorEll[(Distance_to_Center<= Ella)]
            EllAlpha = np.nanmean(AspectInput[ Cell_Inside_Circle[:,0].astype(int), Cell_Inside_Circle[:,1].astype(int)])
            EllAlpha = EllAlpha - 90 ## 90 is the different between the values 
            ###########
        elif (EllAlpha_Calc == "No"):
            EllAlpha    = EllParam[3]
        
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
        
        # # See the current sliding suface
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

            ## Save previous Coordinates as old coordinates. 
            if (nrows==nrows_org):
                CoorG_old = CoorG
            else:
                CoorG_old = CoorG_new ## Ignore undefined name error here. It will not require it in the first loop.
                
            ## Halve the cellsize for this part only
            cellsize = cellsize / 2
            nrows    = nrows    * 2  ## Double the number of rows
            ncols    = ncols    * 2  ## Double the number of columns
            
            ## Cell center coordinate according to the left bottom option-1 (Global)
            CoorG_new =[]
            for i in range(nrows):
                for j in range(ncols):
                    CoorG_new.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
            CoorG_new = np.asarray(CoorG_new) # (row #, column #, x coordinate, y coordinate)     
            
            ## Follow the index numbers
            IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
            
            ## Arrangement of the soil strength parameters according to new cellsize
            if (AnalysisType == 'Drained'):
                ## {c, phi, uws, ksat, diffus} for drained analysis
                c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
                phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
                Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
            elif (AnalysisType == 'Undrained'):
                ## {Su, uws} for drained analysis
                Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                
            ## Arrangments with new cellsize 
            ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
            ## "numpy.kron"
            SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
            ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
            DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
            HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
            rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input

            # ## Linear interpolation
            # ## Slope input
            # SlopeInput[SlopeInput==NoData]   = np.nan
            # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
            # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
            # ## Zmax input
            # ZmaxInput[ZmaxInput==NoData]     = np.nan
            # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
            # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
            # ## DEM input
            # DEMInput[DEMInput==NoData]       = np.nan
            # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # DEMInput = np.reshape(DEMInput, (nrows,ncols))
            # np.nan_to_num(DEMInput,copy=False, nan=NoData)
            # ## Ground water table input
            # HwInput[HwInput==NoData]         = np.nan
            # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # HwInput = np.reshape(HwInput, (nrows,ncols))
            # np.nan_to_num(HwInput,copy=False, nan=NoData)
            # ## Background infiltration rate input
            # rizeroInput[rizeroInput==NoData] = np.nan
            # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
            # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          

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
            ### !!!
   
    
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
            Cell_row, Cell_column = int(CellsInside[i,0]),int(CellsInside[i,1])
            ## Depth of the sliding surface at each cell
            DEMdiff2 = (DEMInput[Cell_row, Cell_column] - DepthDEM[i,0])  
            ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
            Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
            Weight[i]    = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
            
            
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
        
        ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))   ## Reshape 
        condzmax  = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
        
        ## Arrangement of the parameters using truncation condition above
        ## Thickness becomes ZmaxInput for the truncated cells.
        Thickness[condzmax] = ZmaxInput[indexes][condzmax]
        
        ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
        ## Area is recalculated. 
        
        TempA = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                            (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
        A[condzmax] = np.reshape(TempA, (np.shape(TempA)[0],1))
        
        ## Weight is recalculated.                     
        for i in condzmax:
            # print(i)
            Weight[i]    = ZmaxInput[indexes][i]  * cellsize**2 * Gamma.flatten()[indexes][i]   ## Weight
            
    
        # Reshape and truncate to the slope of the current cell
        ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
        Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
        AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
        
        TempSlope = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
        ThetaAvr[condzmax]         = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        Theta[condzmax]            = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        AngleTangentXZE1[condzmax] = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        
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
        if (AnalysisType == 'Drained'):
            PoreWaterForce = HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
        elif (AnalysisType == 'Undrained'):
            # PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
            PoreWaterForce = [np.zeros((np.shape(CellsInside)[0],1))]*np.shape(TimeToAnalyse)[0]
        
        ## !!!
        ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
        if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):   
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction 
                PoreWaterForce[corr_n] = np.zeros((np.shape(CellsInside)[0],1)) 
            
        if (ProblemName =='Pr3S2Wet'):      
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction
                Temp = PoreWaterForce[corr_n]
                Temp[Temp<0] = 0.
                PoreWaterForce[corr_n] = Temp
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
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(c, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current) 
                    FS3D.append(FS3D_current)
                                     
            elif (FSCalType=='Bishop3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)  
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)           
            
        elif (AnalysisType == 'Undrained'):
      
            ## Arrange the undrained shear strength and friction angle with corresponding indexes 
            Su = Su.flatten()[indexes]
            Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
            phi = np.zeros((np.shape(CellsInside)[0],1))
            
            ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
            if (FSCalType=='Normal3D'):  
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(Su, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current)            
                    FS3D.append(FS3D_current)
                
            elif (FSCalType=='Bishop3D'):     
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce: 
                    FS3D_current = root(FSBishop3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'): 
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)
                    FS3D.append(FS3D_current)      
        
        ## Changing these values are only related to the indexing. 
        ##I changed overall shared array to the sub-shared arrays. Therefore, indexing needs this adjustment.
        MCnumber, MC_current = 1,0 ### (Shared array change)         
    
        ## Original indexes of the cells inside the sliding surface
        indexesOriginal = np.asarray(indexesOriginal,dtype=int)
        
        ## For each time instance, assign the min FS to the cells 
        for TimeInd in range(np.shape(TimeToAnalyse)[0]):
            # print(TimeInd)
            ## FS3D value at the current time instance
            FS3D_current = np.round(FS3D[TimeInd],5)
        
            ## Global indexes
            GlobalIndexFS   = TimeInd * nrows_org * ncols_org * MCnumber + \
                                        nrows_org * ncols_org * MC_current + \
                                        (indexesOriginal//ncols_org) * ncols_org + (indexesOriginal%ncols_org)
            GlobalIndexFS = np.asarray(GlobalIndexFS,dtype=int)
            
            mutex = Lock()
            mutex.acquire()
            ## Assign the min FS to the cells inside the sliding surface
            for i in GlobalIndexFS:
                FS_All_MC[i] = FS3D_current if ((FS_All_MC[i]==0) or (FS_All_MC[i] > FS3D_current)) else FS_All_MC[i]
            mutex.release()
        
        print("************************************************************ MC:", int(queue_data[4]),  
              "  Ellipsoid center location:", nrows_org * ncols_org * MC_current + EllRow * ncols_org + EllColumn , 
              "  FS: ", np.round(FS3D,5))
        
        queue_data = queue.get()
        # return()

#---------------------------------------------------------------------------------
def Ellipsoid_Generate_Main(InZone, SubDisNum, \
                            nrows, ncols, nel, cellsize, EllParam, \
                            SlopeInput,ZmaxInput, DEMInput, AspectInput, \
                            NoData,ProblemName = ''):
    """
    This is main function to obtain sliding surfaces information over the area of interest. 

    Parameters
    ----------
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
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
    EllParam : array of float
        Ellipsoidal parameters.
    SlopeInput : array
         Slope angle.
    ZmaxInput: array
        Map of depth to bedrock over the study area. 
    DEMInput : array
         Digital elevation map data.
    AspectInput : array
         Values of aspect.
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    AllInf : list
        Sliding surfaces' information

    """

    ## Allocate
    AllInf = []
    
    
    ########### (Change aspect)
    ## Change aspect of NoData to np.nan 
    AspectInput[AspectInput == NoData] = np.nan 
    ###########

    count=0

    ## Elllipsoidal sliding surface is generated for a given row and column number.
    # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest    (example:range(10,108), np.arange(11,108,2)) 
    #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest 
    #         ## Current Row and Column for the ellipsoid center
    #         EllRow    = i
    #         EllColumn = j
    #         # print(EllRow,EllColumn)
    for i in range(np.shape(InZone)[0]): ## (InZone change)
            ## Current Row and Column for the ellipsoid center
            EllRow, EllColumn = InZone[i]
            # print(EllRow,EllColumn)   
             
            # EllRow, EllColumn = 10, 20
            # t_start = time.time() ## Time before FS calculation for 1 ellipdoidal sliding surface
            Inf_ellipsoid = Ellipsoid_Generate(count, SubDisNum, nrows, ncols, nel, cellsize, \
                                        EllParam, EllRow, EllColumn, \
                                        SlopeInput, ZmaxInput, DEMInput, AspectInput, \
                                        NoData,ProblemName)
            
            AllInf.append(Inf_ellipsoid)
            count += 1
        
    return(AllInf)

#---------------------------------------------------------------------------------
def Ellipsoid_Generate(count, SubDisNum, nrows, ncols, nel, cellsize, \
                            EllParam, EllRow, EllColumn, \
                            SlopeInput, ZmaxInput, DEMInput,AspectInput, \
                            NoData,ProblemName=''):
    """
    This is the function obtaining current sliding surface information. 

    Parameters
    ----------
    count : int
        Sliding surface number.
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
    EllParam : array of float
        Ellipsoidal parameters.
    EllRow : int
        Current row for the ellipsoid center.
    EllColumn : int
        Current column for the ellipsoid center.  
    SlopeInput : array
         Slope angle.
    ZmaxInput: array
        Map of depth to bedrock over the study area. 
    DEMInput : array
         Digital elevation map data.
    AspectInput : array
         Values of aspect.
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    Inf : list
        Information on the current sliding surface

    """
    
    nrows_org, ncols_org =  int(nrows), int(ncols)
    
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
    Ellz        = EllParam[4]  ## Offset of the ellipsoid
    
    ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
    ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
    ## Coordinates according to the center of allipsoid, e
    CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
      
    ## Check if EllAlpha should be calculated or assigned. 
    EllAlpha_Calc = EllParam[5]
    if (EllAlpha_Calc == "Yes"):        
        ########### (Change aspect)
        ## Calculate the distance (over xy) to the center of the ellipoid
        Distance_to_Center = np.sqrt(CoorEll[:,2]**2 + CoorEll[:,3]**2)
        Cell_Inside_Circle = CoorEll[(Distance_to_Center<= Ella)]
        EllAlpha = np.nanmean(AspectInput[ Cell_Inside_Circle[:,0].astype(int), Cell_Inside_Circle[:,1].astype(int)])
        EllAlpha = EllAlpha - 90 ## 90 is the different between the values 
        ###########
    elif (EllAlpha_Calc == "No"):
        EllAlpha    = EllParam[3]
        
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

    # See = np.zeros((nrows,ncols))
    # for i in range(np.shape(Cell_Inside_Circle)[0]):
    #     See[int(Cell_Inside_Circle[i][0]),int(Cell_Inside_Circle[i][1])] = 1
    # # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 1)
    # #plt.title('')
    # plt.imshow(See)
    # plt.colorbar() 
    # plt.show()
    
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
    
    # # See the current sliding suface
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
    
    CountSubDis = 0
    ## Subdiscretisize if the number of cells are low (by halving the cell size and arranging the data). 
    ## SubDisNum is the minimum number of cells defined by the user in the main code.         
    while (np.shape(CellsInside)[0] < SubDisNum):

        CountSubDis +=1 ## Check how many times subdiscretisized
        
        ## Save previous Coordinates as old coordinates. 
        ## Old and new coordinates are only for linear interpolation below.
        if (nrows==nrows_org):
            CoorG_old = CoorG
        else:
            CoorG_old = CoorG_new ## Ignore undefined name error here. It will not require it in the first loop.
            
        ## Halve the cellsize for this part only
        cellsize = cellsize / 2
        nrows    = nrows    * 2  ## Double the number of rows
        ncols    = ncols    * 2  ## Double the number of columns
        
        ## Cell center coordinate according to the left bottom option-1 (Global)
        CoorG_new =[]
        for i in range(nrows):
            for j in range(ncols):
                CoorG_new.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
        CoorG_new = np.asarray(CoorG_new) # (row #, column #, x coordinate, y coordinate)     
        
        ## Follow the index numbers
        IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
        
        # ## Arrangement of the soil strength parameters according to new cellsize
        # if (AnalysisType == 'Drained'):
        #     ## {c, phi, uws, ksat, diffus} for drained analysis
        #     c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
        #     phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
        #     Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
        #     Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
        #     Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
        # elif (AnalysisType == 'Undrained'):
        #     ## {Su, uws} for drained analysis
        #     Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
        #     Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
            
        ## Arrangments with new cellsize 
        ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
        ## "numpy.kron"
        SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
        ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
        DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
        # HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
        # rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input

        # ## Linear interpolation
        # ## Slope input
        # SlopeInput[SlopeInput==NoData]   = np.nan
        # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
        # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
        # ## Zmax input
        # ZmaxInput[ZmaxInput==NoData]     = np.nan
        # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
        # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
        # ## DEM input
        # DEMInput[DEMInput==NoData]       = np.nan
        # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # DEMInput = np.reshape(DEMInput, (nrows,ncols))
        # np.nan_to_num(DEMInput,copy=False, nan=NoData)
        # ## Ground water table input
        # HwInput[HwInput==NoData]         = np.nan
        # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # HwInput = np.reshape(HwInput, (nrows,ncols))
        # np.nan_to_num(HwInput,copy=False, nan=NoData)
        # ## Background infiltration rate input
        # rizeroInput[rizeroInput==NoData] = np.nan
        # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
        # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
        # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          

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
        ### !!!
   

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
        Cell_row, Cell_column = int(CellsInside[i,0]),int(CellsInside[i,1])
        ## Depth of the sliding surface at each cell
        DEMdiff2 = (DEMInput[Cell_row, Cell_column] - DepthDEM[i,0])  
        ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
        Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
        # Weight[i]    = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
        
        
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
    
    ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))   ## Reshape 
    condzmax  = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
    
    ## Arrangement of the parameters using truncation condition above
    ## Thickness becomes ZmaxInput for the truncated cells.
    Thickness[condzmax] = ZmaxInput[indexes][condzmax]
    
    ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
    ## Area is recalculated. 
    
    TempA = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                        (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
    A[condzmax] = np.reshape(TempA, (np.shape(TempA)[0],1))
    
    # ## Weight is recalculated.                     
    # for i in condzmax:
    #     # print(i)
    #     Weight[i]    = ZmaxInput[indexes][i]  * cellsize**2 * Gamma.flatten()[indexes][i]   ## Weight
        
    # Weight   = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
    # Reshape and truncate to the slope of the current cell
    ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
    Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
    AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
    
    TempSlope = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
    ThetaAvr[condzmax]         = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
    Theta[condzmax]            = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
    AngleTangentXZE1[condzmax] = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
    
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
        
        
    ## !!!
    
    ## Remove very high area values again. 
    LimitValue = 1 + np.max(Thickness)/cellsize  ## By trials
    A[A>LimitValue*cellsize**2] = LimitValue*cellsize**2 
    
    indexesOriginal = np.asarray(indexesOriginal,dtype=int)
    
    ## Round the values
    A                = np.round(A, 8)
    Thickness        = np.round(Thickness, 8)
    Theta            = np.round(Theta, 8)
    ThetaAvr         = np.round(ThetaAvr, 8)
    AngleTangentXZE1 = np.round(AngleTangentXZE1, 8)
    
    ## Information on the current ellipdoidal sliding surface
    Inf = [count, EllRow, EllColumn, CountSubDis, indexesOriginal, indexes, CellsInside, A, Thickness, Theta, ThetaAvr, AngleTangentXZE1]
     
    
    return(Inf)

#---------------------------------------------------------------------------------
def Ellipsoid_Generate_Main_Multi(TOTAL_PROCESSES_EllGen, InZone, SubDisNum, \
                            nrows, ncols, nel, cellsize, EllParam, \
                            SlopeInput,ZmaxInput, DEMInput,AspectInput,\
                            NoData,ProblemName = ''):
    """
    This is main function to obtain sliding surfaces information over the area of interest with multiple processor. 

    Parameters
    ----------
    TOTAL_PROCESSES_EllGen : int	
        Numbers of processors used for gathering information on ellipsoidal sliding surfes.	
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).	
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
    EllParam : array of float	
        Ellipsoidal parameters.	
    SlopeInput : array	
         Slope angle.	
    ZmaxInput: array	
        Map of depth to bedrock over the study area. 	
    DEMInput : array	
         Digital elevation map data.	
    AspectInput : array	
         Values of aspect.	
    NoData : int	
         The value of No Data. 	
    ProblemName : str	
         Name of the problem to reproduce results. The default is ''. 	


    Returns
    -------
    shared_list: list
        Sliding surfaces' information.

    """

    
    manager = Manager()
    shared_list = manager.list()
    
    ########### (Change aspect)
    ## Change aspect of NoData to np.nan 
    AspectInput[AspectInput == NoData] = np.nan 
    ###########
    
    if __name__ == 'Functions_3DPLS_v1_0':
        queue_EllGen = Queue(maxsize=1000)
    
        processes_EllGen = []
        for i in range(TOTAL_PROCESSES_EllGen):
            p_EllGen = Process(target=Ellipsoid_Generate_Multi, args=(queue_EllGen,shared_list, \
                                                            nrows, ncols, cellsize,SubDisNum, \
                                                            EllParam, \
                                                            SlopeInput, ZmaxInput, DEMInput, AspectInput, \
                                                            NoData,ProblemName))
            processes_EllGen.append(p_EllGen)
            p_EllGen.start()  
        
        count=0
        # for i in range (InZone[0,0],InZone[0,1]+1):      ## Row numbers of interest
        #     for j in range (InZone[1,0],InZone[1,1]+1):   ## Column numbers of interest                                                         
        #         EllRow, EllColumn = i,j
        for i in range(np.shape(InZone)[0]): ## (InZone change)
                ## Current Row and Column for the ellipsoid center
                EllRow, EllColumn = InZone[i]
                # print(EllRow,EllColumn)
                queue_EllGen.put([EllRow,EllColumn,nrows,ncols,count])
                # print("queue_ell.put([,]):",EllRow,",",EllColumn)
                count +=1
                
        for _ in range(TOTAL_PROCESSES_EllGen): queue_EllGen.put(None)
        for p_EllGen in processes_EllGen: p_EllGen.join() ## For Processes

    return(shared_list)

#---------------------------------------------------------------------------------
def Ellipsoid_Generate_Multi(queue_EllGen,inf_list, \
                            nrows, ncols, cellsize,SubDisNum, \
                            EllParam, \
                            SlopeInput, ZmaxInput, DEMInput,AspectInput,\
                            NoData,ProblemName=''):
    """
    This is the function obtaining current sliding surface information with function using multiple processor . 

    Parameters
    ----------
    queue_EllGen : -
        Queue of the tasks for multiprocessing.
    inf_list : list
        Information on the current sliding surface
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size.
    SubDisNum : int
         Minimum number of cells desired inside the ellipsoidal sliding surface.
    EllParam : array of float
        Ellipsoidal parameters.
    SlopeInput : array
         Slope angle.
    ZmaxInput: array
        Map of depth to bedrock over the study area. 
    DEMInput : array
         Digital elevation map data.
    AspectInput : array
         Values of aspect.
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """
    
    
    ## Keep the original parameters
    arg_original_EllGen = (nrows, ncols, cellsize,SubDisNum, \
                            EllParam, \
                            SlopeInput, ZmaxInput, DEMInput,AspectInput,\
                            NoData,ProblemName)
    
    queue_data = queue_EllGen.get()
    
    while queue_data is not None:
        
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (nrows, ncols, cellsize,SubDisNum, \
            EllParam, \
            SlopeInput, ZmaxInput, DEMInput, AspectInput,\
            NoData,ProblemName) = arg_original_EllGen[:]
        
        ## Extract the prameters from the queue
        EllRow,EllColumn = int(queue_data[0]), int(queue_data[1])
        nrows_org, ncols_org =  int(queue_data[2]), int(queue_data[3])
        count = int(queue_data[4])
        
        
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
        Ellz        = EllParam[4]  ## Offset of the ellipsoid
        
        ## The inclination of the ellipdoidal sliding surface, Ellbeta value, is calculated as 
        ## the average slope angle within a zone of rectangle with the dimensions of (2*Ella – 2* Ellb).    
        ## Coordinates according to the center of allipsoid, e
        CoorEll = CoorG[:]-np.array((0,0,EllCenterX,EllCenterY))
        
        ## Check if EllAlpha should be calculated or assigned. 
        EllAlpha_Calc = EllParam[5]
        if (EllAlpha_Calc == "Yes"):        
            ########### (Change aspect)
            ## Calculate the distance (over xy) to the center of the ellipoid
            Distance_to_Center = np.sqrt(CoorEll[:,2]**2 + CoorEll[:,3]**2)
            Cell_Inside_Circle = CoorEll[(Distance_to_Center<= Ella)]
            EllAlpha = np.nanmean(AspectInput[ Cell_Inside_Circle[:,0].astype(int), Cell_Inside_Circle[:,1].astype(int)])
            EllAlpha = EllAlpha - 90 ## 90 is the different between the values 
            ###########
        elif (EllAlpha_Calc == "No"):
            EllAlpha    = EllParam[3]
        
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
        
        # # See the current sliding suface
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
        
        CountSubDis = 0
        ## Subdiscretisize if the number of cells are low (by halving the cell size and arranging the data). 
        ## SubDisNum is the minimum number of cells defined by the user in the main code.         
        while (np.shape(CellsInside)[0] < SubDisNum):
    
            CountSubDis +=1 ## Check how many times subdiscretisized
            
            ## Save previous Coordinates as old coordinates. 
            ## Old and new coordinates are only for linear interpolation below.
            if (nrows==nrows_org):
                CoorG_old = CoorG
            else:
                CoorG_old = CoorG_new ## Ignore undefined name error here. It will not require it in the first loop.
                
            ## Halve the cellsize for this part only
            cellsize = cellsize / 2
            nrows    = nrows    * 2  ## Double the number of rows
            ncols    = ncols    * 2  ## Double the number of columns
            
            ## Cell center coordinate according to the left bottom option-1 (Global)
            CoorG_new =[]
            for i in range(nrows):
                for j in range(ncols):
                    CoorG_new.append([i, j, cellsize/2+j*cellsize, (nrows-1)*cellsize+cellsize/2-i*cellsize ])
            CoorG_new = np.asarray(CoorG_new) # (row #, column #, x coordinate, y coordinate)     
            
            ## Follow the index numbers
            IndexTemp  = np.kron(IndexTemp, np.ones((2,2)))
            
            # ## Arrangement of the soil strength parameters according to new cellsize
            # if (AnalysisType == 'Drained'):
            #     ## {c, phi, uws, ksat, diffus} for drained analysis
            #     c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
            #     phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
            #     Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
            #     Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
            #     Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
            # elif (AnalysisType == 'Undrained'):
            #     ## {Su, uws} for drained analysis
            #     Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
            #     Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                
            ## Arrangments with new cellsize 
            ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
            ## "numpy.kron"
            SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
            ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
            DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
            # HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
            # rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input
    
            # ## Linear interpolation
            # ## Slope input
            # SlopeInput[SlopeInput==NoData]   = np.nan
            # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
            # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
            # ## Zmax input
            # ZmaxInput[ZmaxInput==NoData]     = np.nan
            # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
            # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
            # ## DEM input
            # DEMInput[DEMInput==NoData]       = np.nan
            # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # DEMInput = np.reshape(DEMInput, (nrows,ncols))
            # np.nan_to_num(DEMInput,copy=False, nan=NoData)
            # ## Ground water table input
            # HwInput[HwInput==NoData]         = np.nan
            # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # HwInput = np.reshape(HwInput, (nrows,ncols))
            # np.nan_to_num(HwInput,copy=False, nan=NoData)
            # ## Background infiltration rate input
            # rizeroInput[rizeroInput==NoData] = np.nan
            # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
            # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          
    
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
            ### !!!
       
    
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
            Cell_row, Cell_column = int(CellsInside[i,0]),int(CellsInside[i,1])
            ## Depth of the sliding surface at each cell
            DEMdiff2 = (DEMInput[Cell_row, Cell_column] - DepthDEM[i,0])  
            ## If the DEM of the sliding surface is higher than the cell's DEM. Thickness becomes zero.
            Thickness[i] = (DEMdiff2 if DEMdiff2>0 else 0)      ## Thickness
            # Weight[i]    = Thickness[i] * cellsize**2 * Gamma[Cell_row, Cell_column]   ## Weight
            
            
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
        
        ZmaxInput = np.reshape(ZmaxInput.flatten(), (nrows*ncols,1))   ## Reshape 
        condzmax  = np.where(ZmaxInput[indexes] < Thickness)[0]        ## Truncation condition 
        
        ## Arrangement of the parameters using truncation condition above
        ## Thickness becomes ZmaxInput for the truncated cells.
        Thickness[condzmax] = ZmaxInput[indexes][condzmax]
        
        ## The slope of the bottom sliding surface is now the slope of the truncated cell at the ground surface. 
        ## Area is recalculated. 
        
        TempA = (cellsize**2) * ( (np.sqrt(1-((np.sin(np.radians(0)))**2) *(    np.square(np.sin(np.radians((SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax]))))    )    )) )/ \
                            (((np.cos(np.radians((0)))))*((np.cos(np.radians(SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax])))))
        A[condzmax] = np.reshape(TempA, (np.shape(TempA)[0],1))
        
        # ## Weight is recalculated.                     
        # for i in condzmax:
        #     # print(i)
        #     Weight[i]    = ZmaxInput[indexes][i]  * cellsize**2 * Gamma.flatten()[indexes][i]   ## Weight
            
    
        # Reshape and truncate to the slope of the current cell
        ThetaAvr = np.reshape(ThetaAvr, (np.shape(CellsInside)[0],1))
        Theta = np.reshape(Theta, (np.shape(CellsInside)[0],1))
        AngleTangentXZE1 = np.reshape(AngleTangentXZE1, (np.shape(CellsInside)[0],1))
        
        TempSlope = SlopeInput[(CellsInside[:,0].astype(int),CellsInside[:,1].astype(int))].T[condzmax] 
        ThetaAvr[condzmax]         = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        Theta[condzmax]            = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        AngleTangentXZE1[condzmax] = np.reshape(TempSlope, (np.shape(TempSlope)[0],1))
        
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
           
        ## !!!
        
        ## Remove very high area values again. 
        LimitValue = 1 + np.max(Thickness)/cellsize  ## By trials
        A[A>LimitValue*cellsize**2] = LimitValue*cellsize**2 
        
        indexesOriginal = np.asarray(indexesOriginal,dtype=int)
        
        ## Round the values
        A                = np.round(A, 8)
        Thickness        = np.round(Thickness, 8)
        Theta            = np.round(Theta, 8)
        ThetaAvr         = np.round(ThetaAvr, 8)
        AngleTangentXZE1 = np.round(AngleTangentXZE1, 8)
        
        ## Append them to the inf list
        inf_list.append((count, EllRow, EllColumn, CountSubDis, indexesOriginal, indexes, CellsInside, A, Thickness, Theta, ThetaAvr, AngleTangentXZE1))

        ## Take another task from the queue
        queue_data = queue_EllGen.get()     
    # return(Inf)

#---------------------------------------------------------------------------------
def IndMC_Main(AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                                    InZone, Results_Directory, Maxrix_Directory, \
                                    nrows, ncols, cellsize, MCnumber, \
                                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                    SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                    TimeToAnalyse, NoData,ProblemName = ''):
    """
    ## This is the main function in which the calculations are done with single processor.

    Parameters
    ----------
    AllInf_sorted : list
        Sliding surfaces' information, sorted based on the list of Inzone.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
    Results_Directory : str
        The folder to store data.
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.
    SaveMat : str, optional
        Whether the mayrix is saved or not. The default is 'NO'.
    SlopeInput : array
         Slope angle.
    ZoneInput : Array
         Zones.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 


    Returns
    -------
    None.

    """

    ## Allocate
    # FS_All_MC = [0]*(MCnumber * nrows * ncols)      ## Only 1 time instance
    # FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * MCnumber * nrows * ncols) ## multiple time instances
    FS_All_MC = []
    
    # MC_current = 0
    for MC_current in range(MCnumber): # One MC analysis in each for loop                
        ## Print current MC number
        print("MC:%d"%(MC_current))     

        # FS_All_MC = IndMC_FS(AllInf_sorted,FS_All_MC,MC_current, AnalysisType, FSCalType, RanFieldMethod, \
        #                                     InZone, MCnumber, Results_Directory, Maxrix_Directory, \
        #                                     nrows, ncols, cellsize, \
        #                                     Parameter_Means, Parameter_CoVs, Parameter_Dist, \
        #                                     Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
        #                                     SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
        #                                     TimeToAnalyse, NoData,ProblemName)
        
        IndMC_FS(AllInf_sorted,FS_All_MC,MC_current, AnalysisType, FSCalType, RanFieldMethod, \
                                                InZone, MCnumber, Results_Directory, Maxrix_Directory, \
                                                nrows, ncols, cellsize, \
                                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                                SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                                TimeToAnalyse, NoData,ProblemName)

    # return(FS_All_MC)
    return()

#---------------------------------------------------------------------------------
def IndMC_FS(AllInf_sorted,FS_All_MC, MC_current, AnalysisType, FSCalType, RanFieldMethod, \
                                    InZone, MCnumber, Results_Directory, Maxrix_Directory, \
                                    nrows, ncols, cellsize, \
                                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                    SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                    TimeToAnalyse, NoData,ProblemName = ''):
    """
    This function calculates the factor of safety with single processor.

    Parameters
    ----------
    AllInf_sorted : list
        Sliding surfaces' information, sorted based on the list of Inzone.
    FS_All_MC : list
        List for storinf all factor of safety values for all Monte Carlo simulations.
    MC_current : int
        Current number of Monte Carlo simulation.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Results_Directory : str
        The folder to store data.
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size.
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.
    SaveMat : str, optional
        Whether the mayrix is saved or not. The default is 'NO'.
    SlopeInput : array
         Slope angle.
    ZoneInput : Array
         Zones.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 

    Returns
    -------
    None.

    """

    
    ## Generate parameter fields
    Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                   Parameter_CorrLenX, Parameter_CorrLenY, \
                                   nrows, ncols, cellsize, \
                                   RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
    
    ## Keep the original values
    nrows_org, ncols_org = nrows, ncols
    cellsize_org = cellsize
    SlopeInput_org, HwInput_org, rizeroInput_org =  SlopeInput, HwInput, rizeroInput
    MC_current_org = MC_current
    
    ## Allocate FS 
    # TempLst = {i:[] for i in list(range(0,nrows*ncols))}
    FS_All_MC = [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols) ### (Shared array change)
    
    
    # EllCurr = 0
    for EllCurr in range(np.shape(AllInf_sorted)[0]):
        
        ## Keep the original values
        SlopeInput, HwInput, rizeroInput = SlopeInput_org, HwInput_org, rizeroInput_org
        cellsize = cellsize_org
        
        ## Data --> count, EllRow, EllColumn, CountSubDis, indexesOriginal, indexes, CellsInside, A, Thickness, Theta, ThetaAvr, AngleTangentXZE1
        SurfaceData = AllInf_sorted[EllCurr]
        
        ## Extract information on the current ellipsoidal sliding surface
        EllRow, EllColumn = SurfaceData[1:3]
        CountSubDis       = SurfaceData[3]
        indexesOriginal   = SurfaceData[4]
        indexes           = SurfaceData[5]
        CellsInside       = SurfaceData[6]
        A                 = SurfaceData[7]
        Thickness         = SurfaceData[8]
        Theta             = SurfaceData[9]
        ThetaAvr          = SurfaceData[10]
        AngleTangentXZE1  = SurfaceData[11]
       
        
      
        ## Assign the parameters
        if (AnalysisType == 'Drained'):   
            ## {c, phi, uws, ksat, diffus} for drained analysis
            c      = Parameter_Fields[0]  ## Cohesion
            phi    = Parameter_Fields[1]  ## Friction angle 
            Gamma  = Parameter_Fields[2]  ## Unit weight 
            Ksat   = Parameter_Fields[3]  ## Hydraulic conductivity
            Diff0  = Parameter_Fields[4]  ## Diffusivity
        elif (AnalysisType == 'Undrained'):
            ## {Su, uws} for drained analysis
            Su     = Parameter_Fields[0]  ## Undrained shear strength 
            Gamma  = Parameter_Fields[1]  ## Unit weight 
        
    
        ## Subdivide the parameters based on the number in the generation of surfaces, CountSubDis
        for SubDisCurr in range(CountSubDis):
            
            cellsize = cellsize / 2
            
            ## Arrangement of the soil strength parameters according to new cellsize
            if (AnalysisType == 'Drained'):
                ## {c, phi, uws, ksat, diffus} for drained analysis
                c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
                phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
                Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
            elif (AnalysisType == 'Undrained'):
                ## {Su, uws} for drained analysis
                Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
                Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                
            ## Arrangments with new cellsize 
            ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
            ## "numpy.kron"
            SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
            # ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
            # DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
            HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
            rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input
    
            # ## Linear interpolation
            # ## Slope input
            # SlopeInput[SlopeInput==NoData]   = np.nan
            # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
            # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
            # ## Zmax input
            # ZmaxInput[ZmaxInput==NoData]     = np.nan
            # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
            # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
            # ## DEM input
            # DEMInput[DEMInput==NoData]       = np.nan
            # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # DEMInput = np.reshape(DEMInput, (nrows,ncols))
            # np.nan_to_num(DEMInput,copy=False, nan=NoData)
            # ## Ground water table input
            # HwInput[HwInput==NoData]         = np.nan
            # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # HwInput = np.reshape(HwInput, (nrows,ncols))
            # np.nan_to_num(HwInput,copy=False, nan=NoData)
            # ## Background infiltration rate input
            # rizeroInput[rizeroInput==NoData] = np.nan
            # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
            # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
            # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          
    
        ## Calculate the weight using the gamma values
        Weight   = Thickness * cellsize**2 * np.expand_dims(Gamma.flatten()[indexes],axis=1)
       
    
        ####################################
        ## Iverson (2000), infiltraion model 
        ####################################
        ## The calculation of the pore water forces using Iverson's solution of infiltration (Iverson, 2000)
        ## Slope parallel flow is assummed.    
        ## If the analysis is undrained, pore water forces are zero. If it is drained, calculations are performed using Iverson's solution-.
        if (AnalysisType == 'Drained'):
            PoreWaterForce = HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
        elif (AnalysisType == 'Undrained'):
            # PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
            PoreWaterForce = [np.zeros((np.shape(CellsInside)[0],1))]*np.shape(TimeToAnalyse)[0]
        
        ## !!!
        ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
        if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):   
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction 
                PoreWaterForce[corr_n] = np.zeros((np.shape(CellsInside)[0],1)) 
            
        if (ProblemName =='Pr3S2Wet'):      
            for corr_n in range(np.shape(PoreWaterForce)[0]):
                ## Correction
                Temp = PoreWaterForce[corr_n]
                Temp[Temp<0] = 0.
                PoreWaterForce[corr_n] = Temp
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
          
        ## !!!
        ##This part is for problem 3 slide 1 dry, problem 3 slide 2 dry  
        if ( ProblemName == 'Pr3S2Dry' or ProblemName == 'Pr3S2Wet'): 
            
            ## Condition for truncation
            condzmax = np.where(A == np.round((cellsize**2),8))
            
            ## Assign weak layer properties for the truncated cells.
            ## Cohesion
            c = np.reshape(c.flatten(), (nrows*ncols,1))
            tempvalues =  c[indexes][:]
            tempvalues[condzmax] = np.zeros((np.shape(condzmax)[1]))
            c[indexes] = tempvalues
            c = np.reshape(c, (nrows,ncols))
            
            ## Friction angle
            phi = np.reshape(phi.flatten(), (nrows*ncols,1))
            # phi[indexes][condzmax] = np.zeros((9520, 1))
            tempvalues =  phi[indexes][:]
            tempvalues[condzmax] = np.ones((np.shape(condzmax)[1])) * 10.
            phi[indexes] = tempvalues
            phi =np.reshape(phi, (nrows,ncols))
        ## !!!
            
        ## Factor of safety calculation for 3D ellipsoidal shape. 
        if (AnalysisType == 'Drained'):
            
            ## Arrange the cohesion and friction angle with corresponding indexes 
            c = c.flatten()[indexes]
            c = np.reshape( c , (np.shape(CellsInside)[0],1) )
            phi = phi.flatten()[indexes]
            phi = np.reshape( phi , (np.shape(CellsInside)[0],1) )
    
            ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'        
            if (FSCalType=='Normal3D'):
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(c, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current) 
                    FS3D.append(FS3D_current)
                                     
            elif (FSCalType=='Bishop3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)  
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'):            
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)             
        
            
        
        elif (AnalysisType == 'Undrained'):
      
            ## Arrange the undrained shear strength and friction angle with corresponding indexes 
            Su = Su.flatten()[indexes]
            Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
            phi = np.zeros((np.shape(CellsInside)[0],1))
            
            ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
            if (FSCalType=='Normal3D'):  
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = FSNormal3D(Su, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                    # print( "Normal 3D FS is %.5f"%FS3D_current)            
                    FS3D.append(FS3D_current)
                
            elif (FSCalType=='Bishop3D'):     
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce: 
                    FS3D_current = root(FSBishop3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Bishop 3D FS is %.5f"%FS3D_current)   
                    FS3D.append(FS3D_current)
            
            elif (FSCalType=='Janbu3D'): 
                
                FS3D = []
                for PoreWaterForceCurrent in PoreWaterForce:
                    FS3D_current = root(FSJanbu3D, 1.5,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                    FS3D_current = FS3D_current.x[0]
                    # print( "Janbu 3D FS is %.5f"%FS3D_current)
                    FS3D.append(FS3D_current)          
    
       
        ## Changing these values are only related to the indexing. 
        ## I changed overall FS array to the sub-shared arrays. Therefore, indexing needs this adjustment.
        MCnumber, MC_current = 1,0 ### (Shared array change)     
    
        ## Original indexes of the cells inside the sliding surface
        indexesOriginal = np.asarray(indexesOriginal,dtype=int)
        
        ## For each time instance, assign the min FS to the cells 
        for TimeInd in range(np.shape(TimeToAnalyse)[0]):
            # print(TimeInd)
            ## FS3D value at the current time instance
            FS3D_current = np.round(FS3D[TimeInd],5)
        
            ## Global indexes
            GlobalIndexFS   = TimeInd * nrows_org * ncols_org * MCnumber + \
                                        nrows_org * ncols_org * MC_current + \
                                            indexesOriginal                           
            GlobalIndexFS = np.asarray(GlobalIndexFS,dtype=int)
            
            ## Assign the min FS to the cells inside the sliding surface
            for i in GlobalIndexFS:
                FS_All_MC[i] = FS3D_current if ((FS_All_MC[i]==0) or (FS_All_MC[i] > FS3D_current)) else FS_All_MC[i]
    
    
    
    ## Write FS for the current MC simulation
    FS_All_MC_InZone = np.asarray(FS_All_MC)
    FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) ### (Shared array change)

    os.chdir(Results_Directory)
    NameResFile = 'MC_%.4d_FS_Values'%(MC_current_org)
    np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
    
    # return(FS_All_MC)
    return()

#---------------------------------------------------------------------------------
def IndMC_Main_Multi(TOTAL_PROCESSES_IndMC, AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                                    InZone, Results_Directory, Maxrix_Directory, \
                                    nrows, ncols, cellsize, MCnumber, \
                                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                    SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                    TimeToAnalyse, NoData,ProblemName = ''):
    """
    ## This is the main function in which the calculations are done with multiple processor.

    Parameters
    ----------
    TOTAL_PROCESSES_IndMC : int
        Numbers of processors used for calculating factor of safety of the ellipsoidal sliding surfes.
    AllInf_sorted : list
        Sliding surfaces' information, sorted based on the list of Inzone.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
    Results_Directory : str
        The folder to store data.
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size.
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.
    SaveMat : str, optional
        Whether the mayrix is saved or not. The default is 'NO'.
    SlopeInput : array
         Slope angle.
    ZoneInput : Array
         Zones.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 


    Returns
    -------
    None.

    """

    ## Allocate array for multiprocessing 
    ## Lock is necessary because multiple processors need to access the same cell to asssign the min. 
    # FS_All_MC = Array('d', [0]*(MCnumber * nrows * ncols), lock=True)  
    # FS_All_MC = Array('d', [0]*(np.shape(TimeToAnalyse)[0] * MCnumber * nrows * ncols), lock=True)  

    if __name__ == 'Functions_3DPLS_v1_0':
        queue_mc_ind = Queue(maxsize=1000)
        
        processes_mc_ind = []
        
        for i in range(TOTAL_PROCESSES_IndMC):
            
            # p_mc_ind = Process(target=IndMC_FS_Multi, args=(queue_mc_ind, FS_All_MC, \
            #                                     AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
            #                                     InZone, MCnumber, Results_Directory, Maxrix_Directory, \
            #                                     nrows, ncols, cellsize, \
            #                                     Parameter_Means, Parameter_CoVs, Parameter_Dist, \
            #                                     Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
            #                                     SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
            #                                     TimeToAnalyse, NoData,ProblemName))
                
            p_mc_ind = Process(target=IndMC_FS_Multi, args=(queue_mc_ind, \
                                                AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                                                InZone, MCnumber, Results_Directory, Maxrix_Directory, \
                                                nrows, ncols, cellsize, \
                                                Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                                Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                                                SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                                                TimeToAnalyse, NoData,ProblemName))
            processes_mc_ind.append(p_mc_ind)
            p_mc_ind.start() 
       
                                                         
        for i in range(MCnumber):      
            queue_mc_ind.put(i)
            # print("queue_mc.put([,]): MC ",i)       

        for _ in range(TOTAL_PROCESSES_IndMC): queue_mc_ind.put(None)
        for p_mc_ind in processes_mc_ind: p_mc_ind.join()
                                                                
    

    
    return()
    # return(FS_All_MC)

#---------------------------------------------------------------------------------
def IndMC_FS_Multi(queue_mc_ind, \
                    AllInf_sorted, AnalysisType, FSCalType, RanFieldMethod, \
                    InZone, MCnumber, Results_Directory, Maxrix_Directory, \
                    nrows, ncols, cellsize, \
                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                    SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                    TimeToAnalyse, NoData,ProblemName=''):
    """
    This function calculates the factor of safety with multiple processor.
    
    Parameters
    ----------
    queue_mc_ind : -
                Queue of the tasks for multiprocessing for Monte Carlo simulations.
    AllInf_sorted : list
        Sliding surfaces' information, sorted based on the list of Inzone.
    AnalysisType : str
        It might be either 'Drained' or 'Undrained'.
    FSCalType : str
        It shows which method will be used for FS calculation: 'Normal3D' - 'Bishop3D' - 'Janbu3D'.
    RanFieldMethod : str
        It shows which method will be used for random field generation: 'CMD' - 'SCMD' .
    InZone : Matrix
        Matrix for zone of interest: ((rowstart, rowfinish), (columnstart, columnfinish)).
    MCnumber : int
        Monte Carlo number (suggested: 1000).
    Results_Directory : str
        The folder to store data.
    Maxrix_Directory : str
        The folder for the matrix
    nrows : int
        Number of rows.
    ncols : int
        Number of columns. 
    cellsize : int
        Cell size.
    Parameter_Means : Array
        List of mean values for given model parameters.
    Parameter_CoVs : Array
        List of coefficient of variation values for given model parameters.
    Parameter_Dist : Array
        List of distributions for given model parameters..
    Parameter_CorrLenX : Array
        List of values of correlation length in x direction for given model parameters..
    Parameter_CorrLenY : Array
        List of values of correlation length in y direction for given model parameters.
    SaveMat : str, optional
        Whether the mayrix is saved or not. The default is 'NO'.
    SlopeInput : array
         Slope angle.
    ZoneInput : Array
         Zones.
    HwInput : array
         Ground water table depth.
    rizeroInput : Array
        Steady, background infiltration.
    riInp : array
         the rainfall data.
    TimeToAnalyse : array
         The time of analysis (in the range of rainfall).
    NoData : int
         The value of No Data. 
    ProblemName : str
         Name of the problem to reproduce results. The default is ''. 


    Returns
    -------
    None.

    """
    
    ## Keep the original parameters
    arg_original_mc_ind = ( AnalysisType, FSCalType, RanFieldMethod, \
                            InZone, MCnumber,  Results_Directory, Maxrix_Directory, \
                            nrows, ncols, cellsize, \
                            Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                            Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                            SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                            TimeToAnalyse, NoData,ProblemName)
    
    queue_data_mc_ind = queue_mc_ind.get() 
    
    
    while queue_data_mc_ind is not None:
        
        ## Here, the original parameters are assigned again in order to not change the parameters in the while loop.
        (AnalysisType, FSCalType, RanFieldMethod, \
                    InZone, MCnumber, Results_Directory, Maxrix_Directory, \
                    nrows, ncols, cellsize, \
                    Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                    Parameter_CorrLenX, Parameter_CorrLenY, SaveMat, \
                    SlopeInput, ZoneInput, HwInput, rizeroInput, riInp, \
                    TimeToAnalyse, NoData,ProblemName) = arg_original_mc_ind[:]
        
        MC_current = int(queue_data_mc_ind)
        print("---->------>----->>>>>",int(MC_current))
     
        ## Generate parameter fields
        Parameter_Fields = Par_Fields(Parameter_Means, Parameter_CoVs, Parameter_Dist, \
                                       Parameter_CorrLenX, Parameter_CorrLenY, \
                                       nrows, ncols, cellsize, \
                                       RanFieldMethod, ZoneInput, Maxrix_Directory, NoData, SaveMat)
        
        ## Keep the original values
        nrows_org, ncols_org = nrows, ncols
        cellsize_org = cellsize
        SlopeInput_org, HwInput_org, rizeroInput_org =  SlopeInput, HwInput, rizeroInput
        
        ## Allocate
        FSValues_MC_Current = [0]*(np.shape(TimeToAnalyse)[0] * nrows * ncols)


        ## Allocate FS 
        # TempLst = {i:[] for i in list(range(0,nrows*ncols))}
        # EllCurr = 0
        for EllCurr in range(np.shape(AllInf_sorted)[0]):
            
            ## Keep the original values
            SlopeInput, HwInput, rizeroInput = SlopeInput_org, HwInput_org, rizeroInput_org
            cellsize = cellsize_org
            
            ## Data --> count, EllRow, EllColumn, CountSubDis, indexesOriginal, indexes, CellsInside, A, Thickness, Theta, ThetaAvr, AngleTangentXZE1
            SurfaceData = AllInf_sorted[EllCurr]
            
            ## Extract information on the current ellipsoidal sliding surface
            EllRow, EllColumn = SurfaceData[1:3]
            CountSubDis       = SurfaceData[3]
            indexesOriginal   = SurfaceData[4]
            indexes           = SurfaceData[5]
            CellsInside       = SurfaceData[6]
            A                 = SurfaceData[7]
            Thickness         = SurfaceData[8]
            Theta             = SurfaceData[9]
            ThetaAvr          = SurfaceData[10]
            AngleTangentXZE1  = SurfaceData[11]
           
            
            ## Assign the parameters
            if (AnalysisType == 'Drained'):   
                ## {c, phi, uws, ksat, diffus} for drained analysis
                c      = Parameter_Fields[0]  ## Cohesion
                phi    = Parameter_Fields[1]  ## Friction angle 
                Gamma  = Parameter_Fields[2]  ## Unit weight 
                Ksat   = Parameter_Fields[3]  ## Hydraulic conductivity
                Diff0  = Parameter_Fields[4]  ## Diffusivity
            elif (AnalysisType == 'Undrained'):
                ## {Su, uws} for drained analysis
                Su     = Parameter_Fields[0]  ## Undrained shear strength 
                Gamma  = Parameter_Fields[1]  ## Unit weight 
        
            ## Subdivide the parameters based on the number in the generation of surfaces, CountSubDis
            for SubDisCurr in range(CountSubDis):
                
                cellsize = cellsize / 2
                
                ## Arrangement of the soil strength parameters according to new cellsize
                if (AnalysisType == 'Drained'):
                    ## {c, phi, uws, ksat, diffus} for drained analysis
                    c      = np.kron(c,     np.ones((2,2)))  ## Cohesion
                    phi    = np.kron(phi,   np.ones((2,2)))  ## Friction angle 
                    Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                    Ksat   = np.kron(Ksat,  np.ones((2,2)))  ## Hydraulic conductivity
                    Diff0  = np.kron(Diff0, np.ones((2,2)))  ## Diffusivity
                elif (AnalysisType == 'Undrained'):
                    ## {Su, uws} for drained analysis
                    Su     = np.kron(Su,    np.ones((2,2)))  ## Undrained shear strength 
                    Gamma  = np.kron(Gamma, np.ones((2,2)))  ## Unit weight 
                    
                ## Arrangments with new cellsize 
                ## You can either use numpy.kron or linear interpolation. Select the desired method below: 
                ## "numpy.kron"
                SlopeInput = np.kron(SlopeInput, np.ones((2,2)))  ## Slope input
                # ZmaxInput  = np.kron(ZmaxInput, np.ones((2,2)))   ## Zmax input
                # DEMInput   = np.kron(DEMInput, np.ones((2,2)))    ## DEM input
                HwInput    = np.kron(HwInput, np.ones((2,2)))     ## Ground water table input
                rizeroInput= np.kron(rizeroInput, np.ones((2,2))) ## Background infiltration rate input
        
                # ## Linear interpolation
                # ## Slope input
                # SlopeInput[SlopeInput==NoData]   = np.nan
                # SlopeInput = griddata(CoorG_old[:,(2,3)], SlopeInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
                # SlopeInput = np.reshape(SlopeInput, (nrows,ncols))
                # np.nan_to_num(SlopeInput,copy=False, nan=NoData)
                # ## Zmax input
                # ZmaxInput[ZmaxInput==NoData]     = np.nan
                # ZmaxInput = griddata(CoorG_old[:,(2,3)], ZmaxInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
                # ZmaxInput = np.reshape(ZmaxInput, (nrows,ncols))
                # np.nan_to_num(ZmaxInput,copy=False, nan=NoData)
                # ## DEM input
                # DEMInput[DEMInput==NoData]       = np.nan
                # DEMInput = griddata(CoorG_old[:,(2,3)], DEMInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
                # DEMInput = np.reshape(DEMInput, (nrows,ncols))
                # np.nan_to_num(DEMInput,copy=False, nan=NoData)
                # ## Ground water table input
                # HwInput[HwInput==NoData]         = np.nan
                # HwInput = griddata(CoorG_old[:,(2,3)], HwInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
                # HwInput = np.reshape(HwInput, (nrows,ncols))
                # np.nan_to_num(HwInput,copy=False, nan=NoData)
                # ## Background infiltration rate input
                # rizeroInput[rizeroInput==NoData] = np.nan
                # rizeroInput = griddata(CoorG_old[:,(2,3)], rizeroInput.flatten().T, CoorG_new[:,(2,3)], method='linear') 
                # rizeroInput = np.reshape(rizeroInput, (nrows,ncols))
                # np.nan_to_num(rizeroInput,copy=False, nan=NoData)          
        
            ## Calculate the weight using the gamma values
            Weight   = Thickness * cellsize**2 * np.expand_dims(Gamma.flatten()[indexes],axis=1)
           
        
            ####################################
            ## Iverson (2000), infiltraion model 
            ####################################
            ## The calculation of the pore water forces using Iverson's solution of infiltration (Iverson, 2000)
            ## Slope parallel flow is assummed.    
            ## If the analysis is undrained, pore water forces are zero. If it is drained, calculations are performed using Iverson's solution-.
            if (AnalysisType == 'Drained'):
                PoreWaterForce = HydrologyModel_v1_0(TimeToAnalyse, CellsInside, A, HwInput, rizeroInput, riInp, Ksat, Diff0, Thickness, SlopeInput)
            elif (AnalysisType == 'Undrained'):
                # PoreWaterForce = np.zeros((np.shape(CellsInside)[0],1))
                PoreWaterForce = [np.zeros((np.shape(CellsInside)[0],1))]*np.shape(TimeToAnalyse)[0]
            
            ## !!!
            ##This part is for validation problem 1, poblem 2, problem 3 slide 1 dry, problem 3 slide 2 dry  
            if (ProblemName == 'Pr1' or ProblemName == 'Pr2' or ProblemName == 'Pr3S1Dry' or ProblemName =='Pr3S2Dry'):   
                for corr_n in range(np.shape(PoreWaterForce)[0]):
                    ## Correction 
                    PoreWaterForce[corr_n] = np.zeros((np.shape(CellsInside)[0],1)) 
                
            if (ProblemName =='Pr3S2Wet'):      
                for corr_n in range(np.shape(PoreWaterForce)[0]):
                    ## Correction
                    Temp = PoreWaterForce[corr_n]
                    Temp[Temp<0] = 0.
                    PoreWaterForce[corr_n] = Temp
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
          
            ## !!!
            ##This part is for problem 3 slide 1 dry, problem 3 slide 2 dry  
            if ( ProblemName == 'Pr3S2Dry' or ProblemName == 'Pr3S2Wet'): 
                
                ## Condition for truncation
                condzmax = np.where(A == np.round((cellsize**2),8))
                
                ## Assign weak layer properties for the truncated cells.
                ## Cohesion
                c = np.reshape(c.flatten(), (nrows*ncols,1))
                tempvalues =  c[indexes][:]
                tempvalues[condzmax] = np.zeros((np.shape(condzmax)[1]))
                c[indexes] = tempvalues
                c = np.reshape(c, (nrows,ncols))
                
                ## Friction angle
                phi = np.reshape(phi.flatten(), (nrows*ncols,1))
                # phi[indexes][condzmax] = np.zeros((9520, 1))
                tempvalues =  phi[indexes][:]
                tempvalues[condzmax] = np.ones((np.shape(condzmax)[1])) * 10.
                phi[indexes] = tempvalues
                phi =np.reshape(phi, (nrows,ncols))
            ## !!!
                        
            ## Factor of safety calculation for 3D ellipsoidal shape. 
            if (AnalysisType == 'Drained'):
                
                ## Arrange the cohesion and friction angle with corresponding indexes 
                c = c.flatten()[indexes]
                c = np.reshape( c , (np.shape(CellsInside)[0],1) )
                phi = phi.flatten()[indexes]
                phi = np.reshape( phi , (np.shape(CellsInside)[0],1) )
                
                ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'        
                if (FSCalType=='Normal3D'):
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce:
                        FS3D_current = FSNormal3D(c, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                        # print( "Normal 3D FS is %.5f"%FS3D_current) 
                        FS3D.append(FS3D_current)
                                         
                elif (FSCalType=='Bishop3D'):            
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce:
                        FS3D_current = root(FSBishop3D, 1.5,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                        FS3D_current = FS3D_current.x[0]
                        # print( "Bishop 3D FS is %.5f"%FS3D_current)  
                        FS3D.append(FS3D_current)
                
                elif (FSCalType=='Janbu3D'):            
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce:
                        FS3D_current = root(FSJanbu3D, 1.,args=(c, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                        FS3D_current = FS3D_current.x[0]
                        # print( "Janbu 3D FS is %.5f"%FS3D_current)   
                        FS3D.append(FS3D_current)             
            
                
            elif (AnalysisType == 'Undrained'):
          
                ## Arrange the undrained shear strength and friction angle with corresponding indexes 
                Su = Su.flatten()[indexes]
                Su = np.reshape( Su , (np.shape(CellsInside)[0],1) )
                phi = np.zeros((np.shape(CellsInside)[0],1))      
                
                ## Calculations are done depending on the method: 'Normal3D' - 'Bishop3D' - 'Janbu3D'                   
                if (FSCalType=='Normal3D'):  
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce:
                        FS3D_current = FSNormal3D(Su, phi, A, Weight, PoreWaterForceCurrent, Theta, ThetaAvr, AngleTangentXZE1 )
                        # print( "Normal 3D FS is %.5f"%FS3D_current)            
                        FS3D.append(FS3D_current)
                    
                elif (FSCalType=='Bishop3D'):     
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce: 
                        FS3D_current = root(FSBishop3D, 1.,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                        FS3D_current = FS3D_current.x[0]
                        # print( "Bishop 3D FS is %.5f"%FS3D_current)   
                        FS3D.append(FS3D_current)
                
                elif (FSCalType=='Janbu3D'): 
                    
                    FS3D = []
                    for PoreWaterForceCurrent in PoreWaterForce:
                        FS3D_current = root(FSJanbu3D, 1.,args=(Su, phi, CellsInside, Weight, PoreWaterForceCurrent,ThetaAvr, Theta, A,AngleTangentXZE1))
                        FS3D_current = FS3D_current.x[0]
                        # print( "Janbu 3D FS is %.5f"%FS3D_current)
                        FS3D.append(FS3D_current)    
        
           

            ## Original indexes of the cells inside the sliding surface
            indexesOriginal = np.asarray(indexesOriginal,dtype=int)
            
            ## For each time instance, assign the min FS to the cells 
            for TimeInd in range(np.shape(TimeToAnalyse)[0]):
                # print(TimeInd)
                ## FS3D value at the current time instance
                FS3D_current = np.round(FS3D[TimeInd],5)
                
                '''
                ## Old way to write using shared array
                ''' 
                
                # ## Global indexes
                # GlobalIndexFS   = TimeInd * nrows_org * ncols_org * MCnumber + \
                #                             nrows_org * ncols_org * MC_current + \
                #                                 indexesOriginal                           
                # GlobalIndexFS = np.asarray(GlobalIndexFS,dtype=int)
                
                # ## Assign the min FS to the cells inside the sliding surface
                # for i in GlobalIndexFS:
                #     FS_All_MC[i] = FS3D_current if ((FS_All_MC[i]==0) or (FS_All_MC[i] > FS3D_current)) else FS_All_MC[i]
            
            
                '''
                ## New way to write 
                '''                
                ## Index for data in a Monte Carlo, not shared 
                GlobalIndex_MC_Current   = TimeInd * nrows_org * ncols_org + indexesOriginal                           
                GlobalIndex_MC_Current = np.asarray(GlobalIndex_MC_Current,dtype=int)
                
                ## Assign the min FS to the cells inside the sliding surface
                for i in GlobalIndex_MC_Current:
                    FSValues_MC_Current[i] = FS3D_current if ((FSValues_MC_Current[i]==0) or (FSValues_MC_Current[i] > FS3D_current)) else FSValues_MC_Current[i]

        
        '''
        ## New way to write 
        '''     
        ## Write FS of current MC simulation
        FS_All_MC_InZone = np.asarray(FSValues_MC_Current)
        FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],nrows,ncols)) 
         
        # FS_All_MC_InZone = FS_All_MC_InZone[:,InZone[0,0]:InZone[0,1]+1, InZone[1,0]:InZone[1,1]+1] ## (InZone change)
        ## Write FS for the current MC simulation
        os.chdir(Results_Directory)
        NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
        
        
        '''
        ## Old way to write using shared array
        ''' 
        
        # ## Write FS of current MC simulation
        # FS_All_MC_InZone = np.asarray(FS_All_MC)
        # FS_All_MC_InZone = np.reshape(FS_All_MC_InZone, (np.shape(TimeToAnalyse)[0],MCnumber,nrows,ncols)) 
         
        # FS_All_MC_InZone = FS_All_MC_InZone[:,MC_current,InZone[0,0]:InZone[0,1]+1, InZone[1,0]:InZone[1,1]+1]
        # ## Write FS for the current MC simulation
        # os.chdir(Results_Directory)
        # NameResFile = 'MC_%.4d_FS_Values'%(MC_current)
        # np.save(NameResFile, FS_All_MC_InZone) #Save as .npy
        
        
        ## Take another task from the queue
        queue_data_mc_ind = queue_mc_ind.get()
    # return()
