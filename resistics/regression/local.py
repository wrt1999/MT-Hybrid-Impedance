import os
import random
import math
import numpy as np
import numpy.linalg as linalg
import scipy.interpolate as interp
from scipy.stats import chi2
from typing import List, Dict, Tuple

from resistics.common.base import ResisticsBase
from resistics.window.selector import WindowSelector
from resistics.transfunc.data import TransferFunctionData
from resistics.transfunc.io import TransferFunctionWriter
from resistics.common.io import checkAndMakeDir, fileFormatSampleFreq
from resistics.common.smooth import smooth1d
from resistics.regression.compute import spectralMatrices
from resistics.regression.robust import (
    sampleMAD0,
    hermitianTranspose,
    olsModel,
    weightLS,
    chatterjeeMachler,
    mestimateModel,
    mmestimateModel,
    mahalanobisDistance,
    robustmahalanobisDistance,
)


class LocalRegressor(ResisticsBase):
    """Performs single site (or intersite) transfer function calculations 

    By default, the LocalRegression is setup to calculate the impedance tensor using Hx, Hy as input channels and Ex, Ey as output channels. To calculate the Tipper, the appropriate input and output channels have to be set.

    Attributes
    ----------
    winSelector : WindowSelector
        A window selector object which defines which windows to use in the linear model
    decParams : DecimationParameters
        DecimationParameters object with information about the decimation scheme
    winParams : WindowParameters
        WindowParameters object with information about the windowing
    outpath : str 
        Location to put the calculated transfer functions (Edi files)
    inSite : str 
        The site to use for the input channels 
    inChannels: List[str] (["Hx", "Hy"])
        List of hannels to use as input channels for the linear system
    inSize : int 
        Number of input channels
    outSite : str 
        The site to use for the output channels
    outChannels : List[str] (["Ex", "Ey"])
        List of channels to use as output channels for the linear system
    outSize : int
        Number of output channels
    allChannels : List[str] 
        inChannels and outChannels combined into a single list
    crossChannels : List[str] 
        The channels to calculate the cross spectra out for
    intercept : bool (default False)
        Flag for including an intercept (static) term in the linear system
    method : str (options, "ols", "cm") 
        String for describing what solution method to use
    win : str (default hanning)
        Window function to use in robust solution
    winSmooth : int (default -1)
        The size of the window smoother. If -1, this will be autocalculated based on data size
    postpend : str (default "")
        String to postpend to the output filename to help file management
    evalFreq : List[float] or np.ndarray
        The evaluation frequencies
    impedances : List
    variances : List

    Methods
    -------
    __init__(proj, winSelector, outpath)
        Initialise with a Project instance and MaskData instance
    setInput(inSite, inChannels)
        Set the input site and channels
    setOutput(outSite, outChannels)
        Set the output site and channels
    process()
        Process the spectra to calculate the transfer function
    getWindowSmooth()
        Get the window smooth length
    checkForBadValues(numWindows, data)
        Check the spectral data for bad values that might cause an error   
    prepareLinearEqn(data)
        Prepare regressors and observations for regression from cross-power data
    robustProcess(numWindows, obs, reg)      
        Robust regression processing   
    olsProcess(numWindows, obs, reg)      
        Ordinary least squares processing
    stackedProcess(data)
        Stacked processing                  
    printList()
        Class status returned as list of strings
    """

    def __init__(self, winSelector: WindowSelector, outpath: str):
        """Intialise the processor
        初始化processor，大地电磁传输函数transferfunction处理器的初始化，专门用于站点阻抗张量计算
        Parameters
        ----------
        winSelector : WindowSelector
            A window selector instance
        outpath : str
            The path to write the transfer function data to
        """
        self.winSelector = winSelector
        self.decParams = winSelector.decParams              
        self.winParams = winSelector.winParams              
        self.outpath: str = outpath                         
        # default parameters for user options
        self.inSite: str = "dummy"                          
        self.inChannels: List[str] = ["Hx", "Hy"]           
        self.inSize: int = len(self.inChannels)             
        self.outSite: str = "dummy"                         
        self.outChannels: List[str] = ["Ex", "Ey"]          
        self.outSize: int = len(self.outChannels)           
        self.allChannels: List[str] = self.inChannels + self.outChannels        
        self.crossChannels: List[str] = self.allChannels    
        # solution options
        self.intercept: bool = False                
        self.method: str = "cm"                     
        # smoothing options
        self.win: str = "hann"                      
        self.winSmooth: int = 9                     
        # output filename
        self.postpend: str = ""                     
        # evaluation frequency data
        self.evalFreq = []                          
        self.impedances = []                        
        self.variances = []                         


    def setInput(self, inSite: str, inChannels: List[str]) -> None:
        """Set information about input site and channels
    
        Parameters
        ----------
        inSite : str
            Site to use for input channel data
        inChannels : List[str]
            Channels to use as the input in the linear system
        """
        self.inSite = inSite
        self.inChannels = inChannels
        self.inSize = len(inChannels)
        # set all and cross channels
        self.allChannels = self.inChannels + self.outChannels
        self.crossChannels = self.allChannels

    def setOutput(self, outSite: str, outChannels: List[str]) -> None:
        """Set information about output site and channels
    
        Parameters
        ----------
        inSite : str
            Site to use for output channel data
        inChannels : List[str]
            Channels to use as the output in the linear system
        """
        self.outSite = outSite
        self.outChannels = outChannels
        self.outSize = len(outChannels)
        # set all and cross channels
        self.allChannels = self.inChannels + self.outChannels
        self.crossChannels = self.allChannels

    def process(self, ncores: int = 0) -> None:
        """Process spectra data
        
        The processing sequence for each decimation level is as below:
        
        1. Get shared (unmasked) windows for all relevant sites (inSite and outSite) 
        2. For shared unmasked windows 
            
            - Calculate out the cross-power spectra. 
            - Interpolate calculated cross-power data to the evaluation frequencies for the decimation level. 
        
        3. For each evaluation frequency 
            
            - Do the robust processing to calculate the transfer function at that evaluation frequency.

        The spectral power data is smoothed as this tends to improve results. The smoothing can be changed by setting the smoothing parameters. This method is still subject to change in the future as it is an area of active work
        
        """
        numLevels: int = self.decParams.numLevels
        for declevel in range(0, numLevels):                
            self.printText("Processing decimation level {}".format(declevel))
            numWindows = self.winSelector.getNumSharedWindows(declevel)
            unmaskedWindows = self.winSelector.getUnmaskedWindowsLevel(declevel)                                         
            numUnmasked = len(unmaskedWindows)
            self.printText(
                "Total shared windows for decimation level = {}".format(numWindows)
            )
            self.printText(
                "Total unmasked windows for decimation level = {}".format(numUnmasked)
            )
            if numUnmasked == 0:
                self.printText(
                    "No unmasked windows found at this decimation level ({:d}), continuing to next level".format(
                        declevel
                    )
                )
                continue
            self.printText("{} windows will be processed".format(numUnmasked))

            # set variables
            evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
            totalSize: int = self.inSize + self.outSize
            numEvalFreq: int = len(evalFreq)
            dataSize: int = self.winSelector.getDataSize(declevel)        
            smoothLen: int = self.getWindowSmooth(datasize=dataSize)      
            # data array for each evaluation frequency and keep spectral power information for all windows
            evalFreqData: np.ndarray = np.empty(                                            
                shape=(numEvalFreq, numWindows, totalSize, totalSize), dtype="complex"      
            )           
            
            # global to local map to help choose windows for each evaluation frequency
            localWin: int = 0
            global2local: Dict = {}
            # process spectral batches
            spectraBatches = self.winSelector.getSpecReaderBatches(declevel)
            numBatches = len(spectraBatches)            
            for batchIdx, batch in enumerate(spectraBatches):
                # find the unmasked batched windows and add the data
                batchedWindows = unmaskedWindows.intersection(
                    set(range(batch["globalrange"][0], batch["globalrange"][1] + 1))
                )
                self.printText(
                    "Processing batch {:d} of {:d}: Global window range {:d} to {:d}, {:.3%} of data".format(
                        batchIdx + 1,
                        numBatches,
                        batch["globalrange"][0],                                                                            
                        batch["globalrange"][1],
                        len(batchedWindows) / numWindows,
                    )
                )

                # collect spectrum data and compute spectral matrices
                # set batchedWindows from return to ensure proper order
                inReader = batch[self.inSite]
                inData, batchedWindows = inReader.readBinaryBatchGlobal(
                    globalIndices=batchedWindows                    
                )
                if self.outSite != self.inSite:
                    outReader = batch[self.outSite]
                    outData, _gIndicesOut = outReader.readBinaryBatchGlobal(
                        globalIndices=batchedWindows
                    )
                else:
                    outData = inData                                

                out = spectralMatrices(                             
                    ncores,                                         
                    inData,
                    outData,
                    self.inChannels,                                
                    self.outChannels,                               
                    smoothLen,
                    self.win,
                    evalFreq,                                       
                )

                for batchWin, globalWin in enumerate(batchedWindows):       
                    evalFreqData[:, localWin] = out[batchWin]               
                    # local to global map and increment local window
                    global2local[globalWin] = localWin                      
                    localWin = localWin + 1                                 

            # close spectra files for decimation level
            for batch in spectraBatches:
                for site in self.winSelector.sites:
                    batch[site].closeFile()

            # data has been collected for each evaluation frequency, perform robust processing
            for eIdx in range(0, numEvalFreq):              
                self.printText(
                    "Processing evaluation frequency = {:.6f} [Hz], period = {:.6f} [s]".format(
                        evalFreq[eIdx], 1 / evalFreq[eIdx]
                    )
                )
                # get the constrained windows for the evaluation frequency
                evalFreqWindows = self.winSelector.getWindowsForFreq(declevel, eIdx)        
                if len(evalFreqWindows) == 0:
                    # no windows meet constraints
                    self.printText("No windows found - possibly due to masking")
                    continue
                localWinIndices = []
                for iW in evalFreqWindows:
                    localWinIndices.append(global2local[iW])            

                self.printText(
                    "{:d} windows will be solved for".format(len(localWinIndices))
                )
                # restrict processing to data that meets constraints for this evaluation frequency
                self.evalFreq.append(evalFreq[eIdx])        
                # solution using all components
                numSolveWindows, obs, reg = self.prepareLinearEqn(                          
                    evalFreqData[eIdx, localWinIndices]                     
                )           
                # ============================================================================================================
                out, var = self.newProcess(numSolveWindows, obs, reg)                     
                # out, var = self.robustProcess(numSolveWindows, obs, reg)                    
                # out, var = self.olsProcess(numSolveWindows, obs, reg)                     
                # out, var = self.stackedProcess(evalFreqData[eIdx, localWinIndices])       
                self.impedances.append(out)                                         
                self.variances.append(var)                                          
                

        if len(self.evalFreq) == 0:
            self.printWarning(
                "No data was found at any decimation level for insite {}, outsite {} and specdir {}".format(
                    self.inSite, self.outSite, self.winSelector.specdir
                )
            )
            return

        
        # =========================================================================
        Z_array = np.array(self.impedances)
        var_array = np.array(self.variances)
        freq_array = np.array(self.evalFreq)
        
        
        Z_smoothed = self.MT_smooth(freq_array, Z_array, var_array, lam=1e-1)
        self.impedances = list(Z_smoothed)
        # =========================================================================

        # write out all the data
        self.writeTF(
            self.winSelector.specdir,
            self.postpend,
            self.evalFreq,
            self.impedances,
            self.variances,
        )

        

    def MT_smooth(self, freqs: np.ndarray, Z_array: np.ndarray, var_array: np.ndarray,
                       lam: float = 1e-1) -> np.ndarray:                                                
        """
        smooth
        """
        n_freq = len(freqs)                                                                     
        omega = 2.0 * np.pi * freqs                                                             

        
        tau_min = 1.0 / (np.max(omega) * 50.0)                                                  
        tau_max = 1.0 / (np.min(omega) / 50.0)                                                  
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), num=n_freq + 40)                
        n_tau = len(tau)                                                                        

        
        A_re = np.zeros((n_freq, n_tau + 2))                                                    
        A_im = np.zeros((n_freq, n_tau + 2))                                                    
        A_re[:, 0] = 1.0                                                                        
        A_im[:, 1] = omega                                                                      

        for i in range(n_freq):                                                                 
            for k in range(n_tau):                                                              
                wt = omega[i] * tau[k]                                                          
                den = 1.0 + wt ** 2                                                             
                A_re[i, k + 2] = (wt ** 2) / den                                                
                A_im[i, k + 2] = wt / den                                                       

        A_stack = np.vstack((A_re, A_im))                                                       

        
        L = np.zeros((n_tau + 2, n_tau + 2))                                                    
        for k in range(3, n_tau + 1):                                                           
            L[k, k - 1] = -1.0                                                                  
            L[k, k] = 2.0                                                                       
            L[k, k + 1] = -1.0                                                                  

        Z_kk = np.empty_like(Z_array, dtype="complex")                                          

        for i in range(Z_array.shape[1]):                                                       
            for j in range(Z_array.shape[2]):                                                   
                Z_real = Z_array[:, i, j].real                                                  
                Z_imag = Z_array[:, i, j].imag                                                  
                Z_data = Z_real + 1j * Z_imag                                                   
                abs_Z = np.abs(Z_data) + 1e-12                                                  

                
                std_dev = np.sqrt(var_array[:, i, j])                                           
                rel_err = std_dev / abs_Z                                                       

                
                rel_err_clipped = np.clip(rel_err, a_min=0.02, a_max=None)                      

                
                std_dev_adjusted = rel_err_clipped * abs_Z                                      
                w_initial = 1.0 / (std_dev_adjusted + 1e-12)                                    
                w_initial = w_initial / np.median(w_initial)                                    
                w_plot = w_initial / np.max(w_initial)

                weights = w_initial.copy()                                                      

                
                for _iter in range(5):                                                          
                    W_diag = np.concatenate((weights, weights))                                 
                    W_matrix = np.diag(W_diag)                                                  

                    A_weighted = np.dot(W_matrix, A_stack)                                      
                    y_stack = np.concatenate((Z_real, Z_imag))                                  
                    y_weighted = np.dot(W_matrix, y_stack)                                      

                    A_reg = np.vstack((A_weighted, lam * L))                                    
                    y_reg = np.concatenate((y_weighted, np.zeros(n_tau + 2)))                   

                    
                    x_fit, _, _, _ = linalg.lstsq(A_reg, y_reg, rcond=None)                     

                    Z_re_fit = np.dot(A_re, x_fit)                                              
                    Z_im_fit = np.dot(A_im, x_fit)                                              
                    Z_fit = Z_re_fit + 1j * Z_im_fit                                            

                    
                    rel_resids = np.abs(Z_data - Z_fit) / abs_Z                                 

                    
                    scale = np.median(rel_resids) * 1.4826 + 1e-12                              
                    u = rel_resids / (4.685 * scale)                                            
                    w_robust = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)                  

                    
                    weights = w_initial * w_robust                                              
                    weights = weights / (np.median(weights) + 1e-12)                            

                
                Z_kk[:, i, j] = Z_fit                                                           

        return Z_kk

    def getWindowSmooth(self, **kwargs):
        """Window smoothing length
        
        Power spectra data is smoothed. This returns the size of the smoothing window.
        
        Parameters
        ----------
        datasize : int
            The size of the data

        Returns
        -------
        smoothLen : int
            Smoothing size
        """
        # check if window size specified by user
        if self.winSmooth != -1 and self.winSmooth > 1:
            return self.winSmooth
        # if not, calculate based on datasize
        if "datasize" in kwargs:
            winSmooth = kwargs["datasize"] * 1.0 / 16.0
            if winSmooth < 3:
                return 3  # minimum smoothing
            # otherwise round to nearest odd number
            winSmooth = np.ceil(winSmooth) // 2  # this is floor division
            return int(winSmooth * 2 + 1)
        # otherwise, return a default value
        return 15

    def checkForBadValues(self, numWindows: int, data: np.ndarray):
        """Check data for bad values and remove
        
        Parameters
        ----------
        numWindows : int
            The number of windows
        data : np.ndarray 
            Cross-spectra data

        Returns
        -------
        numGoodWindows : int
            The number of good windows
        goodData : np.ndarray
            The cross-spectra data with bad windows removed
        """
        finiteArray = np.ones(shape=(numWindows))
        for iW in range(0, numWindows):
            if not np.isfinite(data[iW]).all():
                finiteArray[iW] = 0
        numGoodWindows = sum(finiteArray)
        if numGoodWindows == numWindows:
            return numWindows, data
        self.printWarning(
            "Bad data found...number of windows reduced from {} to {}".format(
                numWindows, numGoodWindows
            )
        )
        goodWindowIndices = np.where(finiteArray == 1)
        return numGoodWindows, data[goodWindowIndices]

    def prepareLinearEqn(self, data: np.ndarray):
        r"""Prepare data as a linear equation for the robust regression
        
        This prepares the data for the following type of solution,
        
        .. math::
            y = Ax,

        where :math:`y` is the observations, :math:`A` is the regressors and :math:`x` is the unknown. 
                                                         
        The number of observations is number of windows * number of cross-power channels 
        The shapes of the arrays are as follows:
        
            - y is (number of output channels, number of observations)                              
            - A is (number of output channels, number of observations, number of input channels)    
            - x is (number of output channels, number of input channels)                            

        Consider the impedance tensor,  

        .. math::
            :nowrap:

            \begin{eqnarray}
            E_x & = & Z_{xx} H_x + Z_{xy} H_y \\
            E_y & = & Z_{yx} H_x + Z_{yy} H_y 
            \end{eqnarray}  

        Here, there are two input channels, :math:`H_x`, :math:`H_y` and two output channels :math:`E_x` and :math:`E_y`. In total, there are four components of the unknown impedance tensor, :math:`Z_{xx}`, :math:`Z_{xy}`, :math:`Z_{yx}`, :math:`Z_{yy}` (number of input channels * number of output channels). The number of observations is the number of windows multiplied by the number of channels used for cross-power spectra.     
        
        Parameters
        ----------
        data : np.ndarray
            Cross-power spectral data at evaluation frequencies
            
        Returns
        -------
        numWindows : int
            The number of windows included in the regression (after bad value removal)
        obs : np.ndarray
            Observations array
        reg : np.ndarray 
            Regressors array
        """
        numWindows = data.shape[0]                      
        # check for bad values
        numWindows, data = self.checkForBadValues(numWindows, data)
        crossSize = len(self.crossChannels)             
        # for each output variable, have number of input regressor variables
        # construct our arrays
        obs = np.empty(shape=(self.outSize, crossSize * numWindows), dtype="complex")       
        reg = np.empty(
            shape=(self.outSize, crossSize * numWindows, self.inSize), dtype="complex"      
        )
        for iW in range(0, numWindows):             
            iOffset = iW * crossSize                
            for i in range(0, self.outSize):                            
                for j, crossChan in enumerate(self.crossChannels):      
                    # this is the observation row where i is the observed output
                    crossIndex = self.allChannels.index(crossChan)
                    obs[i, iOffset + j] = data[iW, self.inSize + i, crossIndex]         
                    for k in range(0, self.inSize):
                        reg[i, iOffset + j, k] = data[iW, k, crossIndex]                
        return numWindows, obs, reg                         

    

    def newProcess(
            self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Robust regression processing
        
        """
        crossSize = len(self.crossChannels)
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
        varOutput = np.empty(shape=(self.outSize, self.inSize), dtype="float")

        for i in range(0, self.outSize):
            observation = obs[i, :]
            predictors = reg[i, :, :]

            
            # ---------------------------------------------------------
            out, resids, weights = robustmahalanobisDistance(
                predictors, observation, intercept=self.intercept
            )

            if self.intercept:
                out_impedance = out[1:]  
            else:
                out_impedance = out

            
            # ---------------------------------------------------------
            num_boot = 500  
            n_obs = predictors.shape[0]
            p_vars = predictors.shape[1]
            boot_params = np.empty((num_boot, p_vars), dtype="complex")

            W_sqrt = np.sqrt(weights)[:, np.newaxis]
            A_w = predictors * W_sqrt
            y_w = observation * np.sqrt(weights)

            for b in range(num_boot):
                
                idx = np.random.randint(0, n_obs, n_obs)
                params_b, _, _, _ = linalg.lstsq(A_w[idx, :], y_w[idx], rcond=None)
                boot_params[b, :] = params_b

            cov_params_fw = np.cov(boot_params.T)

            psi_prime_vals = weights  

            W_mat = np.dot(predictors.conj().T, predictors * weights[:, np.newaxis])

            D_mat = np.dot(predictors.conj().T, predictors * psi_prime_vals[:, np.newaxis])

            
            try:
                D_inv = np.linalg.inv(D_mat)
                K_FRB = np.dot(D_inv, W_mat)
            except np.linalg.LinAlgError:
                
                K_FRB = np.eye(p_vars)

            cov_params_corrected = np.dot(np.dot(K_FRB, cov_params_fw), K_FRB.conj().T)

            varOut = np.diag(cov_params_corrected).real

            output[i] = out_impedance

            if self.intercept:
                varOutput[i] = varOut[1:]
            else:
                varOutput[i] = varOut

        return output, varOutput


    def robustProcess(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Robust regression processing
        
        Perform robust regression processing using observations and regressors for a single evaluation frequency. 
        
        Parameters
        ----------
        numWindows : int
            The number of windows   
        obs : np.ndarray
            The observations    
        reg : np.ndarray
            The regressors      

        Returns
        -------
        output : np.ndarray
            The solution to the regression problem  
        varOutput : np.ndarray
            The variance    
        """
        crossSize = len(self.crossChannels)
        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")       
        varOutput = np.empty(shape=(self.outSize, self.inSize), dtype="float")      
        # solve
        for i in range(0, self.outSize):        
            observation = obs[i, :]             
            predictors = reg[i, :, :]           
            # save the output
            # out, resids, weights = chatterjeeMachler(                               
            #     predictors, observation, intercept=self.intercept                   
            # )                                                                       
            out, resids, scale, weights = mmestimateModel(predictors, observation, intercept=self.intercept)          
            
            # now take the weights, apply to the observations and predictors, stack the appropriate rows
            observation2 = np.zeros(shape=(crossSize), dtype="complex")                 
            predictors2 = np.zeros(shape=(crossSize, self.inSize), dtype="complex")     
            for iChan in range(0, crossSize):                     
                # now need to have my indexing array
                indexArray = np.arange(iChan, numWindows * crossSize, crossSize)        
                weightsLim = weights[indexArray]                                        
                # weightsLim = weightsLim/np.sum(weightsLim) # normalise weights to 1   
                observation2[iChan] = (
                    np.sum(obs[i, indexArray] * weightsLim) / numWindows                
                )
                # now for the regressors
                for j in range(0, self.inSize):             
                    predictors2[iChan, j] = (
                        np.sum(reg[i, indexArray, j] * weightsLim) / numWindows         
                    )
            
            # out, resids, weights = chatterjeeMachler(                                   
            #     predictors2, observation2, intercept=self.intercept                                           
            # )                    
            out, resids, scale, weights = mmestimateModel(                            
                predictors2, observation2, intercept=self.intercept)
            
            if self.intercept:
                out = out[1:]               
            
            # now calculate out the varainces - have the solution out, have the weights 
            # recalculate out the residuals with the final solution 
            # calculate standard deviation of residuals 
            # and then use chatterjee machler formula to estimate variances
            # this needs work - better to use an empirical bootstrap method, but this will do for now 
            
            resids = np.absolute(observation - np.dot(predictors, out))         # 残差计算：使用最终阻抗估计计算所有原始数据点的残差。计算最终残差r=|y-Ax|：使用原始数据和最终阻抗估计
            scale = sampleMAD0(                 # 计算标准差scale：标准差σ的稳健估计。使用样本中位数绝对偏差(MAD)估计残差尺度（稳健的标准差估计），MAD是标准差的一种稳健估计，对异常值不敏感。
                resids                          # 稳健尺度估计：sampleMAD0使用中位数绝对偏差，对异常值稳健
            )  # some measure of standard deviation, rather than using the standard deviation(采用某种标准差类的度量指标，而非直接使用标准差)
            residsVar = scale * scale           # 计算方差residsVar：方差σ²的稳健估计。方差估计：residsVar作为误差方差的稳健估计。计算残差方差,方差 = 标准差的平方。Var(r) = σ²，其中σ是尺度估计。注意：这是稳健的方差估计，不是样本方差。
            # 接下来对设计矩阵predictors的信息矩阵varPred的运算；对信息矩阵求逆，得到精度矩阵varPred；最后计算参数的方差-协方差矩阵varOut。
            # varPred = np.dot(hermitianTranspose(predictors), weights*predictors) # need to fix this(注释方案使用加权版本，当前使用未加权版本)     # hermitianTranspose()：计算共轭转置（对复数数据）
            varPred = np.dot(hermitianTranspose(predictors), predictors)                        # 计算predictors的信息矩阵矩阵。 XᵀX矩阵,其中X是predictors。在MT中：HᵀH，反映磁场数据的变异性及相关性
            varPred = np.linalg.inv(varPred)  # this is a pxp matrix(这是一个 pxp 矩阵)           # 对信息矩阵求逆，得到精度矩阵，反映参数估计的精度。(XᵀX)⁻¹矩阵，参数协方差矩阵的核心部分。在MT中：反映阻抗估计的潜在精度，受磁场变异性的影响。
            varOut = 1.91472 * residsVar * varPred                # 计算参数的方差-协方差矩阵varOut：使用稳健方差估计公式：Var(β) = k * σ² * (XᵀX)⁻¹,系数1.91472是针对MAD估计的校正因子，用于小样本校正。σ²=residsVar（残差方差）。MT中：阻抗估计的方差-协方差矩阵
            varOut = np.diag(varOut).real  # this should be a real number   # 提取参数方差的对角线元素，并取实部。在MT中：每个阻抗元素的方差，如Var(Zxx), Var(Zxy)等
            # if self.intercept:                                            # 非对角线: Cov(Zxx, Zxy) - 参数间的协方差
            #     output[i] = out[1:]                                       # 对角线: Var(Zxx), Var(Zxy) - 各参数的方差
            #     varOutput[i] = varOut[1:]                                 # 参数方差-协方差矩阵的结构：
            # else:                                                         # Cov(β) = [[Var(Zxx),    Cov(Zxx, Zxy)]
            #     output[i] = out                                           #           [Cov(Zxy, Zxx), Var(Zxy)]]
            #     varOutput[i] = varOut
            # 存储当前输出通道的估计结果
            output[i] = out             # 阻抗估计（复数）
            varOutput[i] = varOut       # 方差估计（实数）

        return output, varOutput

    

    def olsProcess(
        self, numWindows: int, obs: np.ndarray, reg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ordinary least squares regression processing
        普通最小二乘法回归处理
        Perform ordinary least regression processing using observations and regressors for a single evaluation frequency. 
        使用单个评估频率的观测值和回归量执行普通最小二乘回归处理
        Parameters
        ----------
        numWindows : int
            The number of windows
        obs : np.ndarray
            The observations
        reg : np.ndarray
            The regressors

        Returns
        -------
        output : np.ndarray
            The solution to the regression problem
        varOutput : np.ndarray
            The variance        
        """
        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")                       # 创建输出数组：output存储回归解（复数类型），形状为(输出维度, 输入维度)；
        varOutput = np.empty(shape=(self.outSize, self.inSize), dtype="float")                      # varOutput存储方差（浮点类型）
        # solve
        for i in range(0, self.outSize):                                                            # 遍历outSize个输出维度(Ex,Ey)
            observation = obs[i, :]
            predictors = reg[i, :, :]
            # save the output
            out, resids, squareResid, rank, s = olsModel(                                           # 调用OLS模型求解回归系数：out=回归系数，resids=残差，squareResid=残差平方和，rank=矩阵秩，s=奇异值
                predictors, observation, intercept=self.intercept                                   # 对所有窗口的功率谱统一最小二乘求解得到阻抗（2个：Zxx，Zxy）
            )            
            if self.intercept:
                out = out[1:]                                                                       # 如果设置了截距项（intercept=True），olsModel返回的系数包含截距（第一个元素），需剔除截距只保留自变量系数

            # now calculate out the varainces - have the solution out, have the weights(现在计算出方差——得出解决方案，确定权重)
            # recalculate out the residuals with the final solution(用最终解重新计算出残差)
            # calculate standard deviation of residuals(计算残差的标准差)
            # and then use chatterjee machler formula to estimate variances(然后使用chatterjee machler公式来估计方差)
            # this needs work - better to use an empirical bootstrap method, but this will do for now(这还需要改进——最好使用经验自助法，但目前这样也凑合)
            resids = np.absolute(observation - np.dot(predictors, out))                             # 重新计算残差：|观测值 - 预测值|（y-Ax）（预测值=回归系数×回归变量的线性组合）。np.dot做矩阵乘法（向量×矩阵）
            scale = sampleMAD0(
                resids                                                                              # 用样本MAD（中位数绝对偏差）估计残差的尺度参数（替代标准差，对异常值更稳健）
            )  # some measure of standard deviation, rather than using the standard deviation(采用一定程度的标准差指标，而非直接使用标准差)
            residsVar = scale * scale                                                               # 残差的方差 = 尺度参数的平方（将MAD转换为方差估计）
            # varPred = np.dot(hermitianTranspose(predictors), weights*predictors) # need to fix this
            varPred = np.dot(hermitianTranspose(predictors), predictors)                            # 计算预测变量矩阵的Hermitian转置（共轭转置，适用于复数矩阵；若为实数则与普通转置等价）与自身的乘积
            varPred = np.linalg.inv(varPred)  # this is a pxp matrix(结果是p×p矩阵（p为预测变量个数）)   # varPred：设计矩阵的方差-协方差矩阵。对上述乘积矩阵求逆：得到(X^H X)^-1（X为预测变量矩阵，H表示Hermitian转置）
            varOut = 1.91472 * residsVar * varPred                                                  # 计算回归系数的方差-协方差矩阵（应用 Chatterjee-Mächler 公式）。方差 = 常数1.91472 × 残差方差 × (X^H X)^-1 （注：此处公式可能为特定场景的经验修正）
            varOut = np.diag(varOut).real  # this should be a real number                           # 提取对角线元素并取实部
            # if self.intercept:
            #     output[i] = out[1:]
            #     varOutput[i] = varOut[1:]
            # else:
            #     output[i] = out
            #     varOutput[i] = varOut
            output[i] = out                                                                         # 保存当前输出维度的回归系数和方差
            varOutput[i] = varOut            

        return output, varOutput

    def stackedProcess(self, data: np.ndarray) -> np.ndarray:
        """Ordinary least squares processing after stacking
        叠加后的普通最小二乘处理
        Parameters
        ----------
        data : np.ndarray
            Cross-spectra data      功率谱数据

        Returns
        -------
        output : np.ndarray
            The solution to the regression problem
        varOutput : np.ndarray
            The variance        
        """
        # then do various sums
        numWindows = data.shape[0]                                                              # 计算时间窗口数量：通过数据的第一个维度获取窗口总数
        crossSize = len(self.crossChannels)                                                     # 获取通道的数量：即参与功率谱计算的通道数
        # unweighted sum (i.e. normal solution)
        unWeightedSum = np.sum(data, axis=0)                                                    # 计算未加权的和：对所有时间窗口的功率谱数据求和（沿第0轴）
        unWeightedSum = unWeightedSum / numWindows                                              # 计算平均值：将总和除以窗口数量，得到平均功率谱矩阵

        # for each output variable, have ninput regressor variables(对于每个输出变量，都有n个输入回归变量。)
        # let's construct our arrays(让我们构建我们的数组)
        obs = np.empty(shape=(self.outSize, crossSize), dtype="complex")                        # 创建观测数据数组
        reg = np.empty(shape=(self.outSize, crossSize, self.inSize), dtype="complex")           # 创建回归变量数组
        for i in range(0, self.outSize):                                                        # 遍历所有输出维度['Ex','Ey']
            for j, crossChan in enumerate(self.crossChannels):                                  # 遍历所有交叉通道['Hx','Hy','Ex','Ey']
                crossIndex = self.allChannels.index(crossChan)                                  # 获取当前交叉通道在所有通道列表中的索引位置
                obs[i, j] = unWeightedSum[self.inSize + i, crossIndex]                          # 从平均功率谱矩阵中提取观测数据obs
                for k in range(0, self.inSize):                                                 # 遍历所有输入维度['Hx','Hy']
                    reg[i, j, k] = unWeightedSum[k, crossIndex]                                 # # 从平均功率谱矩阵中提取回归变量reg

        # create array for output
        output = np.empty(shape=(self.outSize, self.inSize), dtype="complex")                   # 创建输出数组：用于存储回归系数x结果，复数类型

        for i in range(0, self.outSize):                                                        # 对每个输出维度进行稳健回归计算
            observation = obs[i, :]                                                             # 提取当前输出维度的观测数据向量
            predictors = reg[i, :, :]                                                           # 提取当前输出维度的预测变量矩阵
            # save the output
            out, resids, scale, weights = mmestimateModel(                                      # 调用MM估计稳健回归模型进行计算
                predictors, observation, intercept=self.intercept                               # 返回：回归系数、残差、尺度参数(标准差)、权重
            )
            if self.intercept:
                output[i] = out[1:]
            else:
                output[i] = out
        return output, output*0.1                                                               # 返回回归系数和方差估计（这里方差简单设为系数的10%）

    def writeTF(
        self,
        specdir: str,
        postpend: str,
        freq,
        data: np.ndarray,
        variances: np.ndarray,
        **kwargs
    ):
        """Write the transfer function file

        Parameters
        ----------
        specdir : str
            The spectra data being used for the transfer function estimate
        postpend : str
            The optional postpend to the transfer function file
        data : np.ndarray
            The transfer function estimates
        variances : np.ndarray
            The transfer function variances
        remotesite : str, optional
            Optionally, if there is a remote site
        remotechans : List[str], optional
            Optionally add the remote channels if there is a remote site
        """
        # path for writing out to
        sampleFreqStr = fileFormatSampleFreq(self.decParams.sampleFreq)
        if postpend == "":
            filename = "{}_fs{:s}_{}".format(self.outSite, sampleFreqStr, specdir)
        else:
            filename = "{}_fs{:s}_{}_{}".format(
                self.outSite, sampleFreqStr, specdir, postpend
            )
        datapath = os.path.join(self.outpath, sampleFreqStr)
        checkAndMakeDir(datapath)
        outfile = os.path.join(datapath, filename)
        # now construct the transferFunctionData object
        numFreq = len(freq)
        dataDict = {}
        varDict = {}
        for i in range(0, self.outSize):
            for j in range(0, self.inSize):
                key = "{}{}".format(self.outChannels[i], self.inChannels[j])
                dataArray = np.empty(shape=(numFreq), dtype="complex")
                varArray = np.empty(shape=(len(freq)), dtype="float")
                for ifreq in range(0, numFreq):
                    dataArray[ifreq] = data[ifreq][i, j]
                    varArray[ifreq] = variances[ifreq][i, j]
                dataDict[key] = dataArray
                varDict[key] = varArray
        tfData = TransferFunctionData(freq, dataDict, varDict)
        # now make the writer and write out
        tfWriter = TransferFunctionWriter(outfile, tfData)
        tfWriter.setHeaders(
            sampleFreq=self.decParams.sampleFreq,
            insite=self.inSite,
            inchans=self.inChannels,
            outsite=self.outSite,
            outchans=self.outChannels,
        )
        if "remotesite" in kwargs:
            tfWriter.addHeader("remotesite", kwargs["remotesite"])
        if "remotechans" in kwargs:
            tfWriter.addHeader("remotechans", kwargs["remotechans"])
        tfWriter.write()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        list
            List of strings with information
        """
        textLst = []
        textLst.append("In Site = {}".format(self.inSite))
        textLst.append("In Channels = {}".format(self.inChannels))
        textLst.append("Out Site = {}".format(self.outSite))
        textLst.append("Out Channels = {}".format(self.outChannels))
        textLst.append("Sample frequency = {:.3f}".format(self.decParams.sampleFreq))
        return textLst
