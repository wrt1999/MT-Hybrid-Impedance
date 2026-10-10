import numpy as np
import scipy.interpolate as interp
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from resistics.common.base import ResisticsBase
from resistics.common.print import listToString, arrayToString
from resistics.common.smooth import smooth1d
from resistics.spectra.data import SpectrumData
from resistics.regression.robust import olsModel
from resistics.regression.robust import mestimateModel
from resistics.regression.robust import mmestimateModel
from resistics.regression.robust import mahalanobisDistance
from resistics.regression.robust import mahalanobisDistance_fixed
from resistics.regression.robust import chatterjeeMachler
from resistics.regression.robust import chatterjeeMachler_fixed
from resistics.regression.robust import mestimateModel_fixed
from resistics.regression.robust import mmestimateModel_fixed
from resistics.regression.robust import robustmahalanobisDistance
from resistics.regression.robust import robustmahalanobisDistance_fixed


class StatisticCalculator(ResisticsBase):
    """Calculate statistics for data restriction
    计算数据限制（抽取）的统计量
    Statistics are calculated out for each evaluation frequency in each window. Therefore, there are nwindow*nfreq statistics in total.
    统计量为每个窗口的每个评估频率计算，因此总共有 nwindow*nfreq 个统计量
    This class was written to speed up statistic calculations. Many statistics need the same data, for example power spectra. This class calculates and reuses some common values amongst the various statistics to improve calculation speed.
    这个类是为了加速统计量计算而编写的。许多统计量需要相同的数据，例如功率谱。这个类计算并重用各种统计量之间的公共值以提高计算速度。
    Attributes
    ----------
    evalFreq : List
        List of evaluation frequencies
    winLen : int
        Window length for spectra calculations
    winType : str
        Window function to apply to time data before fourier transform
    inChans : List[str]
        Input channels
    inSize : int
        Number of input channels
    outChans : List[str]
        Output channels 
    outSize : int
        Number of output channels
    specChans : List[str] 
        The channels for which to calculate auto and cross power spectra
    remoteChans : List[str]
        Remote reference channels
    psdChans : List[str] 
        Power spectral density channels
    cohPairs : List[List[str]]
        Pairs of channels for coherence calculations
    polDirs : List[List[str]]
        Pairs of channels of polarisation direction calculation
    spec: Dict[str, np.ndarray]
        The spectra data dictionary mapping channel to data array
    tfCalculated : bool 
        Boolean flag to show that the transfer function has been calculated for a window
    remoteCalculated : bool 
        Boolean flag to show if the remote transfer function has been calculated
    intercept : bool   
        Boolean flag to include an intercept into the 
    outData: Dict = {}

    Methods
    -------
    __init__()
    getEvalFreq()
        Get a list of the evaluation frequency
    getInChans()
        Get a list of the input channels for the transfer function calculation
    getOutChans()
        Get a list of the output channels for the transfer function calculation
    getSpecChans()
        Get a list of the spectra channels
    getRemoteChans()
        Get a list of the remote channels
    getPSDChans()
        Get a list of channels for which to calculate power spectral density
    getCohPairs()
        Get a list of coherence pairs to calculate
    getPolDirs()
        Get a list of polarisation direction channels to calculate
    getAutoPower(chan)
        Get auto power data for a channel
    getAutoPowerEval(chan, eIdx)
        Get auto power data for a channel calculated at evaluation frequencies
    getCrossPower(chan1, chan2)
        Get cross power data between two channels
    getCrossPowerEval(chan1, chan2, eIdx)
        Get cross power data between two channels at the evaluation frequencies
    getOutData()
        Get the output data
    setInChans(inChans)
        Set the input channels for transfer function statistics
    setOutChans(outChans)
        Set the output channels for the transfer function statistics
    setSpectra(freq, winData, evalFreq)
        Set the spectra data
    setIntercept(intercept)
        Boolean flag for adding an intercept term
    calculateSpectralMatrix()
        Calculate the cross power spectra matrix
    calculateEvalMatrix()
        Calculate the cross power spectra matrix at the evaluation frequencies
    addRemoteSpec(remoteData, **kwargs)
        Add a remote reference spectra
    calculateRemoteSpectralMatrix()
        Calculate the remote data power spectral matrix
    calculateRemoteEvalMatrix()
        Calculate the remote data power spectra matrix at the evaluation frequencies
    calculateReferenceSpectralMatrix()
        Calculate the power spectral matrix for remote reference processing
    calculateReferenceEvalMatrix()
        Calculate the power spectral matrix for remote reference processing at the evaluation frequencies
    getRemoteAutoPower(chan)
        Get remote data auto power spectra
    getRemoteAutoPowerEval(chan, eIdx)
        Get remote data auto power spectra at evaluation frequency
    getRemoteCrossPower(chan1, chan2)
        Get remote cross power spectra data
    getRemoteCrossPowerEval(chan1, chan2, eIdx)
        Get remote cross power spectra value at evaluation frequency
    getReferenceCrossPower(dataChan, remoteChan)
        Get the remote reference formulation cross power spectra
    getReferenceCrossPowerEval(dataChan, remoteChan, eIdx)
        Get the remote reference formulation cross power spectra at a given evaluation frequency
    interpolateToEvalFreq(data)
        Interpolate data to evaluation frequencies
    prepareOutDict()
        Prepare the output dictionary which will be returned
    getDataForStatName(statName)
        Get the statistic data for a statistic
    winPSD()
        Calculate window power spectral density
    winCoherence()
        Calculate window coherence for coherence pairs
    winPolarisations()
        Calculate polarisation directions
    winPartials()
        Calculate window partial coherences
    winAbsVal()
        Calculate window absolute values for the evaluation frequencies        
    winTransferFunction()
        Calculate window transfer function
    winRemoteCoherence()
        Calculate window remote coherence with local data
    winRemoteEqnCoherence()
        Calculate remote equation coherence
    winRemoteAbsVal()
        Calculate remote absolute value
    winRemoteTransferFunction()
        Calculate remote transfer function
    printList()
        Class information returned as list of strings  
    """

    def __init__(self) -> None:
        """Initialise the statistic calculator"""
        """初始化统计量计算器"""
        # default evaluation frequencies
        self.evalFreq: List = []                                            # 默认评估频率（空列表，需要后续设置）
        # power smoothing vals(功率谱平滑参数)
        self.winLen: int = 9                                                # 平滑窗口长度，默认9个频率点
        self.winType: str = "hanning"                                       # 窗函数类型，默认汉宁窗
        # set some defaults
        self.inChans: List[str] = ["Hx", "Hy"]                              # 输入通道（水平磁场分量）
        self.inSize: int = len(self.inChans)                                # 输入通道数量：2
        self.outChans: List[str] = ["Ex", "Ey"]                             # 输入通道（水平磁场分量）
        self.outSize: int = len(self.outChans)                              # 输入通道数量：2
        self.specChans: List[str] = self.inChans + self.outChans            # 频谱计算通道（输入+输出）['Hx', 'Hy', 'Ex', 'Ey']
        self.remoteChans: List[str] = self.inChans                          # 远参考通道（默认使用输入通道作为远程参考）['Hx', 'Hy']
        self.psdChans: List[str] = ["Ex", "Ey", "Hx", "Hy"]                 # 功率谱密度通道（所有场分量）
        self.cohPairs: List[List[str]] = [                                  # 相干性计算通道对
            ["Ex", "Hx"],                                                   # Ex-Hx相干性
            ["Ex", "Hy"],
            ["Ey", "Hx"],
            ["Ey", "Hy"],
        ]
        self.polDirs: List[List[str]] = [["Ex", "Ey"], ["Hx", "Hy"]]        # 极化方向计算通道对([电场极化方向],[磁场极化方向])，分析场的极化特性
        # set data presets(设置数据预设值)
        self.freq: Union[np.ndarray, None] = None                           # 频率数组，初始为None
        self.spec: Dict[str, np.ndarray] = {}                               # 频谱数据字典，初始为空
        # output data and marker for transfer function calculated(输出数据和传递函数计算标记)
        self.tfCalculated: bool = False                                     # 传递函数计算标志,tfCalculated标记阻抗张量是否已计算
        self.remoteCalculated: bool = False                                 # 远程传递函数计算标志,remoteCalculated标记远程参考处理状态
        self.intercept: bool = False                                        # 截距项标志,intercept控制是否在回归中包含截距项
        self.outData: Dict = {}                                             # 输出数据字典,outData存储所有计算结果的字典

    def getEvalFreq(self):
        """Get a copy of the evaluation frequency
        
        Returns
        -------
        List[float]
            List of evaluation frequencies
        """
        return deepcopy(self.evalFreq)

    def getInChans(self) -> List[str]:
        """Get a copy of the input channels
        
        Returns
        -------
        List[str]
            List of input channels
        """
        return deepcopy(self.inChans)

    def getOutChans(self) -> List[str]:
        """Get a copy of the output channels
        
        Returns
        -------
        List[str]
            List of output channels
        """
        return deepcopy(self.outChans)

    def getSpecChans(self):
        """Get a copy of the channels for which to calculate spectra
        
        Returns
        -------
        List[str]
            List of spectra channels
        """
        return deepcopy(self.specChans)

    def getRemoteChans(self) -> List[str]:
        """Get a copy of the remote reference channels
        
        Returns
        -------
        List[str]
            List of remote reference channels
        """
        return deepcopy(self.remoteChans)

    def getPSDChans(self) -> List[str]:
        """Get a copy of the channels to include power spectral density
        
        Returns
        -------
        List[str]
            List of power spectral density channels
        """
        return deepcopy(self.psdChans)

    def getCohPairs(self) -> List[List[str]]:
        """Get a copy of coherence channel pairs to calculate out
        
        Returns
        -------
        List[List[str]]
            List of coherence pairs
        """
        return deepcopy(self.cohPairs)

    def getPolDirs(self) -> List[List[str]]:
        """Get a list of polarisation direction pairs
        
        Returns
        -------
        List[List[str]]
            List of polarisation direction pairs
        """
        return deepcopy(self.polDirs)

    def getAutoPower(self, chan: str) -> np.ndarray:
        """Get the auto power for a channel

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        
        Returns
        -------
        np.ndarray
            The auto power for the channel
        """
        idx = self.specChans.index(chan)
        # then return the autopower
        return self.spectralMatrix[idx, idx].real

    def getAutoPowerEval(self, chan: str, eIdx: int) -> float:
        """Get the auto power value for an particular evaluation frequency

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        eIdx : int
            The index for the evaluation frequency
        
        Returns
        -------
        np.ndarray
            The auto power for the channel
        """
        idx = self.specChans.index(chan)
        # then return the autopower
        return self.evalMatrix[idx, idx, eIdx].real

    def getCrossPower(self, chan1: str, chan2: str) -> np.ndarray:
        """Get the cross power between two channels

        Parameters
        ----------
        chan1 : str
            The first channel for the cross channels
        chan2 : str
            The second channl for the cross channels
        
        Returns
        -------
        np.ndarray
            The cross power spectral density
        """
        idx1 = self.specChans.index(chan1)
        idx2 = self.specChans.index(chan2)
        # then return the autopower
        return self.spectralMatrix[idx1, idx2]

    def getCrossPowerEval(self, chan1: str, chan2: str, eIdx: int) -> float:
        """Get the cross power between two channels
        获取两个通道之间的功率谱
        Parameters
        ----------
        chan1 : str
            The first channel for the cross channels
        chan2 : str
            The second channl for the cross channels
        eIdex : int
            The index of the evaluation frequency
        
        Returns
        -------
        np.ndarray
            The cross power spectral density
        """
        idx1 = self.specChans.index(chan1)                  # 获取第一个通道的索引(输入或输出通道)
        idx2 = self.specChans.index(chan2)                  # 获取第二个通道的索引(所有4个通道)
        # then return the autopower
        return self.evalMatrix[idx1, idx2, eIdx]            # 从评估频率功率谱矩阵中返回对应的功率谱值

    def getOutData(self) -> Dict:
        """Get the output data

        Returns
        -------
        Dict
            The statistic output data
        """
        return deepcopy(self.outData)

    def setInChans(self, inChans: List[str]) -> None:
        """Set the input channels

        Parameters
        ----------
        inChans : List[str]
            Input channels for the magnetotelluric linear system
        """
        self.inChans = inChans
        self.inSize = len(self.inChans)

    def setOutChans(self, outChans: List[str]):
        """Set the output channels

        Parameters
        ----------
        inChans : List[str]
            Output channels for the magnetotelluric linear system
        """
        self.outChans = outChans
        self.outSize = len(self.outChans)

    def setRemoteChans(self, remoteChans):
        """Set the input channels

        Parameters
        ----------
        inChans : List[str]
            Input channels for the magnetotelluric linear system
        """
        self.remoteChans = remoteChans

    def setPSDChans(self, psdChans: List[str]) -> None:
        """Set the power spectral density channels

        Parameters
        ----------
        psdChans : List[str]
            Power spectral density channels. An example input would be: ["Ex", "Ey", "Hx", "Hy"]
        """
        self.psdChans = psdChans

    def setCohPairs(self, cohPairs: List[List[str]]) -> None:
        """Set the power spectral density channels

        If cohPairs of [["Ex", "Hx"], ["Ex", "Hy"], ["Ey", "Hx"], ["Ey", "Hy"]] are set, the following coherences will be calculated:
        ExHx
        ExHy
        EyHx
        EyHy

        Parameters
        ----------
        cohPairs : List[List[str]]
            Set the coherence pairs using a list of channel pairs, for example: [["Ex", "Hx"], ["Ex", "Hy"], ["Ey", "Hx"], ["Ey", "Hy"]]
        """
        self.cohPairs = cohPairs

    def setPolDirs(self, polDirs: List[List[str]]) -> None:
        """Set the polarisation direction pairs to calculate

        If polDirs of [["Ex", "Ey"], ["Hx", "Hy"]] are set, the following polarisation directions will be calculated:
        Ex Ey
        Hx Hy

        Parameters
        ----------
        polDirs : List[List[str]]
            Set polarisation direction channel pairs, for example: [["Ex", "Ey"], ["Hx", "Hy"]]
        """
        self.polDirs = polDirs

    def setSpectra(
        self, freq: np.ndarray, specData: SpectrumData, evalFreq: np.ndarray
    ) -> None:
        """Set the spectra data
        设置频谱数据
        Parameters
        ----------
        freq : np.ndarray
            The frequency points in the spectra data
        specData : SpectrumData
            Spectrum data (i.e. spectrum data for a window)
        evalFreq : np.ndarray
            Evaluation frequency array
        """
        self.freq = freq                                                                # 存储原始频谱的频率点
        self.spec: Dict[str, np.ndarray] = specData.data                                # 从频谱数据对象提取各通道的频谱数据(当前窗口)
        self.evalFreq = evalFreq                                                        # 存储评估频率（实际计算统计量的频率点）
        self.numChans = len(self.specChans)                                             # 计算通道数量
        self.dataSize = specData.dataSize                                               # 获取数据大小（频率点数257）
        # calculate the power matrix(计算功率谱矩阵)
        self.calculateSpectralMatrix()                                                  # 计算频谱每个频率的功率谱矩阵
        self.calculateEvalMatrix()                                                      # 计算插值到评估频率点的功率谱矩阵
        # clear the out dictionary and set that transfer function not calculated(清空输出字典并设置传递函数未计算标志)
        self.prepareOutDict()                                                           # 清空之前的计算结果,将tfCalculated等标志重置为False,确保每次计算都是基于新加载的数据

    def setIntercept(self, intercept: bool):
        """Set the intercept boolean

        Parameters
        ----------
        intercept : bool
            Boolean flag for having an intercept in the transfer function calculation
        """
        self.intercept = intercept

    def calculateSpectralMatrix(self) -> None:
        """Calculate out the cross power spectral matrix
        计算功率谱矩阵
        The method calculates out the cross powers which will then be used in the other statistic calculations.
        该方法计算功率谱，这些功率谱将用于其他统计量的计算
        The cross powers spectral matrix is a 3-D matrix of size:互功率谱矩阵是一个三维矩阵，尺寸为：
        numChans * numChans * numFrequencies                        通道数 × 通道数 × 频率点数
        The elements of this are calculated by multiplying the spectra of one channel by the complex conjugate of the spectra of another channel.
        矩阵元素通过将一个通道的频谱与另一个通道频谱的复共轭相乘得到
        """
        # create the 3d array(创建3D数组存储功率谱矩阵4×4×257)
        self.spectralMatrix = np.empty(
            shape=(self.numChans, self.numChans, self.dataSize), dtype="complex"
        )
        # now need to go through the chans(现在需要遍历所有通道['Hx','Hy','Ex','Ey'])
        for ii in range(0, self.numChans):                                      # 从第一个通道到最后一个通道
            for jj in range(ii, self.numChans):                                 # 内层循环：从当前通道到最后一个通道（只计算上三角部分）
                chan1 = self.specChans[ii]                                      # 第一个通道名称
                chan2 = self.specChans[jj]                                      # 第二个通道名称
                self.spectralMatrix[ii, jj] = smooth1d(                         # 计算功率谱并平滑
                    self.spec[chan1] * np.conjugate(self.spec[chan2]),          # 功率谱计算：X * conj(Y)
                    self.winLen,                                                # 平滑窗口长度(9点滑动平均)
                    self.winType,                                               # 窗函数类型(汉宁窗加权平滑)
                )
                if ii != jj:                                                    # 如果不是对角线元素(互功率谱)
                    # conjugate symmtry(共轭对称性)
                    self.spectralMatrix[jj, ii] = np.conjugate(
                        self.spectralMatrix[ii, jj]                             # 利用共轭对称性补全下三角元素
                    )

    def calculateEvalMatrix(self):
        """Calculate out the cross power spectral matrix at the evaluation frequencies
        计算评估频率点的功率谱矩阵
        The method calculates out the cross powers which will then be used in the other statistic calculations at the evaluation frequencies
        该方法计算功率谱，这些功率谱将用于在评估频率点计算其他统计量
        The cross powers spectral matrix for evaluation frequencies is a 3-D matrix of size:    评估频率的功率谱矩阵是一个三维矩阵，尺寸为：
        numChans * numChans * numEvaluationFrequencies                                          通道数 × 通道数 × 评估频率点数
        The elements of this are calculated by taking the cross powers spectral matrix and using the result there to interpolate the values at the evaluation frequencies.
        矩阵元素通过获取功率谱矩阵并使用其结果插值到评估频率点来计算
        """
        # create the array
        self.evalMatrix = np.empty(
            shape=(self.numChans, self.numChans, len(self.evalFreq)), dtype="complex"           # 创建评估频率功率谱矩阵
        )
        for ii in range(0, self.numChans):                                                      # 从第一个通道到最后一个通道
            for jj in range(ii, self.numChans):                                                 # 内层循环：从当前通道到最后一个通道（只计算上三角部分）
                self.evalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.spectralMatrix[ii, jj]                                                 # 将原始频谱频率的功率谱插值到评估频率
                )
                if ii != jj:                                                                    # 如果是非对角线元素（互功率谱）
                    # conjugate symmtry(共轭对称性)
                    self.evalMatrix[jj, ii] = np.conjugate(self.evalMatrix[ii, jj])             # 利用共轭对称性补全下三角元素

    def addRemoteSpec(
        self, remoteData: SpectrumData, remoteChans: List[str] = []
    ) -> None:
        """Add coincident remote reference spectrum data

        Parameters
        ----------
        remoteData : SpectrumData
            Spectrum data (i.e. spectrum data for a window)
        remoteChans : List[str]
            The channels to use from remote reference data
        """
        self.remoteSpec = remoteData.data
        if len(remoteChans) > 0:
            self.remoteChans = remoteChans
        # now calculate some remote reference related values
        self.calculateRemoteSpectralMatrix()
        self.calculateRemoteEvalMatrix()
        self.calculateReferenceSpectralMatrix()
        self.calculateReferenceEvalMatrix()

    def calculateRemoteSpectralMatrix(self):
        """Calculate out the cross power spectral matrix for the remote reference data

        The method calculates out the cross powers for the remote reference channels which will then be used in the other statistic calculations.

        The remote reference cross powers spectral matrix is a 3-D matrix of size:
        numRemoteChans * numRemoteChans * numFrequencies
        The elements of this are calculated by multiplying the spectra of one channel by the complex conjugate of the spectra of another channel.
        """
        # create the 3d array
        numRemoteChans = len(self.remoteChans)
        self.remoteSpectralMatrix = np.empty(
            shape=(numRemoteChans, numRemoteChans, self.dataSize), dtype="complex"
        )
        # now need to go through the chans
        for ii in range(0, numRemoteChans):
            for jj in range(ii, numRemoteChans):
                chan1 = self.remoteChans[ii]
                chan2 = self.remoteChans[jj]
                self.remoteSpectralMatrix[ii, jj] = smooth1d(
                    self.remoteSpec[chan1] * np.conjugate(self.remoteSpec[chan2]),
                    self.winLen,
                    self.winType,
                )
                if ii == jj:
                    # conjugate symmtry
                    self.remoteSpectralMatrix[jj, ii] = np.conjugate(
                        self.remoteSpectralMatrix[ii, jj]
                    )

    def calculateRemoteEvalMatrix(self):
        """Calculate out the cross power spectral matrix for the remote reference data at the evaluation frequencies

        Takes the cross power spectral data calculate for the remote reference channels and interpoaltes it to the evaluation frequencies
        """
        # create the array
        numRemoteChans = len(self.remoteChans)
        self.remoteEvalMatrix = np.empty(
            shape=(numRemoteChans, numRemoteChans, len(self.evalFreq)), dtype="complex"
        )
        for ii in range(0, numRemoteChans):
            for jj in range(ii, numRemoteChans):
                self.remoteEvalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.remoteSpectralMatrix[ii, jj]
                )
                if ii != jj:
                    # conjugate symmtry
                    self.remoteEvalMatrix[jj, ii] = np.conjugate(
                        self.remoteEvalMatrix[ii, jj]
                    )

    def calculateReferenceSpectralMatrix(self):
        """Calculate out the cross power spectral matrix between the site spectral data and the remote reference spectral data

        The reference cross powers spectral matrix is a 3-D matrix of size:
        numChans * numRemoteChans * numFrequencies
        The elements of this are calculated by multiplying a channel of the site spectral data by the complex conjugate of a channel from the remote reference. 
        """
        # cannot use conjugate symmetry in this case
        self.referenceSpectralMatrix = np.empty(
            shape=(self.numChans, len(self.remoteChans), self.dataSize), dtype="complex"
        )
        for ii, chan1 in enumerate(self.specChans):
            for jj, chan2 in enumerate(self.remoteChans):
                self.referenceSpectralMatrix[ii, jj] = smooth1d(
                    self.spec[chan1] * np.conjugate(self.remoteSpec[chan2]),
                    self.winLen,
                    self.winType,
                )

    def calculateReferenceEvalMatrix(self):
        """Interpolate the remote and site cross powers spectral matrix to the evaluation frequencies.
        """
        self.referenceEvalMatrix = np.empty(
            shape=(self.numChans, len(self.remoteChans), len(self.evalFreq)),
            dtype="complex",
        )
        for ii, chan1 in enumerate(self.specChans):
            for jj, chan2 in enumerate(self.remoteChans):
                self.referenceEvalMatrix[ii, jj] = self.interpolateToEvalFreq(
                    self.referenceSpectralMatrix[ii, jj]
                )

    def getRemoteAutoPower(self, chan: str) -> np.ndarray:
        """Get the auto power of a remote reference channel

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        
        Returns
        -------
        np.ndarray
            The autopower array (real for autopowers)
        """
        idx = self.remoteChans.index(chan)
        return self.remoteSpectralMatrix[idx, idx].real

    def getRemoteAutoPowerEval(self, chan: str, eIdx: int) -> float:
        """Get the auto power of a remote reference channel at an evaluation frequency

        Parameters
        ----------
        chan : str
            The channel for which to get the autopower
        eIdx : int
            The evaluation frequency index
        
        Returns
        -------
        float
            The autopower of the channel at the evaluation frequency
        """
        idx = self.remoteChans.index(chan)
        return self.remoteEvalMatrix[idx, idx, eIdx].real

    def getRemoteCrossPower(self, chan1: str, chan2: str) -> np.ndarray:
        """Get the cross power of two remote reference channels

        Parameters
        ----------
        chan1 : str
            The first channel for the cross power
        chan2 : str
            The second channel for the cross power
        
        Returns
        -------
        np.ndarray
            The cross power array 
        """
        idx1 = self.remoteChans.index(chan1)
        idx2 = self.remoteChans.index(chan2)
        return self.remoteSpectralMatrix[idx1, idx2]

    def getRemoteCrossPowerEval(self, chan1: str, chan2: str, eIdx: int) -> float:
        """Get the cross power of two remote reference channels at a single evaluation frequency

        Parameters
        ----------
        chan1 : str
            The first channel for the cross power
        chan2 : str
            The second channel for the cross power
        eIdx : int
            The evaluation frequency index            
        
        Returns
        -------
        float
            The value of the cross power at the evaluation frequency 
        """
        idx1 = self.remoteChans.index(chan1)
        idx2 = self.remoteChans.index(chan2)
        return self.remoteSpectralMatrix[idx1, idx2, eIdx]

    def getReferenceCrossPower(self, dataChan: str, remoteChan: str) -> np.ndarray:
        """Get the cross power of a data channel and a remote reference channel

        Parameters
        ----------
        dataChan : str
            The data channel
        remoteChan : str
            The remote reference channel
        
        Returns
        -------
        np.ndarray
            The cross power array 
        """
        idx1 = self.specChans.index(dataChan)
        idx2 = self.remoteChans.index(remoteChan)
        return self.referenceSpectralMatrix[idx1, idx2]

    def getReferenceCrossPowerEval(
        self, dataChan: str, remoteChan: str, eIdx: int
    ) -> float:
        """Get the cross power of a data channel and a remote reference channel at a single evaluation frequency

        Parameters
        ----------
        dataChan : str
            The data channel
        remoteChan : str
            The remote reference channel
        eIdx : int
            The evaluation frequency index              
        
        Returns
        -------
        float
            The value of the cross power at the evaluation frequency 
        """
        idx1 = self.specChans.index(dataChan)
        idx2 = self.remoteChans.index(remoteChan)
        return self.referenceEvalMatrix[idx1, idx2, eIdx]

    def interpolateToEvalFreq(self, data: np.ndarray) -> np.ndarray:
        """Interpolate data on to the evaluation frequency

        Parameters
        ----------
        data : np.ndarray
            Power spectral data defined at frequency points given in the freqs array

        Returns
        -------
        np.ndarray
            Data interpolated to evaluation frequencies
        """
        interpFunc = interp.interp1d(self.freq, data)                           # self.freq：fft后的频率数组(226个) data：fft后的对应频域数据
        interpData = interpFunc(self.evalFreq)                                  # self.evalFreq：评估频率数组
        return interpData

    def prepareOutDict(self) -> None:
        """Prepare output statistic output dictionary
        构建输出统计量的输出字典
        The outData dictionary is indexed in the following way:         outData字典的索引方式如下：
        outData[evaluation frequency][statistic component] = value      outData[评估频率][统计分量] = 数值
        """
        self.outData = {}                                       # 初始化输出数据字典
        for e in self.evalFreq:
            self.outData[e] = {}                                # 为每个评估频率创建子字典
        # set various calculated flags to false(设置各种计算标志为False,确保每次计算都是基于新加载的数据)
        self.tfCalculated = False
        self.remoteCalculated = False

    def getDataForStatName(self, statName: str):
        """Return the data for a statistic
        返回对应统计量的数据
        Given a statitic name, this method returns data from the correct internal method.
        给定统计量名称，本方法将从正确的内部方法返回对应数据
        Parameters
        ----------
        statName : str
            The name of the statistic to calculate out
        
        Returns
        -------
        Dict
            The output dictionary
        """
        if statName == "absvalEqn":
            return self.winAbsVal()                                         # 功率谱矩阵的绝对值(模)
        elif statName == "coherence":
            return self.winCoherence()                                      # 计算频谱相干度对
        elif statName == "powerSpectralDensity":
            return self.winPSD()                                            # 计算功率谱密度(PSD表示信号功率在频率域的分布,用于信号强度分析，这里采用周期图法)
        elif statName == "polarisationDirection":
            return self.winPolarisations()                                  # 计算极化方向
        elif statName == "partialCoherence":
            return self.winPartials()                                       # 计算偏相干度(和双变量相干度)
        elif statName == "transferFunction" or statName == "resPhase":
            if self.tfCalculated:
                return self.getOutData()                                    # "transferFunction"里已经计算，这里"resPhase"直接返回结果
            return self.winTransferFunction()                               # 计算频谱数据的传递函数(默认是普通最小二乘法ols)
        elif statName == "RR_coherence":
            return self.winRemoteCoherence()
        elif statName == "RR_coherenceEqn":
            return self.winRemoteEqnCoherence()
        elif statName == "RR_absvalEqn":
            return self.winRemoteAbsVal()
        elif statName == "RR_transferFunction" or statName == "RR_resPhase":
            if self.remoteCalculated:
                return self.getOutData()
            return self.winRemoteTransferFunction()
        else:
            self.printError(
                "Statistic in getDataForStatName not recognised", quitRun=True
            )
            return self.winCoherence()

    def winPSD(self):
        """Calculate power spectral densities
        计算功率谱密度
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]    统计量值字典，按[评估频率][统计量分量]索引
        """
        # need to divide by length of time too(需要除以时间长度进行归一化)
        freqLen = self.freq.size                                                                # 获取原始频率点数257
        timeLen = (freqLen - 1) * 2  # minus 1 because time sections are usually even           # 时间窗口样本长度512
        fs = self.freq[-1] * 2  # sampling frequency                                            # 采样率15.0
        # and then calculate amount of time(然后计算时间长度)
        duration = timeLen / fs                                                                 # 窗口持续时间34.13s
        # interpolate onto evaluation frequency and output to outData(插值到评估频率并输出到outData)
        for eIdx, eF in enumerate(self.evalFreq):
            for chan in self.getPSDChans():                                                     # ['Ex','Ey','Hx','Hy']
                key = "psd{}".format(chan)                                                      # 生成键名，如"psdEx", "psdHx"
                self.outData[eF][key] = self.getAutoPowerEval(chan, eIdx) / duration            # PSD计算：自功率谱 / 窗口持续时间   (如ExEx* / 34.13)    从周期图到功率谱密度的转换
        return self.getOutData()

    def winCoherence(self):
        """Calculate spectral coherence pairs
        计算频谱相干性对
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]    统计量值字典，按[评估频率][统计量分量]索引
        """
        for idx, p in enumerate(self.getCohPairs()):                                        # 遍历所有需要计算相干性的通道对(如['Ex','Hx'])
            c1 = p[0]  # chan1                                                              # 第一个通道名称(如'Ex')
            c2 = p[1]  # chan2                                                              # 第二个通道名称(如'Hx')
            for eIdx, eF in enumerate(self.evalFreq):                                       # 遍历所有评估频率点
                # calculate the nominator and denominator(计算分子和分母)
                cohNom = np.power(
                    np.absolute(self.getCrossPowerEval(c1, c2, eIdx)), 2                    # 计算分子(反映通道间能量耦合强度)：|Pxy|²。取实部，确保结果为实数(如：|ExHx*|²)，理论上已经是实数
                ).real
                cohDenom = self.getAutoPowerEval(c1, eIdx) * self.getAutoPowerEval(         # 计算分母(两通道总信号能量的乘积)：Pxx * Pyy(如：ExEx* × HxHx*)
                    c2, eIdx
                )
                # save in outData(生成统计量键名并存储结果)
                key = "coh{}".format(c1 + c2)                                               # 如 "cohExHx"
                self.outData[eF][key] = cohNom / cohDenom                                   # 相干度 = 分子/分母
        return self.getOutData()

    def winPolarisations(self):
        """Calculate polarisation directions
        计算极化方向
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]    统计量值字典，按[评估频率][统计量分量]索引
        """
        for idx, p in enumerate(self.getPolDirs()):                                         # 遍历所有需要计算极化方向的通道对[['Ex','Ey'],['Hx','Hy']]
            c1 = p[0]  # chan1                                                              # 第一个通道名称
            c2 = p[1]  # chan2                                                              # 第二个通道名称
            for eIdx, eF in enumerate(self.evalFreq):
                # now calculate the nominator and denominator(计算分子和分母)
                cohNom = (
                    2 * self.getCrossPowerEval(c1, c2, eIdx).real                           # 计算分子(反映两个通道之间的同相耦合强度)：2 × Re(Pxy)  取实部     如：2 × Re(ExEy*)
                )  # take the real part of this
                cohDenom = self.getAutoPowerEval(c1, eIdx) - self.getAutoPowerEval(         # 计算分母(反映两个通道的功率差异)：Pxx - Pyy    如：ExEx* - EyEy*
                    c2, eIdx
                )
                # save to out dictionary(计算极化角度并存储)
                key = "pol{}".format(c1 + c2)                                               # 生成键名，如"polExEy"
                self.outData[eF][key] = np.arctan(cohNom / cohDenom) * (180.0 / np.pi)      # 公式：arctan(2*Re(ExEy*)/([ExEx*]−[EyEy*]))，* (180.0 / np.pi)将弧度转换为度，结果在-90°到+90°之间。
        return self.getOutData()

    def winPartials(self):
        """Calculate partial coherencies
        计算偏相干性
        Based on paper Weckmann, Magunia Ritter 2005.   基于Weckmann, Magunia Ritter 2005年的论文
        e.g. coherence Ex, Hx w.r.t Hy                  例如：Ex和Hx之间相对于Hy的偏相干性
        This currently only works for impedance tensor calculations and higher power partial coherencies are not supported.
        当前仅适用于阻抗张量计算，不支持更高阶的偏相干性
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
            当前仅适用于阻抗张量计算，不支持更高阶的偏相干性
        Notes
        -----
        Based on paper by Weckmann, Magunia Ritter 2005.        统计量值字典，按[评估频率][统计量分量]索引
        """
        # get the coherences - these will be required later
        winCoherence = self.winCoherence()                                                  # 获取相干性 - 后续计算需要

        for i, outChan in enumerate(self.outChans):                                         # 遍历所有输出通道['Ex','Ey']
            for eIdx, eFreq in enumerate(self.evalFreq):                                    # 遍历所有评估频率点
                inChan1 = self.inChans[0]                                                   # 第一个输入通道，通常是Hx
                inChan2 = self.inChans[1]                                                   # 第二个输入通道，通常是Hy
                xOutIn1 = self.getCrossPowerEval(outChan, inChan1, eIdx)                    # 计算必要的互功率谱,例如：S_ExHx*
                xOutIn2 = self.getCrossPowerEval(outChan, inChan2, eIdx)                    # 例如：S_ExHy*
                xIn1In2 = self.getCrossPowerEval(inChan1, inChan2, eIdx)                    # 例如：S_HxHy*
                xIn2In1 = self.getCrossPowerEval(inChan2, inChan1, eIdx)                    # 例如：S_HyHx*
                # calculate out transFunc components(计算传递函数分量)
                denom = (                                                                   # 计算分母D(值越大表示两个输入越独立)
                    self.getAutoPowerEval(inChan1, eIdx)
                    * self.getAutoPowerEval(inChan2, eIdx)                                  # D 如：(HxHx* × HyHy*) - (HxHy* × HyHx*)
                    - xIn1In2 * xIn2In1
                )
                # Z1(张量阻抗第一分量:Zxx或Zyx)
                Z1nom = (
                    xOutIn1 * self.getAutoPowerEval(inChan2, eIdx) - xIn2In1 * xOutIn2      # 如：(ExHx* × HyHy*) - (HyHx* × ExHy*)
                )
                Z1 = Z1nom / denom                                                          # Z1是阻抗张量的Zxx或Zyx分量（取决于输出通道outChan）
                # Z2(张量阻抗第二分量:Zxy或Zyy)
                Z2nom = (
                    self.getAutoPowerEval(inChan1, eIdx) * xOutIn2 - xIn1In2 * xOutIn1      # 如：<HxHx*><ExHy*> - <HxHy*><ExHx*>
                )
                Z2 = Z2nom / denom                                                          # Z2是阻抗张量的Zxy或Zyy分量（取决于输出通道outChan）
                # calculate bivariate coherency(双变量相干性：表示电场可由两个磁场分量解释的程度)
                rb = Z1 * self.getCrossPowerEval(
                    inChan1, outChan, eIdx
                ) + Z2 * self.getCrossPowerEval(inChan2, outChan, eIdx)
                rb = rb / self.getAutoPowerEval(outChan, eIdx)                              # rb：bivarEx = (Zxx<Hx,Ex*> + Zxy<Hy,Ex*>)/<Ex,Ex*>
                # now calculate out partials(计算偏相干度)
                # calculate partial inChan, outChan1 with respect to outChan2(计算 inChan 和 outChan1 对 outChan2 的偏相干度)
                cohkey = "coh{}".format(outChan + inChan2)                                  # 例如："cohExHy"
                rp1 = (rb - winCoherence[eFreq][cohkey]) / (                                # rp1 = (rb - coherence(Ex,Hy)) / (1 - coherence(Ex,Hy))
                    1.0 - winCoherence[eFreq][cohkey]                                       # parExHx：去除inChan2(Hy)影响后，outChan(Ex)与inChan1(Hx)的独立相关性
                )
                # calculate partial inChan, outChan2 with respect to outChan1
                cohkey = "coh{}".format(outChan + inChan1)                                  # 例如："cohExHx"
                rp2 = (rb - winCoherence[eFreq][cohkey]) / (                                # rp2 = (rb - coherence(Ex,Hx)) / (1 - coherence(Ex,Hx))
                    1.0 - winCoherence[eFreq][cohkey]                                       # parExHy：去除inChan1(Hx)影响后，outChan(Ex)与inChan2(Hy)的独立相关性
                )
                # now save in outDict(存储结果到输出字典)
                self.outData[eFreq]["bivar{}".format(outChan)] = rb                         # 二元相干性(如bivarEx,bivarEy)
                self.outData[eFreq]["par{}".format(outChan + inChan1)] = rp1                # 偏相干性1(如parExHx,parEyHx)
                self.outData[eFreq]["par{}".format(outChan + inChan2)] = rp2                # 偏相干性2(如parExHy,parEyHy)
        return self.getOutData()

    def winAbsVal(self):
        """Absolute values of the cross power spectral matrix
        功率谱矩阵的绝对值(模)
        This data is often useful for cross plotting
        这些数据通常用于交叉绘图
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]    统计量值字典，按[评估频率][统计量分量]索引
        """
        for eIdx, eFreq in enumerate(self.evalFreq):                                        # 遍历该层所有评估频率点
            for iChan, chan in enumerate(self.specChans):                                   # 遍历所有频谱通道['Hx','Hy','Ex','Ey']
                # first do the outchans multiplied by every other channel(首先将每个输出通道与其他所有通道相乘)
                for iOut, outChan in enumerate(self.outChans):                              # 首先计算输出通道(Ex和Ey)与每个其他通道(Hx*,Hy*,Ex*,Ey*)的互功率绝对值(模)
                    absval = np.absolute(self.getCrossPowerEval(outChan, chan, eIdx))       # 获取评估频率点的功率谱值(模)
                    key = "abs{}{}".format(outChan, chan)                             # 生成统计量键名，如"absExHx", "absEyHy"等
                    self.outData[eFreq][key] = absval                                       # 存储到输出字典。如：4.6875：{'absExHx':221440}

                # then  do the inchans multiplied by every other channel(然后将每个输入通道与其他所有通道相乘)
                for iIn, inChan in enumerate(self.inChans):                                 # 然后计算输入通道(Hx和Hy)与每个其他通道的互功率绝对值(模)
                    absval = np.absolute(self.getCrossPowerEval(inChan, chan, eIdx))        # 同上
                    key = "abs{}{}".format(inChan, chan)
                    self.outData[eFreq][key] = absval
        # return the dictionary
        return self.getOutData()                                                            # 返回字典形式的计算结果

    def winTransferFunction(self):
        """Calculate transfer function for the spectral data
        计算频谱数据的传递函数
        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]    统计量值字典，按[评估频率][统计量分量]索引
        """
        totalSize = self.inSize + self.outSize                                              # 计算总通道数（输入通道 + 输出通道）
        # now want to calculate the transfer function for each evaluation frequency(现针对各评估频率计算传递函数)
        output = np.empty(                                                                  # 预分配传递函数输出数组
            shape=(self.evalFreq.size, self.outSize, self.inSize), dtype="complex"          # 形状：频率×输出×输入   复数类型，传递函数是复数
        )
        for eIdx, eFreq in enumerate(self.evalFreq):                                        # 遍历所有评估频率点[4.6875,4.0625,3.4375,2.8125]
            # solve transfer function
            obs = np.empty(shape=(self.outSize, totalSize), dtype="complex")                # 观测矩阵(obs存储电场与所有场的功率谱)
            reg = np.empty(
                shape=(self.outSize, totalSize, self.inSize), dtype="complex"               # 回归矩阵(reg存储磁场与所有场的功率谱)
            )
            for i in range(0, self.outSize):                                                # 遍历输出通道(Ex=...,Ey=...)
                for j in range(0, totalSize):                                               # 遍历所有通道(ExHx*=...,ExHy*=...,ExEx*=...,ExEy*=...)
                    # this is the observation row where,i is the observed output(这是观测行，其中 i 是观测到的输出)
                    # idx in the evaluation frequency(评估频率对应的索引)
                    obs[i, j] = self.getCrossPowerEval(                                     # 观测矩阵：输出通道与第j个通道的功率谱
                        self.outChans[i], self.specChans[j], eIdx                           # 输出通道（如Ex）,所有频谱通道,频率索引
                    )
                    for k in range(0, self.inSize):                                         # 遍历输入通道(Hx,Hy)
                        reg[i, j, k] = self.getCrossPowerEval(                              # 回归矩阵：输入通道与第j个通道的功率谱
                            self.inChans[k], self.specChans[j], eIdx                        # 输入通道（如Hx,Hy）,所有频谱通道,频率索引
                        )
            ##################################################################################################################
            # 最小二乘求解(默认)
            for i in range(0, self.outSize):                                                # 对每个输出通道进行最小二乘求解
                observation = obs[i, :]                                                     # 第i个输出通道的观测向量
                predictors = reg[i, :, :]                                                   # 对应的预测矩阵
                # now do the solution(执行最小二乘求解)
                # out, resids, squareResid, rank, s = olsModel(                               # 执行普通最小二乘估计
                #     predictors, observation, intercept=self.intercept                       # 预测变量矩阵,观测向量,是否包含截距项
                # )
                # out, resids, weights = mestimateModel_fixed(predictors, observation, intercept=self.intercept)
                # out, resids, weights = chatterjeeMachler_fixed(predictors, observation, intercept=self.intercept)
                # out, resids, weights = mahalanobisDistance_fixed(predictors, observation, intercept=self.intercept)
                out, resids, weights = robustmahalanobisDistance_fixed(predictors, observation, intercept=self.intercept)
                # not interested in the intercept (const) term(本研究/分析中不关注截距项（常数项）)
                if self.intercept:
                    output[eIdx, i] = out[1:]                                               # 去掉截距项，取传递函数系数(阻抗)
                else:
                    output[eIdx, i] = out                                                   # 直接使用所有系数(阻抗)

            # calculate components of transfer function and res and phase(计算传递函数（阻抗）的分量、视电阻率（res）及相位)
            for i in range(0, self.outSize):                                                # 输出通道循环(Ex,Ey)
                for j in range(0, self.inSize):                                             # 输入通道循环(Hx,Hy)
                    period = 1.0 / eFreq                                                    # 计算周期 T = 1/f
                    res = 0.2 * period * np.power(np.absolute(output[eIdx, i, j]), 2)       # 计算视电阻率：ρ = 0.2*T*|Z|²
                    phase = np.angle(output[eIdx, i, j], deg=True)                          # 计算相位（角度制）
                    keyRes = self.outChans[i] + self.inChans[j] + "Res"                     # 如"ExHxRes"
                    keyPhase = self.outChans[i] + self.inChans[j] + "Phase"                 # 如"ExHxPhase"
                    self.outData[eFreq][keyRes] = res                                       # 存储视电阻率(普通最小二乘估计)
                    self.outData[eFreq][keyPhase] = phase                                   # 存储相位(普通最小二乘估计)
                    # add the components(同时存储实部和虚部)
                    keyReal = self.outChans[i] + self.inChans[j] + "Real"                   # 如"ExHxReal"
                    keyImag = self.outChans[i] + self.inChans[j] + "Imag"                   # 如"ExHxImag"
                    self.outData[eFreq][keyReal] = output[eIdx, i, j].real                  # 实部
                    self.outData[eFreq][keyImag] = output[eIdx, i, j].imag                  # 虚部
        # set transfer function calculated as true(设置传递函数计算标志为真)
        # saves having to do it again(避免重复计算)
        self.tfCalculated = True                                                            # 设置传递函数计算标志为真,避免重复计算
        return self.getOutData()

    def blockTransferFunction(self, block_spectra: List[Any], evalFreq: np.ndarray):
        """
        计算滑动数据块的传递函数 (Block-by-Block Regression)
        累积块内全部窗口的互功率谱，解决微样本(n=4)导致的崩溃与协方差病态问题。
        """
        # 使用块内首个窗口初始化通道参数
        self.setSpectra(block_spectra[0].freqArray, block_spectra[0], evalFreq)
        totalSize = self.inSize + self.outSize                                              # 计算总通道数（输入通道 + 输出通道）
        num_win = len(block_spectra)
        # now want to calculate the transfer function for each evaluation frequency(现针对各评估频率计算传递函数)
        output = np.empty(                                                                  # 预分配传递函数输出数组
            shape=(self.evalFreq.size, self.outSize, self.inSize), dtype="complex"          # 形状：频率×输出×输入   复数类型，传递函数是复数
        )
        # 为了保证结构不被破坏，保留 outData 字典形式
        self.prepareOutDict()

        for eIdx, eFreq in enumerate(self.evalFreq):
            # 预分配能够容纳整个块（如50窗 x 4通道对 = 200行）的观测与回归矩阵
            block_obs = np.empty(shape=(self.outSize, num_win * totalSize), dtype="complex")
            block_reg = np.empty(shape=(self.outSize, num_win * totalSize, self.inSize), dtype="complex")

            # 遍历块内每一个时间窗，提取交叉功率谱并堆积成列
            for wIdx, winSpecData in enumerate(block_spectra):
                self.setSpectra(winSpecData.freqArray, winSpecData, evalFreq)
                for i in range(0, self.outSize):                                                # 遍历输出通道(Ex=...,Ey=...)
                    for j in range(0, totalSize):                                               # 遍历所有通道(ExHx*=...,ExHy*=...,ExEx*=...,ExEy*=...)
                        # this is the observation row where,i is the observed output(这是观测行，其中 i 是观测到的输出)
                        # idx in the evaluation frequency(评估频率对应的索引)
                        row_idx = wIdx * totalSize + j
                        block_obs[i, row_idx] = self.getCrossPowerEval(                         # 观测矩阵：输出通道与第j个通道的功率谱
                            self.outChans[i], self.specChans[j], eIdx                           # 输出通道（如Ex）,所有频谱通道,频率索引
                        )
                        for k in range(0, self.inSize):                                         # 遍历输入通道(Hx,Hy)
                            block_reg[i, row_idx, k] = self.getCrossPowerEval(                  # 回归矩阵：输入通道与第j个通道的功率谱
                                self.inChans[k], self.specChans[j], eIdx                        # 输入通道（如Hx,Hy）,所有频谱通道,频率索引
                            )
            ##################################################################################################################
            # 矩阵堆叠完毕，针对各输出通道执行全局大样本回归
            # 最小二乘求解(默认)
            for i in range(0, self.outSize):                                                # 对每个输出通道进行最小二乘求解
                observation = block_obs[i, :]                                               # 第i个输出通道的观测向量
                predictors = block_reg[i, :, :]                                             # 对应的预测矩阵
                # now do the solution(执行最小二乘求解)
                out, resids, squareResid, rank, s = olsModel(                               # 执行普通最小二乘估计
                    predictors, observation, intercept=self.intercept                       # 预测变量矩阵,观测向量,是否包含截距项
                )
                # out, resids, scale, weights = mestimateModel(predictors, observation, intercept=self.intercept)
                # out, resids, weights = chatterjeeMachler(predictors, observation, intercept=self.intercept)
                # out, resids, weights = mahalanobisDistance(predictors, observation, intercept=self.intercept)
                # out, resids, weights = robustmahalanobisDistance(predictors, observation, intercept=self.intercept)
                # not interested in the intercept (const) term(本研究/分析中不关注截距项（常数项）)
                if self.intercept:
                    output[eIdx, i] = out[1:]                                               # 去掉截距项，取传递函数系数(阻抗)
                else:
                    output[eIdx, i] = out                                                   # 直接使用所有系数(阻抗)
            ##################################################################################################################
            # calculate components of transfer function and res and phase(计算传递函数（阻抗）的分量、视电阻率（res）及相位)
            for eIdx, eFreq in enumerate(self.evalFreq):
                for i in range(0, self.outSize):                                                # 输出通道循环(Ex,Ey)
                    for j in range(0, self.inSize):                                             # 输入通道循环(Hx,Hy)
                        period = 1.0 / eFreq                                                    # 计算周期 T = 1/f
                        res = 0.2 * period * np.power(np.absolute(output[eIdx, i, j]), 2)       # 计算视电阻率：ρ = 0.2*T*|Z|²
                        phase = np.angle(output[eIdx, i, j], deg=True)                          # 计算相位（角度制）
                        keyRes = self.outChans[i] + self.inChans[j] + "Res"                     # 如"ExHxRes"
                        keyPhase = self.outChans[i] + self.inChans[j] + "Phase"                 # 如"ExHxPhase"
                        self.outData[eFreq][keyRes] = res                                       # 存储视电阻率(普通最小二乘估计)
                        self.outData[eFreq][keyPhase] = phase                                   # 存储相位(普通最小二乘估计)
                        # add the components(同时存储实部和虚部)
                        keyReal = self.outChans[i] + self.inChans[j] + "Real"                   # 如"ExHxReal"
                        keyImag = self.outChans[i] + self.inChans[j] + "Imag"                   # 如"ExHxImag"
                        self.outData[eFreq][keyReal] = output[eIdx, i, j].real                  # 实部
                        self.outData[eFreq][keyImag] = output[eIdx, i, j].imag                  # 虚部
        # set transfer function calculated as true(设置传递函数计算标志为真)
        # saves having to do it again(避免重复计算)
        self.tfCalculated = True                                                            # 设置传递函数计算标志为真,避免重复计算
        return self.getOutData()

    def winRemoteCoherence(self):
        """Calulate coherence between data channels and remote channels

        For example, this is the coherence of Ex-HxR, Ex-HyR, Ey-HxR, Ey-HyR, Hx-HxR, Hx-HyR, Hy-HxR, Hy-HyR

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """
        # now let's calculate coherency
        # abs(crosspower(A,B))^2/autopower(A)*autpower(B)
        for dataChan in self.specChans:
            for remoteChan in self.remoteChans:
                key = "{}{}RR".format(dataChan, remoteChan)
                for eIdx, eFreq in enumerate(self.evalFreq):
                    cohNom = np.power(
                        np.absolute(
                            self.getReferenceCrossPowerEval(dataChan, remoteChan, eIdx)
                        ),
                        2,
                    )
                    cohDenom = self.getAutoPowerEval(
                        dataChan, eIdx
                    ) * self.getRemoteAutoPowerEval(remoteChan, eIdx)
                    coh = cohNom / cohDenom
                    self.outData[eFreq][key] = coh
        return self.getOutData()

    def winRemoteEqnCoherence(self):
        """Calulates coherences for the remote reference solver equations

        For example, this is the coherence of Ex-HxR, Ex-HyR, Ey-HxR, Ey-HyR, Hx-HxR, Hx-HyR, Hy-HxR, Hy-HyR

        todo:
        Write more information in these comments

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """
        # now calculate out the relevant coherencies
        # here we calculate the coherency between <Ex,HyR> and <Hy,HyR> for example
        for iOut, outChan in enumerate(self.outChans):
            for iIn, inChan in enumerate(self.inChans):
                for iRemote, remoteChan in enumerate(self.remoteChans):
                    # calculate powers
                    c1c1 = smooth1d(
                        self.getReferenceCrossPower(outChan, remoteChan)
                        * np.conjugate(
                            self.getReferenceCrossPower(outChan, remoteChan)
                        ),
                        self.winLen,
                        self.winType,
                    )
                    c2c2 = smooth1d(
                        self.getReferenceCrossPower(inChan, remoteChan)
                        * np.conjugate(self.getReferenceCrossPower(inChan, remoteChan)),
                        self.winLen,
                        self.winType,
                    )
                    c1c2 = smooth1d(
                        self.getReferenceCrossPower(outChan, remoteChan)
                        * np.conjugate(self.getReferenceCrossPower(inChan, remoteChan)),
                        self.winLen,
                        self.winType,
                    )
                    # now interpolate
                    c1c1 = self.interpolateToEvalFreq(c1c1)
                    c2c2 = self.interpolateToEvalFreq(c2c2)
                    c1c2 = self.interpolateToEvalFreq(c1c2)
                    # now calculate the nominator and denominator
                    cohNom = np.power(np.absolute(c1c2), 2)
                    cohDenom = c1c1 * c2c2
                    coh = (
                        cohNom / cohDenom
                    )  # cast as float - discard complex part (complex part should be zero anyway)
                    # now need the coherencies for the evaluation frequencies
                    # this involves interpolation
                    key = "{}{}R-{}{}R".format(outChan, remoteChan, inChan, remoteChan)
                    for iFreq, eFreq in enumerate(self.evalFreq):
                        self.outData[eFreq][key] = coh[iFreq]
        return self.getOutData()

    def winRemoteAbsVal(self):
        """Absolute values of cross power spectral densities between remote refence channels and data channels

        This data can be useful for cross plotting

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """
        for eIdx, eFreq in enumerate(self.evalFreq):
            for iOut, outChan in enumerate(self.outChans):
                for iRemote, remoteChan in enumerate(self.remoteChans):
                    absOut = np.absolute(
                        self.getReferenceCrossPowerEval(outChan, remoteChan, eIdx)
                    )
                    keyOut = "abs{}{}R".format(outChan, remoteChan)
                    self.outData[eFreq][keyOut] = absOut
                    for iIn, inChan in enumerate(self.inChans):
                        absIn = np.absolute(
                            self.getReferenceCrossPowerEval(inChan, remoteChan, eIdx)
                        )
                        keyIn = "abs{}{}R".format(inChan, remoteChan)
                        self.outData[eFreq][keyIn] = absIn
        return self.getOutData()

    def winRemoteTransferFunction(self):
        """Calculate transfer function for the spectral data when remote reference is included too

        Returns
        -------
        Dict : 
            Dictionary with statistic values, indexed by [evaluation frequency][statistic component]
        """
        output = np.empty(
            shape=(self.evalFreq.size, self.outSize, self.inSize), dtype="complex"
        )
        for eIdx, eFreq in enumerate(self.evalFreq):
            # solve transfer function
            obs = np.empty(shape=(self.outSize, self.inSize), dtype="complex")
            reg = np.empty(
                shape=(self.outSize, self.inSize, self.inSize), dtype="complex"
            )
            for i, outChan in enumerate(self.outChans):
                for j, remoteChan in enumerate(self.remoteChans):
                    # this is the observation row where,i is the observed output
                    # eIdx in the evaluation frequency
                    obs[i, j] = self.getReferenceCrossPowerEval(
                        outChan, remoteChan, eIdx
                    )
                    for k, inChan in enumerate(self.inChans):
                        reg[i, j, k] = self.getReferenceCrossPowerEval(
                            inChan, remoteChan, eIdx
                        )

            for i in range(0, self.outSize):
                observation = obs[i, :]
                predictors = reg[i, :, :]
                # now do the solution
                out, resids, squareResid, rank, s = olsModel(
                    predictors, observation, intercept=self.intercept
                )
                # out, resids, scale, weights	= mmestimateModel(predictors, observation, intercept=False)
                # not interested in the intercept (const) term
                if self.intercept:
                    output[eIdx, i] = out[1:]
                else:
                    output[eIdx, i] = out

            # calculate components of transfer function and res and phase
            for i in range(0, self.outSize):
                for j in range(0, self.inSize):
                    period = 1.0 / eFreq
                    res = 0.2 * period * np.power(np.absolute(output[eIdx, i, j]), 2)
                    phase = np.angle(output[eIdx, i, j], deg=True)
                    keyRes = self.outChans[i] + self.inChans[j] + "ResRR"
                    keyPhase = self.outChans[i] + self.inChans[j] + "PhaseRR"
                    self.outData[eFreq][keyRes] = res
                    self.outData[eFreq][keyPhase] = phase
                    # add the components
                    keyReal = self.outChans[i] + self.inChans[j] + "RealRR"
                    keyImag = self.outChans[i] + self.inChans[j] + "ImagRR"
                    self.outData[eFreq][keyReal] = output[eIdx, i, j].real
                    self.outData[eFreq][keyImag] = output[eIdx, i, j].imag
        # set transfer function calculated as true
        # saves having to do it again
        self.remoteCalculated = True
        return self.getOutData()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Default options")
        textLst.append("\tInput Chans = {}".format(listToString(self.getInChans())))
        textLst.append("\tOutput Chans = {}".format(listToString(self.getOutChans())))
        textLst.append(
            "\tRemote Chans = {}".format(listToString(self.getRemoteChans()))
        )
        textLst.append("\tPowers = {}".format(listToString(self.getPSDChans())))
        textLst.append(
            "\tCoherence pairs = {}".format(listToString(self.getCohPairs()))
        )
        textLst.append(
            "\tPartial coherence = {}".format(listToString(self.getPolDirs()))
        )
        if len(self.getEvalFreq()) == 0:
            textLst.append("Evaluation frequencies = {}")
        else:
            textLst.append(
                "Evaluation frequencies = {}".format(arrayToString(self.getEvalFreq()))
            )
        return textLst
