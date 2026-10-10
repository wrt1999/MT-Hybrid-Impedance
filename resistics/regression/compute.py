import numpy as np
from typing import List, Tuple
import scipy.interpolate as interp

from resistics.common.smooth import smooth1d
from resistics.spectra.data import SpectrumData


def evalFrequencyData(freq, evalFreq, winDataMatrix):
    """Calculate spectral power data at evaluation frequencies
    在各评估频率点处计算频谱功率数据
    Parameters
    ----------
    freq : np.ndarray
        Frequency array of spectra data
    evalFreq : np.ndarray
        Evaluation frequencies for the decimation level
    winDataMatrix : np.ndarray
        Array holding spectral power data at frequencies freq        

    Returns
    -------
    out : np.ndarray
        Spectral power data interpolated to evaluation frequencies 插值至各评估频率点的功率谱数据
    """
    inShape: Tuple = winDataMatrix.shape
    data: np.ndarray = np.empty(                                            # 初始化输出数组
        shape=(evalFreq.size, inShape[0], inShape[1]), dtype="complex"      # 保持复数类型(输出数组形状：评估频率点×行通道×列通道)
    )
    # get data from winDataMatrix
    for i in range(0, inShape[0]):
        for j in range(0, inShape[1]):
            interpFunc = interp.interp1d(freq, winDataMatrix[i, j])         # winDataMatrix[0, 0]：257个HxHx*功率谱数据。
            interpVals = interpFunc(evalFreq)                               # 在评估频率点上进行插值
            for eIdx in range(len(evalFreq)):
                data[eIdx, i, j] = interpVals[eIdx]                     # eIdx：有7个评估频率点       data:三维数组 data[0,0,0]:3.75Hz的HxHx*  data[1,0,0]:2.65Hz的HxHx* data[0,1,0]:3.75Hz的HxHy*
    return data                                                         # data：每个评估频率点（7个）的功率谱矩阵（每个矩阵4×4个数据）


def spectralMatricesWindow(
    inData: SpectrumData,       # 输入频谱数据（磁场H）
    outData: SpectrumData,      # 输出频谱数据（电场E）
    inChannels: List[str],      # 输入通道名（如["Hx", "Hy"]）
    outChannels: List[str],     # 输出通道名（如["Ex", "Ey"]）
    smoothLen: int,             # 平滑窗口长度（如9）
    smoothWin: str,             # 平滑窗类型(hanning)
    evalFreq: List[float],      # 目标评估频率列表[3.75,....,0.46875]
):
    """Compute the spectral matrices 
    计算频谱矩阵（从原始频谱数据到功率谱矩阵）
    Parameters
    ----------
    inData : SpectrumData
        The input spectrum data
    outData : SpectrumData
        The output spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to
        待插值的评估频率
    Returns
    -------
    out : np.ndarray
        Cross spectral matrices interpolated to evaluation frequencies 功率谱矩阵插值到评估频率
    """
    inSize = len(inChannels)
    outSize = len(outChannels)
    totalSize = inSize + outSize    # 总通道数
    dataSize = inData.dataSize      # 频谱数据长度(由于共轭对称性512只用保存一半257)
    # arrays for the window data(创建窗口数据的存储数组)
    winDataArray: np.ndarray = np.empty(shape=(totalSize, dataSize), dtype="complex")       # 创建二维复数数组(单通道频谱)，用于存储通道(Hx,Hy,Ex,Ey)的频谱数据（幅度和相位信息）
    winSpectraMatrix: np.ndarray = np.empty(
        shape=(totalSize, totalSize, dataSize), dtype="complex"                             # 互功率谱(通道间互功率谱)，创建三维复数数组，用于存储多通道之间的功率谱矩阵，
    )
    # get data into the right part of the arrays(频谱数据装配：合并电场和磁场数据，Hx：0 Hy：1 Ex：2 Ey：3)
    for iChan in range(0, inSize):
        winDataArray[iChan] = inData.data[inChannels[iChan]]        # 填充Hx, Hy
    for iChan in range(0, outSize):
        winDataArray[inSize + iChan] = outData.data[outChannels[iChan]]     # 填充Ex, Ey, (Hz)
    # power spectra(计算功率谱矩阵)三维
    for i in range(0, totalSize):
        for j in range(i, totalSize):                       # 上三角部分+对角线
            winSpectraMatrix[i, j] = smooth1d(              # 计算平滑后的功率谱        # np.conjugate()函数：计算复数的共轭
                winDataArray[i] * np.conjugate(winDataArray[j]), smoothLen, smoothWin   # smooth1d(原始功率谱，平滑窗口长度，平滑窗函数类型)函数：对频谱数据应用滑动窗口平滑，降低噪声，提高信噪比，同时保持数据长度不变
            )
            if i != j:              # Hermitian对称性
                # complex symmetry(复共轭对称性)
                winSpectraMatrix[j, i] = np.conjugate(winSpectraMatrix[i, j])
    # return data interpolated to evaluation frequencies(返回插值到评估频率的数据)
    return evalFrequencyData(inData.freqArray, evalFreq, winSpectraMatrix)      # 返回：插值至各评估频率点的功率谱数据。evalFrequencyData：每个评估频率点（7个）的功率谱矩阵（每个矩阵4×4个数据）


def spectralMatrices(
    ncores: int,
    inData: List[SpectrumData],
    outData: List[SpectrumData],
    inChannels: List[str],
    outChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: np.ndarray,
):
    """Parallel calculation of spectral matrices
    功率谱矩阵的并行计算
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input spectrum data
    outData : List[SpectrumData]
        The output spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : List[np.ndarray]
        List of spectral matrices
    """
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (iD, oD, inChannels, outChannels, smoothLen, smoothWin, evalFreq)
            for iD, oD in zip(inData, outData)
        ]
        with mp.Pool(ncores) as pool:
            out = pool.starmap(spectralMatricesWindow, multiTuples)
    else:
        out = []
        for iD, oD in zip(inData, outData):                                                 # 2893个窗口，每个窗口都要计算(循环2893次)
            out.append(
                spectralMatricesWindow(                                                     # 计算功率谱矩阵(将电场和磁场的原始频谱数据转换为功率谱矩阵，并插值到目标评估频率)
                    iD, oD, inChannels, outChannels, smoothLen, smoothWin, evalFreq         # out：每个评估频率点（7个）的功率谱矩阵（每个矩阵4×4个数据），共2893个窗口
                )
            )
    return out


def remoteMatricesWindow(
    inData: SpectrumData,
    outData: SpectrumData,
    remoteData: SpectrumData,
    inChannels: List[str],
    outChannels: List[str],
    remoteChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: List[float],
):
    """Compute the spectral matrices 

    Parameters
    ----------
    inData : SpectrumData
        The input spectrum data
    outData : SpectrumData
        The output spectrum data
    remoteData : SpectrumData
        The remote spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    remoteChannels : List[str]
        The remote channels to use            
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : np.ndarray
        Cross spectral matrices interpolated to evaluation frequencies
    """
    inSize = len(inChannels)
    outSize = len(outChannels)
    remoteSize = len(remoteChannels)
    totalSize = inSize + outSize
    dataSize = inData.dataSize
    # an array for the in and out channels fourier data
    winDataArray = np.empty(shape=(totalSize, dataSize), dtype="complex")
    # an array for the remote reference fourier data
    winRemoteArray = np.empty(shape=(remoteSize, dataSize), dtype="complex")
    # an array for the power spectra data
    winSpectraMatrix = np.empty(
        shape=(totalSize, remoteSize, dataSize), dtype="complex"
    )
    # get data into the right part of the arrays
    for i in range(0, inSize):
        winDataArray[i] = inData.data[inChannels[i]]
    for i in range(0, outSize):
        winDataArray[inSize + i] = outData.data[outChannels[i]]
    for i in range(0, remoteSize):
        winRemoteArray[i] = remoteData.data[remoteChannels[i]]

    # power spectra
    for iD in range(totalSize):
        for iR in range(remoteSize):
            # cannot use conjugate symmetry unlike single site processor
            winSpectraMatrix[iD, iR] = smooth1d(
                winDataArray[iD] * np.conjugate(winRemoteArray[iR]),
                smoothLen,
                smoothWin,
            )
    # return data interpolated to evaluation frequencies
    return evalFrequencyData(inData.freqArray, evalFreq, winSpectraMatrix)


def remoteMatrices(
    ncores: int,
    inData: List[SpectrumData],
    outData: List[SpectrumData],
    remoteData: List[SpectrumData],
    inChannels: List[str],
    outChannels: List[str],
    remoteChannels: List[str],
    smoothLen: int,
    smoothWin: str,
    evalFreq: np.ndarray,
):
    """Parallel calculation of spectral matrices for remote reference data
    
    Parameters
    ----------
    ncores : int
        The number of cores to run on
    inData : List[SpectrumData]
        The input spectrum data
    outData : List[SpectrumData]
        The output spectrum data
    remoteData : List[SpectrumData]
        The remote reference spectrum data
    inChannels : List[str]
        The input channels to use
    outChannels : List[str]
        The output channels to use
    remoteChannels : List[str]
        The remote channels to use        
    smoothLen : int
        The smoothing length
    smoothWin : str
        The smoothing window
    evalFreq : np.ndarray
        The evaluation frequencies to interpolate to

    Returns
    -------
    out : List[np.ndarray]
        List of spectral matrices
    """
    if ncores > 0:
        import multiprocessing as mp

        multiTuples = [
            (
                iD,
                oD,
                rD,
                inChannels,
                outChannels,
                remoteChannels,
                smoothLen,
                smoothWin,
                evalFreq,
            )
            for iD, oD, rD in zip(inData, outData, remoteData)
        ]
        with mp.Pool(ncores) as pool:
            out = pool.starmap(remoteMatricesWindow, multiTuples)
    else:
        out = []
        for iD, oD, rD in zip(inData, outData, remoteData):
            out.append(
                remoteMatricesWindow(
                    iD,
                    oD,
                    rD,
                    inChannels,
                    outChannels,
                    remoteChannels,
                    smoothLen,
                    smoothWin,
                    evalFreq,
                )
            )
    return out
