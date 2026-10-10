import numpy as np
import scipy.signal as signal
import pyfftw
from typing import List, Dict, Union

from resistics.common.base import ResisticsBase
from resistics.spectra.data import SpectrumData
from resistics.time.data import TimeData
from resistics.config.io import loadConfig


class SpectrumCalculator(ResisticsBase):
    """Class for calculating spectra from time data windows
    基于时间数据窗口计算频谱的类
    The class requires the pyfftw and FFTW3 libraries, which allow for fast fourier transforms for spectra calculations
    此类需依赖pyfftw与FFTW3库，这两个库能够实现快速傅里叶变换，用于频谱计算。
    Attributes
    ----------
    sampleFreq 采样频率: float
        Sample frequency of the data
    numSamples 每窗口样本数: int
        The number of samples in each window
    window 窗函数类型: bool (true)
        Flag for applying a window function before fourier transform
    windowFunc 窗函数系数: np.ndarray
        Window function to apply to time data before fourier transform
    dataArray 预分配内存: np.ndarray
        numpy array to copy data into before performing fourier transform
    ffotObj FFT计算对象: pyfftw.FFTW
        fourier transform object

    Methods
    -------
    __init__(sampleFreq, winSamples, config=None)
        Initialise with time data sampling frequency and the number of samples in a window      使用时间数据采样频率和窗口样本数完成初始化
    calcFourierTransform(timeData)
        Perform fourier transform for timeData and return specData object       对timeData执行傅里叶变换并返回specData对象
    printList()
        Class status returned as list of strings        以字符串列表形式返回类状态
    """

    def __init__(
        self, sampleFreq: float, winSamples: int, config: Union[Dict, None] = None
    ) -> None:
        """Initialise 
    
        Parameters
        ----------
        sampleFreq : float
            Sampling frequency of time data
        winSamples : int
            Number of samples in a window 
        """
        self.sampleFreq: float = sampleFreq
        self.numSamples: int = winSamples
        if config is None:
            config = loadConfig()
        self.window: str = config["Spectra"]["windowfunc"]              # 窗函数名称
        self.applywindow: bool = config["Spectra"]["applywindow"]       # 是否应用窗函数的开关
        self.windowFunc: np.ndarray = signal.get_window(                # signal.get_window生成指定类型和长度的窗函数，返回值：长度为 Nx的NumPy数组，包含窗函数系数
            config["Spectra"]["windowfunc"], winSamples                 # 返回值：长度为512的NumPy数组，包含汉明窗系数
        )
        # create an pyfftw plan(生成一个pyfftw运算规划)
        self.dataArray: np.ndarray = pyfftw.empty_aligned(
            self.numSamples, dtype="float64"                            # 创建内存对齐的NumPy数组，为FFT计算提供最优的内存访问性能，提升FFT计算速度2-3倍
        )
        self.fftObj: pyfftw.FFTW = pyfftw.builders.rfft(self.dataArray) # 创建预优化的FFT计算对象，通过一次性的算法选择和优化，为后续的大量FFT计算提供最优性能

    def calcFourierCoeff(self, timeData: TimeData) -> SpectrumData:
        """Fast fourier transform of timeData(对时域数据进行快速傅里叶变换（FFT）生成频谱数据)
        输入：TimeData 对象（含多通道时域数据）      输出：SpectrumData 对象（含复数频谱和元数据）
        Parameters
        ----------
        timeData : TimeData
            A TimeData object

        Returns
        -------
        specData : SpectrumData
            A SpectrumData object
        """
        fftData: Dict = {}                                  # 创建空字典存储频谱数据
        for c in timeData.chans:
            # first copy data into dataArray(首先将数据复制到 dataArray 中)
            self.dataArray[:] = timeData.data[c][:]         # 将Ex数据复制到处理数组，覆盖掉之前的内存对齐numpy数组。
            # no need to pad, these are usually multiples of two(无需填充，这些通常是 2 的倍数)
            # detrend and apply window function if set(如果设置了，则进行去趋势处理并应用窗口函数)
            self.dataArray[:] = signal.detrend(self.dataArray, type="linear")   # 线性去趋势，消除线性趋势对FFT的影响。实际采集的时域数据往往包含缓慢变化的线性趋势，这会破坏FFT的周期性假设，导致频谱泄漏
            if self.applywindow:
                self.dataArray[:] = self.dataArray * self.windowFunc       # 时域加窗处理，将时域数据与窗函数相乘，平滑数据边界（自然过渡），减少FFT计算中的频谱泄漏。通过逐点乘法，用窗函数的权重系数调制原始信号。
            # use pytfftw here(在此处使用 pytfftw)
            fftData[c] = np.array(self.fftObj())           # FFT计算(时域信号 → 频域频谱（实数 → 复数）)，转换为NumPy数组。调用预优化的FFTW计算器执行傅里叶变换，并将复数频谱结果存储到结果字典中。
        # calculate frequency array(计算频率数组)
        dataSize = fftData[timeData.chans[0]].size         # 获取频谱数据长度257，所有通道的频谱长度相同（对称性）。实数信号的傅里叶变换具有共轭对称性，只需要存储一半的频谱数据即可完整表示。

        # comments(生成处理注释)
        specComments = []
        if self.applywindow:
            specComments.append(
                "Time data multiplied by {} window function".format(self.window)
            )
        specComments.append(
            "Fourier transform performed, time data size = {}, spectra data size = {}".format(
                self.numSamples, dataSize
            )
        )
        # 创建频谱数据对象
        return SpectrumData(
            windowSize=self.numSamples,                     # 窗口大小512
            dataSize=dataSize,                              # 频谱数据长度257
            sampleFreq=timeData.sampleFreq,                 # 采样频率(15->1.875)
            startTime=timeData.startTime,                   # 开始时间
            stopTime=timeData.stopTime,                     # 结束时间
            data=fftData,                                   # 频谱数据字典
            comments=timeData.comments + specComments,      # 合并注释
        )

    def printList(self) -> List:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = ["Sample frequency = {:f} [Hz]".format(self.sampleFreq)]
        return textLst
