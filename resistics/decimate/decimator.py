import numpy as np
import scipy.signal as signal
from datetime import timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.math import intdiv
from resistics.common.print import (
    generalPrint,
    warningPrint,
    blockPrint,
    arrayToStringInt,
)
from resistics.config.io import loadConfig
from resistics.decimate.parameters import DecimationParameters
from resistics.time.data import TimeData
from resistics.time.filter import downsampleData


class Decimator(ResisticsBase):
    """Decimate time data

    Decimates time data by factors until the minimum number of required samples is reached. When a downsample factor is too large, downsampling is performed in multiple steps to maintain accuracy of result.
    对时间数据按指定倍数进行降采样，直至达到所需的最小样本数。当降采样倍数过大时，将分多步进行降采样以保持结果的准确性
    Attributes
    ----------
    timeData : TimeData
        timeData object to decimate
    sampleFreq : float
        Sampling frequency of time data in Hz
    chans : List[str]
        Channels in time data
    numSamples : int
        Number of samples in timeData
    decParams : DecimationParams
        A DecimationParams object holding decimation information
    minSamples : int
        Minimum required samples to decimate
    level : int
        Current decimation level
    maxDownSampleFactor : int
        Max allowable downsampling in one go. Downsampling becomes less accurate at large downsample factors

    Methods
    -------
    __init__(timeData, decParams)
        Initialise Decimator with a TimeData object and DecimationParams object
    incrementLevel()
        Downsample the timeData to the next decimation level
    downsample(downsampleFactor)
        Do the downsampling
    printList()
        Class status returned as list of strings  
    """

    def __init__(self, timeData: TimeData, decParams: DecimationParameters) -> None:
        """Initialise with timeData and decimation parameters
        使用时间数据和降采样参数进行初始化
        Parameters
        ----------
        timeData : TimeData
            The time data to decimate
        decParams : DecimationParams
            Decimation parameters for performing the decimation
        """
        self.timeData: TimeData = timeData                          # timeData数据
        self.sampleFreq: float = timeData.sampleFreq * 1.0          # 采样率
        self.chans: List = timeData.chans                           # 通道
        self.numSamples: int = timeData.numSamples                  # 采样点数
        self.decParams: DecimationParameters = decParams            # 降采样参数
        config = loadConfig()                                       # 加载默认配置文件
        self.minSamples: int = config["Decimation"]["minsamples"]   # 最小样本点
        self.level: int = -1                                        # 初始化降采样层数
        self.maxDownsampleFactor: int = 8                           # 单步最大降采样因子

    def incrementLevel(self) -> bool:
        """Downsample to the next decimation level
        降采样至下一个抽取级别
        Returns
        -------
        out : bool
            True if downsampling completed successfully. False otherwise

        Notes
        -----
        When the downsampling factor is too large, downsampling is performed in multiple steps. Downsampling will become increasingly inaccurate using the scipy routine when factor is too large        
        当降采样因子过大时，会分多步进行降采样，如16 → [8, 2]。当因子过大时，使用 scipy 路线进行降采样会变得越来越不准确。
        """
        # increment level, 0 is the first level(递增层级，其中0表示首个层级)
        self.level = self.level + 1           # 级别递增（0-based），初始为-1，0表示第1级
        downsampleFactor = self.decParams.getIncrementalFactor(self.level)         # 获取从上一级到当前级的增量降采样因子

        # if downsample factor is greater than maxDownsampleFactor, downsample in multiple steps(如果降采样因子大于最大降采样因子，则分多步进行降采样,如16 → [8, 2])
        numDownsamples = 1
        downsampleList = [downsampleFactor]
        if downsampleFactor > self.maxDownsampleFactor:
            # this should give an integer(此操作应输出整数结果)
            numDownsamples = intdiv(downsampleFactor, self.maxDownsampleFactor)            # 返回整商2，例如：目标因子16 → 拆分为 [8, 2],intdiv 确保分解后的因子为整数
            downsampleList = [self.maxDownsampleFactor, numDownsamples]                     # [8, 2]
            # print info
            self.printText(
                "Downsample factor of {:d} greater than max decimation factor {:d}.".format(
                    downsampleFactor, self.maxDownsampleFactor
                )
            )
            self.printText(
                "Downsampling in multiple decimations given by factors: {}".format(
                    arrayToStringInt(downsampleList)
                )
            )
        # 分步降采样执行，若目标因子超过设定的最大降采样因子8
        if numDownsamples <= 2:
            for iDS in range(0, numDownsamples):
                check = self.downsample(downsampleList[iDS])        # (这一步里进行数据降采样)判断是否可以继续对timeData进行降采样,一般返回True。当降采样导致样本数量低于 minSamples 时，将不会执行降采样。在这种情况下，该函数返回 False。
                if not check:  # check outcome of decimation(检验降采样的成效)
                    return False
                # otherwise, everything ok, update class vars and return True(否则，一切正常，更新类变量并返回 True)
                self.sampleFreq = self.timeData.sampleFreq           # 降采样后频率
                self.numSamples = self.timeData.numSamples           # 降采样后样本数
        else:
            for iDS in range(0, 2):
                check = self.downsample(downsampleList[iDS])        # (这一步里进行数据降采样)判断是否可以继续对timeData进行降采样,一般返回True。当降采样导致样本数量低于 minSamples 时，将不会执行降采样。在这种情况下，该函数返回 False。
                if not check:  # check outcome of decimation(检验降采样的成效)
                    return False
                # otherwise, everything ok, update class vars and return True(否则，一切正常，更新类变量并返回 True)
                self.sampleFreq = self.timeData.sampleFreq           # 降采样后频率
                self.numSamples = self.timeData.numSamples           # 降采样后样本数
        return True

    def downsample(self, downsampleFactor: int) -> bool:
        """Downsample time data
        对timeData进行降采样
        Parameters
        ----------
        downsampleFactor : int
            Downsampling factor        

        Returns
        -------
        out : bool
            True if downsampling completed successfully. False otherwise

        Notes
        -----
        When the downsampling causes number of simples to fall below minSamples, downsampling is not performed. The function returns False in this situation
        当降采样导致样本数量低于 minSamples 时，将不会执行降采样。在这种情况下，该函数返回 False 。
        """
        # check to see not at max level(检查一下，确保未达到最大级别)
        if self.level >= self.decParams.numLevels:
            self.printWarning(
                "Error, number of decimation levels exceeded, returning no data"
            )
            return False
        # if downsample factor is 1, nothing to do(若降采样因子为1，则无需进行任何操作)
        if downsampleFactor == 1:
            return True

        # downsampling reduces the number of samples by downsample factor   降采样，通过降采样因子降低样本数量
        # if new number of samples is too small, return False       若新样本数量过小，则返回False
        if self.numSamples / downsampleFactor < self.minSamples:
            self.printWarning(
                "Next decimation level has less than {} samples. Decimation is exiting.\nSet minimum of samples required using decimator.setMinSamples().".format(
                    self.minSamples
                )
            )
            return False

        # do the resampling（执行重采样操作）
        self.timeData.data = downsampleData(self.timeData.data, downsampleFactor)           # (这一步里进行数据降采样),将timeData里每个通道的时域数据进行降采样。
        # update the rest of the timeData object(更新 timeData 对象的其余字段)
        self.timeData.sampleFreq = self.timeData.sampleFreq / downsampleFactor              # 降采样后的新采样率15/8=1.875
        self.timeData.numSamples = self.timeData.data[self.chans[0]].size                   # 降采样后timeData每个通道的时域数据个数
        # start time stays the same, but end time needs to change(第一个样本数据开始时间保持不变，但最后一个样本数据结束时间需要更改)
        self.timeData.stopTime = self.timeData.startTime + timedelta(
            seconds=(1.0 / self.timeData.sampleFreq) * (self.timeData.numSamples - 1)
        )
        self.timeData.addComment(
            "Time data decimated from {} Hz to {} Hz, new start time {}, new end time {}".format(
                self.sampleFreq,
                self.timeData.sampleFreq,
                self.timeData.startTime,
                self.timeData.stopTime,
            )
        )
        return True

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Current level = {:d}".format(self.level))
        if self.level == -1:
            textLst.append("This is the initial level - no decimation has occured")
        textLst.append("Current sample freq. [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Current sample rate [s] = {:.6f}".format(1.0 / self.sampleFreq))
        textLst.append("Current number of samples = {:d}".format(self.numSamples))
        return textLst
