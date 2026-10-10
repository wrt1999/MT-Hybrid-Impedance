import numpy as np
import math
from datetime import datetime, timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.print import blockPrint
from resistics.time.data import TimeData
from resistics.config.io import loadConfig


class Windower(ResisticsBase):
    """Class for windowing time data with overlaps 
    用于带重叠的时间数据加窗处理的类
    Given time data for a decimation level and a reference time, the windower calculates the number of windows and returns them.
    给定一个抽取级别和参考时间的时间数据，窗口生成器会计算窗口数量并返回它们
    Attributes
    ----------
    timeData : TimeData
        TimeData object to window
    numSamples : int
        The number of samples in each window
    sampleFreq : float
        Sampling frequency of data in Hz
    winSize : int
        Number of samples in a window
    winDuration : float
        Duration of a window in seconds
    winOlap : np.ndarray
        Number of overlap samples
    chans : List[str]
        List of chans in timeData
    refTime : datetime
        Project reference time
    dataTime : datetime
        Time of timeData first sample
    minWindows : int
        Minimum number of windows required
    winOffset : int
        Offset between local and global window indexing
    firstWindowTime: datetime
        Time of the first window in the TimeData
    winSamples : np.ndarray[int]
        Two dimensional, the sample start and end for each window
    winTimes : 
    winActive : np.ndarray[bool]
        Boolean flagging whether window is active or not

    Methods
    -------
    __init__(refTime, timeData, winSize, winOlap)
        Initialise windower
    getWindowActive(iW)

    getGlobalIndex(iW)
        Return the global index (relative to reference time) given local window index
    getData(iWindow)
        Get timeData for local window iWindow
    getDataGlobal(iGlobal)
        Get timeData for glonal index iGlobal
    initialiseWindows()
        Calculate number of windows, local to global index mapping and local window offset from global
    printList()
        Class status returned as list of strings  
    printWindowTimes()
        Print window times  
    printWindowTimeList()
        Window time information returned as List of strings               
    """

    def __init__(
        self, refTime: datetime, timeData: TimeData, winSize: int, winOlap: int
    ) -> None:
        """Initialise the windower
        对窗函数生成器进行初始化
        Parameters
        ----------
        refTime : datetime
            The reference time
        timeData : TimeData
            The time series data to window
        winSize : int
            The window size in samples
        winOlap : int
            The overlap size between windows in samples
        """
        self.timeData: TimeData = timeData                              # timeData数据(时域)
        self.numSamples: int = timeData.numSamples                      # 样本数(最开始1111575->138947)
        self.sampleFreq: float = timeData.sampleFreq                    # 采样率(15->1.875)
        self.winSize: int = winSize                                     # 窗口大小(512)
        self.winDuration: float = (winSize - 1) / timeData.sampleFreq   # 窗口持续时间,为什么是 winSize - 1？因为N个样本点覆盖的时间跨度是 (N-1)/sampleFreq
        self.winOlap: int = winOlap                                     # 重叠大小(128)
        self.chans: List[str] = timeData.chans                          # 通道
        # refTime and dataTime are already datetime objects(参考时间（refTime）与数据时间（dataTime）均已为日期时间对象)
        self.refTime: datetime = refTime
        self.dataTime: datetime = timeData.startTime
        # min window warning setting(最小窗口警告设置)
        config = loadConfig()
        self.minWindows: int = config["Window"]["minwindows"]           # 最小窗口数
        # initialise
        self.initialiseWindows()                # 计算窗口分割参数，包括全局窗口索引等。空间分割 - 计算如何在样本索引空间分割窗口[[258, 769], [642, 1153]...[1111170, 1111681]],winOffset窗口偏移量507
        self.calcWindowTimes()                  # 计算时间信息。时间对齐 - 计算每个窗口的时间信息(计算每个窗口的开始时间和结束时间)

    def getWindowActive(self, iWindow: int) -> bool:
        """Returns active status of local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        active : bool
            True if active, false if not
        """
        return self.winActive[iWindow]

    def getGlobalIndex(self, iWindow: int) -> int:
        """Returns global index for local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        iGlobal : int
            Global index
        """
        return iWindow + self.winOffset

    def getData(self, iWindow: int) -> TimeData:
        """Returns time window data for local index

        Parameters
        ----------
        iWindow : int
            Local index of window

        Returns
        -------
        windowData : TimeData
            TimeData object with the window data
        """
        winSamples = self.winSamples[iWindow]
        winData = {}
        for c in self.chans:
            winData[c] = self.timeData.data[c][
                winSamples[0] : winSamples[1] + 1
            ]  # add 1 because numpy indexing like this is not inclusive
        globalWindow = self.winTimes[iWindow][0]
        winStartTime = self.winTimes[iWindow][1]
        winStopTime = self.winTimes[iWindow][2]
        return TimeData(
            sampleFreq=self.sampleFreq,
            startTime=winStartTime,
            stopTime=winStopTime,
            data=winData,
            comments=self.timeData.comments
            + ["Local window iW, global window {}".format(globalWindow)],
        )

    def getDataGlobal(self, iGlobal: int) -> TimeData:
        """Returns time window data for global index

        Parameters
        ----------
        iGlobal : int
            Global index of window

        Returns
        -------
        windowData : TimeData
            TimeData object with the window data
        """
        iWindow = iGlobal - self.winOffset
        return self.getData(iWindow)

    def initialiseWindows(self):
        """Calculate all the window information
        计算所有窗口信息
        For timeData and referenceTime, initialiseWindows calculates the number of windows (after the reference time) and the global indices of the windows relative to the reference time
        对于 timeData 和 referenceTime，initialiseWindows 会计算出（参考时间之后的）窗口数量以及相对于参考时间的全局窗口索引
        Stores the offset between local indices and the global indices
        储存局部索引与全局索引之间的偏移量
        """
        # have a reference time
        # the first window starts there
        deltaRefStart = self.dataTime - self.refTime
        if deltaRefStart.total_seconds() < 0:
            self.printWarning(
                "Reference time is after start of recording. Stuff may go wrong!"
            )
        # increment of window start times(窗口起始时刻的增量)
        # -1 because inclusive of sample at start(减1，这是由于起始处样本也包含在内)
        winStartIncrement = 1.0 * (self.winSize - self.winOlap) / self.sampleFreq  # 计算窗口步长，相邻窗口起始时间的时间间隔(秒)
        # calculate number of windows started before reference time(确定从哪个窗口开始包含有效数据，确保时间对齐精度)
        # and then by taking the ceiling, find the global index of the first window in the data(计算首个有效分析窗口相对于参考时间的全局索引)
        self.winOffset = int(
            math.ceil(deltaRefStart.total_seconds() / winStartIncrement)          # winOffset窗口偏移量计算(从0开始，第一个有效窗口相对于参考时间的窗口偏移量(全局索引)507->64)
        )                     # deltaRefStart.total_seconds()：数据开始时间相对于参考时间的偏移（秒），/ winStartIncrement：转换为窗口数，math.ceil()：向上取整，确保窗口完全包含数据
        # calculate start time of first global window(计算第一个全局窗口的起始时间)
        offsetSeconds = self.winOffset * winStartIncrement            # offsetSeconds窗口开始时间(相对于参考时间reftime)
        # calculate the first window time(第一个包含数据的窗口开始时间转换为年-月-日)
        self.firstWindowTime = self.refTime + timedelta(seconds=offsetSeconds)  # 第一个有效窗口开始时间年-月-日
        # calculate first sample(计算首个样本,计算数据窗口相对于原始数据流的精确起始位置，确保时间对齐精度)
        deltaStart = self.firstWindowTime - self.dataTime               # deltaStart = 第一个有效窗口开始时间 - 数据开始时间，(数据在前，第一个窗口在后)
        sampleStart = deltaStart.total_seconds() * self.sampleFreq      # sampleStart：数据点的样本索引(相对于第一个数据)，从第一个窗口开始的样本点
        # next calculate number of windows(接下来计算窗口数量)
        # sample start is the first sample(样本起始点为第一个样本)
        # window size is window size inclusive of first sample(窗口大小包含第一个样本)
        winStart = sampleStart                          # 起始样本索引258(1次降采样272.25)     一个窗口内
        winEnd = sampleStart + self.winSize - 1         # 结束样本索引769(1次降采样783.25)     一个窗口内
        winStartOff = self.winSize - self.winOlap       # 窗口步长（样本数）384 = 512-128
        winSamples = []                                 # 存储窗口列表
        while winEnd < self.numSamples:                 # 确保窗口不超出数据范围，窗口最后一个样本点 < 总样本数1111575
            winSamples.append([winStart, winEnd])       # 添加当前窗口，[[258, 769], [642, 1153]...[1111170, 1111681]] 各窗口的[start_sample, end_sample]
            winStart = winStart + winStartOff           # 移动到下一个窗口，下一个窗口开始样本点 = 窗口开始点 + 偏移量(窗口步长)
            winEnd = winStart + self.winSize - 1        # 更新结束位置，下一个窗口结束样本点 = 上一行代码窗口开始样本点 + 窗口大小(512) - 1
        self.numWindows = len(winSamples)               # 窗口个数2893（1次降采样后360）
        # warn if number of windows is small(若窗口数量较少，则给出警示)
        if self.numWindows < self.minWindows:           # 如果窗口个数 < 最小窗口数5
            self.printWarning(
                "Number of windows in data is small - consider stopping decimation"
            )
        # save winSamples as numpy list in class(在类中将winSamples保存为NumPy数组列表)
        self.winSamples = np.array(winSamples, dtype=int)                # 输出属性self.winSamples数组
        # set all windows initially to active(将所有窗口初始设置为活动状态)
        self.winActive = np.ones(shape=(self.numWindows), dtype=bool)    # 输出属性self.winActive，窗口全为有效窗口True

    def calcWindowTimes(self) -> None:
        """Calculate start and stop times for each window
        计算每个窗口的开始时间和结束时间
        """
        self.winTimes = []
        iW = 0
        for samples in self.winSamples:
            start = samples[0]          # 窗口起始样本索引
            stop = samples[1]           # 窗口结束样本索引
            win = []          # [窗口全局索引, 窗口开始时间, 窗口结束时间]
            # global index(全局索引)相对于参考时间
            win.append(self.winOffset + iW)
            # start time
            deltaStart = timedelta(seconds=start / self.sampleFreq)     # start / self.sampleFreq：将样本索引转换为时间偏移（秒），相对于第一个样本数据
            timeStart = self.dataTime + deltaStart                      # 第一个窗口(窗口里第一个样本)开始时间年-月-日-时-分-秒
            deltaEnd = timedelta(seconds=stop / self.sampleFreq)
            timeEnd = self.dataTime + deltaEnd                          # 第一个窗口结束时间年-月-日-时-分-秒
            # samples2end = self.winSize - 1 # need to remove initial sample(需要移除初始样本)，替代计算方法（注释中）
            # timeEnd = timeStart + timedelta(seconds=samples2end/self.sampleFreq)
            win.append(timeStart)
            win.append(timeEnd)
            self.winTimes.append(win)               # self.winTimes列表里添加win窗口信息
            iW = iW + 1

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sample freq. [Hz] = {:f}".format(self.sampleFreq))
        textLst.append("Window size = {:d}".format(self.winSize))
        textLst.append("Overlap size = {:d}".format(self.winOlap))
        textLst.append("Window duration [s] = {:.3f}".format(self.winDuration))
        textLst.append("Reference time {}".format(self.refTime))
        textLst.append("Data start time {}".format(self.dataTime))
        textLst.append(
            "Number of complete windows in data = {:d}".format(self.numWindows)
        )
        if self.numWindows < self.minWindows:
            textLst.append(
                "Number of windows in data is small - consider stopping decimation"
            )
        if self.numWindows > 0:
            textLst.append(
                "Global index of first window from reference time = {}".format(
                    self.winOffset
                )
            )
            textLst.append(
                "First window starts at time {}, sample {:d}".format(
                    self.firstWindowTime, self.winSamples[0, 0]
                )
            )
        return textLst

    def printWindowTimes(self) -> None:
        """Print the times of the windows"""
        blockPrint("Windower::window times", self.printWindowTimeList())

    def printWindowTimeList(self) -> List:
        """Window time information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        winTimes = self.winTimes
        winSamples = self.winSamples
        textLst = []
        textLst.append(
            "NOTE: Sample ranges are inclusive, to get number of samples, use: sample end - sample start + 1"
        )
        for win, winS in zip(winTimes, winSamples):
            textLst.append(
                "Global index = {:d}, start time = {}, end time = {}, start sample = {:d}, end sample = {:d}".format(
                    win[0], win[1], win[2], winS[0], winS[1]
                )
            )
        return textLst