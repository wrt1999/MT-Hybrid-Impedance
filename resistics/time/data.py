import os
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Union

from resistics.common.base import ResisticsBase
from resistics.common.checks import isElectric
from resistics.common.plot import getViewFonts


class TimeData(ResisticsBase):
    """Class for holding time data

    Attributes
    ----------
    numSamples : int
        The number of samples in the data
    sampleFreq : float
        The sampling frequency
    startTime : datetime.datetime
        The time of the first sample
    stopTime : datetime.datetime
        The time of the last sample 
    chans : List[str]
        The channels in the data
    numChans : int
        The number of channels
    data : Dict
        The time data with channels as keys and arrays as values
    comments : List[str]
        Information about the time data as a list of strings

    Methods
    -------
    __init__(kwargs)
        Initialise the time data
    setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        Set data with parameters
    getDateArray() : np.ndarray
        A datetime array of the sample times
    getComments()
        Get a deepcopy of the comments
    addComment(comment)
        Add a comment to the dataset
    copy()
        Get a copy of the timeseries data
    view(kwargs)
        View the time data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
        comments: Union[str, List[str]] = [],
    ) -> None:
        """Initialise and set object parameters

        Parameters
        ----------
        sampleFreq : float
            The sampling frequency in Hz
        startTime : datetime, str
            The startTime of the window
        stopTime : datetime, str
            The stopTime of the window 
        data : Dict
            The data dictionary with keys as channels and values as spectra data
        comments : List[str]
            Dataset comments
        """
        self.setData(sampleFreq, startTime, stopTime, data)
        self.comments = comments
        if isinstance(self.comments, str):
            self.comments = [self.comments]

    def setData(
        self,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
    ):
        """Set the object parameters

        Parameters
        ----------
        sampleFreq : float
            The sampling frequency in Hz
        startTime : datetime, str
            The startTime of the window
        stopTime : datetime, str
            The stopTime of the window 
        data : Dict
            The data dictionary with keys as channels and values as spectra data
        """
        self.sampleFreq = sampleFreq
        # start time
        self.startTime = (
            datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(startTime, str)
            else startTime
        )
        # stop time
        self.stopTime = (
            datetime.strptime(stopTime, "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(stopTime, str)
            else stopTime
        )
        # other properties
        self.chans = sorted(data.keys())
        self.numChans = len(self.chans)
        self.data = data
        self.numSamples = data[self.chans[0]].size

    def getDateArray(self) -> np.ndarray:
        """Get the date array

        Returns
        -------
        np.ndarray
            The date array of the time samples
        """
        x = np.empty(shape=(self.numSamples), dtype=datetime)
        for i in range(0, self.numSamples):
            x[i] = self.startTime + timedelta(seconds=1.0 * i / self.sampleFreq)
        return x

    def getComments(self) -> List[str]:
        """Get a deepcopy of the comments
        
        Returns
        -------
        List[str]
            Dataset comments as a list of strings
        """
        return deepcopy(self.comments)

    def addComment(self, comment: str) -> None:
        """Add a new comment

        Parameters
        ----------
        comment : str
            A new comment
        """
        self.comments.append(comment)

    def copy(self):
        """Get a copy of the time data object

        Returns
        -------
        TimeData
            A copy of the time data object
        """
        return TimeData(
            self.sampleFreq,
            self.startTime,
            self.stopTime,
            deepcopy(self.data),
            self.getComments(),
        )

    def view(self, **kwargs) -> Figure:
        """View timeseries data as a line plot

        Parameters
        ----------
        sampleStart : int, optional
            Sample to start plotting from
        sampleStop : int, optional
            Sample to plot to                   
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        chans : List[str]
            Channels to plot
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        # the number of samples to plot(设置默认的样本范围（从第0个样本到最后一个样本）)
        sampleStart = 0
        sampleStop = 432000
        if "sampleStart" in kwargs:                     # 使用用户传入的参数覆盖默认值
            sampleStart = kwargs["sampleStart"]         # 用户指定的开始样本
        if "sampleStop" in kwargs:
            sampleStop = kwargs["sampleStop"]           # 用户指定的结束样本
        if sampleStop >= self.numSamples:               # 边界检查：确保结束样本不超过数据总长度
            sampleStop = self.numSamples - 1            # 调整为最后一个有效样本
        # get the x axis ready(获取完整的时间戳数组（所有样本对应的时间）)
        x = self.getDateArray()                         # 返回datetime对象数组
        start = x[sampleStart]                          # 开始样本对应的时间
        stop = x[sampleStop]                            # 结束样本对应的时间

        # now plot(创建或复用图形对象)
        fig = (
            plt.figure(kwargs["fig"].number)                    # 复用现有图形
            if "fig" in kwargs
            else plt.figure(figsize=(20, 2 * self.numChans))    # 创建新图形
        )
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()        # 获取字体设置（用户指定或默认值）
        # suptitle(设置图形总标题)
        st = fig.suptitle(
            "Time data from {} to {}, samples {} to {}".format(
                start, stop, sampleStart, sampleStop
            ),
            fontsize=plotfonts["suptitle"],         # 使用标题字体大小
        )
        st.set_y(0.98)                  # 调整标题垂直位置（避免与子图重叠）
        # now plot the data
        dataChans = kwargs["chans"] if "chans" in kwargs else self.chans        # 确定要绘制的通道列表(用户指定或默认所有通道)
        numDataChans = len(dataChans)                                           # 实际要绘制的通道数量
        for idx, chan in enumerate(dataChans):                                  # 遍历每个通道，创建子图
            ax = plt.subplot(numDataChans, 1, idx + 1)                    # 创建子图：行数=通道数，列数=1，当前位置=idx+1
            plt.title("Channel {}".format(chan), fontsize=plotfonts["title"])   # 设置子图标题（通道名称）

            # check if channel exists in data and if not, leave empty plot so it's clear(检查通道数据是否存在（处理数据缺失情况）)
            if chan not in self.data:
                continue

            # label for plot(生成图例标签（用户指定或自动生成）)
            lab = (
                kwargs["label"]                                         # 用户自定义标签
                if "label" in kwargs
                else "{}: {} to {}".format(chan, start, stop)     # 自动生成标签
            )
            # plot data(绘制时间序列数据（核心绘图操作）)
            plt.plot(
                x[sampleStart : sampleStop + 1],                  # X轴数据：时间序列
                self.data[chan][sampleStart : sampleStop + 1],          # Y轴数据：通道数值
                label=lab,                                              # 图例标签
            )
            # add time label（只在最下面的子图显示X轴标签（避免重复））
            if idx == numDataChans - 1:                                             # 最后一个通道（最下方）
                plt.xlabel("Time", fontsize=plotfonts["axisLabel"])
            # set the xlim
            xlim = kwargs["xlim"] if "xlim" in kwargs else [start, stop]            # 设置X轴范围（用户指定或使用数据时间范围）
            plt.xlim(xlim)
            # y axis options                                                        # 设置Y轴标签（根据通道类型）
            if isElectric(chan):
                plt.ylabel("mV/km", fontsize=plotfonts["axisLabel"])            # 电场通道
            else:
                plt.ylabel("nT or mV", fontsize=plotfonts["axisLabel"])         # 磁场通道
            plt.grid(True)                                                            # 添加网格线便于读数
            # set tick sizes(设置刻度标签字体大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            # legend
            if "legend" in kwargs and kwargs["legend"]:                             # 根据参数决定是否显示图例(这里没有)
                plt.legend(loc=4)                                                   # loc=4表示右下角

        # show if the figure is not in keywords(如果是在新创建的图形上绘图，进行布局优化并显示)
        if "fig" not in kwargs:                                     # 新图形（非复用）
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])               # 自动调整布局，为标题留空间
            plt.show()                                              # 显示图形

        return fig                                                  # 返回图形对象（便于进一步处理或保存）

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Sampling frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Sample rate [s] = {}".format(1.0 / self.sampleFreq))
        textLst.append("Number of samples = {}".format(self.numSamples))
        textLst.append("Number of channels = {}".format(self.numChans))
        textLst.append("Channels = {}".format(self.chans))
        textLst.append("Start time = {}".format(self.startTime))
        textLst.append("Stop time = {}".format(self.stopTime))
        if len(self.comments) == 0:
            textLst.append("No comments")
        else:
            textLst.append("Comments")
            for comment in self.comments:
                textLst.append("\t{}".format(comment))
        return textLst
