from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Union

from resistics.common.base import ResisticsBase
from resistics.common.checks import isElectric
from resistics.common.math import getFrequencyArray
from resistics.common.plot import getViewFonts


class SpectrumData(ResisticsBase):
    """Class for holding spectra data

    Attributes
    ----------
    windowSize : int
        The size of the time window in samples
    dataSize : int
        The number of samples in the frequency spectra
    sampleFreq : float
        The sampling frequency
    startTime : datetime.datetime
        The time of the first sample of the time data which was fourier transformed
    stopTime : datetime.time
        The time of the last sample of the time data which was fourier transformed
    chans : List[str]
        The channels in the data
    numChans : int
        The number of channels
    data : Dict
        The spectra data with channels as keys and arrays as values
    comments : List[str]
        Information about the spectra data as a list of strings

    Methods
    -------
    __init__(kwargs)
        Initialise spectra data
    setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        Set data with parameters
    getComments()
        Get a deepcopy of the comments        
    addComment(comment)
        Add a comment to the dataset
    copy()
        Get a copy of the spectrum data
    view(kwargs)
        View the spectra data 
    printList()
        Class status returned as list of strings          
    """

    def __init__(
        self,
        windowSize: int,
        dataSize: int,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
        comments: List[str] = [],
    ) -> None:
        """Initialise and set object parameters

        Parameters
        ----------
        windowSize : int
            The window size in samples of the time data
        dataSize : int
            The spectra size in samples
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
        self.setData(windowSize, dataSize, sampleFreq, startTime, stopTime, data)
        self.comments = comments

    def setData(
        self,
        windowSize: int,
        dataSize: int,
        sampleFreq: float,
        startTime: Union[datetime, str],
        stopTime: Union[datetime, str],
        data,
    ) -> None:
        """Set the object parameters

        Parameters
        ----------
        windowSize : int
            The window size in samples of the time data
        dataSize : int
            The spectra size in samples
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
        self.windowSize = windowSize
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
        self.dataSize = data[self.chans[0]].size

    @property
    def nyquist(self) -> float:
        """Get the nyquist frequency of the spectra data

        Returns
        -------
        nyquist : float
            The nyquist frequency in Hz
        """
        return self.sampleFreq / 2.0

    @property
    def freqArray(self) -> np.ndarray:
        """Get the frequency array of the spectra data

        Returns
        -------
        freqArray : np.ndarray
            Array of frequencies
        """
        return getFrequencyArray(self.sampleFreq, self.dataSize)

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
        comment : float
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
        return SpectrumData(
            self.windowSize,
            self.dataSize,
            self.sampleFreq,
            self.startTime,
            self.stopTime,
            deepcopy(self.data),
            self.getComments(),
        )

    def view(self, **kwargs) -> Figure:
        """Plot spectra data

        Parameters
        ----------
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        chans : List[str], optional
            A list of channels to plot
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        color : str, rgba Tuple
            The color for the line plot
        legend : bool
            Boolean flag for adding a legend

        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        f = self.freqArray                                      # 从对象属性获取频率数组(257)
        fig = (                                                 # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)                    # 如果提供了图形，复用它
            if "fig" in kwargs
            else plt.figure(figsize=(20, 2 * self.numChans))    # 否则创建新图形
        )
        plotFonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()        # 获取字体设置，或使用默认字体
        color = kwargs["color"] if "color" in kwargs else None                              # 获取线条颜色
        # suptitle
        st = fig.suptitle(                                                                  # 设置图形总标题
            "Spectra data from {} to {}".format(self.startTime, self.stopTime),       # 显示时间范围
            fontsize=plotFonts["suptitle"],                                                 # 使用标题字体大小
        )
        st.set_y(0.98)                                                                      # 调整标题位置，避免与子图重叠
        # now plot the data
        dataChans = kwargs["chans"] if "chans" in kwargs else self.chans                    # 确定要绘制的通道
        numPlotChans = len(dataChans)                                                       # 实际要绘制的通道数
        for idx, chan in enumerate(dataChans):                                              # 循环绘制每个通道
            ax = plt.subplot(numPlotChans, 1, idx + 1)                                # 创建子图：numPlotChans行1列，当前通道在idx+1位置
            plt.title("Channel {}".format(chan), fontsize=plotFonts["title"])               # 设置子图标题
            # plot the data（绘制数据）
            if "label" in kwargs:                                                                   # 如果提供了标签
                plt.plot(
                    f, np.absolute(self.data[chan]), color=color, label=kwargs["label"]       # X轴：频率数组。Y轴：通道数据的幅度(复数的模)。线条颜色。图例标签
                )
            else:
                plt.plot(f, np.absolute(self.data[chan]), color=color)                      # 没有标签
            # add frequency label(添加频率标签（只在最后一个子图）)
            if idx == numPlotChans - 1:                                                           # 如果是最后一个通道
                plt.xlabel("Frequency [Hz]", fontsize=plotFonts["axisLabel"])               # 添加标签Frequence[Hz]
            # x axis options(X轴范围设置)
            xlim = kwargs["xlim"] if "xlim" in kwargs else [f[0], f[-1]]                        # X轴范围设置,默认使用全频率范围
            plt.xlim(xlim)                                                                      # 设置X轴范围
            # y axis options(Y轴单位标签)
            if isElectric(chan):
                plt.ylabel("[mV/km]", fontsize=plotFonts["axisLabel"])                  # 电场单位
            else:
                plt.ylabel("[nT]", fontsize=plotFonts["axisLabel"])                     # 磁场单位
            plt.grid()                                                                        # 添加网格
            # set tick sizes(设置刻度字体大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotFonts["axisTicks"])
            # legend(图例设置)
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)                                                             # 位置4：右下角

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            plt.show()

        return fig

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Sample rate [s] = {}".format(1.0 / self.sampleFreq))
        textLst.append("Sampling frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Nyquist frequency [Hz] = {}".format(self.nyquist))
        textLst.append("Number of time samples = {}".format(self.windowSize))
        textLst.append("Number of frequency samples = {}".format(self.dataSize))
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

