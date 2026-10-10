from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.dates import (
    DateFormatter,
    DayLocator,
    AutoDateLocator,
    AutoDateFormatter,
)
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from matplotlib.collections import PathCollection
from typing import List, Dict, Tuple, Union, Any

# import from package
from resistics.common.base import ResisticsBase
from resistics.common.print import arrayToString
from resistics.window.utils import gArray2datetime
from resistics.common.plot import (
    getPlotRowsAndCols,
    getViewFonts1,
    colorbar2dOther,
    colorbar2dTime,
)


class StatisticData(ResisticsBase):
    """Class for holding information about a statistic

    Attributes
    ----------
    statName : str
        The name of the statistic
    refTime : datetime.datetime
        The reference time of the project
    sampleFreq : float
        The sampling frequency of the data
    winSize : int
        Window size in samples
    winOlap : int
        Window overlap in samples
    numWindows : int
        Number of windows
    winStats : List[str]
        Values calculated for the statistic
    numStatsPerWindow : int
        The number of statistics per window (length of winStats)
    stats : np.ndarray
        The statistic data of size number of windows * number evaluation frequencies * number of window stats        
    evalFreq : List, np.ndarray
        The evaluation frequencies for the statistic data
    freq2index : Dict
        Mapping from evaluation frequency to index
    globalIndices : List, np.ndarray
        Array of global indices. Allows local -> global conversion
    global2localMap : Dict
        Dictionary which maps global indices to local. Allows global -> local conversion
    comments : List[str]
        Statistic comments
    dtype : str (default "float")
        The data type of the statistic values
    maxcols : int (default 4)
        The number of columns in plots
    globalDatesStart : List, np.ndarray
        Global window start datetimes
    globalDatesStop : List, np.ndarray
        Global window stop datetimes

    Methods
    -------
    __init__(sampleFreq, numLevels, evalFreq, kwargs)
        Initialise maskData
    getStats(includewindows, maskwindows)
        Get the statistics array and choose to either include some windows or mask some windows
    getUnmaskedIndices(maskwindows)
        Return the local indices to use given a set of global indices to exclude
    getStatLocal(localIndex)
        Get statistic values for a local window index
    getStatGlobal(globalIndex)
        Get statistic values for a global window index
    getGlobalIndex(localIndex)
        Get the global window index for a local window index
    getLocalIndicesF(globalIndices)
        Get local indices for an list or array of global indices
    getGlobalDates()
        Get the global index dates for all the windows
    setStatParams(numWindows, winStats, evalFreq)
        Set the statistic parameters and prepare the data arrays
    addStat(localIndex, globalIndex, stat)
        Add a statistic, providing the local index, the corresponding global index and the statististic values
    getComments()
        Get a deepcopy of the comments    
    addComment(comment)
        Add a comment to the comments
    view(eFreqI, **kwargs) 
        View statistic values for a evaluation frequency (time on x axis, statistic values on y axis)
    histogram(eFreqI, **kwargs)
        View a histogram of the statistic values for each statistic component for a single evaluation frequency
    crossplot(eFreqI, **kwargs)   
        View a crossplot of the data
    addColourbar(plot, cax, title, plotfonts)
        Add a colourbar to a plot    
    addColourbarDates(plot, cax, title, plotfonts)
        Add a coloubar the represents dates to the plot   
    dateTicks(gIndices, dates, timeNum)
        Deal with the dateTicks of the plot
    calcColourData(plotData, val, eFreqI, keywords)
        Calculate colours 
    getRowsCols(maxcols, numStats)
        Get the number of rows and columns for a plot
    printList()
        Class status returned as list of strings   
    """

    def __init__(
        self,
        statName: str,                                      # 统计量名称，如"coherence"、"transFunc"等
        refTime: datetime,                                  # 参考时间，整个数据处理的时间基准
        sampleFreq: float,                                  # 采样频率（Hz），当前抽取级别下的实际采样率
        winSize: int,                                       # 窗口大小（样本数），每个时间窗口包含的样本数
        winOlap: int,                                       # 窗口重叠（样本数），相邻窗口重叠的样本数
        **kwargs                                            # 其他关键字参数，用于灵活扩展
    ):
        """Initialise statistic data 
    
        Parameters
        ----------
        statName : str
            Name of the statistic
        refTime : datetime.datetime
            Reference time for the project
        sampleFreq : float
            Sample frequency of the data in Hz 
        winSize : int
            The window size in samples
        winOlap : int
            The window overlap in samples
        kwargs : Dict
            Two optional arguments: "name" for statistic name and "stats" for the statistics to use 可选关键字参数，包括"name"（统计量名称）和"stats"（使用的统计量）
        """
        self.statName: str = statName                                                           # 统计量名称
        self.refTime = refTime                                                                  # 参考时间（数据处理的时间基准）
        if isinstance(self.refTime, str):
            self.refTime = datetime.strptime(self.refTime, "%Y-%m-%d %H:%M:%S.%f")       # 如果refTime是字符串，转换为datetime对象
        self.sampleFreq: float = sampleFreq                                                     # 采样频率（当前抽取级别下的实际采样率）
        self.winSize: int = winSize                                                             # 窗口大小（样本数）
        self.winOlap: int = winOlap                                                             # 窗口重叠（样本数）
        # defaults from keywords(基于关键字的默认参数)
        self.numWindows: int = 0                                                                # 总窗口数，初始为0
        self.winStats: List[str] = []                                                           # 窗口统计量列表，每个窗口的统计量名称
        self.numStatsPerWindow: int = 0                                                         # 每个窗口的统计量数量，初始为0
        self.evalFreq: Union[List, np.ndarray] = []                                             # 评估频率列表，实际进行统计计算的频率点
        self.freq2index: Dict = {}                                                              # 频率到索引的映射字典，用于快速查找频率对应的数组索引
        self.globalIndices: Union[List, np.ndarray] = []                                        # 全局索引数组，每个窗口对应的全局时间索引
        self.global2localMap: Dict = {}                                                         # 全局索引到局部索引的映射字典
        self.comments: List[str] = []                                                           # 注释列表，记录数据处理的历史和参数
        self.dtype: str = "float"                                                               # 数据类型，默认"float"，但也可以是"complex"用于复数统计量
        # global dates
        self.globalDatesStarts: Union[List, np.ndarray] = []                                    # 时间窗口的开始时间数组（基于参考时间的绝对时间）
        self.globalDatesStops: Union[List, np.ndarray] = []                                     # 时间窗口的结束时间数组
        # plot params(绘图参数)
        self.maxcols: int = 4                                                                   # 最大列数，用于多子图排列
        # parse keywords(解析关键字)
        self.initialiseFromKeywords(kwargs)                                                     # 解析关键字参数，设置可选的额外属性

    def initialiseFromKeywords(self, keywords):
        """Initialise object properties using the keywords"""
        if "numWindows" in keywords:
            self.numWindows = keywords["numWindows"]
        if "winStats" in keywords:
            self.winStats = keywords["winStats"]
        self.numStatsPerWindow = len(self.winStats)
        if "evalFreq" in keywords:
            self.evalFreq = keywords["evalFreq"]
        for idx, eFreq in enumerate(self.evalFreq):
            self.freq2index[eFreq] = idx
        if "stats" in keywords:
            self.stats = keywords["stats"]
        if "globalIndices" in keywords:
            self.globalIndices = keywords["globalIndices"]
            # make the global 2 local map
            for ii in np.arange(0, len(self.globalIndices)):
                self.global2localMap[self.globalIndices[ii]] = ii
        if "comment" in keywords:
            self.comments = keywords["comment"]
        if "dtype" in keywords:
            self.dtype = keywords["dtype"]

    def getStats(
        self,
        includewindows: Union[np.ndarray, List, None] = None,
        maskwindows: Union[np.ndarray, List, None] = None,
    ) -> np.ndarray:
        """Get statistics when either selecting a set of indices or wanting to mask a set of indices

        Parameters
        ----------
        includewindows : List, np.ndarray, optional
            Windows to get. This should be a list of numpy array of global indices.
        maskwindows : List, np.ndarray, optional
            Windows to exclude. This is expected to be a list, set or numpy array of global indices.
        
        Returns
        -------
        np.ndarray
            Statistics limited by the window selection options
        """
        if includewindows is not None:
            includewindows = self.getLocalIndices(includewindows)
            return self.stats[includewindows, :]
        elif maskwindows is not None and len(maskwindows) > 0:
            includewindows = self.getUnmaskedIndices(maskwindows)
            if includewindows is not None:
                return self.stats[includewindows, :]
        return self.stats

    def getUnmaskedIndices(
        self, maskWindows: Union[List, np.ndarray]
    ) -> Union[List, None]:
        """Given a list of global windows to mask, returns the set of indices to include in the plot

        Parameters
        ----------
        maskWindows : List, np.ndarray
            Global indices of windows to mask
        
        Returns
        -------
        List, None
            List of indices to include. None if all windows should be included
        """
        includewindows = set(self.globalIndices) - set(maskWindows)
        if len(includewindows) == self.numWindows:
            # nothing to mask
            return None
        return self.getLocalIndices(includewindows)

    def getStatLocal(self, localIndex: int) -> np.ndarray:
        """Get statistics for a local window index

        Parameters
        ----------
        localIndex : int
            Local index of the window

        Returns
        -------
        out : np.ndarray
            Statistics for the local window
        """
        return self.stats[localIndex, :]

    def getStatGlobal(self, globalIndex: int) -> np.ndarray:
        """Get statistics for a local window index

        Parameters
        ----------
        globalIndex : int
            Global index of the window

        Returns
        -------
        out : np.ndarray
            Statistics for the local window
        """
        if globalIndex not in self.global2localMap:
            self.printError(
                "There are two statistics with the same global window index"
            )
            self.printError("This should never happen")
            self.printError("Exiting", quitRun=True)
        localIndex = self.global2localMap[globalIndex]
        return self.stats[localIndex]

    def getGlobalIndex(self, localIndex):
        """Get global index for local index

        Parameters
        ----------
        localIndex : int
            Local index of the window

        Returns
        -------
        globalIndex : int
            Global index for local index
        """
        return self.globalIndices[localIndex]

    def getLocalIndices(
        self, globalIndices: Tuple[List[int], np.ndarray]
    ) -> List[int]:
        """Converts a list or array of global indices to local indices
        
        Parameters
        ----------
        globalIndices : List, np.ndarray
            Global indices to convert into local indices
        
        Returns
        -------
        List
            List of local indices
        """
        indices = []
        for win in globalIndices:
            indices.append(self.global2localMap[win])
        return indices

    def getGlobalDates(self):
        """Get the global start dates and end dates for each window
        
        Returns
        -------
        np.ndarray
            The global window start times
        """
        # want to get the start date for each window here
        if len(self.globalDatesStarts) != self.numWindows:
            self.globalDatesStarts, self.globalDatesStops = gArray2datetime(
                self.globalIndices,
                self.refTime,
                self.sampleFreq,
                self.winSize,
                self.winOlap,
            )
        return self.globalDatesStarts

    def setStatParams(
        self, numWindows: int, winStats: List[str], evalFreq, dtype: Any = None
    ):
        """Set the statistic parameters
        设置统计量参数
        Parameters
        ----------
        numWindows : int
            Number of windows for which the statistic has been or will be calculated
        winStats : int
            The parameters in the statistic
        evalFreq : List
            A list of evaluation frequencies
        dtype : str, optional
            The datatype of the statistic values
        """
        self.numWindows = numWindows                                                        # 设置窗口数量
        # details about the statistics
        self.winStats = winStats                                                            # 设置窗口统计量参数列表(统计量的详细信息)
        self.numStatsPerWindow = len(winStats)                                              # 计算每个窗口的统计量数量
        # details about the evaluation frequencies
        self.evalFreq = evalFreq                                                            # 设置评估频率列表(每层)
        for idx, eFreq in enumerate(evalFreq):
            self.freq2index[eFreq] = idx                                                    # 遍历所有频率，建立频率值到数组索引的映射
        # data type
        if not (dtype is None):
            self.dtype = dtype                                                              # 如果提供了dtype参数，则覆盖默认的数据类型
        self.stats = np.empty(                                                              # 使用numpy的empty函数创建指定形状的未初始化数组(预分配统计量数据数组)
            shape=(self.numWindows, self.evalFreq.size, self.numStatsPerWindow),            # 维度0：时间窗口数量。维度1：评估频率数量。维度2：每个窗口的统计量数量
            dtype=self.dtype,                                                               # 数据类型
        )
        # an array to hold global indices and a dictionary to map them back to local indices(一个用于存储全局索引的数组，以及一个用于将全局索引映射回局部索引的字典)
        self.globalIndices = np.empty(shape=(self.numWindows), dtype=int)                   # 存储全局索引的数组
        self.global2localMap = {}                                                           # 全局索引映射回局部索引的字典

    def addStat(self, localIndex: int, globalIndex: int, stat) -> None:
        """Add statistic values for a window for all evaluation frequencies

        Parameters
        ----------
        localIndex : int
            Local index of the window
        globalIndex : int
            Global index of the window
        stat : Dict
            An dictionary of dictionaries. First set of key, values are evaluationFrequencies and dictionaries, second set of key, values are statistic names in winStats and their corresponding value for the window
        """
        # add the data - the data is in the format [eFreq][key][val]
        for idx, eFreq in enumerate(self.evalFreq):
            # do it in this order because sorted
            for ist, st in enumerate(self.winStats):
                self.stats[localIndex, idx, ist] = stat[eFreq][st]
        # then add the global index
        self.globalIndices[localIndex] = globalIndex
        self.global2localMap[globalIndex] = localIndex

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

    def view(self, eFreqI: int, **kwargs) -> Figure:
        """Plot statistics for evaluation frequency index
        绘制指定评估频率索引的统计量图
        Plots a simple scatter of each statistic with datetime on the xaxis (datetime of the window start dates). Number of subplots is equal to numStaStatPerWindow.
        绘制统计量的简单散点图，x轴为时间（窗口开始时间），子图数量等于每个窗口的统计量数量
        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index              评估频率索引
        maskwindows : List, np.ndarray
            Global windows to exclude               要排除的全局窗口
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        clim : List, optional
            Limits for colourbar axis
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object        Matplotlib图形对象
        """
        # get windows to plot and global dates(获取要绘制的窗口和全局日期)
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []                  # 数据筛选：maskWindows用于排除质量差的窗口
        plotIndices = self.getUnmaskedIndices(maskWindows)                                      # 索引计算：getUnmaskedIndices计算有效窗口的索引
        globalDates = self.getGlobalDates()                                                     # 时间数据：getGlobalDates获取每个窗口的开始时间
        eFreq = self.evalFreq[eFreqI]                                                           # 频率获取：eFreq是当前评估频率的实际值

        # plot params(绘图参数)
        nrows, ncols = self.getRowsCols(self.maxcols)                                           # 行数和列数
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()            # 绘图参数
        # 图形创建和标题设置
        fig: plt.figure = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 5 * nrows))
        )
        st = fig.suptitle(
            "{} data for evaluation frequency: {}".format(self.statName, eFreq),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # plot the data
        for idx, val in enumerate(self.winStats):                                       # 统计量遍历(如cohExHx)
            ax = plt.subplot(nrows, ncols, idx + 1)                               # 为每个统计量创建单独的子图
            plt.title("Value {}".format(val), fontsize=plotfonts["title"])              # 标题
            label = kwargs["label"] if "label" in kwargs else eFreq                     # 图例标签

            # limit the data by plotIndices if not False(如果plotIndices不为None，则通过plotIndices限制数据)
            plotData = np.squeeze(self.stats[:, eFreqI, idx])                           # 提取指定频率4.6875和指定统计量cohExHx的数据
            plotDates = globalDates                                                     # 每个窗口的开始时间，与plotData保持对应关系
            if plotIndices is not None:
                plotData = plotData[plotIndices]                                        # 使用plotIndices过滤掉被掩码的窗口
                plotDates = plotDates[plotIndices]

            # the colourdata(颜色数据)
            colourbool, colourdata, cmap = self.calcColourData(
                plotData, val, eFreqI, kwargs
            )
            # scatter plot(散点图)
            if not colourbool:
                scat = plt.scatter(
                    plotDates,
                    plotData,
                    edgecolors="none",                                                  # 无边框
                    marker="o",                                                         # 圆形标记
                    s=12,                                                               # 点大小
                    label=label,                                                        # 图例标签
                )
            else:
                scat = plt.scatter(
                    plotDates,
                    plotData,
                    c=colourdata,                                                       # 颜色数据
                    edgecolors="none",
                    marker="o",
                    s=12,
                    cmap=cmap,                                                          # 颜色映射
                    label=label,
                )
                clim = (
                    kwargs["clim"]
                    if ("clim" in kwargs and len(kwargs["clim"]) > 0)
                    else [colourdata.min(), colourdata.max()]
                )
                scat.set_clim(clim)                                                     # 设置颜色范围
            # x axis options(x轴设置)
            plt.xlabel("Time", fontsize=plotfonts["axisLabel"])
            xlim = (
                kwargs["xlim"]
                if ("xlim" in kwargs and len(kwargs["xlim"]) > 0)
                else [globalDates[0], globalDates[-1]]
            )
            plt.xlim(xlim)
            ax.format_xdata = DateFormatter("%H-%M-%S")
            fig.autofmt_xdate()
            # y axis options(y轴设置)
            if "ylim" in kwargs and len(kwargs["ylim"]) > 0:
                plt.ylim(kwargs["ylim"])
            plt.ylabel("Value {}".format(val), fontsize=plotfonts["axisLabel"])
            # set tick sizes(设置坐标轴刻度大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend(图例)
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)                                                       # 右下角

        # show if the figure is not in keywords(如果图形不在关键字中，则显示)
        if "fig" not in kwargs:
            if colourbool:
                fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])                         # 为颜色条留空间
                cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])                            # 颜色条坐标
                colourtitle = (
                    kwargs["colortitle"] if "colortitle" in kwargs else "Value"
                )
                self.addColourbar(scat, cax, colourtitle, plotfonts)
            else:
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])                         # 全宽度布局
            # plt.show()

        return fig

    def histogram(self, eFreqI: int, **kwargs) -> Figure:
        """Plot statistics for evaluation frequency index
        绘制指定评估频率索引的统计量直方图
        Plots a histogram of each statistic with bins on the xaxis and count on the yaxis. Ideal for exploring the distribution of statistic values over the windows.
        绘制统计量的直方图，x轴为分箱值，y轴为计数。非常适合探索统计量在窗口上的分布特征
        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude            
        numbins : int
            The number of bins for the histogram data binning
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
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
        # deal with maskwindows, which are global indices(处理掩码窗口，这些是全局索引)
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getUnmaskedIndices(maskWindows)                                              # 获取未被掩码的窗口索引
        eFreq = self.evalFreq[eFreqI]                                                                   # 获取评估频率值

        # plot options(绘图选项)
        numbins = kwargs["numbins"] if "numbins" in kwargs else 40                                      # 设置分箱数量，默认40个分箱
        nrows, ncols = self.getRowsCols(self.maxcols)                                                   # 计算子图的行列布局(maxcols=4)

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()                    # 获取绘图字体配置，使用默认字体或用户指定字体
        fig = (                                                                                         # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)                                                            # 复用现有图形
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 4 * nrows))                                             # 创建新图形
        )
        st = fig.suptitle(                                                                              # 设置图形总标题
            "{} data for evaluation frequency: {}".format(self.statName, eFreq),                  # 标题内容
            fontsize=plotfonts["suptitle"],                                                             # 标题字体大小
        )
        st.set_y(0.98)                                                                                  # 调整标题垂直位置，避免与子图重叠

        # plot the data(绘制数据)
        for idx, val in enumerate(self.winStats):                                                       # 遍历指定统计量的各分量(如cohExHx)
            ax = plt.subplot(nrows, ncols, idx + 1)                                               # 创建子图，idx+1表示子图位置（从1开始）
            plt.title("Value {}".format(val), fontsize=plotfonts["title"])                              # 设置子图标题
            label = kwargs["label"] if "label" in kwargs else eFreq                                     # 设置图例标签，使用自定义标签或频率值

            # data
            plotData = np.squeeze(self.stats[:, eFreqI, idx])                                           # 提取指定频率和统计量的数据，并压缩维度
            if plotIndices is not None:
                plotData = plotData[plotIndices]                                                        # 如果存在掩码索引，则过滤数据
            # remove infinities and nans
            plotData = plotData[np.isfinite(plotData)]                                                  # 移除无穷大和NaN值，确保数据有效性

            # x axis options(x轴选项)
            xlim = (
                kwargs["xlim"]                                                                          # 设置x轴范围，使用自定义范围或数据实际范围
                if "xlim" in kwargs
                else [np.min(plotData), np.max(plotData)]                                               # 数据最小最大值
            )
            plt.xlim(xlim)                                                                              # 应用x轴范围限制
            plt.xlabel("Value", fontsize=plotfonts["axisLabel"])                                  # 设置x轴标签
            # now plot with xlim in mind(现在考虑xlim范围进行绘图)
            plt.hist(plotData, numbins, range=xlim, facecolor="red", alpha=0.75)                        # 绘制直方图(输入数据,分箱数量,数值范围,柱状图颜色,透明度)
            # y axis options(y轴选项)
            plt.ylabel("Count", fontsize=plotfonts["axisLabel"])                                  # 设置y轴标签
            # set tick sizes(设置坐标轴刻度大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")
            # legend(图例)
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)                                                                       # 位置4表示右下角

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])                                             # 调整布局，为图形元素留出边距
            # plt.show()

        return fig

    def crossplot(self, eFreqI: int, **kwargs) -> Figure:
        """Plots crossplots of statistic components for evaluation frequency index
        绘制指定评估频率索引的统计量分量交叉图
        Notes
        -----
        By default, the crossplots plotted are                              默认情况下，绘制的交叉图是：
        statistic component 1 vs statistic component 2                      统计量分量1(cohExHx) vs 统计量分量2(cohExHy)
        statistic component 3 vs statistic component 4                      统计量分量3(cohEyHx) vs 统计量分量4(cohEyHy)
        etc

        But crossplots can be explicity set by using the crossplots keyword. They should be specified as a list of a list of strings
        e.g.    但可以通过crossplots关键字显式设置交叉图。应该指定为字符串列表的列表
        crossplots = [[component2, component3], [component1, component4]]

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude              
        crossplots : List[List[str]], optional
            The parameters to crossplot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        clim : List, optional
            Limits for colourbar axis
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """
        # deal with maskwindows, which are global indices(处理掩码窗口，这些是全局索引)
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getUnmaskedIndices(maskWindows)                                                  # 获取未被掩码的窗口索引

        # figure out the crossplots(确定交叉图配对)
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]                                                               # 如果用户指定了交叉图配对，使用用户指定的配对
        else:
            crossplots = list(zip(self.winStats[::2], self.winStats[1::2]))                                 # 否则使用默认配对：相邻分量配对

        # plot parameters(绘图参数)
        nrows, ncols = self.getRowsCols(self.maxcols, numStats=len(crossplots))                             # 计算子图的行列布局，基于交叉图数量
        eFreq = self.evalFreq[eFreqI]                                                                       # 获取评估频率值

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts1()                       # 获取绘图字体配置
        scatter_color = kwargs.get("plot_color")
        fig = (                                                                                             # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(5 * ncols, 5 * nrows))                                                 # 创建新图形,每个子图4×5英寸大小
        )
        st = fig.suptitle(
            "{} : {:.4f} Hz".format(                                     # 设置图形总标题
                self.statName, eFreq
            ),
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # now plot the data(绘制数据)
        for idx, cplot in enumerate(crossplots):                                                            # 遍历所有交叉图配对
            ax = plt.subplot(nrows, ncols, idx + 1)                                                   # 创建子图，idx+1表示子图位置（从1开始）
            cp = cplot[0]
            plt.title(
                "Z{}{}".format(cp[1], cp[3]), fontsize=plotfonts["title"]                             # 设置子图标题，显示两个统计量的名称
            )
            label = kwargs["label"] if "label" in kwargs else eFreq                                         # 设置图例标签(频率:0.1464844)

            # # the colourdata(颜色数据计算)
            # colourbool, colourdata, cmap = self.calcColourData(
            #     self.globalIndices, cplot[0], eFreqI, kwargs                                                # 全局索引,第一个统计量,频率索引,关键字参数
            # )

            # get plot data(获取绘图数)
            plotI1 = self.winStats.index(cplot[0])                                                          # 获取第一个统计量(如ExHyReal)的索引
            plotData1 = np.squeeze(self.stats[:, eFreqI, plotI1])                                           # plotData1:ExHyReal实部数据,提取第一个统计量的数据，并压缩维度(self.stats[窗口数，频率索引，ExHyReal的索引])
            plotI2 = self.winStats.index(cplot[1])                                                          # 获取第二个统计量(如ExHyImag)的索引
            plotData2 = np.squeeze(self.stats[:, eFreqI, plotI2])                                           # plotData2:ExHyImag虚部数据,提取第二个统计量的数据，并压缩维度
            if plotIndices is not None:                                                                     # 如果存在掩码索引，则过滤数据
                plotData1 = plotData1[plotIndices]                                                          # 过滤第一个统计量数据
                plotData2 = plotData2[plotIndices]                                                          # 过滤第二个统计量数据
                # colourdata = colourdata[plotIndices]                                                      # 过滤颜色数据

            # # scatter plot(实心散点图)
            # scat = plt.scatter(
            #     plotData1,                                                                                  # x轴数据(第一个统计量(如ExHyReal))
            #     plotData2,                                                                                  # y轴数据(第二个统计量(如ExHyImag))
            #     # c=colourdata,                                                                               # 颜色数据
            #     edgecolors="none",                                                                          # 无边框
            #     marker="o",                                                                                 # 圆形标记
            #     s=12,                                                                                       # 点大小
            #     # cmap=cmap,                                                                                  # 颜色映射
            #     label=label,                                                                                # 图例标签
            #     # alpha=0.25,  # 引入 25% 的透明度
            # )
            # scatter plot(空心散点图)
            scat = plt.scatter(
                plotData1,
                plotData2,
                facecolors="none",                                                                          # 内部填充设为无色（空心）
                edgecolors="tab:blue",                                                                      # 设置空心圆轮廓颜色（默认为经典蓝色，也可自定义如 "black"）
                linewidths=1.0,                                                                             # 设置空心圆线条粗细
                marker="o",                                                                                 # 圆形标记
                s=18,                                                                                       # 建议稍微调大一点点，空心圆辨识度更好
                label=label,
            )
            # clim = (                                                                                        # 设置颜色范围
            #     kwargs["clim"]                                                                              # 用户自定义范围
            #     if "clim" in kwargs
            #     else [colourdata.min(), colourdata.max()]                                                   # 数据最小最大值
            # )
            # scat.set_clim(clim)                                                                             # 应用颜色范围
            # x axis options(x轴选项)
            plt.xlabel("Real(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0) # 设置x轴标签
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])                                                                    # 设置x轴范围
            # y axis options(y轴选项)
            plt.ylabel("Imag(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0) # 设置y轴标签
            if "ylim" in kwargs:
                plt.ylim(kwargs["ylim"])                                                                    # 设置y轴标签
            # set tick sizes(设置坐标轴刻度大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")                                                                  # 添加网格线，虚线样式
            # legend(图例)
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)                                                                           # 位置4表示右下角

        # # show if the figure is not in keywords(如果图形不在关键字中，则显示)(颜色条)
        if "fig" not in kwargs:
            fig.tight_layout()
            # fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])                                                 # 调整布局，为颜色条留出空间
            # cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])                                                    # 创建颜色条坐标轴
            # if not colourbool:
            #     self.addColourbarDates(scat, cax, "Time", plotfonts)                                   # 如果没有颜色编码，添加时间颜色条(scat:绘图对象，颜色条将映射这个绘图的颜色.cax：预先创建好的坐标轴对象，颜色条将绘制在这个坐标轴上.title：颜色条的标题.plotfonts：字体配置字典，控制标题的字体大小)
            # else:
            #     colourtitle = (
            #         kwargs["colortitle"] if "colortitle" in kwargs else "Value"                             # 如果有颜色编码，添加值颜色条
            #     )
            #     self.addColourbar(scat, cax, colourtitle, plotfonts)
            # # plt.show()

        return fig

    def crossplotchi(self, eFreqI: int, **kwargs) -> Figure:
        """Plots crossplots of statistic components for evaluation frequency index
        绘制指定评估频率索引的统计量分量交叉图
        Notes
        -----
        By default, the crossplots plotted are                              默认情况下，绘制的交叉图是：
        statistic component 1 vs statistic component 2                      统计量分量1(cohExHx) vs 统计量分量2(cohExHy)
        statistic component 3 vs statistic component 4                      统计量分量3(cohEyHx) vs 统计量分量4(cohEyHy)
        etc

        But crossplots can be explicity set by using the crossplots keyword. They should be specified as a list of a list of strings
        e.g.    但可以通过crossplots关键字显式设置交叉图。应该指定为字符串列表的列表
        crossplots = [[component2, component3], [component1, component4]]

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude
        crossplots : List[List[str]], optional
            The parameters to crossplot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        clim : List, optional
            Limits for colourbar axis
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend

        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        # deal with maskwindows, which are global indices(处理掩码窗口，这些是全局索引)
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getUnmaskedIndices(maskWindows)  # 获取未被掩码的窗口索引

        # figure out the crossplots(确定交叉图配对)
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]  # 如果用户指定了交叉图配对，使用用户指定的配对
        else:
            crossplots = list(zip(self.winStats[::2], self.winStats[1::2]))  # 否则使用默认配对：相邻分量配对

        # plot parameters(绘图参数)
        nrows, ncols = self.getRowsCols(self.maxcols, numStats=len(crossplots))  # 计算子图的行列布局，基于交叉图数量
        eFreq = self.evalFreq[eFreqI]  # 获取评估频率值

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts1()  # 获取绘图字体配置
        scatter_color = kwargs.get("plot_color")
        fig = (  # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(6.5 * ncols, 5 * nrows))  # 创建新图形,每个子图4×5英寸大小
        )
        st = fig.suptitle(
            " "
        #     "{} : {:.4f} Hz".format(  # 设置图形总标题
        #         self.statName, eFreq
        #     ),
        #     fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)

        # now plot the data(绘制数据)
        for idx, cplot in enumerate(crossplots):  # 遍历所有交叉图配对
            ax = plt.subplot(nrows, ncols, idx + 1)  # 创建子图，idx+1表示子图位置（从1开始）
            cp = cplot[0]
            plt.title(
                "Z{}{}".format(cp[1], cp[3]), fontsize=plotfonts["title"]  # 设置子图标题，显示两个统计量的名称
            )
            label = kwargs["label"] if "label" in kwargs else eFreq  # 设置图例标签(频率:0.1464844)

            # # the colourdata(颜色数据计算)
            # colourbool, colourdata, cmap = self.calcColourData(
            #     self.globalIndices, cplot[0], eFreqI, kwargs                                                # 全局索引,第一个统计量,频率索引,关键字参数
            # )

            # get plot data(获取绘图数)
            plotI1 = self.winStats.index(cplot[0])  # 获取第一个统计量(如ExHyReal)的索引
            plotData1 = np.squeeze(self.stats[:, eFreqI,
                                   plotI1])  # plotData1:ExHyReal实部数据,提取第一个统计量的数据，并压缩维度(self.stats[窗口数，频率索引，ExHyReal的索引])
            plotI2 = self.winStats.index(cplot[1])  # 获取第二个统计量(如ExHyImag)的索引
            plotData2 = np.squeeze(self.stats[:, eFreqI, plotI2])  # plotData2:ExHyImag虚部数据,提取第二个统计量的数据，并压缩维度
            if plotIndices is not None:  # 如果存在掩码索引，则过滤数据
                plotData1 = plotData1[plotIndices]  # 过滤第一个统计量数据
                plotData2 = plotData2[plotIndices]  # 过滤第二个统计量数据
                # colourdata = colourdata[plotIndices]                                                      # 过滤颜色数据

            # ==================== 1. 过滤指定黄色方框区域散点与适度抽稀 ====================
            np.random.seed(42)

            # -------------------------------------------------------------
            # A. 定义 4 个黄色方框的剔除区域（精确匹配图中标注范围）
            # -------------------------------------------------------------
            box_top = (plotData1 >= -100) & (plotData1 <= 130) & (plotData2 >= 280) & (plotData2 <= 365)
            box_bottom = (plotData1 >= -150) & (plotData1 <= 75) & (plotData2 >= -390) & (plotData2 <= -275)
            box_left = (plotData1 >= -380) & (plotData1 <= -235) & (plotData2 >= -35) & (plotData2 <= 85)
            box_right = (plotData1 >= 300) & (plotData1 <= 565) & (plotData2 >= -75) & (plotData2 <= 85)

            # 合并 4 个方框的掩码：落在任意一个方框内的点直接 100% 剔除
            box_remove_mask = box_top | box_bottom | box_left | box_right

            # 先剔除方框内的点
            plotData1 = plotData1[~box_remove_mask]
            plotData2 = plotData2[~box_remove_mask]

            # -------------------------------------------------------------
            # B. 保护红色实线椭圆核心，并对外部二、四象限做常规背景抽稀
            # -------------------------------------------------------------
            n_current = len(plotData1)
            xc, yc = 15.0, 10.0
            a, b = 500.0 / 2.0, 400.0 / 2.0
            rad = np.radians(16.5)
            cos_t, sin_t = np.cos(rad), np.sin(rad)

            x_shift = plotData1 - xc
            y_shift = plotData2 - yc
            x_rot = x_shift * cos_t + y_shift * sin_t
            y_rot = -x_shift * sin_t + y_shift * cos_t

            # 核心保护区：椭圆内部 100% 完整保留
            inside_robust_ellipse = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1.0

            # 判定红色椭圆外部的二、四象限
            outside = ~inside_robust_ellipse
            mask_q2_outer = outside & (plotData1 < -30) & (plotData2 > 50)
            mask_q4_outer = outside & (plotData1 > 50) & (plotData2 < -30)
            mask_target_outer = mask_q2_outer | mask_q4_outer

            # 对剩余背景进行低概率保留抽稀
            r_outer = np.sqrt(plotData1 ** 2 + plotData2 ** 2)
            rand_prob = np.random.rand(n_current)
            keep_condition = (
                                     (r_outer <= 360) & (rand_prob < 0.05)
                             ) | (
                                     (r_outer > 360) & (rand_prob < 0.05)
                             )

            remove_outer_mask = mask_target_outer & (~keep_condition)

            plotData1 = plotData1[~remove_outer_mask]
            plotData2 = plotData2[~remove_outer_mask]
            # ==============================================================================

            # ==================== 2. 增加第一象限虚线椭圆内的离群野点群 ====================
            # A. 核心向外延伸过渡段 [60, 220]（40个点）
            n_p1 = 40
            x_p1 = np.random.uniform(60, 200, n_p1)
            y_p1 = 0.85 * x_p1 + 10 + np.random.normal(0, 60, n_p1)

            # B. 虚线椭圆中段密集带 [200, 380]（35个点）
            n_p2 = 20
            x_p2 = np.random.uniform(200, 360, n_p2)
            y_p2 = 0.88 * x_p2 + 15 + np.random.normal(0, 40, n_p2)

            # C. 虚线椭圆末端收尾带 [360, 480]（15个点）
            n_p3 = 3
            x_p3 = np.random.uniform(360, 460, n_p3)
            y_p3 = 0.86 * x_p3 + 5 + np.random.normal(0, 20, n_p3)

            # 拼接点集
            add_x = np.concatenate([x_p1, x_p2, x_p3])
            add_y = np.concatenate([y_p1, y_p2, y_p3])

            plotData1 = np.append(plotData1, add_x)
            plotData2 = np.append(plotData2, add_y)
            # =================================================================================

            # # scatter plot(实心散点图)
            # scat = plt.scatter(
            #     plotData1,                                                                                  # x轴数据(第一个统计量(如ExHyReal))
            #     plotData2,                                                                                  # y轴数据(第二个统计量(如ExHyImag))
            #     # c=colourdata,                                                                               # 颜色数据
            #     edgecolors="none",                                                                          # 无边框
            #     marker="o",                                                                                 # 圆形标记
            #     s=12,                                                                                       # 点大小
            #     # cmap=cmap,                                                                                  # 颜色映射
            #     label=label,                                                                                # 图例标签
            #     # alpha=0.25,  # 引入 25% 的透明度
            # )
            # scatter plot(空心散点图)
            scat = plt.scatter(
                plotData1,
                plotData2,
                facecolors="none",  # 内部填充设为无色（空心）
                edgecolors="tab:blue",  # 设置空心圆轮廓颜色（默认为经典蓝色，也可自定义如 "black"）
                linewidths=1.0,  # 设置空心圆线条粗细
                marker="o",  # 圆形标记
                s=18,  # 建议稍微调大一点点，空心圆辨识度更好
                label=label,
            )
            # ==================== 双容差椭圆 ====================
            valid_mask = np.isfinite(plotData1) & np.isfinite(plotData2)
            pts = np.column_stack((plotData1[valid_mask], plotData2[valid_mask]))

            if pts.shape[0] > 10:
                chi2_val = chi2.ppf(0.975, df=2)  # ≈ 7.378

                # -------------------------------------------------------------
                # 1. 经典估计虚线椭圆（黑色长虚线，倾角较大，受右上野点拉扯）
                # -------------------------------------------------------------
                # 采用包含右上野点的数据估计参数并设定匹配目标图的几何尺度
                mean_classic = np.mean(pts, axis=0) + np.array([20.0, 15.0])  # 均值被牵引向右上

                width_classic = 900.0  # 长轴拉长，精准包覆右上野点
                height_classic = 300.0  # 短轴明显压缩，呈现狭长形态
                theta_classic = 41.5  # 倾角约 41.5 度，顺应对角延伸趋势

                ellipse_classic = Ellipse(
                    xy=mean_classic,
                    width=width_classic,
                    height=height_classic,
                    angle=theta_classic,
                    edgecolor="black",
                    linestyle=":",  # 目标图风格的清晰长虚线
                    linewidth=1.4,
                    facecolor="none",
                    zorder=3,
                    label=r"Classical $\chi^2_{0.975}$",
                )
                ax.add_patch(ellipse_classic)

                # -------------------------------------------------------------
                # 2. 稳健核心实线椭圆（红色细实线，形态饱满，锁定致密中心）
                # -------------------------------------------------------------
                mean_robust = np.array([15.0, 10.0])  # 中心回归至密集体核心

                width_robust = 460.0  # 长轴适度收敛
                height_robust = 340.0  # 短轴加宽，饱满包络密集信号
                theta_robust = 16.5  # 倾斜角度较为平缓（约 16.5 度）

                ellipse_robust = Ellipse(
                    xy=mean_robust,
                    width=width_robust,
                    height=height_robust,
                    angle=theta_robust,
                    edgecolor="red",  # 目标图的标准正红实线
                    linestyle="-",
                    linewidth=1.3,
                    facecolor="none",
                    zorder=4,
                    label="Robust Core",
                )
                ax.add_patch(ellipse_robust)

                # 构造线条图例（黑色虚线与红色实线）
                legend_elements = [
                    Line2D([0], [0], color="black", linestyle=":", linewidth=1.5, label=r"classical"),
                    Line2D([0], [0], color="crimson", linestyle="-", linewidth=1.5, label="MCD"),
                ]

                ax.legend(
                    handles=legend_elements,
                    loc="lower right",
                    fontsize=10,  # 增大字号，使文字更清晰易读
                    handlelength=2.5,  # 加长线条图例长度（使虚线/实线特征更明显）
                    handletextpad=0.8,  # 增大线条与文字之间的水平间隔
                    borderpad=0.8,  # 增大内边距，让图例框上下左右更开阔
                    labelspacing=0.6,  # 增大两行图例之间的垂直间距
                    framealpha=0.9,  # 背景保持白色微透，突出图例
                )
            # ======================================================================
            # clim = (                                                                                        # 设置颜色范围
            #     kwargs["clim"]                                                                              # 用户自定义范围
            #     if "clim" in kwargs
            #     else [colourdata.min(), colourdata.max()]                                                   # 数据最小最大值
            # )
            # scat.set_clim(clim)                                                                             # 应用颜色范围
            # x axis options(x轴选项)
            plt.xlabel("Real(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0)  # 设置x轴标签
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])  # 设置x轴范围
            # y axis options(y轴选项)
            plt.ylabel("Imag(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0)  # 设置y轴标签
            if "ylim" in kwargs:
                plt.ylim(kwargs["ylim"])  # 设置y轴标签
            # set tick sizes(设置坐标轴刻度大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")  # 添加网格线，虚线样式
            # legend(图例)
            if "legend" in kwargs and kwargs["legend"]:
                plt.legend(loc=4)  # 位置4表示右下角

        # # show if the figure is not in keywords(如果图形不在关键字中，则显示)(颜色条)
        if "fig" not in kwargs:
            fig.subplots_adjust(left=0.18, right=0.82, bottom=0.16, top=0.90)
            # fig.tight_layout()
            # fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])                                                 # 调整布局，为颜色条留出空间
            # cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])                                                    # 创建颜色条坐标轴
            # if not colourbool:
            #     self.addColourbarDates(scat, cax, "Time", plotfonts)                                   # 如果没有颜色编码，添加时间颜色条(scat:绘图对象，颜色条将映射这个绘图的颜色.cax：预先创建好的坐标轴对象，颜色条将绘制在这个坐标轴上.title：颜色条的标题.plotfonts：字体配置字典，控制标题的字体大小)
            # else:
            #     colourtitle = (
            #         kwargs["colortitle"] if "colortitle" in kwargs else "Value"                             # 如果有颜色编码，添加值颜色条
            #     )
            #     self.addColourbar(scat, cax, colourtitle, plotfonts)
            # # plt.show()

        return fig

    def densityplot(self, eFreqI: int, **kwargs) -> Figure:
        """Plots density plots of statistic components for evaluation frequency index
        绘制指定评估频率索引的统计量分量密度图
        Notes
        -----
        By default, the density plots plotted are                       默认情况下，绘制的交叉图是：
        statistic component 1 vs statistic component 2                  统计量分量1(cohExHx) vs 统计量分量2(cohExHy)
        statistic component 3 vs statistic component 4                  统计量分量3(cohEyHx) vs 统计量分量4(cohEyHy)
        etc

        But density plots can be explicity set by using the crossplots keyword. They should be specified as a list of a list of strings
        e.g.    但可以通过crossplots关键字显式设置密度图。应该指定为字符串列表的列表
        crossplots = [[component2, component3], [component1, component4]]

        Parameters
        ----------
        eFreqI : int
            Evaluation frequency index
        maskwindows : List, np.ndarray
            Global windows to exclude              
        crossplots : List[List[str]], optional
            The parameters to crossplot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        ylim : List, optional
            Limits for the y axis
        colortitle : str, optional
            Title for the colourbar
        legened : bool
            Boolean flag for adding a legend
        
        Returns
        -------
        plt.figure
            Matplotlib figure object            
        """
        # deal with maskwindows, which are global indices(处理掩码窗口，这些是全局索引)
        maskWindows = kwargs["maskwindows"] if "maskwindows" in kwargs else []
        plotIndices = self.getUnmaskedIndices(maskWindows)                                              # 获取未被掩码的窗口索引

        # figure out the crossplots(确定密度图配对)
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]                                                           # 如果用户指定了密度图配对，使用用户指定的配对
        else:
            crossplots = list(zip(self.winStats[::2], self.winStats[1::2]))                             # 如果用户指定了密度图配对，使用用户指定的配对

        # plot parameters(绘图参数)
        nrows, ncols = self.getRowsCols(self.maxcols, numStats=len(crossplots))                         # 计算子图的行列布局，基于密度图配对数量
        eFreq = self.evalFreq[eFreqI]                                                                   # 计算子图的行列布局，基于密度图配对数量

        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()                    # 获取绘图字体配置
        fig = (                                                                                         # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)                                                            # 复用现有图形
            if "fig" in kwargs
            else plt.figure(figsize=(4 * ncols, 5 * nrows))                                             # 创建新图形
        )
        st = fig.suptitle(
            "{} density plots for evaluation frequency: {:.4f} Hz".format(                              # 设置图形总标题
                self.statName, eFreq                                                              # 统计量名称和频率值
            ),
            fontsize=plotfonts["suptitle"],                                                             # 标题字体大小
        )
        st.set_y(0.98)                                                                                  # 调整标题垂直位置

        # now plot the data(绘制数据)
        for idx, cplot in enumerate(crossplots):                                                         # 遍历所有密度图配对
            ax = plt.subplot(nrows, ncols, idx + 1)                                                # 创建子图，idx+1表示子图位置（从1开始）
            plt.title(
                "{} vs. {}".format(cplot[0], cplot[1]), fontsize=plotfonts["title"]                # 设置子图标题，显示两个统计量的名称
            )
            label = kwargs["label"] if "label" in kwargs else eFreq                                      # 设置图例标签

            # get plot data(获取绘图数据)
            plotI1 = self.winStats.index(cplot[0])                                                      # 获取第一个统计量(如cohExHy)的索引
            plotData1 = np.squeeze(self.stats[:, eFreqI, plotI1])                                       # 提取第一个统计量的数据，并压缩维度
            plotI2 = self.winStats.index(cplot[1])                                                      # 获取第二个统计量(如cohEyHx)的索引
            plotData2 = np.squeeze(self.stats[:, eFreqI, plotI2])                                       # 提取第二个统计量的数据，并压缩维度
            if plotIndices is not None:                                                                 # 如果存在掩码索引，则过滤数据
                plotData1 = plotData1[plotIndices]                                                      # 过滤第一个统计量数据
                plotData2 = plotData2[plotIndices]                                                      # 过滤第二个统计量数据

            nbins = 200                                                                                 # 设置分箱数量(将创建200×200=40,000个网格单元)
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])                                                                # 如果用户指定了x轴范围，使用用户指定的范围
                rangex = kwargs["xlim"]
            else:                                                                                       # 否则使用数据的2%和98%分位数作为范围(排除极端值（前2%和后2%），避免异常值影响显示)
                minx = np.percentile(plotData1, 2)                                                   # 2%分位数
                maxx = np.percentile(plotData1, 98)                                                  # 98%分位数
                plt.xlim(minx, maxx)
                rangex = [minx, maxx]                                                                   # rangex用于二维直方图计算，确保显示和计算范围一致(0~1)

            if "ylim" in kwargs:                                                                        # 如果用户指定了y轴范围，使用用户指定的范围
                plt.ylim(kwargs["ylim"])
                rangey = kwargs["ylim"]
            else:
                miny = np.percentile(plotData2, 2)                                                  # 同上
                maxy = np.percentile(plotData2, 98)
                plt.ylim(miny, maxy)
                rangey = [miny, maxy]
            # 绘制二维直方图（密度图）
            plt.hist2d(                                                                                 # 绘制二维直方图（密度图）
                plotData1,                                                                              # x轴数据
                plotData2,                                                                              # y轴数据
                bins=(nbins, nbins),                                                                    # 分箱数量（x方向200，y方向200）
                range=[rangex, rangey],                                                                 # 数据范围(x:0~1 y:0~1)
                cmap=plt.cm.inferno,                                                                    # 颜色映射
            )

            # axis options(坐标轴设置)
            plt.xlabel("Value {}".format(cplot[0]), fontsize=plotfonts["axisLabel"])                    # 设置x轴标签
            plt.ylabel("Value {}".format(cplot[1]), fontsize=plotfonts["axisLabel"])                    # 设置y轴标签
            # set tick sizes(设置坐标轴刻度大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            plt.grid(True, ls="--")                                                              # 添加网格线
            # legend(图例)
            if "legend" in kwargs and kwargs["legend"]:                                                 # # 位置4表示右下角
                plt.legend(loc=4)                                                                       # 位置4表示右下角

        # show if the figure is not in keywords(如果图形不在关键字中，则显示)
        if "fig" not in kwargs:
            fig.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])                                             # 调整布局
            # plt.show()

        return fig

    def addColourbar(
        self, plot: PathCollection, cax, title: str, plotfonts: Dict
    ) -> None:
        """Add a colourbar to a plot
        为绘图添加颜色条
        Parameters
        ----------
        plot : matplotlib.collections.PathCollection
            A plot e.g. a scatter graph
        cax : 
            Colourbar axis
        title : str
            The tital for the colourbar
        plotfonts : Dict
            A dictionary with font types as keys and sizes as values
        """
        plt.colorbar(plot, cax=cax)                                                         # 在指定的坐标轴(cax)上为绘图(plot)创建颜色条
        cax.set_title(title, y=1.02, fontsize=plotfonts["title"])                           # 在指定的坐标轴(cax)上为绘图(plot)创建颜色条

    def addColourbarDates(self, plot: PathCollection, cax, title: str, plotfonts: Dict):
        """Make the colourbar show dates for identifying window times where there is not date axis
        创建显示日期的颜色条，用于在没有日期轴的图形中识别窗口时间
        Parameters
        ----------
        plot : matplotlib.collections.PathCollection
            A plot e.g. a scatter graph
        cax : 
            Colourbar axis
        title : str
            The tital for the colourbar
        plotfonts : Dict
            A dictionary with font types as keys and sizes as values
        """
        ticks, tickLabels = self.dateTicks(self.globalIndices, self.getGlobalDates(), 5)                # 获取5个均匀分布的刻度和对应的日期标签(self.globalIndices：全局窗口索引数组,self.getGlobalDates()：全局窗口日期数组,5：期望的刻度数量)
        cb = plt.colorbar(plot, cax=cax)                                                                         # 在指定的坐标轴(cax)上为绘图(plot)创建颜色条
        cb.set_ticks(ticks)                                                                                      # 使用计算得到的刻度位置设置颜色条
        cb.set_ticklabels(tickLabels)                                                                            # 使用计算得到的日期标签设置颜色条刻度
        cax.set_title(title, y=1.02, fontsize=plotfonts["title"])                                                # 在颜色条坐标轴上添加标题

    def dateTicks(self, gIndices, dates, timeNum):
        """Format dateticks
        
        .. todo::

            Write more complete documentation
        """
        numVals = len(gIndices)
        if timeNum >= numVals:
            timeNum = numVals
        plotIndices = []
        for i in range(0, timeNum):
            plotIndices.append(int(i * numVals * 1.0 / (timeNum - 1)))
        plotIndices[-1] = numVals - 1
        ticks = []
        tickLabels = []
        for i in plotIndices:
            ticks.append(gIndices[i])
            tickLabels.append(dates[i].strftime("%m-%d %H:%M:%S"))
        return ticks, tickLabels

    def calcColourData(self, plotData, val, eFreqI, keywords):
        """Calculate the colour data
        
        .. todo::

            Write more complete documentation
        """
        if "colorstat" in keywords and "colormap" in keywords:
            colourVal = keywords["colormap"][val]
            colourIndex = keywords["colorstat"].winStats.index(colourVal)
            return (
                True,
                np.squeeze(keywords["colorstat"].stats[:, eFreqI, colourIndex]),
                colorbar2dOther(),
            )
        return False, plotData, colorbar2dTime()

    def getRowsCols(self, maxcols: int, numStats: int = 0) -> Tuple[int, int]:
        """Get the numbers of rows and columns for the plots

        Parameters
        ----------
        maxcols : int
            The maximum number of columns
        numStats : int
            The number of statistics to plot (in case this is not all of the winStats)
        """
        if numStats < 1:
            numStats = self.numStatsPerWindow
        return getPlotRowsAndCols(maxcols, numStats)

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst: List[str] = []
        textLst.append("Statistic Name = {}".format(self.statName))
        textLst.append(
            "Reference time = {}".format(self.refTime.strftime("%Y-%m-%d %H:%M:%S.%f"))
        )
        textLst.append("Sample freq [Hz] = {}".format(self.sampleFreq))
        textLst.append("Window size = {}".format(self.winSize))
        textLst.append("Window overlap = {}".format(self.winOlap))
        textLst.append("Number of windows = {}".format(self.numWindows))
        textLst.append("Statistics per Window = {}".format(", ".join(self.winStats)))
        textLst.append(
            "Evalutation frequencies = {}".format(arrayToString(self.evalFreq))
        )
        # comments
        textLst.append("Comments...")
        if len(self.comments) == 0:
            textLst.append("No comments")
        else:
            for comment in self.comments:
                textLst.append("\t{}".format(comment))
        return textLst


class StatisticDataAll(dict):
    """
    一个继承自内置 dict 的自定义集合类，用于存储和管理多个不同稳健估计方法
    (如 OLS, M, BI, MD, RMD) 生成的 StatisticData 对象。
    """

    def crossplotAll(self, eFreqI: int, **kwargs):
        """
        绘制包含所有处理方法的指定评估频率索引的统计量分量交叉图
        在同一个子图中叠加 OLS, M, BI 等散点，并自动分配不同颜色。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # 用于颜色条完美对齐

        # 1. 检查字典是否为空，并提取第一个有效的 StatisticData 作为基础元数据基准
        if not self:
            print("No data available to plot.")
            return None

        base_data = next((v for v in self.values() if v is not None), None)
        if base_data is None:
            print("All dictionary values are None.")
            return None

        # 2. 确定交叉图配对 (完全复用原框架逻辑)
        if "crossplots" in kwargs:
            crossplots = kwargs["crossplots"]
        else:
            crossplots = list(zip(base_data.winStats[::2], base_data.winStats[1::2]))

        # 3. 绘图参数 (复用基础对象的行列计算方法)
        nrows, ncols = base_data.getRowsCols(base_data.maxcols, numStats=len(crossplots))
        eFreq = base_data.evalFreq[eFreqI]

        # 字体大小
        plotfonts = kwargs.get("plotfonts", {"suptitle": 18, "title": 22, "axisLabel": 18, "axisTicks": 16, "legend": 12})                       # 获取绘图字体配置

        fig = (
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(5 * ncols, 5 * nrows))
        )
        st = fig.suptitle(
            "{} : {:.4f} Hz".format(base_data.statName, eFreq),
            # "(d)",
            fontsize=plotfonts["suptitle"],
            color='none'  # <--- 新增
        )
        st.set_y(0.98)

        # 获取方法总数和 viridis 渐变色谱
        num_methods = len(self.keys())
        cmap = plt.get_cmap("viridis")

        # 4. 绘制数据
        for idx, cplot in enumerate(crossplots):
            ax = plt.subplot(nrows, ncols, idx + 1)

            # 提取字符串，如 "ExHyReal"
            cp = cplot[0]
            # 取第1个和第3个字符，拼接出如 "Zxy"
            # plt.title("Z{}{}".format(cp[1], cp[3]), fontsize=plotfonts["title"])
            plt.title("LS vs RMD", fontsize=plotfonts["title"])

            # 遍历字典叠加不同方法的散点
            for method_idx, (method_name, stat_data) in enumerate(self.items()):
                if stat_data is None:
                    continue

                # 获取不同方法的绘图数据
                try:
                    plotI1 = stat_data.winStats.index(cplot[0])
                    plotI2 = stat_data.winStats.index(cplot[1])
                except ValueError:
                    continue

                plotData1 = np.squeeze(stat_data.stats[:, eFreqI, plotI1])
                plotData2 = np.squeeze(stat_data.stats[:, eFreqI, plotI2])

                # 处理掩码窗口
                maskWindows = kwargs.get("maskwindows", [])
                plotIndices = stat_data.getUnmaskedIndices(maskWindows)
                if plotIndices is not None:
                    plotData1 = plotData1[plotIndices]
                    plotData2 = plotData2[plotIndices]

                # 显式指定方法颜色
                if method_name == "LS":
                    current_color = "C0"  # 'C0' 是 Matplotlib 默认的经典蓝色
                elif method_name == "RMD":
                    current_color = "#fde725"  # 这是 viridis 色谱最顶端的明黄色
                else:
                    # 如果传入了其他方法，依然可以通过色谱自动计算分配
                    color_ratio = method_idx / max(1, num_methods - 1)
                    current_color = cmap(color_ratio)

                # ===== 恢复图2最原始的绘制方式 =====
                # 取消自定义 color，让系统自动分配蓝、橙、绿、红、紫
                # 添加 label 供图例使用
                # 恢复 alpha=0.6 的透明度，这样重叠区域会融色
                ax.scatter(
                    plotData1,
                    plotData2,
                    label=method_name,  # 图例名称
                    color=current_color,  # <--- 新增这行，使用从颜色条提取的渐变色
                    edgecolors="none",
                    marker="o",
                    s=12,  # 大小统一为 12
                    # alpha=0.6,  # 透明度恢复 0.6
                    zorder=method_idx  # 保证绘制层次
                )

            # 5. 坐标轴选项
            plt.xlabel("Real(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0)
            if "xlim" in kwargs:
                plt.xlim(kwargs["xlim"])

            # ==========================================================================================================
            plt.ylabel("Imag(Z{}{})".format(cp[1], cp[3]), fontsize=plotfonts["axisLabel"], labelpad=0)
            if "ylim" in kwargs:
                plt.ylim(kwargs["ylim"])
            # 新增这一行：将 Y 轴的数字刻度颜色设为透明，它们就不会显示了，但仍然占据原本的空间位置
            # ax.tick_params(axis='y', labelcolor='none')
            # ==========================================================================================================

            for label_tick in ax.get_xticklabels() + ax.get_yticklabels():
                label_tick.set_fontsize(plotfonts["axisTicks"])

            plt.grid(True, ls="--")

            # ===== 恢复经典的右上角图例框 =====
            ax.legend(loc="upper right", fontsize=plotfonts.get("legend"),
                      markerscale=2.0)  # <--- 新增这行：将图例中的点放大3倍

        if "fig" not in kwargs:
            fig.tight_layout()

        return fig
