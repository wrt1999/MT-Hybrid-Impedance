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
        statName: str,                                      
        refTime: datetime,                                  
        sampleFreq: float,                                  
        winSize: int,                                       
        winOlap: int,                                       
        **kwargs                                            
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
        self.statName: str = statName                                                           
        self.refTime = refTime                                                                  
        if isinstance(self.refTime, str):
            self.refTime = datetime.strptime(self.refTime, "%Y-%m-%d %H:%M:%S.%f")       
        self.sampleFreq: float = sampleFreq                                                     
        self.winSize: int = winSize                                                             
        self.winOlap: int = winOlap                                                             
        # defaults from keywords
        self.numWindows: int = 0                                                                
        self.winStats: List[str] = []                                                           
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
            # plt.show()

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
