import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from typing import List, Dict, Set

from resistics.common.base import ResisticsBase
from resistics.common.print import arrayToString, blockPrint
from resistics.common.math import getFrequencyArray


class MaskData(ResisticsBase):
    """Class for holding information about window masking

    Attributes
    ----------
    maskName : str
        The name of the mask
    sampleFreq : float  
        The sampling frequency of the data in Hz
    numLevels : int
        Number of decimation levels
    evalFreq : List[float]
        The evaluation frequnecies
    constraints : Dict
        Constraints index by evaluation frequency, statistic and component
    insideOut : Dict
        Flag stating whether to mask values inside constraints (False) or outside constraints (True), indexed by evaluation frequency, statistic and component
    maskWindows : Dict
        Dictionary for storing the masked windows for each evaluation frequency
    stats : List
        Statistics to use in the masking

    Methods
    -------
    __init__(sampleFreq, numLevels, evalFreq, kwargs)
        Initialise maskData
    setStats(stats)
        Set the statistics used in the MaskData
    addConstraint(stat, constraint, insideOut)
        Add constraints for statistic stat for all evaluation frequencies
    addConstraintLevel(stat, constraint, level, insideOut)
        Add constraints for statistic stat for all evaluation frequencies for a single decimation level  
    addConstraintFreq(stat, constraint, level, eIdx, insideOut)
        Add a constraint for statistic stat for a particular evaluation frequency               
    getConstraintFreq(level, eIdx, stat = "")
        Get the constraints for a evaluation frequency provided by the level and the evaluation frequency index and optional request constraints only for a particular statistic
    getMaskWindowsFreq(level, eIdx)
        Get a set of the masked windows for an evaluation frequency identified by decimation level and evaluation frequency index
    prepareDicts()
        Prepare the storage dictionaries given a new set of statistics
    resetMaskWindows()
        Clear the masked windows
    view(level)
        View the masked windows at a particular decimation level
    printList()
        Class status returned as list of strings
    printConstraints()
        Print constraint information to the console
    printConstraintsList()
        Return constraint information as a list of strings
    """

    def __init__(self, sampleFreq: float, numLevels: int, evalFreq, **kwargs):
        """Initialise mask data 
        初始化掩码数据对象
        Parameters
        ----------
        sampleFreq : float
            Data sampling frequency in Hz
        numLevels : int
            Number of decimation levels
        evalFreq : np.ndarray
            2-D Matrix, numlevels * evaluations frequencies per level. Contents are evaluations frequencies
        kwargs : Dict
            Two optional arguments: "name" for statistic name and "stats" for the statistics to use     可选参数："name"表示统计量名称，"stats"表示要使用的统计量列表
        """
        self.maskName: str = "mask"                                             # 设置掩码名称，默认为"mask"
        if "name" in kwargs:
            self.maskName = kwargs["name"]                                      # 如果提供了名称参数，使用提供的名称
        self.sampleFreq: float = sampleFreq                                     # 设置采样频率属性
        self.numLevels: int = numLevels                                         # 设置分频级数属性
        self.evalFreq: List[float] = evalFreq                                   # 设置评估频率属性
        # constraints and masked windows(约束条件和掩码窗口字典)
        self.constraints: Dict = {}                                             # 初始化约束条件字典,存储掩码生成的约束条件
        self.insideOut: Dict = {}                                               # 初始化内外标记字典,标记是否内外翻转掩码逻辑(翻转(True):取阈值范围(如[0.7, 1.0])外的窗口;不翻转(False):取阈值范围(如[0.7, 1.0])内的窗口)
        self.maskWindows: Dict = {}                                             # 初始化掩码窗口字典,存储实际的掩码窗口索引
        # stats is the stats to use(统计量：要使用的统计量列表)
        self.stats: List[str] = []                                              # 初始化统计量列表为空
        if "stats" in kwargs:
            self.stats = kwargs["stats"]                                        # 如果提供了统计量参数，使用提供的统计量
            self.prepareDicts()                                                 # 如果提供了统计量参数，使用提供的统计量

    def setStats(self, stats):
        """Set the statistics to use for the constraints

        Parameters
        ----------
        stats : List[str]
            A list of statistics for which constraints will be provided
        """
        self.stats = stats
        self.prepareDicts()

    def addConstraint(
        self, stat: str, constraint: Dict[str, List], insideOut: List[str] = []
    ) -> None:
        """Add a constraint for all levels and evaluation frequencies
        为所有分频级别和评估频率添加约束条件
        Parameters
        ----------
        stat : str
            The statistic to be constrained                                                                                 要添加约束的统计量名称
        constraint : Dict
            The constraint parameter specified as {"statistic component": [valueMin, valueMax]}                             约束参数，格式为{"统计量分量": [最小值, 最大值]}
        insideOut : List[str], optional with default as range specified is range to include
            The constraints for which the range specified in the constraint is the range to exclude rather than include     内外标记列表，指定哪些约束的范围是排除而非包含，默认为空表示包含
        """
        for declevel in range(0, self.numLevels):                                                           # 遍历所有分频级别
            self.addConstraintLevel(stat, constraint, declevel, insideOut)                                  # 为当前分频级别添加约束条件

    def addConstraintLevel(
        self,
        stat: str,
        constraint: Dict[str, List],
        declevel: int,
        insideOut: List[str] = [],                                                                          # 内外标记列表，默认为空
    ) -> None:
        """Add a constraint for a whole decimation level
        为整个分频级别添加约束条件
        Parameters
        ----------
        stat : str
            The statistic to be constrained
        constraint : Dict
            The constraint parameter specified as {"statistic component": [valueMin, valueMax]}
        declevel : int
            The decimation level
        insideOut : List[str], optional with default as range specified is range to include
            The constraints for which the range specified in the constraint is the range to exclude rather than include
        """
        for eIdx, _ in enumerate(self.evalFreq[declevel]):                                                  # 遍历指定分频级别的所有评估频率
            self.addConstraintFreq(stat, constraint, declevel, eIdx, insideOut)                             # 为当前评估频率添加约束条件

    def addConstraintFreq(
        self,
        stat: str,
        constraint: Dict[str, List],
        declevel: int,
        eIdx: int,
        insideOut: List[str] = [],                                                                          # 内外标记列表，默认为空
    ) -> None:
        """Add a constraint for an evaluation frequency
        为单个评估频率添加约束条件
        Parameters
        ----------
        stat : str
            The statistic to be constrained                                                                                 要添加约束的统计量名称
        constraint : Dict
            The constraint parameter specified as {"statistic component": [valueMin, valueMax]}                             约束参数，格式为{"统计量分量": [最小值, 最大值]}
        declevel : int
            The decimation level                                                                                            分频级别索引
        eIdx : int
            Evaluation Frequency index                                                                                      评估频率索引
        insideOut : List[str], optional with default as range specified is range to include
            The constraints for which the range specified in the constraint is the range to exclude rather than include     内外标记列表，指定哪些约束的范围是排除而非包含，默认为空表示包含
        """
        eFreq = self.evalFreq[declevel][eIdx]                                                               # 从分频级别和评估频率索引获取具体的频率值
        for key in constraint:                                                                              # 遍历约束字典中的所有分量键(如cohExHy,cohEyHx)
            self.constraints[eFreq][stat][key] = constraint[key]                                            # 为当前分量设置约束条件(如[0.7, 1.0])
            if key in insideOut:                                                                            # 判断当前分量是否在insideOut列表中
                self.insideOut[eFreq][stat][key] = True                                                     # 如果分量在insideOut中(True)，则排除阈值范围内(如[0.7, 1.0])的值
            else:
                self.insideOut[eFreq][stat][key] = False                                                    # 如果分量不在insideOut中(False)，则只包含阈值范围内(如[0.7, 1.0])的值

    def getConstraintFreq(self, declevel: int, eIdx: int, stat: str = ""):
        """Get constraints for an evaluation frequency

        Parameters
        ----------
        declevel : int
            The decimation level
        eIdx : int
            Evaluation Frequency index
        stat : str
            The statistic for which to return the constraints

        Returns
        -------
        out : list
            Constraints for the evaluation frequency
        """
        eFreq = self.evalFreq[declevel][eIdx]
        if stat != "":
            return self.constraints[eFreq][stat]
        return self.constraints[eFreq]

    def getMaskWindowsFreq(self, declevel: int, eIdx: int) -> Set[int]:
        """Get a set of the masked windows (used by WindowSelector)

        Parameters
        ----------
        declevel : int
            The decimation level
        eIdx : int
            Evaluation Frequency index

        Returns
        -------
        out : Set[int]
            A set with the indices of the masked windows
        """
        eFreq = self.evalFreq[declevel][eIdx]
        return self.maskWindows[eFreq]

    def prepareDicts(self) -> None:
        """Prepare dictionaries for storing masking information
        准备用于存储掩码信息的字典结构
        """
        for declevel in range(0, self.numLevels):                                       # 遍历所有分频级别
            for eFreq in self.evalFreq[declevel]:                                       # 遍历当前分频级别的所有评估频率
                # empty set for maskWindows to start off with
                # i.e. remove none
                self.maskWindows[eFreq] = set()                                         # 开始时为空集合，即不排除任何窗口
                self.constraints[eFreq] = {}                                            # 初始化约束条件字典
                self.insideOut[eFreq] = {}                                              # 初始化约束条件字典
                for stat in self.stats:                                                 # 为每个统计量初始化约束和内外标记
                    self.constraints[eFreq][stat] = {}                                  # 初始化该统计量的约束字典
                    self.insideOut[eFreq][stat] = {}                                    # 初始化该统计量的内外标记

    def resetMaskWindows(self) -> None:
        """Reset/Clear the masked windows"""
        for declevel in range(0, self.numLevels):
            for eFreq in self.evalFreq[declevel]:
                # empty set for maskWindows to start off with
                # i.e. remove none
                self.maskWindows[eFreq] = set()

    def view(self, declevel: int = 0, **kwargs) -> Figure:
        """Produces a 2-D plot with all the masked windows along the bottom for a decimation level
        为指定分频级别生成二维图，显示所有被掩码的窗口
        Parameters
        ----------
        declevel : int, optional
            The decimation level. Default is 0.
        fig : matplotlib.pyplot.figure, optional
            A figure object
        plotfonts : Dict, optional
            A dictionary of plot fonts            
        """
        from resistics.common.plot import getViewFonts

        fig: plt.figure = (                                                                             # 创建或复用图形对象
            plt.figure(kwargs["fig"].number)
            if "fig" in kwargs
            else plt.figure(figsize=(10, 10))
        )
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()                    # 设置绘图字体，使用自定义字体或默认字体

        evalFreq = self.evalFreq[declevel]                                                              # 获取指定分频级别的评估频率列表
        numFreq = len(evalFreq)                                                                         # 获取评估频率的数量
        freqIndex = range(0, numFreq)                                                                   # 创建评估频率索引范围range(0, 4)
        levelMasked: Set = set()                                                                        # 初始化一个空集合，用于收集该级别所有被掩码的窗口(masked windows)（全局索引）
        numMasked: List[int] = list()                                                                   # 初始化空列表，存储每个频率被掩码的窗口数量
        freqLabel: List[str] = list()                                                                   # 初始化空列表，存储频率标签（格式化后的频率字符串）
        for eFreq in evalFreq:                                                                          # 遍历当前级别的所有评估频率
            levelMasked.update(self.maskWindows[eFreq])                                                 # 将该频率的所有掩码窗口添加到集合中（自动去重）
            numMasked.append(len(self.maskWindows[eFreq]))                                              # 记录该频率的掩码窗口数量(不在[0.7,1.0]内的窗口)
            freqLabel.append("{:.7f}".format(eFreq))                                                    # 将频率值格式化为字符串，保留7位小数

        numMaskedTotal = len(levelMasked)                                                               # 计算该级别总共被掩码的窗口数量(不在[0.7,1.0]内的窗口)（去重后）
        levelMasked = np.array(sorted(list(levelMasked)))                                               # 将集合转换为排序后的NumPy数组
        global2localMap = {}                                                                            # 创建空字典，用于映射全局索引到局部索引
        for idx in range(0, levelMasked.size):                                                          # 遍历排序后的掩码窗口(masked windows)数组
            global2localMap[levelMasked[idx]] = idx                                                     # 建立映射：全局索引(1197,1198,1199,...) -> 局部索引（0,1,2,...）
        # make array
        data = np.zeros(shape=(levelMasked.size, numFreq), dtype=int)                                   # 创建二维零矩阵，形状为（掩码窗口数3668 × 频率数4）
        for idx, eFreq in enumerate(evalFreq):                                                          # 遍历所有评估频率及其索引[4.6875,4.0625,3.4375,2.8125]
            for gI in self.maskWindows[eFreq]:                                                          # 遍历当前频率的所有掩码窗口全局索引
                lI = global2localMap[gI]                                                                # 将全局索引转换为局部索引
                data[lI, idx] = idx + 1                                                                 # 在对应位置标记频率索引（避免0值，0在颜色映射中通常表示白色（未掩码(在[0.7,1.0]内的窗口)的频率的窗口））[4.6875：1,4.0625：2,3.4375：3,2.8125：4],data里有值的表示掩码窗口masked windows

        st = fig.suptitle(
            "Masked windows for decimation level: {}".format(declevel),                                 # 设置图形总标题
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)                                                                                  # 调整标题垂直位置，避免与图形边框重叠

        # now plot
        ax = plt.subplot(2, 1, 1)                                                                 # 创建第一个子图（2行1列的第1个位置）
        cmap = cm.get_cmap("Paired", numFreq)                                                     # 获取"Paired"颜色映射，颜色数量等于频率数
        cmap.set_under("white")                                                                         # 设置小于vmin值(1)的颜色为白色
        plt.pcolor(np.transpose(data), cmap=cmap, vmin=1, vmax=numFreq + 1)                       # 绘制伪彩色图（热图），转置数据使频率在y轴(vmin=1, vmax=numFreq+1定义颜色映射范围)
        ax.set_yticks(np.array(freqIndex) + 0.5)                                                        # 设置Y轴刻度位置（在单元格中心）
        ax.set_yticklabels(freqLabel)                                                                   # 设置Y轴刻度标签为频率值
        plt.ylabel(
            "Masked windows for decimation level", fontsize=plotfonts["axisLabel"]                # 设置标题
        )
        plt.ylabel("Evaluation Frequency [Hz]", fontsize=plotfonts["axisLabel"])                  # 设置Y轴标签
        plt.title(
            "Relative masked windows for decimation level {}".format(declevel),                         # 设置子图标题
            fontsize=plotfonts["title"],
        )
        # set tick sizes(设置坐标轴刻度标签字体大小)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])
        # colourbar
        cb = plt.colorbar()                                                                             # 添加颜色条
        cb.ax.set_title("Masked\nwindows")                                                              # 设置颜色条标题
        cb.set_ticks(np.arange(1, numFreq + 1) + 0.5)                                                   # 设置颜色条刻度位置
        cb.set_ticklabels(freqLabel)                                                                    # 设置颜色条刻度标签

        # bar graph(柱状图)
        ax = plt.subplot(2, 1, 2)                                                                 # 创建第二个子图（2行1列的第2个位置）
        # add the total number masked(累加掩码化数据的总数)
        freqIndex = range(0, numFreq + 1)                                                               # 扩展频率索引范围，包含总计项
        numMasked.append(numMaskedTotal)                                                                # 在掩码数量列表中添加总计数量
        freqLabel.append("Total masked for\n decimation level")                                         # 在频率标签列表中添加总计标签
        plt.bar(freqIndex, numMasked)                                                                   # 绘制柱状图
        ax.set_xticks(freqIndex)                                                                        # 设置X轴刻度位置
        ax.set_xticklabels(freqLabel)                                                                   # 设置X轴刻度标签
        plt.xlabel("Evaluation Frequency [Hz]", fontsize=plotfonts["axisLabel"])                  # 设置X轴标签
        plt.ylabel("Number of masked windows", fontsize=plotfonts["axisLabel"])                   # 设置Y轴标签
        plt.title(
            "Number of masked windows per evaluation frequency",                                  # 设置子图标题
            fontsize=plotfonts["title"],
        )
        # set tick sizes(设置坐标轴刻度标签字体大小)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])                                                 # 调整子图布局，为标题留出空间
        # plt.show()

        return fig

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst: List[str] = []
        textLst.append("Mask name = {}".format(self.maskName))
        textLst.append("Sampling frequency = {:.6f}".format(self.sampleFreq))
        textLst.append(
            "Statistics to use for constraints = {}".format(", ".join(self.stats))
        )
        textLst.append("Evaluation frequencies [Hz]")
        for il in range(0, self.numLevels):
            freqForLevel = self.evalFreq[il]
            for eFreq in freqForLevel:
                textLst.append(
                    "Decimation level = {:d}, Frequency = {:.6f} [Hz], number of masked windows = {:d}".format(
                        il, eFreq, len(self.maskWindows[eFreq])
                    )
                )
        return textLst

    def printConstraints(self) -> None:
        """Print information about the constraints"""
        blockPrint(
            "{} Constraints".format(self.__class__.__name__),
            self.printConstraintsList(),
        )

    def printConstraintsList(self) -> List[str]:
        """Constraint information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        evalFreq = sorted(list(self.constraints.keys()))
        for eFreq in evalFreq:
            textLst.append("Frequency = {:.6f} [Hz]".format(eFreq))
            for stat in self.stats:
                textLst.append("\tStatistic = {}".format(stat))
                # check to see if any constraints
                if len(list(self.constraints[eFreq][stat].keys())) == 0:
                    textLst.append("\tNo constraints for this statistic")
                    continue
                # if there are, print out
                for component in self.constraints[eFreq][stat]:
                    minVal = self.constraints[eFreq][stat][component][0]
                    maxVal = self.constraints[eFreq][stat][component][1]
                    textLst.append(
                        "\t{}\t{}\t{}\t{}".format(
                            component,
                            minVal,
                            maxVal,
                            self.insideOut[eFreq][stat][component],
                        )
                    )
        return textLst
