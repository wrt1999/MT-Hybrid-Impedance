import numpy as np
from typing import List, Union, Tuple

from resistics.common.base import ResisticsBase
from resistics.config.io import loadConfig
from resistics.decimate.evalfreqs import getEvaluationFreqSize
from resistics.common.print import (
    generalPrint,
    warningPrint,
    errorPrint,
    blockPrint,
    arrayToString,
)


class DecimationParameters(ResisticsBase):
    """Class to calculate and hold decimation parameters

    Attributes
    ----------
    sampleFreq : float
        Sampling frequency in Hz.
    divFactor : int
        Minimum division factor when working out decimation factors
    numLevels : int
        Number of decimation levels
    freqPerLevel : int
        Number of frequencies per decimation level
    decFactors : np.ndarray
        Incremental decimation factors
    decFrequencies : np.ndarray
        Sampling frequencies at each decimation level
    evalFreq : np.ndarray 
        Array of evaluation frequencies
    evalFreqPerLevel : np.ndarray
        2-D array (numLevels,freqPerLevel) of evaluation frequencies

    Methods
    -------
    __init__(sampleFreq)
        Initialise with sampling frequency
    getSampleFreqLevel(decLevel)
        Get the sample frequency of the level
    getDecFactor(decLevel)
        Get the decimation factor from level 0 to decLevel
    getIncrementalFactor(decLevel)
        Get the decimation factor from decLevel-1 to decLevel
    getEvalFrequenciesForLevel(decLevel)
        Returns evaluation frequencies for decLevel
    setFrequencyParams(evalFreq, freqPerLevel, maxLevel)
        Set decimation parameters by evaluation frequencies and frequencies per level
    setDecimationParams(numLevels, freqPerLevel)
        Set decimation parameters by number of levels and frequencies per level
    calcFrequencyParams(numLevels, freqPerLevel)
        Calculate decimation parameters based on number of levels and frequencies per level
    calcDecimationParams(evalFreq, maxLevel, freqPerLevel)
        Calculate decimation paratmers based on evaluation frequencies, frequecies per level and max level
    calcNearestFactor(freq)
        Calculate nearest decimation factor
    printList()
        Class status returned as list of strings
    """

    def __init__(self, sampleFreq: float):
        """Initialise decimation parameters with sampling frequency
        基于采样频率初始化抽取参数
        Calculates decimation factors and evaluation frequencies based on defaults
        基于默认值计算抽取因子与评估频率
        Parameters
        ----------
        sampleFreq : float
            Sampling frequency
        """
        self.sampleFreq = sampleFreq
        config = loadConfig()
        self.divFactor: int = 2
        self.numLevels: int = config["Decimation"]["numlevels"]
        self.freqPerLevel: int = config["Frequencies"]["perlevel"]
        self.decFactors: Union[np.ndarray, None] = None
        self.decFrequencies: Union[np.ndarray, None] = None
        self.evalFreq: Union[np.ndarray, None] = None
        self.evalFreqPerLevel: Union[np.ndarray, None] = None

        # calculate some initial values decimation parameters based on defaults(基于默认值defaults计算抽取参数的一些初始值)
        self.calcFrequencyParams(self.numLevels, self.freqPerLevel)             # 计算评估频率和降采样参数

    def getSampleFreqLevel(self, decLevel: int) -> float:
        """Get sampling frequency for decimation level

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : float
            Sample frequency for decimation level decLevel
        """
        self.checkDecimationLevel(decLevel)
        return self.sampleFreq / self.getDecFactor(decLevel)

    def getDecFactor(self, decLevel: int) -> float:
        """Get decimation factor for decimation level

        Returns decimation factor relative to level 0

        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : float
            Decimation factor for level decLevel
        """
        self.checkDecimationLevel(decLevel)
        return self.decFactors[decLevel]

    def getIncrementalFactor(self, decLevel: int) -> int:
        """Get decimation factor for decimation level
        获取降采样级别对应的降采样因子
        Returns decimation factor relative to decLevel - 1
        返回相对于 decLevel - 1 的降采样因子
        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : int
            Incremental decimation factor from decLevel - 1
        """
        self.checkDecimationLevel(decLevel)             # 检查用户提供的降采样级别，如果超出范围则退出。(一般没问题)
        if decLevel == 0:
            return int(self.decFactors[decLevel])
        else:
            return int(self.decFactors[decLevel] / self.decFactors[decLevel - 1])

    def getEvalFrequenciesForLevel(self, decLevel: int) -> np.ndarray:
        """Get evaluation frequencies for level decLevel
        获取级别 decLevel 的评估频率
        Parameters
        ----------
        decLevel : int
            Decimation level

        Returns
        -------
        out : np.ndarray
            Array of evaluation frequencies for level decLevel
        """
        self.checkDecimationLevel(decLevel)
        return self.evalFreqPerLevel[decLevel, :]

    def checkDecimationLevel(self, decLevel: int) -> None:
        """Checks user provided decimation levels and quits if out of bounds
        检查用户提供的降采样级别，如果超出范围则退出。
        Parameters
        ----------
        decLevel : int
            Decimation level
        """
        if decLevel < 0:
            self.printError("Decimation level must be greater than 0", quitRun=True)
        if decLevel > self.numLevels - 1:
            self.printError(
                "Decimation level must be less than or equal to {}".format(
                    self.numLevels - 1
                ),
                quitRun=True,
            )

    def setFrequencyParams(
        self, evalFreq: List, freqPerLevel: int, maxLevel: int
    ) -> None:
        """Sets frequency parameters and calculates decimation paramters based on those

        Parameters
        ----------
        evalFreq : List
            Evaluation frequencies
        freqPerLevel : int
            Number of evaluation frequencies per level
        maxLevel : int
            Maximum decimation level
        """
        evalFreq[::-1].sort()  # sort in descending order
        self.calcDecimationParams(evalFreq, freqPerLevel, maxLevel)

    def setDecimationParams(self, numLevels: int, freqPerLevel: int) -> None:
        """Sets number of levels and frequencies per levels to calculate decimation parameters
        设置要计算抽取参数的级数和每级的频率数
        Parameters
        ----------
        numLevels : int
            Number of decimation levels
        freqPerLevel : int
            Number of evaluation frequencies per level
        """
        self.calcFrequencyParams(numLevels, freqPerLevel)

    def calcFrequencyParams(self, numLevels: int, freqPerLevel: int) -> None:
        """Calculates evaluation frequencies and decimation parameters
        计算评估频率和降采样参数
        Uses number of levels and frequencies per levels to auto-calculate evaluation frequencies and then the decimation parameters.
        根据层数和每层的频率自动计算评估频率，进而计算降采样参数
        Parameters
        ----------
        numLevels : int
            Number of decimation levels降采样层数
        freqPerLevel : int
            Number of evaluation frequencies per level每层的评估频率数量
        """
        numFreq = numLevels * freqPerLevel
        evalFreq = getEvaluationFreqSize(self.sampleFreq, numFreq)          # 生成evalFreq数组
        self.calcDecimationParams(evalFreq, numLevels, freqPerLevel)        # evalFreq评估频率列表，numLevels每个抽取级的频率数量，freqPerLevel最大允许抽取级数

    def calcDecimationParams(
        self, evalFreq: Union[np.ndarray, List], maxLevel: int, freqPerLevel: int
    ) -> None:
        """Calculate decimation parameters from evaluation frequencies
        基于评估频率计算降采样参数
        Uses evaluation frequencies, number of frequencies per decimation level and max allowable decimation level to calculate decimation parameters.
        使用评估频率、每个抽取级的频率数量以及最大允许抽取级来计算抽取参数
        Parameters
        ----------
        evalFreq : List, np.ndarray
            Number of decimation levels
        maxLevel : int
            Maximum allowable number of decimation levels
        freqPerLevel :
            Number of frequencies per level
        """
        evalFreq = np.array(evalFreq)                   # 将输入频率转换为numpy数组，便于数值计算
        # calculating decimation parameters from evaluation frequencies(基于评估频率计算抽取参数)
        maxf = self.sampleFreq / 3.2                    # 计算最大可分析频率,保守一点用1/4避免混叠
        # find the first evaluation frequency less than or equal to maxf(找到第一个小于或等于最大频率 maxf 的评估频率)
        fHigh = evalFreq[0]                             # fHigh初始化,在评估频率数组中找到第一个小于等于maxf的频率,作为实际处理的最高频率（避免混叠）,初始设为第一个频率
        for ifreq in range(0, evalFreq.size):           # 遍历所有评估频率，找到第一个满足条件的
            if evalFreq[ifreq] <= maxf:                 # 找到合适的最高频率
                fHigh = evalFreq[ifreq]                 # 赋值给fHigh
                break                                   # 找到即停止
        iHigh = evalFreq.tolist().index(fHigh)              # 找到最高频率在原始数组evalFreq中的索引位置
        evalFreqSub = evalFreq[iHigh:]                      # 提取从最高频率到最低频率的子数组（有效的分析频率范围）evalFreqSub。这个子数组包含了所有可以被安全、有效处理的频率点
        # calculate number of levels(计算抽取级别数量)
        numLevels = maxLevel                                                    # 初始化numLevels，初始设为最大允许级别数
        # check if enough evaluation frequencies(检查评估频率是否足够)
        if len(evalFreqSub) < freqPerLevel * maxLevel:                          # 检查可用的评估频率数量是否足够支持要求的级别数(没用上)
            # numLevels = int(math.floor(len(evalFreqSub)/freqPerLevel))
            numLevels = int(np.ceil(1.0 * len(evalFreqSub) / freqPerLevel))     # 如果频率点不足，重新计算实际可支持的级别数.使用向上取整，确保充分利用所有可用频率点.

        # do another subslice(进行另一次子切片操作，组织频率点到各层级)
        evalFreqSub = evalFreqSub[: numLevels * freqPerLevel]       # 根据实际级别数进一步修剪频率子集：确保 evalFreqSub的长度是 numLevels * freqPerLevel的整数倍，方便均匀分配

        # now create an array of evalation frequencies per decimation level(现在针对每个抽取级别创建一个评估频率数组)
        # evalFreqPerLevel = np.ones(shape=(numLevels, freqPerLevel))
        evalFreqPerLevel = np.ones(shape=(numLevels, freqPerLevel)) * -1      # 创建二维数组存储每个级别的频率点，初始值设为-1（无效值标记）
        for ilevel in range(0, numLevels):                                      # 遍历每个抽取级别
            for ifreq in range(0, freqPerLevel):                                # 遍历当前级别的每个频率位置
                if ilevel * freqPerLevel + ifreq >= len(evalFreqSub):           # 检查是否超出可用频率点范围
                    break                                                       # 超出范围则终止内层循环
                evalFreqPerLevel[ilevel, ifreq] = evalFreqSub[                  # 将一维频率数组evalFreqSub按顺序填充到二维结构evalFreqPerLevel中
                    ilevel * freqPerLevel + ifreq
                ]
        # 现在为每个级别计算抽取参数，调用辅助函数计算最近的合适抽取因子
        # now calculate decimation factors(现在计算降采样因子),创建一个长度为 numLevels的一维数组，并用1.0填充所有元素
        decFactors = np.ones(shape=(numLevels))         # 初始化抽取因子（降采样倍数）,decFactors：降采样因子数组。将原始采样率(15Hz)降采样的倍数
        decFrequencies = np.ones(shape=(numLevels))     # 初始化各级别的采样频率，decFrequencies：新采样频率数组。每个元素将存储对应层级的新采样频率
        for ilevel in range(0, numLevels):              # 为每个级别计算抽取参数，调用辅助函数计算最近的合适抽取因子
            decFactors[ilevel], decFrequencies[ilevel] = self.calcNearestFactor(        # decFactors[ilevel]：该层级的降采样因子D（整数）。例如，D=4表示该层数据采样率降为原始的1/4
                evalFreqPerLevel[0][0]             # 使用当前级别的最高频率作为参考      # decFrequencies[ilevel]：降采样后数据对应的新采样率
            )

        # finally, set all parameters(将计算出的所有参数存储到对象属性中)
        self.evalFreq = evalFreqSub                     # 最终使用的评估频率子集，一维数组
        self.freqPerLevel = freqPerLevel                # 每级的频率点数
        self.numLevels = numLevels                      # 实际级别数
        self.evalFreqPerLevel = evalFreqPerLevel        # 分级后的评估频率，二维数组
        self.decFactors = decFactors                    # 每层降采样因子数组
        self.decFrequencies = decFrequencies            # 每层的新采样率

    def calcNearestFactor(self, freq: float):
        """Calculates nearest decimation factor given frequency
        根据频率计算最近的降采样因子
        Parameters
        ----------
        freq : float
            Frequency for which to calculate decimation factor

        Returns
        -------
        fac : int
            The decimation factor
        f : float
            Sampling frequency in Hz given the decimation factor (sampleFreq/decimationFactor)
        """
        # want sampling frequency to be 4 times greater than highest freq(希望采样频率是最高频率的 3.2 倍)
        fsMin = freq * 3.2                           # 根据奈奎斯特采样定理，想要分析的频率为freq，那么采样频率至少需要2*freq，这里取3.2倍提供安全裕度
        # set to initial sampling frequency(设置为初始采样频率)
        f = float(self.sampleFreq)                  # 初始化f为原始采样率
        fac = 1                                     # 初始化降采样因子为1（不降采样）
        # 循环：如果当前采样率 f 仍然大于最低安全采样率 fsMin的 self.divFactor倍，就继续降采样
        while f > fsMin * self.divFactor:           # 两边同时除以self.divFactor等价于f / self.divFactor > fsMin，确保了执行一次降采样后仍然满足安全要求，循环不使用>=是因为留足充分的安全余量
            f = f / self.divFactor                  # f：降采样后的新采样频率（单位：Hz）。self.divFactor：降采样基数（通常是 2或 4）。一个常数，决定了降采样的步长
            fac = fac * self.divFactor              # fac：降采样因子（整数）。推荐的降采样倍数
        return fac, f                               # 循环结束后，fac就是能找到的最大降采样因子

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sampling frequency = {:f}".format(self.sampleFreq))
        textLst.append("Number of decimation levels = {:d}".format(self.numLevels))
        # decimation factors
        for il in range(0, self.numLevels):
            textLst.append(
                "Level = {:d}\tsample freq. [Hz] = {:.6f}\tsample rate [s] = {:.6f}\tdec. factor = {:07d}\tinc. factor = {:d}".format(
                    il,
                    self.decFrequencies[il],
                    1.0 / self.decFrequencies[il],
                    int(self.decFactors[il]),
                    self.getIncrementalFactor(il),
                )
            )
        # evaluation frequencies
        textLst.append("Evaluation frequencies [Hz]")
        for il in range(0, self.numLevels):
            freqForLevel = self.getEvalFrequenciesForLevel(il)
            eFreqStr = arrayToString(freqForLevel)
            textLst.append("Level = {:d}: {}".format(il, eFreqStr))
        return textLst
