import os
from datetime import date, time, datetime, timedelta
from typing import List

from resistics.common.base import ResisticsBase
from resistics.common.print import blockPrint, generalPrint, warningPrint
from resistics.project.data import ProjectData
from resistics.statistics.data import StatisticData
from resistics.statistics.io import StatisticIO
from resistics.mask.data import MaskData


class MaskCalculator(ResisticsBase):
    """Calculate window masking given constraints

    Calculate masks for time windows. The window masker reads in statistics and writes out a file with windows which match the given constraints for each statistic. The masked file will then go into the window WindowSelector, which passes on shared windows to the processor.    

    Attributes
    ----------
    proj : Project
        A Project object
    maskData : MaskData
        A MaskData object
    specDir : str
        The spectra directory

    Methods
    -------
    __init__(proj, maskData)
        Initialise with a Project instance and MaskData instance
    clearMaskWindows()
        Clear calculated masks
    applyConstraints(site)
        Apply masking constraints
    printList()
        Class status returned as list of strings
    """

    def __init__(self, projData: ProjectData, maskData: MaskData, **kwargs) -> None:
        """Initialise with project and maskData
    
        Parameters
        ----------
        proj : ProjectData
            The project to which masking should be applied
        maskData : MaskData
            A MaskData instance 
        """
        self.projData = projData
        # stats is the stat to use
        self.maskData = maskData
        self.sampleFreq = maskData.sampleFreq
        self.specdir: str = "spectra"
        if "specdir" in kwargs:
            self.specdir = kwargs["specdir"]

    def clearMaskWindows(self) -> None:
        """Clear mask windows in maskData object """
        self.maskData.maskWindows = {}
        self.maskData.resetMaskWindows()

    def applyConstraints(self, site: str) -> None:
        """Apply the masking constraints to find windows to be excluded
        应用掩码约束条件来找到需要排除的窗口
        WindowsSelector will later exclude the masked windows before processing
        窗口选择器将在后续处理中排除被掩码的窗口
        Parameters
        ----------
        site : str
            The site in the project to mask
        """
        # run through all the statfiles listed(遍历所列出的全部统计文件)
        siteData = self.projData.getSiteData(site)                                                  # 获取站点数据对象
        measurements = siteData.getMeasurements(self.sampleFreq)                                    # 获取该采样频率下的所有测量文件夹
        evalFreqs = self.maskData.evalFreq                                                          # 获取所有评估频率
        # create a single statisticIO instance to do the reading(创建一个statisticIO单实例以执行读取操作)
        statIO = StatisticIO()                                                                      # 创建单个统计量IO实例用于读取数据
        # loop over stats
        for stat in self.maskData.stats:                                                            # 遍历所有统计量
            # loop over decimation levels
            for iDec in range(0, self.maskData.numLevels):                                          # 遍历分频级别
                # loop over measurement folders
                for meas in measurements:                                                           # 遍历测量文件夹
                    # try and open the file
                    statIO.setDatapath(                                                             # 尝试打开统计量数据文件,设置统计量IO对象的数据路径
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), self.specdir
                        )
                    )
                    statData = statIO.read(stat, iDec)                                              # 读入文件,读取指定统计量和分频级别的数据
                    if statData is None:
                        # to next measurement(转到下一个测量文件)
                        self.printWarning(
                            "No statistic data found for {}, measurement {} and decimation level {}".format(            # 如果没有找到数据，打印警告信息,并跳转到下一个测量文件夹
                                site, meas, iDec
                            )
                        )
                        continue
                    numWindows = statData.numWindows                                                # 获取统计量数据中的窗口总数4243
                    winStats = statData.winStats                                                    # 获取统计量分量的名称列表['cohExHx','cohExHy','cohEyHx','cohEyHy']
                    for iW in range(0, numWindows):                                                 # 遍历所有时间窗口
                        # now check each window(现在检查每个窗口的统计量值)
                        winVals = statData.getStatLocal(iW)                                         # 获取当前窗口的局部统计量值winVals（二维数组4×4）
                        # now loop over evaluation frequencies
                        for eIdx, eFreq in enumerate(evalFreqs[iDec]):                              # 现在遍历当前分频级别的所有评估频率
                            constraintCheck = True                                                  # 包含逻辑（使用与运算）
                            constraintCheckInsideOut = False                                        # 排除逻辑（使用或运算）
                            insideOutFlag = False                                                   # 标记是否使用了排除逻辑
                            freqVal = winVals[eIdx]                                                 # 获取当前评估频率下的所有统计量分量值[cohExHx,cohExHy,cohEyHx,cohEyHy]
                            for component in self.maskData.constraints[eFreq][stat]:                # 遍历当前频率和统计量的所有约束分量cohExHy
                                index = winStats.index(component)                                               # 查找当前分量在统计量分量列表中的索引位置
                                componentVal = freqVal[index]                                                   # 获取当前分量的具体数值cohExHy
                                lowerLimit = self.maskData.constraints[eFreq][stat][component][0]               # 从约束数据中获取当前分量的下限阈值0.7
                                upperLimit = self.maskData.constraints[eFreq][stat][component][1]               # 从约束数据中获取当前分量的上限阈值1.0
                                test = (componentVal > lowerLimit) and (componentVal < upperLimit)              # 布尔类型,检查分量值是否在约束范围内（大于下限且小于上限）

                                if self.maskData.insideOut[eFreq][stat][component]:                             # 检查当前分量是否使用排除逻辑（InsideOut为True）
                                    test = not test
                                    constraintCheckInsideOut = constraintCheckInsideOut or test                 # 使用或运算累积排除逻辑的结果（任一条件满足即排除）
                                    insideOutFlag = True
                                else:
                                    constraintCheck = constraintCheck and test                                  # 如果是包含逻辑（默认），使用与运算累积结果（所有条件必须满足）
        
                            if insideOutFlag:                                                                   # 检查是否存在排除逻辑
                                constraintCheck = constraintCheck and constraintCheckInsideOut                  # 如果存在排除逻辑，将包含逻辑和排除逻辑的结果进行与运算
                            
                            if not constraintCheck:                                                             # 检查约束检查是否失败（constraintCheck为False）
                                # if the stat fails, add it - this is a list of windows to exclude(如果统计量检查失败，将该窗口添加到排除列表中)
                                self.maskData.maskWindows[eFreq].add(                                           # 使用集合的add方法添加窗口的全局索引（自动去重）
                                    statData.getGlobalIndex(iW)                                                 # 获取当前窗口的全局唯一索引
                                )

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sample Frequency [Hz] = {}".format(self.sampleFreq))
        textLst.append("Spectra directory = {}".format(self.specdir))
        textLst.append("Mask Data information")
        textLst = textLst + self.maskData.printList()
        return textLst
