import os
from datetime import date, time, datetime, timedelta
from typing import List, Dict, Set, Union, Any, Tuple

from openpyxl.styles.builtins import total
# import from package
from resistics.common.base import ResisticsBase
from resistics.common.print import list2ranges, blockPrint
from resistics.decimate.parameters import DecimationParameters
from resistics.window.parameters import WindowParameters
from resistics.window.utils import datetime2gIndex, gIndex2datetime
from resistics.project.data import ProjectData
from resistics.site.data import SiteData
from resistics.mask.data import MaskData
from resistics.spectra.io import SpectrumReader
from resistics.mask.io import MaskIO


class WindowSelector(ResisticsBase):
    """Select windows for further processing
    选择窗口以进行后续处理
    Finds windows for further processing. Given more than one site, WindowSelector will find the shared windows. For example, when processing with a remote reference, WindowSelector will find the shared global windows (i.e. referenced to the project reference time) of the local site and the remote site.
    定位用于后续处理的窗口。当涉及多个站点时，WindowSelector 会识别共享窗口。例如，在采用远程参考进行处理时，WindowSelector 将定位本地站点与远程站点的共享全局窗口（即相对于项目参考时间的窗口）
    Shared windows will accept maskData when constraints are to be used. Further, datetime constraints can be added to the selection process, in the case where, for example, only night time data is to be processed.
    当启用约束条件时，共享窗口将接收掩码数据（maskData）。此外，若需限定特定时间范围（例如仅处理夜间数据），可在选择过程中添加日期时间约束。
    For datetime constraints, the order of priorities are: 针对日期时间约束，优先级顺序如下
    1. datetime constraints 日期时间约束
    2. date constraints 日期约束
    3. time constraints 时间约束


    Attributes
    ----------
    projData : ProjectData
        A ProjectData instance
    sampleFreq : float
        Sampling frequency of the data
    decParams : DecimationParameters
        A DecimationParameters instance detailing the decimation schemes
    winParams : WindowParameters
        A WindowParameters instance detailing the windowing information 
    sites : List[str]
        List of sites for which to calculate shared windows
    sharedWindows : Dict
        Dictionary to store shared window information between sites. The keys of the dictionary are the decimation levels.
    siteMasks : Dict
        Masks to use for each site
    siteSpecFolders : Dict
        Spectra folders for each site at sampleFreq
    siteSpecReaders : Dict
        SpectraReaders for each site at sampleFreq
    siteSpecRanges : Dict
        Global window ranges for all the spectra files for the sites at sampleFreq
    siteGlobalIndices : Dict
        Global window sets for all the spectra files for the sites at sampleFreq
    specdir : str
        The spectra data to use for calculating shared windows and the subsequent processing
    prepend : str
        The string prepending the spectra data files. This is usually "spectra" and not expected to change
    datetimes : Dict
        User supplied datetime constraints. These are the highest priority in the time constraints. This is a list of constraints for each decimation level.
    dates : Dict
        User supplied date constraints. These are the second highest priority in the time constraints. This is a list of constraints for each decimation level.
    times : Dict
        User supplied time constraints. These are the last priority in the time constraints. This is a list of constraints for each decimation level.
    datetimeConstraints : Dict
        Final combined date and time constraints. This is a list of constraints for each decimation level

    Methods
    -------
    __init__(decParams)
        Initialise window selector with information about the decimation parameters
    getSharedWindowsLevel(declevel)
        Get the shared windows for a decimation level as a python set
    getNumSharedWindows(declevel)
        Get the number of shared windows for a decimation level
    getWindowsForFreq(declevel, eIdx)
        Get the number of shared windows for a decimation level and evaluation frequency
    getUnmaskedWindowsLevel(declevel)
        Get unmasked windows for a decimation level
    getDatetimeConstraints()
        Get the datetime constraints
    getLevelDatetimeConstraints(declevel)
        Get the datetime constraints for a decimation level
    getMasks()
        Get a dictionary with masks to use for each site in the window selector
    getSpecReaderForWindow(site, declevel, iWin)
        Get the spectrum reader for a window
    getDataSize(declevel)
        Get the spectrum reader for a window
    setSites(sites)
        Set the sites for which to find the shared windows
    addDatetimeConstraint(start, stop)
        Add a datetime constraint
    addDateConstraint(dateC)
        Add a date constraint
    addTimeConstraint(start, stop)
        Add a time constraint. This will recur on every day of recording
    addWindowMask(site, maskName)
        Add a window mask
    resetDatetimeConstraints()
        Clear and reset all datetime constraints
    resetMasks()
        Clear and reset site masks        
    calcSharedWindows()
        Calculate shared windows between sites
    calcGlobalIndices()
        Find all the global indices for the sites
    calcDatetimeConstraints()
        Calculate overall datetime constraints
    calcSiteDates()
        Calculate a list of days that all the sites were operating
    printList()
        Class information as a list of strings
    printSiteInfo()
        Print out information about the sites included in the window selection
    printSiteInfoList(site)
        Return site window information as a list of strings
    printSharedWindows() 
        Print out the shared windows
    printSharedWindowsList()
        Shared window information as a list of strings
    printDatetimeConstraints()
        Print out the datetime constraints
    printDatetimeConstraintsList()
        Datetime constraint information as a list of strings
    printWindowMasks()
        Print information about masks being used in the window selection
    printWindowMasksList()
        Window mask information as a list of strings
    printWindowsForFrequency(listwindows=False):
        Print information about the windows for each evaluation frequency
    printWindowsForFrequencyList(listwindows=False)
        Information about windows for each evaluation frequency as a list of strings
    """

    def __init__(
        self,
        projData: ProjectData,
        sampleFreq: float,
        decParams: DecimationParameters,
        winParams: WindowParameters,
        **kwargs
    ) -> None:
        """Initialise window selector

        Parameters
        ----------
        projData : ProjectData
            A ProjectData instance
        sampleFreq : float
            The sampling frequency of the raw time data
        decParams : DecimationParameters
            A decimation parameters instance detailing decimaion scheme
        winParams : WindowParameters
            A window parameters instance detailing window schemes
        specdir : str, optional
            The spectra directories to use 
        """
        self.projData: ProjectData = projData
        self.sampleFreq: float = sampleFreq
        self.decParams = decParams
        self.winParams = winParams
        self.sites: List = []
        # shared indices(共享索引)
        self.sharedWindows: Dict = {}
        # the masks to use for each site - there can be multiple masks for each site(各站点使用的掩码——每个站点可对应多个掩码)
        self.siteMasks: Dict[str, List[str]] = {}
        # the spec files for each site at fs(每个站点在 fs 中的规格文件)
        self.siteSpecFolders: Dict = {}
        self.siteSpecReaders: Dict = {}
        # global indices (ranges and sets)(全局索引（范围与集合）)
        self.siteSpecRanges: Dict = {}
        self.siteGlobalIndices: Dict = {}
        # spectra directory information(频谱目录信息)
        self.specdir = kwargs["specdir"] if "specdir" in kwargs else "spectra"      # self.specdir = kwargs["specdir"] 如果"specdir" in kwargs，否则self.specdir = "spectra"
        self.prepend: str = "spectra"
        # time constraints: priority is datetimes > dates > times(时间限制：优先级为日期时间 > 日期 > 时间)
        self.datetimes: Dict[int, List] = {}
        self.dates: Dict[int, List] = {}
        self.times: Dict[int, List] = {}
        # dictionary for datetime constraints(日期时间约束字典)
        self.datetimeConstraints: Dict[int, List] = {}
        # set all datetime constraints to empty（将所有日期时间约束设置为空）
        self.resetDatetimeConstraints()             # 重置日期时间约束

    def getSharedWindowsLevel(self, declevel: int) -> Set:
        """Get the shared windows for a decimation level

        Parameters
        ----------
        declevel : int
            The decimation level (0 is the first level)
        
        Returns
        -------
        set
            The shared windows for the decimation level
        """
        return self.sharedWindows[declevel]

    def getNumSharedWindows(self, declevel: int) -> int:
        """Get the number of shared windows for a decimation level

        Parameters
        ----------
        declevel : int
            The decimation level (0 is the first level)
        
        Returns
        -------
        int
            The number of shared windows for the decimation level
        """
        return len(self.sharedWindows[declevel])

    def getWindowsForFreq(self, declevel: int, eIdx: int) -> Set:
        """Get the number of shared windows for a decimation level and evaluation frequency
        获取给定抽取级别与评估频率对应的共享窗口数量
        Parameters
        ----------
        declevel : int
            The decimation level (0 is the first level)
        eIdx : int
            The evaluation frequency index
        
        Returns
        -------
        set
            The shared windows for evaluation frequency eIdx at decimation level declevel 抽取(降采样)级别declevel下对应评估频率eIdx的共享窗口
        """
        sharedWindows = self.getSharedWindowsLevel(declevel)                                                                # 获取该分频级别下所有站点共有的窗口
        # now mask for the particular frequency - mask for each given site(给每个特定频率进行掩码——给每个站点)
        for s in self.sites:
            for mask in self.siteMasks[s]:                                                                                  # 获取当前站点的所有掩码对象
                # remove the masked windows from shared indices(从共享索引集中剔除经掩码标记的窗口)
                sharedWindows = sharedWindows - mask.getMaskWindowsFreq(declevel, eIdx)                                     # 从共享窗口中移除被当前掩码标记的窗口(当前分频级别的当前频点,1个)
        return sharedWindows                                                                                                # 返回最终的共享窗口集合

    def getUnmaskedWindowsLevel(self, declevel: int) -> Set:
        """Get unmasked windows for a decimation level
        获取指定分频级别下的未掩码窗口
        Calculate the number of non masked windows for the decimation level. This should speed up processing when constraints are applied.
        计算分频级别的非掩码窗口数量。这应该在应用约束时加快处理速度
        Parameters
        ----------
        declevel : int
            The decimation level

        Returns
        -------
        set
            Unmasked windows for the decimation level
        """
        indices = set()                                                                         # 创建一个空集合，用于存储该分频级别下所有未掩码窗口的全局索引
        evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)                          # 从分频参数对象中获取指定分频级别的所有评估频率
        for eIdx, eFreq in enumerate(evalFreq):                                                 # 遍历该分频级别下的所有评估频率
            indices.update(self.getWindowsForFreq(declevel, eIdx))                              # 获取当前分频级别和频率索引对应的未掩码窗口unmaskedWindows集合(自动去重,取并集(每个频点各自的unmasked))
        return indices

    def getDatetimeConstraints(self) -> Dict:
        """Get the datetime constraints
        
        Returns
        -------
        Dict
            Dictionary of datetime constraints at all decimation levels
        """
        self.calcDatetimeConstraints()
        return self.datetimeConstraints

    def getLevelDatetimeConstraints(self, declevel: int) -> List[List[datetime]]:
        """Get the datetime constraints for a decimation level
        获取某个降采样级别的日期时间约束条件
        Returns
        -------
        List[List[datetime]]
            Returns a list of datetime constraints, where each is a 2 element list with a start and stop 返回时间约束条件列表，其中每个约束条件为包含起始时间与结束时间的二元列表
        """
        self.calcDatetimeConstraints()              # 计算全局日期时间约束条件
        return self.datetimeConstraints[declevel]   # 返回时间约束条件二元列表

    def getMasks(self) -> Dict:
        """Get a dictionary with masks to use for each site in the window selector
        
        Returns
        -------
        Dict
            Dictionary with masks to use for each site in the window selector
        """
        return self.siteMasks

    def getSpecReaderForWindow(self, site: str, declevel: int, iWin: int):
        """Get the spectrum reader for a window

        Parameters
        ----------
        site : str
            The name of the site to get the spectrum reader for
        declevel : int
            The decimation level
        iWin : int
            The window index
        
        Returns 
        -------
        specFile : str, bool
            The name of the spectra file or False if the window is not found in any spectra file
        specReader : SpectrumReader, bool
            The spectrum reader or False if the window is not found in any spectra file
        """
        specRanges = self.siteSpecRanges[site][declevel]
        specReaders = self.siteSpecReaders[site][declevel]
        for specFile in specRanges:
            if iWin >= specRanges[specFile][0] and iWin <= specRanges[specFile][1]:
                return specFile, specReaders[specFile]

        # if here, no window found
        self.printWarning(
            "Shared window {}, decimation level {} does not appear in any files given the constraints applied".format(
                iWin, declevel
            )
        )
        return False, False

    def getSpecReaderBatches(self, declevel: int) -> Tuple[List, List]:
        """Batch the readers into wndow groups that can be read as required in a more efficient way
        将读取器分批划分为窗口组，以实现按需高效读取。
        First sorts all the site spectra files by global window range, putting them in ascending order. Loops over all the sites and constructs spectra batches to read in.
        首先依据全局窗口范围对所有站点频谱文件进行升序排序；随后遍历所有站点，并构建光谱批次以读取。
        Parameters
        ----------
        declevel : int
            The decimation level

        Returns
        -------
        batches : List[Dict[str, All]]
            The list of batches
        """
        # sort spectra files for the decimation level(针对降采样层级，对频谱文件进行排序)
        files: Dict[str, List[str]] = dict()
        readers: Dict[str, List[SpectrumReader]] = dict()   # 读取器readers
        winStarts: Dict[str, List[int]] = dict()
        winStops: Dict[str, List[int]] = dict()
        for site in self.sites:
            tmpFiles: List = list()
            tmpReaders: List = list()
            tmpStarts: List = list()
            tmpStops: List = list()
            for specFile, specRange in self.siteSpecRanges[site][declevel].items():
                tmpFiles.append(specFile)
                tmpReaders.append(self.siteSpecReaders[site][declevel][specFile])
                tmpStarts.append(specRange[0])
                tmpStops.append(specRange[1])
            # now sort on winstarts using zip(当前基于winstarts（窗口起始信息），通过zip（数据打包函数）实现排序)
            zipped = list(zip(tmpStarts, tmpFiles, tmpReaders, tmpStops))       # 按开始时间排序
            zipped.sort()           # # 按第一个元素（tmpStarts）排序
            winStarts[site], files[site], readers[site], winStops[site] = zip(*zipped)      # 解压排序后的结果

        # create batches(创建批次，站点分组)
        mainSite: str = self.sites[0]               # 主站点（参考站点）
        otherSites: List[str] = self.sites[1:]      # 其他站点
        # for saving batches(保存存储批次)
        batches: List[Dict[str, Any]] = list()
        for mainIdx, mainStart in enumerate(winStarts[mainSite]):       # 遍历主站点的每个时间窗口
            lastBatchWin = mainStart - 1                                # 上一个批次的结束位置
            while True:                                                 # 在当前主窗口内寻找重叠批次
                batch = dict()                                          # 创建新批次
                batch["globalrange"] = [lastBatchWin + 1, winStops[mainSite][mainIdx]]      # 初始化批次范围（基于主站点当前窗口）
                batch[mainSite] = readers[mainSite][mainIdx]            # 添加主站点数据
                for site in otherSites:                                 # 寻找其他站点与当前批次的重叠(暂时没用上)
                    for otherIdx, (start, stop) in enumerate(
                        zip(winStarts[site], winStops[site])
                    ):
                        if (
                            start >= batch["globalrange"][1]            # 检查时间重叠条件
                            or stop <= batch["globalrange"][0]
                        ):
                            continue                                    # 无重叠，继续查找
                        # else there is an overlap and it is the first overlap we are interested in(反之则存在重叠区间，且我们聚焦的是首个重叠区间)
                        # amend the batch range as required(按需修正批次范围)
                        if start > batch["globalrange"][0]:             # 有重叠，调整批次范围
                            batch["globalrange"][0] = start             # 调整开始时间
                        if stop < batch["globalrange"][1]:
                            batch["globalrange"][1] = stop              # 调整结束时间
                        batch[site] = readers[site][otherIdx]           # 添加该站点数据
                        break                                           # 找到第一个重叠即停止

                # now need to decide if the batch is complete - the batch dict has an entry for every site and one globalrange(当前需校验批次是否完整：批次字典需包含所有站点对应的条目及一个全局范围条目)
                if len(batch.keys()) == len(self.sites) + 1:            # 检查批次完整性，所有站点 + globalrange
                    batches.append(batch)                               # 完整批次，保存
                lastBatchWin = batch["globalrange"][1]                  # 更新位置，继续寻找下一个批次
                if lastBatchWin >= winStops[mainSite][mainIdx]:
                    break                                               # 当前主窗口处理完成
                # otherwise, continue in the while loop batching up(反之，则在while循环内继续执行批次构建)

        # print information
        self.printText("Spectra batches")
        for batchI, batch in enumerate(batches):        # 外层循环：遍历每个批次
            self.printText(
                "SPECTRA BATCH {}, Window Range {}".format(batchI, batch["globalrange"])    # 输出批次基本信息
            )
            for site in self.sites:                     # 内层循环：遍历批次中的每个站点
                self.printText(
                    "SPECTRA BATCH {}, Site {}, Path {}, Window Range {}".format(                 # 输出站点详细信息
                        batchI, site, batch[site].datapath, batch[site].getGlobalRange()
                    )
                )
        return batches

    def getDataSize(self, declevel: int) -> int:
        """Get the spectrum reader for a window
        获取一个窗口内的specReaders
        Parameters
        ----------
        declevel : str
            The decimation level

        Returns 
        -------
        int 
            The data size (number of points in the spectrum) at the decimation level 返回该抽取级别的数据大小（频谱点数）
        """
        # return data size of first file
        site = self.sites[0]
        specReaders = self.siteSpecReaders[site][declevel]          # 从站点频谱读取器字典中获取指定抽取级别的读取器字典
        for sF in specReaders:
            return specReaders[sF].getDataSize()

    def setSites(self, sites: List[str]) -> None:
        """Set the sites for which to find the shared windows
        设置用于查找共享窗口的站点集合
        Parameters
        ----------
        sites : List[str]
            List of sites
        """
        # first remove repeated sites(首先移除重复站点)
        sitesSet = set(sites)       # {'site1'}
        sites = list(sitesSet)      # 转为列表
        # now continue
        self.sites = sites
        for s in self.sites:
            self.siteMasks[s] = []
            self.siteSpecFolders[s] = []
            self.siteSpecReaders[s] = {}
            self.siteSpecRanges[s] = {}
            # use sets to hold gIndices(使用集合存储gIndices)
            # optimised to find intersections(经优化以实现交集查找)
            self.siteGlobalIndices[s] = {}
        # at the end, calculate global indices(最终计算全局索引)
        self.calcGlobalIndices()            # 确定所有站点对应的全局索引

    def addDatetimeConstraint(
        self, start: str, stop: str, declevel: Union[List[int], int, None] = None
    ):
        """Add datetime constraints

        Parameters
        ----------
        start : str
            Datetime constraint start in format %Y-%m-%d %H:%M:%S
        stop : str
            Datetime constraint end in format %Y-%m-%d %H:%M:%S
        declevel : List[int], int, optional
            The decimation level. If left as default, will be applied to all decimation levels.             
        """
        datetimeStart = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        datetimeStop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

        # levels the constraint applies to
        if declevel is None:
            levels = range(0, self.decParams.numLevels)
        elif isinstance(declevel, list):
            levels = declevel
        else:
            levels = [declevel]
        # then add constraints as appropriate
        for declevel in levels:
            self.datetimes[declevel].append([datetimeStart, datetimeStop])

    def addDateConstraint(
        self, dateC: str, declevel: Union[List[int], int, None] = None
    ):
        """Add a date constraint

        Parameters
        ----------
        dateC : str
            Datetime constraint in format %Y-%m-%d
        declevel : List[int], int, optional
            The decimation level. If left as default, will be applied to all decimation levels.             
        """
        datetimeC = datetime.strptime(dateC, "%Y-%m-%d").date()

        # levels the constraint applies to
        if declevel is None:
            levels = range(0, self.decParams.numLevels)
        elif isinstance(declevel, list):
            levels = declevel
        else:
            levels = [declevel]
        # then add constraints as appropriate
        for declevel in levels:
            self.dates[declevel].append(datetimeC)

    def addTimeConstraint(
        self, start: str, stop: str, declevel: Union[List[int], int, None] = None
    ):
        """Add a time constraint. This will recur on every day of recording.

        Parameters
        ----------
        start : str
            Time constraint start in format %H:%M:%S
        stop : str
            Time constraint end in format %H:%M:%S
        declevel : List[int], int, optional
            The decimation level. If left as default, will be applied to all decimation levels.             
        """
        timeStart = datetime.strptime(start, "%H:%M:%S").time()
        timeStop = datetime.strptime(stop, "%H:%M:%S").time()

        # levels the constraint applies to
        if declevel is None:
            levels = range(0, self.decParams.numLevels)
        elif isinstance(declevel, list):
            levels = declevel
        else:
            levels = [declevel]
        # then add constraints as appropriate
        for declevel in levels:
            self.times[declevel].append([timeStart, timeStop])

    def addWindowMask(self, site: str, maskName: str) -> None:
        """Add a window mask
        添加窗口掩码
        This is a mask with values for each evaluation frequency.
        这是一个包含每个评估频率值的掩码
        Parameters
        ----------
        site : str
            The site for which to search for a mask
        maskName : str
            The name of the mask
        """
        siteData = self.projData.getSiteData(site)                                  # 获取指定站点的数据对象
        maskIO = MaskIO(siteData.getSpecdirMaskPath(self.specdir))                  # 创建掩码IO对象，设置数据路径
        maskData = maskIO.read(maskName, self.sampleFreq)                           # 读取指定名称和采样频率的掩码数据
        self.siteMasks[site].append(maskData)                                       # 将掩码数据添加到站点的掩码列表中

    def resetDatetimeConstraints(self) -> None:
        """Reset datetime constraints重置日期时间约束"""
        # add a list for each decimation level(为每个抽取级别添加一个列表)
        for declevel in range(0, self.decParams.numLevels):
            self.datetimes[declevel] = []
            self.dates[declevel] = []
            self.times[declevel] = []
            self.datetimeConstraints[declevel] = []

    def resetMasks(self) -> None:
        """Reset masks"""
        # reset to no masks for any site
        for site in self.siteMasks:
            self.siteMasks[site] = []

    def calcSharedWindows(self):
        """Calculate shared windows between sites
        计算各站点之间的共享窗函数区间
        Calculates the shared windows between sites. Datetime constraints are applied. No masks are applied in this method. Masks are only applied when getting the windows for a particular evaluation frequency.
        计算多站点间的共享窗函数区间。本过程已施加日期时间约束条件；本方法中未引入掩码操作，掩码仅在获取特定评估频率对应的窗函数区间时生效。
        """
        if len(self.sites) == 0:
            self.printWarning(
                "No sites given to Window Selector. At least one site needs to be given."
            )
            return False

        # calculate datetime constraints(计算日期时间约束条件)
        self.calcDatetimeConstraints()          # 计算全局日期时间约束条件
        # initialise the sharedWindows with a set from one site(用一个站点的数据集初始化共享窗口（sharedWindows）)
        sites = self.sites
        siteInit = sites[0]
        numLevels = self.decParams.numLevels
        for declevel in range(0, numLevels):
            self.sharedWindows[declevel] = self.siteGlobalIndices[siteInit][declevel]
        # 下一步是对多个站点计算共享窗口
        # now for each decimation level, calculate the shared windows(现在对于每个降采样级别，计算共享窗口)
        for declevel in range(0, numLevels):                            # 遍历每个降采样层级
            for site in self.sites:                                     # 遍历每个站点(目前只添加site1这一个站点)
                self.sharedWindows[declevel] = self.sharedWindows[      # 计算当前层级所有站点的索引交集
                    declevel                                            # 比如site1[1，2，3，4，5] site2[2,3,4] site3[3,4,5] 那么交集为[3,4]
                ].intersection(self.siteGlobalIndices[site][declevel])  # intersection函数：取交集

        # apply time constraints(施加时间约束条件)
        # time constraints should be formulated as a set(时间约束条件应被形式化为一个集合)
        # and then, find the intersection again(随后，重新求解交集)
        for declevel in range(0, numLevels):
            constraints = self.getLevelDatetimeConstraints(declevel)        # 获取某个降采样级别的日期时间约束条件(暂时没有约束)
            if len(constraints) != 0:
                datetimeIndices = set()
                for dC in constraints:
                    gIndexStart, firstWindowStart = datetime2gIndex(
                        self.projData.refTime,
                        dC[0],
                        self.decParams.getSampleFreqLevel(declevel),
                        self.winParams.getWindowSize(declevel),
                        self.winParams.getOverlap(declevel),
                    )
                    gIndexEnd, firstWindowEnd = datetime2gIndex(
                        self.projData.refTime,
                        dC[1],
                        self.decParams.getSampleFreqLevel(declevel),
                        self.winParams.getWindowSize(declevel),
                        self.winParams.getOverlap(declevel),
                    )
                    gIndexEnd = (
                        gIndexEnd - 1
                    )  # as the function returns the next window starting after time
                    if gIndexEnd < gIndexStart:
                        gIndexEnd = gIndexStart
                    datetimeIndices.update(list(range(gIndexStart, gIndexEnd)))
                    self.printText(
                        "Decimation level = {}. Applying date constraint {} - {}, global index constraint {} - {}".format(
                            declevel, dC[0], dC[1], gIndexStart, gIndexEnd
                        )
                    )
                self.sharedWindows[declevel] = self.sharedWindows[
                    declevel
                ].intersection(datetimeIndices)

    def calcGlobalIndices(self) -> None:
        """Find all the global indices for the sites为所有站点的所有降采样层级，建立跨文件的全局索引映射，实现精确的数据定位和访问"""
        # get all the spectra files with the correct sampling frequency(获取所有具有正确采样频率的频谱文件)
        for site in self.sites:
            siteData = self.projData.getSiteData(site)
            timeFilesFs = siteData.getMeasurements(self.sampleFreq)
            # specFiles = self.proj.getSiteSpectraFiles(s)
            specFolders = siteData.spectra
            specFoldersFs = []
            for specFolder in specFolders:
                if specFolder in timeFilesFs:
                    specFoldersFs.append(specFolder)

            self.siteSpecFolders[site] = specFoldersFs

            # for each decimation level(针对每个降采样层级)
            # loop through each of the spectra folders(遍历所有频谱文件夹)
            # and find the global indices ranges for each decimation level(并确定各降采样层级对应的全局索引区间)
            numLevels = self.decParams.numLevels
            for declevel in range(0, numLevels):
                # get the dictionaries ready(完成字典结构的初始化与数据加载)
                self.siteSpecReaders[site][declevel] = {}
                self.siteSpecRanges[site][declevel] = {}
                self.siteGlobalIndices[site][declevel] = set()      # 创建集合{}
                # loop through spectra folders and figure out global indices(遍历所有频谱文件夹，确定各目标层级对应的全局索引区间)
                for specFolder in self.siteSpecFolders[site]:
                    # here, have to use the specdir option(此处需调用 specdir 参数选项)
                    specReader = SpectrumReader(
                        os.path.join(siteData.specPath, specFolder, self.specdir)
                    )
                    # here, use prepend to open the spectra file(这里，使用 prepend 打开频谱文件)
                    check = specReader.openBinaryForReading(self.prepend, declevel)         # 检查spectra00.bin和00.info文件(第一次循环)是否存在
                    # if file does not exist, continue
                    if not check:
                        continue
                    self.siteSpecReaders[site][declevel][specFolder] = specReader       # 频谱读取器的信息组织存储，将已初始化的频谱读取器对象存储到三级嵌套字典结构中，实现按站点、层级、文件夹的精确管理
                    globalRange = specReader.getGlobalRange()           # [507, 3399] 2893个窗口，[64, 423] 360个窗口，[8, 51] 44个窗口
                    self.siteSpecRanges[site][declevel][specFolder] = globalRange
                    # and save set of global indices(并保存一组全局索引)
                    self.siteGlobalIndices[site][declevel].update(
                        list(range(globalRange[0], globalRange[1] + 1))     # 保存每个窗口(单独)的全局索引{507, 508, ..., 3399}，{64, 65, ..., 423}
                    )

    def calcDatetimeConstraints(self) -> None:
        """Calculate overall datetime constraints
        计算全局日期时间约束条件
        Priority order for datetime constraints is: 日期时间约束的优先级顺序是
        1. datetime constraints 日期时间约束
        2. date constraints 日期约束
        3. time constraints 时间约束
        """
        # calculate site dates if required(如有需要，计算各站点的日期时间区间)
        siteDates = self.calcSiteDates()        # siteDates: [datetime.date(2023,10,22), datetime.date(2023,10,23)]

        # datetime constraints are for each decimation level（日期时间约束需针对每个降采样级别分别设置）
        numLevels = self.decParams.numLevels
        for declevel in range(0, numLevels):
            # calculate date and time constraints for each level(针对每个层级分别计算日期时间约束条件)
            # begin with the datetime constraints - these have highest priority(以日期时间约束条件为起始步骤——该约束条件具有最高优先级)
            self.datetimeConstraints[declevel] = self.datetimes[declevel]

            # check to see whether any date and time constraints(检查是否有任何日期和时间约束)
            if len(self.dates[declevel]) == 0 and len(self.times[declevel]) == 0:
                continue
            # ************后面暂时没用上
            dateConstraints = []
            if len(self.dates[declevel]) != 0:
                # apply time constraints only on specified days
                dateConstraints = self.dates[declevel]
            else:
                dateConstraints = siteDates

            # finally, add the time constraints to the dates
            # otherwise add the whole day
            dateAndTimeConstraints = []
            if len(self.times[declevel]) == 0:
                # add whole days
                for dC in dateConstraints:
                    start = datetime.combine(dC, time(0, 0, 0))
                    stop = datetime.combine(dC, time(23, 59, 59))
                    dateAndTimeConstraints.append([start, stop])
            else:
                # add each time for each day
                for tC in self.times[declevel]:
                    for dC in dateConstraints:
                        start = datetime.combine(dC, tC[0])
                        stop = datetime.combine(dC, tC[1])
                        # check if this goes over a day
                        if tC[1] < tC[0]:
                            # then need to increment the day
                            dCNext = dC + timedelta(days=1)
                            stop = datetime.combine(dCNext, tC[1])
                        # append to constraints
                        dateAndTimeConstraints.append([start, stop])

            # finally, combine datetimes and dateAndTimeConstraints
            self.datetimeConstraints[declevel] = (
                self.datetimeConstraints[declevel] + dateAndTimeConstraints
            )
            self.datetimeConstraints[declevel] = sorted(
                self.datetimeConstraints[declevel]
            )

    def calcSiteDates(self) -> List[datetime]:
        """Calculate a list of days that all the sites were operating
        计算所有站点均处于运行状态的日期列表
        This uses the siteStart and siteEnd datetimes, so does not take into account the start and end of actual time series measurements, which is taken into account later.
        本步骤基于各站点的起始与终止日期时间（siteStart、siteEnd）进行计算，未纳入实际时间序列测量的起止时间；后者将在后续流程中予以考量
        Returns
        -------
        List[datetime]
            A list of dates all the sites were operating
        """
        starts = []
        stops = []
        for site in self.sites:
            siteData = self.projData.getSiteData(site)
            starts.append(siteData.siteStart)
            stops.append(siteData.siteEnd)
        # need all the dates between(需提取[两时间节点间]所有有效日期)
        d1 = max(starts).date()
        d2 = min(stops).date()
        if d1 > d2:
            self.printError(
                "A site passed to the window selector does not overlap with any other sites. There will be no shared windows",
                quitRun=True,
            )
        # now with d2 > d1
        siteDates = []
        delta = d2 - d1
        # + 1 because inclusive of stop and start days(加1，因计算包含起始日与终止日)
        for i in range(delta.days + 1):
            siteDates.append(d1 + timedelta(days=i))        # siteDates: [datetime.date(2023,10,22), datetime.date(2023,10,23)]
        return siteDates

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Spectra directory = {}".format(self.specdir))
        textLst.append("Sites = {}".format(", ".join(self.sites)))
        textLst.append("Site information:")
        for site in self.sites:
            textLst = textLst + self.printSiteInfoList(site)
        return textLst

    def printSiteInfo(self):
        """Print out information about the sites"""
        for site in self.sites:
            blockPrint("WindowSelector::site info", self.printSiteInfoList(site))

    def printSiteInfoList(self, site: str) -> List[str]:
        """Return site window information as a list of strings

        Parameters
        ----------
        site : str
            The site name
        
        Returns
        -------
        List[str]
            Site window information as a list of strings
        """
        textLst = []
        textLst.append("Sampling frequency [Hz] = {:.6f}".format(self.sampleFreq))
        textLst.append("Site = {}".format(site))
        textLst.append("Site global index information")
        numLevels = self.decParams.numLevels
        for declevel in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(declevel))
            ranges = self.siteSpecRanges
            for sF in sorted(list(ranges[site][declevel].keys())):
                startTime1, endTime1 = gIndex2datetime(
                    ranges[site][declevel][sF][0],
                    self.projData.refTime,
                    self.sampleFreq / self.decParams.getDecFactor(declevel),
                    self.winParams.getWindowSize(declevel),
                    self.winParams.getOverlap(declevel),
                )
                startTime2, endTime2 = gIndex2datetime(
                    ranges[site][declevel][sF][1],
                    self.projData.refTime,
                    self.sampleFreq / self.decParams.getDecFactor(declevel),
                    self.winParams.getWindowSize(declevel),
                    self.winParams.getOverlap(declevel),
                )
                textLst.append(
                    "Measurement file = {}\ttime range = {} - {}\tGlobal Indices Range = {:d} - {:d}".format(
                        sF,
                        startTime1,
                        endTime2,
                        ranges[site][declevel][sF][0],
                        ranges[site][declevel][sF][1],
                    )
                )
        return textLst

    def printSharedWindows(self) -> None:
        """Print out the shared windows"""
        blockPrint("WindowSelector::shared windows", self.printSharedWindowsList())

    def printSharedWindowsList(self) -> List[str]:
        """Shared window information as a list of strings

        Returns
        -------
        List[str]
            Shared window information as a list of strings
        """
        textLst = []
        numLevels = self.decParams.numLevels
        for declevel in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(declevel))
            textLst.append(
                "\tNumber of shared windows = {:d}".format(
                    self.getNumSharedWindows(declevel)
                )
            )
            textLst.append(
                "\tShared window indices: {}".format(
                    list2ranges(self.getSharedWindowsLevel(declevel))
                )
            )
            textLst.append(
                "\tNumber of unmasked windows: {}".format(
                    len(self.getUnmaskedWindowsLevel(declevel))
                )
            )
        textLst.append(
            "NOTE: These are the shared windows at each decimation level. Windows for each evaluation frequency might vary depending on masks"
        )
        return textLst

    def printDatetimeConstraints(self):
        """Print out the datetime constraints"""
        blockPrint(
            "WindowSelector::datetime constraints", self.printDatetimeConstraintsList()
        )

    def printDatetimeConstraintsList(self) -> List[str]:
        """Datetime constraint information as a list of strings

        Returns
        -------
        List[str]
            Datetime constraint information as a list of strings
        """
        textLst = []
        # calculate datetime constraints
        self.calcDatetimeConstraints()
        # populate textLst
        textLst.append("Datetime constraints")
        numLevels = self.decParams.numLevels
        for declevel in range(0, numLevels):
            textLst.append("Decimation Level = {:d}".format(declevel))
            for d in self.getLevelDatetimeConstraints(declevel):
                textLst.append("\tConstraint {} - {}".format(d[0], d[1]))
        return textLst

    def printWindowMasks(self) -> None:
        """Print mask information"""
        blockPrint("WindowSelector::window masks", self.printWindowMasksList())

    def printWindowMasksList(self) -> List[str]:
        """Window mask information as a list of strings

        Returns
        -------
        List[str]
            Window mask information as a list of strings
        """
        textLst = []
        for s in self.sites:
            textLst.append("Site = {}".format(s))
            if len(self.siteMasks[s]) == 0:
                textLst.append("\t\tNo masks for this site")
            else:
                for mask in self.siteMasks[s]:
                    textLst.append("\t\tMask = {}".format(mask.maskName))
        return textLst

    def printWindowsForFrequency(self, listwindows=False):
        """Print information about the windows for each evaluation frequency

        Parameters
        ----------
        listwindows : bool
            Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal
        """
        blockPrint(
            "WindowSelector::windows for frequency",
            self.printWindowsForFrequencyList(listwindows),
        )

    def printWindowsForFrequencyList(self, listwindows=False) -> List[str]:
        """Information about windows for each evaluation frequency as a list of strings

        Parameters
        ----------
        listwindows : bool
            Boolean flag to actually write out all the windows. Default is False as this takes up a lot of space in the terminal

        Returns
        -------
        List[str]
            Windows for evaluation frequency information as a list of strings
        """
        textLst = []
        for declevel in range(0, self.decParams.numLevels):
            evalFreq = self.decParams.getEvalFrequenciesForLevel(declevel)
            unmaskedWindows = self.getNumSharedWindows(declevel)
            for eIdx, eFreq in enumerate(evalFreq):
                maskedWindows = self.getWindowsForFreq(declevel, eIdx)
                textLst.append(
                    "Evaluation frequency = {:.6f}, shared windows = {:d}, windows after masking = {:d}".format(
                        eFreq, unmaskedWindows, len(maskedWindows)
                    )
                )
                if listwindows:
                    textLst.append("{}".format(list2ranges(maskedWindows)))
        return textLst
