import sys
import os
import numpy as np
import scipy.linalg as linalg
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from typing import Dict, Union, List, Any

from resistics.project.data import ProjectData
from resistics.statistics.data import StatisticData
from resistics.statistics.data import StatisticDataAll
from resistics.statistics.calculator import StatisticCalculator
from resistics.statistics.utils import getStatElements
from resistics.window.selector import WindowSelector
from resistics.spectra.io import SpectrumReader
from resistics.spectra.data import SpectrumData
from resistics.project.mask import getMaskData
from resistics.common.io import fileFormatSampleFreq
from resistics.common.print import listToString
from resistics.common.checks import parseKeywords
from resistics.project.utils import projectText, projectWarning, projectError


def getStatisticData(
    projData: ProjectData, site: str, meas: str, stat: str, declevel: int = 0, **kwargs
) -> StatisticData:
    """Get the statistic data for a statistic for a site measurement
    获取站点测量数据文件的统计量数据
    Parameters
    ----------
    projData : ProjectData
        Project instance
    site : str
        The site for which to get the statistic data
    meas : str
        The measurement for which to get the statistic
    stat : str
        The statistic for which to get the measurement
    declevel : int, optional
        The decimation level to read in. Default is 0.
    specdir : str, optional
        The spectra directory

    Returns
    -------
    StatisticData, None
        A StatisticData object or None if the statistic data does not exist
    """
    from resistics.statistics.io import StatisticIO

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    if not siteData:
        projectError("Unable to find site {} in project".format(site), quitRun=True)

    # load the statistic data
    statIO = StatisticIO()                                                                      # 创建统计量IO对象，用于读取统计量数据文件
    statIO.setDatapath(
        os.path.join(siteData.getMeasurementStatPath(meas), options["specdir"])                 # 设置统计量IO对象的数据文件路径(D:\resistics\tutorial\tutorialProject\statData\site1\phnx_DZ12_15\spectra)
    )
    statData = statIO.read(stat, declevel)                                                      # 读取指定统计量stat和分频级别declevel的数据(如transferFunction00.npy)
    return statData

def getStatisticDataAll(
    projData: ProjectData, site: str, meas: str, stat: str, declevel: int = 0, **kwargs
):
    """获取所有处理方法的统计量数据，返回一个包含所有结果的字典

    返回字典结构例如:
    {
        "OLS": StatisticData对象,
        "M": StatisticData对象,
        ...
    }
    """
    from resistics.statistics.io import StatisticIO

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    if not siteData:
        projectError("Unable to find site {} in project".format(site), quitRun=True)
        print("Unable to find site {} in project".format(site))
        return None

    # 定义你需要遍历的所有方法对应的文件夹名称
    # methods = ["LS", "M", "BI", "MD", "RMD"]
    methods = ["LS", "RMD"]
    statDataAll = {}

    # load the statistic data
    statIO = StatisticIO()                                                                              # 创建统计量IO对象，用于读取统计量数据文件
    base_spectra_path = os.path.join(siteData.getMeasurementStatPath(meas), options["specdir"])         # 设置统计量IO对象的数据文件路径(D:\resistics\tutorial\tutorialProject\statData\site1\phnx_DZ12_15\spectra)

    # 循环遍历每种方法
    for method in methods:
        # 拼接具体方法的目录，例如: D:\resistics\tutorial\tutorialProject\statData\site1\phnx_ModEM0\spectra\OLS
        method_datapath = os.path.join(base_spectra_path, method)

        # 如果该方法的文件夹不存在，则跳过
        if not os.path.exists(method_datapath):
            print(f"Warning: Directory not found, skipping {method}")
            continue

        # 设置当前的 datapath 为特定方法的目录
        statIO.setDatapath(method_datapath)

        # 调用原生 read。由于 datapath 变了，它会自动去 .../spectra/OLS/transferFunction 寻找文件
        statData = statIO.read(stat, declevel)                                                  # 读取指定统计量和分频级别的数据(如transferFunction00.npy)

        if statData is not None:
            # 将读取到的 StatisticData 对象存入字典
            statDataAll[method] = statData

    return StatisticDataAll(statDataAll)

def getStatisticDataForSampleFreq(
    projData: ProjectData,
    site: str,
    sampleFreq: float,
    stat: str,
    declevel: int = 0,
    **kwargs
) -> List[StatisticData]:
    """Get the statistic data (for a particular decimation level) for all measurements in a site with sampling frequency sampleFreq

    Parameters
    ----------
    projData : ProjectData
        Project instance
    site : str
        The site for which to get the statistic data
    sampleFreq : float
        The sampling frequency
    stat : str
        The statistic for which to get the measurement
    declevel : int, optional
        The decimation level to read in. Default is 0.
    specdir : str, optional
        The spectra directory

    Returns
    -------
    Dict[str, StatisticData]
        A statistic data object
    """
    from resistics.statistics.io import StatisticIO

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    if not siteData:
        projectError("Unable to find site {} in project".format(site), quitRun=True)

    # load the statistic data
    statData: Dict[str, StatisticData] = {}
    statIO = StatisticIO()
    measurements = siteData.getMeasurements(sampleFreq)
    for meas in measurements:
        statIO.setDatapath(
            os.path.join(siteData.getMeasurementStatPath(meas), options["specdir"])
        )
        # make sure some data was found
        chk = statIO.read(stat, declevel)
        if chk is not None:
            statData[meas] = statIO.read(stat, declevel)
        else:
            projectWarning(
                "No {} data found for site {} and measurement {}".format(
                    stat, site, meas
                )
            )
    return statData


def calculateStatistics(projData: ProjectData, **kwargs):
    """Calculate statistics for sites
    为站点计算统计量
    Parameters
    ----------
    projData : ProjectData
        A project data instance
    sites : List[str], optional
        A list of sites to calculate statistics for
    sampleFreqs : List[float], optional
        List of sampling frequencies for which to calculate statistics
    specdir : str, optional
        The spectra directory for which to calculate statistics
    stats : List[str], optional
        The statistics to calculate out. Acceptable values are: "absvalEqn" "coherence", "psd", "poldir", "transFunc", "resPhase", "partialcoh". Configuration file values are used by default.
    ncores : int, optional
        The number of cores to run the transfer function calculations on        
    """
    from resistics.statistics.io import StatisticIO
    from resistics.project.shortcuts import getDecimationParameters

    options = {}
    options["sites"] = projData.getSites()                                              # 默认处理所有站点
    options["sampleFreqs"] = projData.getSampleFreqs()                                  # 默认所有采样频率
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]             # 从配置获取频谱目录
    options["stats"] = projData.config.configParams["Statistics"]["stats"]              # 从配置获取统计量列表
    options["ncores"] = projData.config.getStatisticCores()
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {}".format(
            listToString(options["stats"]), listToString(options["sites"])
        )
    )
    # loop through sites and calculate statistics
    statIO = StatisticIO()                                              # 初始化统计量IO对象,创建统计量读写器。
    for site in options["sites"]:
        siteData = projData.getSiteData(site)                           # 获取站点数据对象
        measurements = siteData.getMeasurements()                       # 获取站点文件

        for meas in measurements:                                       # 循环处理站点的每个测量
            sampleFreq = siteData.getMeasurementSampleFreq(meas)        # 获取测量的采样频率15.0
            if sampleFreq not in options["sampleFreqs"]:
                continue
            projectText(
                "Calculating stats for site {}, measurement {}".format(site, meas)
            )
            decParams = getDecimationParameters(sampleFreq, projData.config)
            numLevels = decParams.numLevels
            specReader = SpectrumReader(                                                            # 创建频谱数据读取器
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])             # 构建频谱文件路径,频谱目录
            )                                                                                       # 读取器创建：SpectrumReader负责读取二进制频谱数据

            # calculate statistics for decimation level if spectra file exists(若频谱文件已存在，则针对降采样层级计算统计量)
            for declevel in range(0, numLevels):                                                    # 循环处理每个抽取级别
                check = specReader.openBinaryForReading("spectra", declevel)                # 尝试打开该级别的.bin频谱文件，check：True
                if not check:                                                                       # 如果文件不存在，跳过该级别
                    continue
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                sampleFreqDec = specReader.getSampleFreq()                                          # 获取当前抽取级别的采样频率
                evalFreq = decParams.getEvalFrequenciesForLevel(declevel)                           # 获取当前层级的评估频率（实际计算的频率点）

                # dictionary for saving statistic data(创建统计量数据存储字典)
                statData = {}
                for stat in options["stats"]:                                                       # 循环并初始化每个统计量
                    statElements = getStatElements(stat)                                            # 获取该统计量的组成元素（如相干性需要指定通道对）
                    statData[stat] = StatisticData(                                                 # 创建统计量数据对象(里面有成图的最大列数maxcols:4)
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    statData[stat].setStatParams(numWindows, statElements, evalFreq)                # 设置统计量参数(窗口数,统计量元素,评估频率)
                    statData[stat].comments = specReader.getComments()                              # 添加注释信息，频谱文件的注释
                    statData[stat].addComment(projData.config.getConfigComment())                   # 配置文件的注释
                    statData[stat].addComment("Calculating statistic: {}".format(stat))             # 统计量的注释
                    statData[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))               # 统计量元素的注释
                    )
                # get all the spectra data in batch and process all the windows(批量获取所有频谱数据并处理全部窗口)
                spectraData, globalIndices = specReader.readBinaryBatchGlobal()                     # 批量读取所有窗口的频谱数据（提高IO效率）
                if options["ncores"] > 0:                                                           # 如果设置了多核并行计算
                    out = multiStatistics(                                                          # 使用多核并行计算统计量
                        options["ncores"], spectraData, evalFreq, options["stats"]                  # 使用的核心数，频谱数据数组，评估频率，要计算的统计量列表
                    )
                    for iW in range(numWindows):                                                    # 处理并行计算的结果，遍历每个窗口
                        for stat in options["stats"]:                                               # 遍历每个统计量
                            statData[stat].addStat(iW, globalIndices[iW], out[iW][stat])            # 将结果添加到统计量数据对象。iW窗口索引，globalIndices全局索引，out并行计算的结果
                else:
                    statCalculator = StatisticCalculator()                                          # 创建统计量计算器并初始化（可复用），避免重复创建对象
                    # ====================================================================================================
                    # 【核心修改区】：引入跨窗口的滑动数据块机制(一段时间，不是单独的一个窗口)
                    # ====================================================================================================
                    # blockSize = 50  # 建议块大小50-100，保证样本数 n 远大于变量数 p
                    #
                    # for start_idx in range(0, numWindows, blockSize):
                    #     end_idx = min(start_idx + blockSize, numWindows)
                    #     # 如果尾部剩余窗口太少（低于4个），则抛弃，避免自由度坍塌
                    #     if end_idx - start_idx < 4:
                    #         continue
                    #
                    #     # 切片获取当前块的所有频域窗口数据
                    #     block_spectra = spectraData[start_idx:end_idx]
                    #
                    #     # 执行跨窗块回归计算
                    #     blockStatData = calculateBlockStatistics(
                    #         block_spectra,
                    #         evalFreq,
                    #         options["stats"],
                    #         statCalculator=statCalculator,
                    #     )
                    #
                    #     # 散点图只需要记录点集，我们将计算得出的"块结果"记在块起始窗口的位置上
                    #     for stat in options["stats"]:
                    #         statData[stat].addStat(
                    #             start_idx, globalIndices[start_idx], blockStatData[stat]
                    #         )
                    # =======================================================================================================
                    # 原本代码(单个窗口)
                    for iW in range(numWindows):                                                    # 遍历每个窗口进行串行计算
                        winSpecData = spectraData[iW]                                               # 计算当前窗口的频谱数据
                        winStatData = calculateWindowStatistics(                                    # ※计算当前窗口的统计量(其中transferfuction用的普通最小二乘ols)
                            winSpecData,                                                            # 单个时间窗口的频谱数据
                            evalFreq,                                                               # 评估频率数组
                            options["stats"],                                                       # 要计算的统计量名称列表
                            statCalculator=statCalculator,                                          # 可选的统计量计算器（可复用）
                        )
                        # 将结果添加到统计量数据
                        for stat in options["stats"]:
                            statData[stat].addStat(
                                iW, globalIndices[iW], winStatData[stat]                            # 窗口索引,全局时间索引,该窗口的统计量结果
                            )

                specReader.closeFile()                                                              # 关闭频谱文件.bin（释放资源）

                # save statistic(保存计算好的统计量数据)
                for stat in options["stats"]:
                    statIO.setDatapath(                                                             # 设置统计量保存路径
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]               # 统计量路径,频谱目录（作为子目录）
                        )
                    )
                    statIO.write(statData[stat], declevel)                                          # 写入统计量数据到文件.npy,按抽取级别保存。


def calculateRemoteStatistics(projData: ProjectData, remoteSite: str, **kwargs):
    """Calculate statistics involving a remote reference site

    Parameters
    ----------
    projData : ProjectData
        A project data instance
    remoteSite : str
        The name of the site to use as the remote site
    sites : List[str], optional
        A list of sites to calculate statistics for
    sampleFreqs : List[float], optional
        List of sampling frequencies for which to calculate statistics
    specdir : str, optional
        The spectra directory for which to calculate statistics
    remotestats : List[str], optional
        The statistics to calculate out. Acceptable statistics are: "RR_coherence", "RR_coherenceEqn", "RR_absvalEqn", "RR_transferFunction", "RR_resPhase". Configuration file values are used by default.
    """
    from resistics.statistics.io import StatisticIO
    from resistics.statistics.calculator import StatisticCalculator
    from resistics.project.shortcuts import (
        getDecimationParameters,
        getWindowParameters,
        getWindowSelector,
    )

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["chans"] = []
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["remotestats"] = projData.config.configParams["Statistics"]["remotestats"]
    options["ncores"] = projData.config.getStatisticCores()
    options = parseKeywords(options, kwargs)

    projectText(
        "Calculating stats: {} for sites: {} with remote site {}".format(
            listToString(options["remotestats"]),
            listToString(options["sites"]),
            remoteSite,
        )
    )

    statIO = StatisticIO()
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        measurements = siteData.getMeasurements()

        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)
            if sampleFreq not in options["sampleFreqs"]:
                continue
            projectText(
                "Calculating stats for site {}, measurement {} with reference {}".format(
                    site, meas, remoteSite
                )
            )
            # decimation and window parameters
            decParams = getDecimationParameters(sampleFreq, projData.config)
            numLevels = decParams.numLevels
            winParams = getWindowParameters(decParams, projData.config)
            # create the window selector and find the shared windows
            winSelector = getWindowSelector(projData, decParams, winParams)
            winSelector.setSites([site, remoteSite])
            winSelector.calcSharedWindows()
            # create the spectrum reader
            specReader = SpectrumReader(
                os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
            )

            # calculate statistics for decimation level if spectra file exists
            for declevel in range(0, numLevels):
                check = specReader.openBinaryForReading("spectra", declevel)
                if not check:
                    continue
                # information regarding only this spectra file
                refTime = specReader.getReferenceTime()
                winSize = specReader.getWindowSize()
                winOlap = specReader.getWindowOverlap()
                numWindows = specReader.getNumWindows()
                evalFreq = decParams.getEvalFrequenciesForLevel(declevel)
                sampleFreqDec = specReader.getSampleFreq()
                globalOffset = specReader.getGlobalOffset()

                # find size of the intersection between the windows in this spectra file and the shared windows
                sharedWindows = winSelector.getSharedWindowsLevel(declevel)
                sharedWindowsMeas = sharedWindows.intersection(
                    set(np.arange(globalOffset, globalOffset + numWindows))
                )
                sharedWindowsMeas = sorted(list(sharedWindowsMeas))
                numSharedWindows = len(sharedWindowsMeas)

                statData = {}
                # create the statistic handlers
                for stat in options["remotestats"]:
                    statElements = getStatElements(stat)
                    statData[stat] = StatisticData(
                        stat, refTime, sampleFreqDec, winSize, winOlap
                    )
                    # with remote reference the number of windows is number of shared windows
                    statData[stat].setStatParams(
                        numSharedWindows, statElements, evalFreq
                    )
                    statData[stat].comments = specReader.getComments()
                    statData[stat].addComment(projData.config.getConfigComment())
                    statData[stat].addComment(
                        "Calculating remote statistic: {}".format(stat)
                    )
                    statData[stat].addComment(
                        "Statistic components: {}".format(listToString(statElements))
                    )

                # collect the spectra data
                spectraData, _globalIndices = specReader.readBinaryBatchGlobal(
                    sharedWindowsMeas
                )
                remoteData = []
                for globalWindow in sharedWindowsMeas:
                    _, remoteReader = winSelector.getSpecReaderForWindow(
                        remoteSite, declevel, globalWindow
                    )
                    remoteData.append(remoteReader.readBinaryWindowGlobal(globalWindow))

                # calculate
                if options["ncores"] > 0:
                    out = multiStatistics(
                        options["ncores"],
                        spectraData,
                        evalFreq,
                        options["remotestats"],
                        remoteData=remoteData,
                    )
                    for iW, globalWindow in enumerate(sharedWindowsMeas):
                        for stat in options["remotestats"]:
                            statData[stat].addStat(iW, globalWindow, out[iW][stat])
                else:
                    statCalculator = StatisticCalculator()
                    for iW, globalWindow in enumerate(sharedWindowsMeas):
                        winStatData = calculateWindowStatistics(
                            spectraData[iW],
                            evalFreq,
                            options["remotestats"],
                            remoteSpecData=remoteData[iW],
                            statCalculator=statCalculator,
                        )
                        for stat in options["remotestats"]:
                            statData[stat].addStat(iW, globalWindow, winStatData[stat])

                # save statistic
                for stat in options["remotestats"]:
                    statIO.setDatapath(
                        os.path.join(
                            siteData.getMeasurementStatPath(meas), options["specdir"]
                        )
                    )
                    statIO.write(statData[stat], declevel)

def calculateBlockStatistics(
    block_spectra: List[Any],                                               # 滑动窗口的频谱数据
    evalFreq: np.ndarray,                                                   # 评估频率数组
    stats: List[str],                                                       # 要计算的统计量名称列表
    remoteSpecData: Union[SpectrumData, None] = None,                       # 可选的远程参考频谱数据
    statCalculator: Union[StatisticCalculator, None] = None,                # 可选的统计量计算器
) -> Dict[str, Any]:                                                        # 可选的统计量计算器
    """Calculate the statistics for a block of windows (跨窗口统计)
    滑动时间窗口的统计量
    This method is meant to be used with multiprocessing
    这个方法专为多进程并行计算设计
    Parameters
    ----------
    winSpecData : SpectrumData
        Spectrum data for a single window
    evalFreq : np.ndarray
        Array for evaluation frequencies
    stats : List[str]
        List of statistics to calculate
    remoteSpecData : None, SpectrumData
        Optionally add remote spectrum data to calculate remote statistics
    statCalculator : None, StatisticCalculator
        Provide a statistic calculator to avoid multiple creation of objects
    """
    if statCalculator is None:
        statCalculator = StatisticCalculator()                                          # 如果未提供统计量计算器，则创建一个新的StatisticCalculator实例
    statCalculator.setSpectra(winSpecData.freqArray, winSpecData, evalFreq)  # ※计算功率谱(矩阵)，设置频谱数据到统计量计算器。winSpecData.freqArray：原始频谱的频率点数组。winSpecData：频谱数据对象（包含各通道的复数频谱）。evalFreq：评估频率数组（实际计算统计量的频率点）。
    data = {}
    for stat in stats:                                                                  # 遍历所有要计算的统计量
        data[stat] = statCalculator.blockTransferFunction(block_spectra, evalFreq)                            # 计算单个对应统计量的数据并存储到字典
    return data

def calculateWindowStatistics(
    winSpecData: SpectrumData,                                              # 单个时间窗口的频谱数据
    evalFreq: np.ndarray,                                                   # 评估频率数组
    stats: List[str],                                                       # 要计算的统计量名称列表
    remoteSpecData: Union[SpectrumData, None] = None,                       # 可选的远程参考频谱数据
    statCalculator: Union[StatisticCalculator, None] = None,                # 可选的统计量计算器
) -> Dict[str, Any]:                                                        # 可选的统计量计算器
    """Calculate the statistics for a single window
    计算单个时间窗口的统计量
    This method is meant to be used with multiprocessing
    这个方法专为多进程并行计算设计
    Parameters
    ----------
    winSpecData : SpectrumData
        Spectrum data for a single window
    evalFreq : np.ndarray
        Array for evaluation frequencies
    stats : List[str]
        List of statistics to calculate
    remoteSpecData : None, SpectrumData
        Optionally add remote spectrum data to calculate remote statistics
    statCalculator : None, StatisticCalculator
        Provide a statistic calculator to avoid multiple creation of objects
    """
    if statCalculator is None:
        statCalculator = StatisticCalculator()                                          # 如果未提供统计量计算器，则创建一个新的StatisticCalculator实例
    statCalculator.setSpectra(winSpecData.freqArray, winSpecData, evalFreq)             # ※计算功率谱(矩阵)，设置频谱数据到统计量计算器。winSpecData.freqArray：原始频谱的频率点数组。winSpecData：频谱数据对象（包含各通道的复数频谱）。evalFreq：评估频率数组（实际计算统计量的频率点）。
    if remoteSpecData is not None:
        statCalculator.addRemoteSpec(remoteSpecData)                                    # 如果提供了远程参考频谱数据，则添加远参考频谱数据到计算器
    data = {}
    for stat in stats:                                                                  # 遍历所有要计算的统计量
        data[stat] = statCalculator.getDataForStatName(stat)                            # 计算单个对应统计量的数据并存储到字典
    return data                                                                         # 返回统计量数据字典


def multiStatistics(
    ncores: int,
    spectraData: List[SpectrumData],
    evalFreq: np.ndarray,
    stats: List[str],
    remoteData: Union[List[SpectrumData], None] = None,
):
    """Multiprocessing of statistics
    
    Parameters
    ----------
    ncores : int
        The number of cores to use    
    spectraData : List[SpectrumData]
        List of spectrum data to process
    evalFreq : np.ndarray
        The evaluation frequencies
    stats : List[str]
        The statistics to calculate
    remoteData : None, List[SpectrumData]
        Remote data in case of remote reference statistics. None is default

    Returns
    -------
    List[Dict[str, Any]]
        The statistic data returned as a list with an entry for every window. The dictionary maps the statistic name to the data for that window.
    """
    import multiprocessing as mp

    # prepare the lists
    if remoteData is None:
        multiTuples = [(winSpecData, evalFreq, stats) for winSpecData in spectraData]
    else:
        multiTuples = [
            (winSpecData, evalFreq, stats, remoteSpecData)
            for winSpecData, remoteSpecData in zip(spectraData, remoteData)
        ]
    # multiprocess
    projectText("Running statistic calculations on {} cores".format(ncores))
    with mp.Pool(ncores) as pool:
        out = pool.starmap(calculateWindowStatistics, multiTuples)
    return out


def viewStatistic(
    projData: ProjectData, site: str, sampleFreq: Union[int, float], stat: str, **kwargs
) -> Union[Figure, None]:
    """View statistic data for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    clim : List, optional
        Limits for colourbar axis
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    colortitle : str, optional
        Title for the colourbar
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, None.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["clim"] = []
    options["xlim"] = []
    options["ylim"] = []
    options["colortitle"] = ""
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting statistic {} for site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )
    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    statMeas = list(statData.keys())
    if len(statMeas) == 0:
        projectWarning(
            "No statistic files for site {}, sampling frequency {}, statistic {} and decimation level {}".format(
                site, sampleFreq, stat, options["declevel"]
            )
        )
        return None
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # setup the figure
    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])

    # get the date limits
    siteData = projData.getSiteData(site)
    if len(options["xlim"]) == 0:
        start = siteData.getMeasurementStart(statMeas[0])
        end = siteData.getMeasurementEnd(statMeas[0])
        for meas in statMeas:
            start = min(start, siteData.getMeasurementStart(meas))
            end = max(end, siteData.getMeasurementEnd(meas))
        options["xlim"] = [start, end]

    # do the plots
    for meas in statMeas:
        statData[meas].view(
            options["eFreqI"],
            fig=fig,
            xlim=options["xlim"],
            ylim=options["ylim"],
            clim=options["clim"],
            label=meas,
            plotfonts=options["plotoptions"]["plotfonts"],
            maskwindows=maskWindows,
        )
    # add a legened
    plt.legend(markerscale=4, fontsize=plotfonts["legend"])

    # do the title after all the plots
    fig.suptitle(
        "{} values for {}, sampling frequency = {:.2f} Hz, decimation level = {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "stat_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticHistogram(
    projData: ProjectData, site: str, sampleFreq: float, stat: str, **kwargs
) -> Union[Figure, None]:
    """View statistic histograms for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index       
    specdir : str
        The spectra directory        
    maskname : str
        Mask name 
    numbins : int
        The number of bins for the histogram data binning
    xlim : List, optional
        Limits for the x axis
    maxcols : int
        The maximum number of columns in the plots
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None. If no data was found, None.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["numbins"] = 40
    options["xlim"] = []
    options["maxcols"] = 4
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting histogram for statistic {}, site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )

    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    statMeas = list(statData.keys())
    if len(statMeas) == 0:
        projectWarning(
            "No statistic files for site {}, sampling frequency {}, statistic {} and decimation level {}".format(
                site, sampleFreq, stat, options["declevel"]
            )
        )
        return None
    # get the statistic components
    statComponents = statData[statMeas[0]].winStats
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # plot information
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(statComponents))
    numbins = options["numbins"]

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} histogram for {}, sampling frequency {} Hz, decimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, val in enumerate(statComponents):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Histogram {}".format(val), fontsize=plotfonts["title"])

        plotData = np.empty(shape=(0))
        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotData = np.concatenate(
                (plotData, np.squeeze(stats[:, options["eFreqI"], idx]))
            )
        # remove infinities and nans
        plotData = plotData[np.isfinite(plotData)]

        # x axis options
        xlim = (
            options["xlim"]
            if len(options["xlim"]) > 0
            else [np.min(plotData), np.max(plotData)]
        )
        plt.xlim(xlim)
        plt.xlabel("Value", fontsize=plotfonts["axisLabel"])
        # now plot with xlim in mind
        plt.hist(plotData, numbins, range=xlim, facecolor="red", alpha=0.75)
        plt.grid()
        # y axis options
        plt.ylabel("Count", fontsize=plotfonts["axisLabel"])
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statHist_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticCrossplot(
    projData: ProjectData,
    site: str,
    sampleFreq: Union[int, float],
    stat: str,
    crossplots: List[List[str]],
    **kwargs
) -> Union[Figure, None]:
    """View statistic crossplot data for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    crossplots : List[List[str]]
        The statistic element pairs to crossplot
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    maxcols : int
        The maximum number of columns in the plots        
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, None.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, getPlotRowsAndCols

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["xlim"] = []
    options["ylim"] = []
    options["maxcols"] = 2
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting crossplot for statistic {}, site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )

    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    statMeas = list(statData.keys())
    if len(statMeas) == 0:
        projectWarning(
            "No statistic files for site {}, sampling frequency {}, statistic {} and decimation level {}".format(
                site, sampleFreq, stat, options["declevel"]
            )
        )
        return None
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # plot information
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(crossplots))

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} crossplots for {}, sampling frequency {} Hz,\ndecimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, cplot in enumerate(crossplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Crossplot {}".format(cplot), fontsize=plotfonts["title"])

        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotI1 = statData[meas].winStats.index(cplot[0])
            plotData1 = np.squeeze(stats[:, options["eFreqI"], plotI1])
            plotI2 = statData[meas].winStats.index(cplot[1])
            plotData2 = np.squeeze(stats[:, options["eFreqI"], plotI2])
            scat = plt.scatter(
                plotData1, plotData2, edgecolors="none", marker="o", s=12, label=meas
            )

        # x axis options
        if len(options["xlim"]) > 0:
            plt.xlim(options["xlim"])
        if len(options["ylim"]) > 0:
            plt.ylim(options["ylim"])
        plt.xlabel(cplot[0], fontsize=plotfonts["axisLabel"])
        plt.ylabel(cplot[1], fontsize=plotfonts["axisLabel"])
        plt.grid(True)
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])
        plt.legend(loc=2, markerscale=4, fontsize=plotfonts["legend"])

    # plot format, show and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statCrossplot_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewStatisticDensityplot(
    projData: ProjectData,
    site: str,
    sampleFreq: Union[int, float],
    stat: str,
    crossplots: List[List[str]],
    **kwargs
) -> Union[Figure, None]:
    """View statistic data as a density plot for a single sampling frequency of a site
    
    Parameters
    ----------
    projData : ProjectData
        A project instance
    site : str
        The site for which to plot statistics
    stat : str
        The statistic to plot
    sampleFreq : float
        The sampling frequency for which to plot statistics
    crossplots : List[List[str]]
        The statistic element pairs to crossplot
    declevel : int
        The decimation level to plot
    eFreqI : int
        The evaluation frequency index
    specdir : str
        The spectra directory
    maskname : str
        Mask name         
    xlim : List, optional
        Limits for the x axis
    ylim : List, optional
        Limits for the y axis
    maxcols : int
        The maximum number of columns in the plots        
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options    

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, None.
    """
    from resistics.common.plot import (
        savePlot,
        plotOptionsSpec,
        getPlotRowsAndCols,
        colorbar2dSpectra,
    )

    options = {}
    options["declevel"] = 0
    options["eFreqI"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["maskname"] = ""
    options["xlim"] = []
    options["ylim"] = []
    options["maxcols"] = 2
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting density plot for statistic {}, site {} and sampling frequency {}".format(
            stat, site, sampleFreq
        )
    )

    statData = getStatisticDataForSampleFreq(
        projData,
        site,
        sampleFreq,
        stat,
        declevel=options["declevel"],
        specdir=options["specdir"],
    )
    statMeas = list(statData.keys())
    if len(statMeas) == 0:
        projectWarning(
            "No statistic files for site {}, sampling frequency {}, statistic {} and decimation level {}".format(
                site, sampleFreq, stat, options["declevel"]
            )
        )
        return None
    # get the evaluation frequency
    eFreq = statData[statMeas[0]].evalFreq[options["eFreqI"]]

    # get the mask data
    maskWindows = []
    if options["maskname"] != "":
        maskData = getMaskData(projData, site, options["maskname"], sampleFreq)
        maskWindows = maskData.getMaskWindowsFreq(
            options["declevel"], options["eFreqI"]
        )

    # plot information
    nrows, ncols = getPlotRowsAndCols(options["maxcols"], len(crossplots))

    plotfonts = options["plotoptions"]["plotfonts"]
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])
    # suptitle
    st = fig.suptitle(
        "{} density plots for {}, sampling frequency {} Hz,\ndecimation level {} and evaluation frequency {} Hz".format(
            stat, site, sampleFreq, options["declevel"], eFreq
        ),
        fontsize=plotfonts["suptitle"],
    )
    st.set_y(0.98)

    # now plot the data
    for idx, cplot in enumerate(crossplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plt.title("Crossplot {}".format(cplot), fontsize=plotfonts["title"])

        plotAll1 = []
        plotAll2 = []
        for meas in statMeas:
            stats = statData[meas].getStats(maskwindows=maskWindows)
            plotI1 = statData[meas].winStats.index(cplot[0])
            plotData1 = np.squeeze(stats[:, options["eFreqI"], plotI1])
            plotI2 = statData[meas].winStats.index(cplot[1])
            plotData2 = np.squeeze(stats[:, options["eFreqI"], plotI2])
            # add to all data
            if plotData1.size == 0:
                continue
            if plotData1.size == 1:
                plotAll1 = plotAll1 + [float(plotData1)]
                plotAll2 = plotAll2 + [float(plotData2)]
            else:
                plotAll1 = plotAll1 + plotData1.tolist()
                plotAll2 = plotAll2 + plotData2.tolist()

        plotAll1 = np.array(plotAll1)
        plotAll2 = np.array(plotAll2)

        nbins = 200
        if len(options["xlim"]) > 0:
            plt.xlim(options["xlim"])
            rangex = options["xlim"]
        else:
            minx = np.percentile(plotAll1, 2)
            maxx = np.percentile(plotAll1, 98)
            ax.set_xlim(minx, maxx)
            rangex = [minx, maxx]

        if len(options["ylim"]) > 0:
            plt.ylim(options["ylim"])
            rangey = options["ylim"]
        else:
            miny = np.percentile(plotAll2, 2)
            maxy = np.percentile(plotAll2, 98)
            ax.set_ylim(miny, maxy)
            rangey = [miny, maxy]

        plt.hist2d(
            plotAll1,
            plotAll2,
            bins=(nbins, nbins),
            range=[rangex, rangey],
            cmap=plt.cm.inferno,
        )

        # axis format
        plt.xlabel(cplot[0], fontsize=plotfonts["axisLabel"])
        plt.ylabel(cplot[1], fontsize=plotfonts["axisLabel"])
        plt.grid(True)
        # set tick sizes
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])

    # plot format, show and save
    # fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    if options["save"]:
        impath = projData.imagePath
        sampleFreqStr = fileFormatSampleFreq(sampleFreq)
        filename = "statDensityplot_{:s}_{:s}_{:s}_dec{:d}_efreq{:d}_{:s}".format(
            stat,
            site,
            sampleFreqStr,
            options["declevel"],
            options["eFreqI"],
            options["specdir"],
        )
        if options["maskname"] != "":
            filename = "{}_{}".format(filename, options["maskname"])
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig
