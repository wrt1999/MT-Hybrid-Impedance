import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
from typing import List, Union, Dict

from resistics.project.data import ProjectData
from resistics.site.data import SiteData
from resistics.time.data import TimeData
from resistics.spectra.io import SpectrumReader
from resistics.common.checks import parseKeywords, isMagnetic
from resistics.common.print import listToString, breakComment
from resistics.project.utils import projectText, projectWarning, projectError


def getSpecReader(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> Union[SpectrumReader, None]:
    """Get the spectrum reader for a measurement

    Parameters
    ----------
    site : str
        Site for which to get the spectra reader
    meas : str
        The measurement
    options : Dict
        Options in a dictionary
    declevel : int, optional
        Decimation level for which to get data
    specdir : str, optional
        String that specifies spectra directory for the measurement

    Returns
    -------
    SpectrumReader
        The SpectrumReader object or None if data does not exist
    """
    options = {}
    options["declevel"]: int = 0
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)

    siteData = projData.getSiteData(site)
    measurements = siteData.getMeasurements()
    if meas not in measurements:
        projectError("Measurement directory {} not found".format(meas), quitRun=True)

    # create the spectrum reader
    specReader = SpectrumReader(
        os.path.join(siteData.getMeasurementSpecPath(meas), options["specdir"])
    )
    specReader.printInfo()

    # open the spectra file for the current decimation level if it exists
    check = specReader.openBinaryForReading("spectra", options["declevel"])
    if not check:
        projectWarning(
            "Spectra file does not exist at level {}".format(options["declevel"])
        )
        return None
    return specReader


def calculateSpectra(projData: ProjectData, **kwargs) -> None:
    """Calculate spectra for the project time data
    计算该项目时间数据的频谱
    The philosophy is that spectra are calculated out for all data and later limited using statistics and time constraints
    其设计理念为：先对所有数据完成频谱计算，再通过统计学方法与时间约束条件对结果进行筛选
    Parameters
    ----------
    projData : ProjectData
        A project data object
    sites : str, List[str], optional
        Either a single site or a list of sites
    sampleFreqs : int, float, List[float], optional
        The frequencies in Hz for which to calculate the spectra. Either a single frequency or a list of them.
    chans : List[str], optional
        The channels for which to calculate out the spectra
    polreverse :  Dict[str, bool]
        Keys are channels and values are boolean flags for reversing
    scale : Dict[str, float]
        Keys are channels and values are floats to multiply the channel data by
    calibrate : bool, optional
        Flag whether to calibrate the data or not
    notch : List[float], optional
        List of frequencies to notch
    filter : Dict, optional
        Filter parameters
    specdir : str, optional
        The spectra directory to save the spectra data in
    ncores : int, optional
        The number of cores to run the transfer function calculations on        
    """
    from resistics.spectra.io import SpectrumWriter
    from resistics.decimate.decimator import Decimator
    from resistics.window.windower import Windower
    from resistics.project.shortcuts import (
        getCalibrator,
        getDecimationParameters,
        getWindowParameters,
    )
    from resistics.project.preprocess import (
        applyPolarisationReversalOptions,
        applyScaleOptions,
        applyCalibrationOptions,
        applyFilterOptions,
        applyNotchOptions,
    )

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"]: List[float] = projData.getSampleFreqs()
    options["chans"]: List[str] = []
    options["polreverse"]: Union[bool, Dict[str, bool]] = False
    options["scale"]: Union[bool, Dict[str, float]] = False
    options["calibrate"]: bool = True
    options["notch"]: List[float] = []
    options["filter"]: Dict = {}
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]
    options["ncores"] = projData.config.getSpectraCores()
    options = parseKeywords(options, kwargs)

    # prepare calibrator
    cal = getCalibrator(projData.calPath, projData.config)
    if options["calibrate"]:
        cal.printInfo()

    datetimeRef = projData.refTime
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        siteData.printInfo()

        # calculate spectra for each frequency(计算每个频率的频谱)
        for sampleFreq in options["sampleFreqs"]:
            measurements = siteData.getMeasurements(sampleFreq)
            projectText(
                "Site {} has {:d} measurement(s) at sampling frequency {:.2f}".format(
                    site, len(measurements), sampleFreq
                )
            )
            if len(measurements) == 0:
                continue  # no data files at this sample rate

            for meas in measurements:
                projectText(
                    "Calculating spectra for site {} and measurement {}".format(
                        site, meas
                    )
                )
                # get measurement start and end times - this is the time of the first and last sample
                reader = siteData.getMeasurement(meas)
                startTime = siteData.getMeasurementStart(meas)
                stopTime = siteData.getMeasurementEnd(meas)
                dataChans = (
                    options["chans"]
                    if len(options["chans"]) > 0
                    else reader.getChannels()
                )
                timeData = reader.getPhysicalData(startTime, stopTime, chans=dataChans)         # ***注意：Ez通道没归一处理(mV->mV/km)***，去除增益gain,减去数据的平均值（去直流分量）,使数据以零为中心。
                timeData.addComment(breakComment())
                timeData.addComment("Calculating project spectra")
                timeData.addComment(projData.config.getConfigComment())
                # apply various options（应用各种选项）
                applyPolarisationReversalOptions(options, timeData)              # 极性反转：解决物理传感器方向问题(默认False)
                applyScaleOptions(options, timeData)                             # 幅度缩放：统一量纲和硬件增益(默认False)
                applyCalibrationOptions(options, cal, timeData, reader)          # 传感器校准(先CLB后CLC)：将电压转为物理量(返回timeData)
                applyFilterOptions(options, timeData)                            # 宽带滤波：提取目标频段(默认{})
                applyNotchOptions(options, timeData)                             # 陷波滤波：抑制文化噪声(默认[])
                # define decimation and window parameters(定义降采样和窗口参数)
                decParams = getDecimationParameters(sampleFreq, projData.config)  # 设置要计算抽取参数的级数和每级的频率数，根据采样率和项目配置，自动规划出这些处理层级（Level），并为每一级计算出合适的降采样参数
                numLevels = decParams.numLevels
                winParams = getWindowParameters(decParams, projData.config)       # 根据降采样参数和项目配置，为每一处理层级智能地设计合适的时窗参数，计算每层的窗口大小[512 512 ... 512] 和重叠率 [128 128 ... 128],不同层级的降采样数据具有不同的采样率和频率特性，因此需要不同的窗口参数
                dec = Decimator(timeData, decParams)                              # 使用时间数据和降采样参数进行初始化
                timeData.addComment(
                    "Decimating with {} levels and {} frequencies per level".format(
                        numLevels, decParams.freqPerLevel
                    )
                )

                # loop through decimation levels(遍历降采样级别)
                for declevel in range(0, numLevels):
                    # get the data for the current level(获取当前层级的数据)
                    check = dec.incrementLevel()            # (这一步里进行数据降采样)判断是否可以将数据从当前层级继续降采样到下一个层级，一般返回True，这是大地电磁（MT）多级降采样处理中层级切换的核心方法。
                    if not check:
                        break  # not enough data
                    timeData = dec.timeData                 # 获取当前层级的降采样数据(相对上一层已经过降采样之后)

                    # create the windower and give it window parameters for current level(构建窗函数生成器，并为其配置当前层级所需的窗口参数)
                    sampleFreqDec = dec.sampleFreq
                    # 窗函数创建(如[[258, 769], [642, 1153]...[1111170, 1111681]])
                    win = Windower(
                        datetimeRef,
                        timeData,
                        winParams.getWindowSize(declevel),
                        winParams.getOverlap(declevel),
                    )
                    if win.numWindows < 2:            # 窗口数 < 2
                        break  # do no more decimation

                    # print information and add some comments
                    projectText(
                        "Calculating spectra for decimation level {}".format(declevel)
                    )
                    timeData.addComment(
                        "Evaluation frequencies for this level {}".format(
                            listToString(decParams.getEvalFrequenciesForLevel(declevel))        # 获取每层declevel的评估频率
                        )
                    )
                    timeData.addComment(
                        "Windowing with window size {} samples and overlap {} samples".format(
                            winParams.getWindowSize(declevel),
                            winParams.getOverlap(declevel),
                        )
                    )
                    if projData.config.configParams["Spectra"]["applywindow"]:
                        timeData.addComment(
                            "Performing fourier transform with window function {}".format(
                                projData.config.configParams["Spectra"]["windowfunc"]
                            )
                        )
                    else:
                        timeData.addComment(
                            "Performing fourier transform with no window function"
                        )

                    # collect time data
                    timeDataList = []                   # timeDataList用于存储所有窗口的时域数据
                    for iW in range(0, win.numWindows):
                        timeDataList.append(win.getData(iW))            # 从窗口处理器中提取所有窗口的时域数据

                    # open spectra file for saving(打开谱文件后保存)
                    specPath = os.path.join(
                        siteData.getMeasurementSpecPath(meas), options["specdir"]             # 频谱文件初始化
                    )
                    specWrite = SpectrumWriter(specPath, datetimeRef)
                    # 输入：频谱计算所需的所有关键参数     输出：初始化的二进制文件（.bin）和信息文件（.info），.bin文件：高效存储大量数据，.info文件：人类可读的参数说明
                    specWrite.openBinaryForWriting(                 # 打开二进制文件.bin并写入头信息
                        "spectra",                         # 文件标识
                        declevel,                                   # 当前降采样层级
                        sampleFreqDec,                              # 当前层级采样率
                        winParams.getWindowSize(declevel),          # 窗口大小
                        winParams.getOverlap(declevel),             # 重叠大小
                        win.winOffset,                              # 相对于参考时间的窗口全局索引值
                        win.numWindows,                             # 总窗口数
                        dataChans,                                  # 通道名称列表
                    )
                    if options["ncores"] > 0:
                        specDataList = multiSpectra(
                            options["ncores"],
                            timeDataList,
                            sampleFreqDec,
                            winParams.getWindowSize(declevel),
                            projData.config.configParams,
                        )
                    else:                                           # specDataList用于存储所有窗口的频域数据(fft变换后)
                        specDataList = calculateWindowSpectra(      # 针对每一个窗口的时间数据计算频谱************************
                            timeDataList,                           # 2893个窗口的时间数据
                            sampleFreqDec,                          # 当前层级的采样频率（Hz）
                            winParams.getWindowSize(declevel),      # 窗口大小（样本点数）
                            projData.config.configParams,           # 处理配置参数
                        )
                    # write out to spectra file(写入频谱文件)
                    for iW in range(0, win.numWindows):
                        specWrite.writeBinary(specDataList[iW])             # 将内存中计算好的所有时间窗口的频谱数据依次写入到二进制文件中.bin，实现数据持久化存储。
                    specWrite.writeCommentsFile(timeData.getComments())     # 创建人类可读的注释文件.info，记录处理参数和历史
                    specWrite.closeFile()


def calculateWindowSpectra(
    timeDataList: List[TimeData],
    sampleFreq: float,
    windowSize: int,
    config: Union[Dict, None] = None,
):
    """Calculate spectra for a list of TimeData
    针对一个窗口的时间数据计算频谱
    Parameters
    ----------
    timeDataList : List[TimeData]
        A list of TimeData objects
    sampleFreq : float
        The sampling frequency of the TimeData
    windowSize : int
        The number of samples in the window
    
    Returns
    -------
    specDataList : List[SpectrumData]
        A list of spectra data
    """
    from resistics.spectra.calculator import SpectrumCalculator

    specCalc = SpectrumCalculator(sampleFreq, windowSize, config=config)        # 创建可重用的频谱计算器，为后续的大量数据计算做准备
    numWindows = len(timeDataList)
    specDataList = list()
    # loop though windows and calculate spectra(遍历每个窗口并计算频谱)
    for iW in range(0, numWindows):             # 对每个窗口时间数据进行fft
        timeData = timeDataList[iW]             # 获取当前窗口时域数据
        specDataList.append(specCalc.calcFourierCoeff(timeData))    # 计算频谱，对当前窗口的时域数据进行快速傅里叶变换（FFT）生成频谱数据
    return specDataList


def multiSpectra(
    ncores: int,
    timeDataList: List[TimeData],
    sampleFreq: float,
    windowSize: int,
    config: Dict[str, None] = None,
):
    """Multiprocessing of spectra

    Parameters
    ----------
    ncores: int
        The number of cores for multiprocessing
    timeDataList : List[TimeData]
        A list of TimeData objects
    sampleFreq : float
        The sampling frequency of the TimeData
    windowSize : int
        The number of samples in the window
    
    Returns
    -------
    specDataList : List[SpectrumData]
        A list of spectra data    
    """
    import multiprocessing as mp

    # separate time data into batches
    numWindows = len(timeDataList)
    batchSize = int(np.ceil(numWindows / ncores))
    batches = []
    sizes = []
    for iB in range(0, ncores):
        batchStartWin = iB * batchSize
        if batchStartWin >= numWindows:
            break
        batchEndWin = batchStartWin + batchSize
        if batchEndWin > numWindows:
            batchEndWin = numWindows
        batch = []
        for iW in range(batchStartWin, batchEndWin):
            batch.append(timeDataList[iW])
        batches.append(batch)
        sizes.append(str(len(batch)))
    # set up tuples
    multiTuples = [(batch, sampleFreq, windowSize, config) for batch in batches]
    # multiprocess
    projectText("Running spectra calculations on {} cores".format(ncores))
    projectText(
        "{} windows being run in {} batches with sizes {}".format(
            numWindows, len(batches), ", ".join(sizes)
        )
    )
    with mp.Pool(ncores) as pool:
        out = pool.starmap(calculateWindowSpectra, multiTuples)
    # format the output into a single list
    specDataList = []
    for outBatch in out:
        specDataList = specDataList + outBatch
    return specDataList


def viewSpectra(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> Union[Figure, None]:
    """View spectra for a measurement
    查看数据文件的频谱
    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view    
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    plotwindow : int, str, Dict, optional
        Windows to plot (local). If int, the window with local index plotwindow will be plotted. If string and "all", all the windows will be plotted if there are less than 20 windows, otherwise 20 windows throughout the whole spectra dataset will be plotted. If a dictionary, needs to have start and stop to define a range.
        要绘制的时间窗口。如果是整数，绘制指定索引的窗口；如果是字符串"all"，如果窗口数少于20个则绘制所有，否则绘制20个均匀分布的窗口；如果是字典，需包含start和stop定义范围。
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, then None is returned.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, colorbarMultiline

    options = {}
    options["chans"]: List[str] = ["Ex", "Ey", "Hx", "Hy", "Hz"]                        # 绘制需要的通道
    options["declevel"]: int = 0                                                        # 默认抽取级别0（原始采样率）
    options["plotwindow"]: Union[int, Dict, str] = [0]                                  # 默认绘制第一个窗口
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]        # 从配置获取频谱目录spectra
    options["show"]: bool = True
    options["save"]: bool = False
    options["plotoptions"]: Dict = plotOptionsSpec()                                    # 获取频谱绘图的默认选项
    options = parseKeywords(options, kwargs)                                            # 用户关键字参数覆盖默认值

    projectText("Plotting spectra for measurement {} and site {}".format(meas, site))   # 输出处理信息到项目日志
    specReader = getSpecReader(projData, site, meas, **options)                         # 获取频谱数据读取器
    if specReader is None:
        return None

    # channels
    dataChans = specReader.getChannels()                                                # 获取数据中可用的通道
    if len(options["chans"]) > 0:
        dataChans = options["chans"]                                                    # 如果用户指定了通道，使用指定的通道
    numChans = len(dataChans)                                                           # 计算通道数量

    # get windows
    numWindows = specReader.getNumWindows()                                             # 总窗口数
    sampleFreqDec = specReader.getSampleFreq()                                          # 当前抽取级别的采样频率

    # get the window data
    windows = options["plotwindow"]                                                     # 根据plotwindow参数选择要绘制的时间窗口
    if isinstance(windows, str) and windows == "all":                                   # 如果参数是字符串"all"，选择所有窗口或代表性窗口
        if numWindows > 20:
            windows = list(
                np.linspace(0, numWindows, 20, endpoint=False, dtype=np.int32)   # 如果窗口数>20，选择20个均匀分布的窗口
            )
        else:
            windows = list(np.arange(0, numWindows))                                        # 如果窗口数<=20，选择所有窗口
    elif isinstance(windows, int):                                                          # 如果参数是整数,将整数转换为列表
        windows = [windows]  # if an integer, make it into a list
    elif isinstance(windows, dict):                                                         # 如果参数是字典，将字典转换为列表,使用start和stop定义范围
        windows = list(np.arange(windows["start"], windows["stop"] + 1))

    # create a figure
    plotfonts = options["plotoptions"]["plotfonts"]                                         # 绘图字体配置
    cmap = colorbarMultiline()                                                              # 获取多线颜色映射（用于区分不同窗口）
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])                             # 创建图形对象，使用预设的图形尺寸
    for iW in windows:                                                      # 循环处理每个选中的时间窗口
        if iW >= numWindows:
            break
        if isinstance(options["plotwindow"], dict):
            color = cmap(iW / len(windows))                                       # 根据窗口索引获取颜色(实现颜色渐变)
        else:
            color = cmap(iW / numWindows)
        winData = specReader.readBinaryWindowLocal(iW)                      # 读取指定窗口的二进制频谱数据
        winData.view(                                                       # 绘制窗口的频谱数据
            fig=fig,
            chans=dataChans,
            label="{} to {}".format(                                        # 图例标签：显示时间范围
                winData.startTime.strftime("%m-%d %H:%M:%S"),         # 开始时间
                winData.stopTime.strftime("%m-%d %H:%M:%S"),                # 结束时间
            ),
            plotfonts=plotfonts,                                            # 字体设置
            color=color,                                                    # 线条颜色
        )

    st = fig.suptitle(                                                      # 设置图形总标题
        "Spectra plot, site = {}, meas = {}, fs = {:.2f} [Hz], decimation level = {:2d}".format(
            site, meas, sampleFreqDec, options["declevel"]            # 包含关键参数信息
        ),
        fontsize=plotfonts["suptitle"],                                     # 使用标题字体大小
    )
    st.set_y(0.98)                                                          # 调整标题位置（靠近顶部）

    # put on axis labels etc(加上坐标轴标签等)
    for idx, chan in enumerate(dataChans):                                      # 为每个通道设置子图属性
        ax = plt.subplot(numChans, 1, idx + 1)                            # 选择当前通道的子图,创建numChans行1列的子图布局
        plt.title("Amplitude {}".format(chan), fontsize=plotfonts["title"])     # 设置子图标题
        if len(options["plotoptions"]["amplim"]) == 2:                          # 设置Y轴范围（如果指定了幅度限制,这里没有）
            ax.set_ylim(options["plotoptions"]["amplim"])
        ax.set_xlim(0, specReader.getSampleFreq() / 2.0)                        # 设置X轴范围（0到奈奎斯特频率）
        plt.grid(True)                                                          # 添加网格线

    # fig legend and formatting(图例和格式设置)
    ax = plt.gca()                                                 # 获取当前轴（最后一个子图）
    h, l = ax.get_legend_handles_labels()                          # 获取图例句柄和标签
    fig.tight_layout(rect=[0.02, 0.02, 0.77, 0.92])                # 调整图形布局，为右侧图例留出空间。左侧2%，右侧23%空间
    # legend axis(创建专门的图例轴)
    legax = plt.axes(position=[0.77, 0.02, 0.23, 0.88], in_layout=False)                    # 右侧23%宽度
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom="False")         # 隐藏刻度
    plt.box(False)                                                                          # 隐藏边框
    legax.legend(h, l, loc="upper left", fontsize=plotfonts["legend"])                      # 在图例轴中添加图例(图的右侧)

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraData_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])                                     # block参数控制图形显示是否阻塞程序执行
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewSpectraSection(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> Union[Figure, None]:
    """View spectra section for a measurement

    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view    
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, then None is returned.
    """
    from matplotlib.colors import LogNorm                                               # 对数颜色映射

    from resistics.common.plot import savePlot, plotOptionsSpec, colorbar2dSpectra      # 绘图工具

    options = {}
    options["chans"] = ["Ex", "Ey", "Hx", "Hy", "Hz"]
    options["declevel"] = 0
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting spectra section for measurement {} and site {}".format(meas, site)
    )
    specReader = getSpecReader(projData, site, meas, **options)                         # 获取频谱数据读取器
    if specReader is None:                                                              # 如果未找到数据，返回None
        return None

    # channels
    dataChans = specReader.getChannels()                                                # 获取数据中可用的通道
    if len(options["chans"]) > 0:
        dataChans = options["chans"]                                                    # 如果用户指定了通道，使用指定的通道

    # get windows
    numWindows = specReader.getNumWindows()                                             # 总窗口数
    sampleFreqDec = specReader.getSampleFreq()                                          # 当前抽取级别的采样频率
    f = specReader.getFrequencyArray()                                                  # 获取频率数组(257)

    # if plotting a section, ignore plotwindow(如果绘制断面图,则忽略plotwindow)
    if numWindows > 250:                                                                                # 如果窗口数超过250，均匀采样250个窗口
        windows = list(np.linspace(0, numWindows, 250, endpoint=False, dtype=np.int32))      # 限制最大窗口数为250，避免图形过于密集
    else:
        windows = np.arange(0, numWindows)                                                              # 否则使用所有窗口

    # create figure(创建图形对象)
    plotfonts = options["plotoptions"]["plotfonts"]                                                     # 获取字体设置
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])                                         # 使用预设图形尺寸
    st = fig.suptitle(                                                                                  # 设置图形总标题（包含详细信息）
        "Spectra section, site = {}, meas = {}, fs = {:.2f} [Hz], decimation level = {:2d}, windows = {:d}, {} to {}".format(
            site,                                                                                 # 站点名称
            meas,                                                                                       # 测量编号
            sampleFreqDec,                                                                              # 采样频率（2位小数）
            options["declevel"],                                                                        # 抽取级别（2位整数）
            len(windows),                                                                               # 窗口数量
            windows[0],                                                                                 # 第一个窗口索引
            windows[-1],                                                                                # 最后一个窗口索引
        ),
        fontsize=plotfonts["suptitle"],                                                                 # 使用标题字体大小
    )
    st.set_y(0.98)                                                                                      # 调整标题位置（靠近顶部）

    # collect the data
    specData = np.empty(                                                                                # 创建数据存储数组
        shape=(len(windows), len(dataChans), specReader.getDataSize()), dtype="complex"                 # 形状：窗口数(250)×通道数(5)×频率点数(257)
    )
    dates = []                                                                      # 存储每个窗口的开始时间
    for idx, iW in enumerate(windows):                                              # 循环读取每个窗口的数据
        winData = specReader.readBinaryWindowLocal(iW)                              # 读取指定窗口的二进制数据
        for cIdx, chan in enumerate(dataChans):                                     # 存储每个通道的数据
            specData[idx, cIdx, :] = winData.data[chan]                             # 存储复数频谱数据
        dates.append(winData.startTime)                                             # 存储窗口开始时间

    ampLim = options["plotoptions"]["amplim"]                                       # 获取幅度范围限制
    for idx, chan in enumerate(dataChans):                                          # 为每个通道创建子图
        ax = plt.subplot(1, len(dataChans), idx + 1)                          # 创建子图：1行×通道数列，当前通道位置
        plotData = np.transpose(np.absolute(np.squeeze(specData[:, idx, :])))       # 提取当前通道(如Ex)的数据并转置，plotData形状为（频率×时间）。specData[:, idx, :]：获取当前通道的所有窗口数据。np.squeeze()：去除单维度。np.absolute()：取幅度（复数模）。np.transpose()：转置为频率×时间格式
        if len(ampLim) == 2:                                                        # 如果设置了幅度范围
            plt.pcolor(
                dates,                                                        # X轴：时间数组
                f,                                                                  # Y轴：频率数组
                plotData,                                                           # Z轴：幅度数据（颜色表示
                norm=LogNorm(vmin=ampLim[0], vmax=ampLim[1]),                       # 对数颜色映射，指定范围(用户指定的amplim)
                cmap=colorbar2dSpectra(),                                           # 使用专业的频谱颜色映射
            )
        else:                                                                       # 如果没有设置幅度范围，使用数据自动范围
            plt.pcolor(
                dates,
                f,
                plotData,
                norm=LogNorm(vmin=plotData.min(), vmax=plotData.max()),             # 自动范围(数据的最小最大值)
                cmap=colorbar2dSpectra(),
            )
        cb = plt.colorbar()                                                         # 添加颜色条
        cb.ax.tick_params(labelsize=plotfonts["axisTicks"])                         # 设置颜色条刻度字体
        # set axis limits(设置坐标轴范围)
        ax.set_ylim(0, specReader.getSampleFreq() / 2.0)                            # 设置Y轴范围（0到奈奎斯特频率）
        ax.set_xlim([dates[0], dates[-1]])                                          # 设置X轴范围（250个：第一个到最后一个时间点）
        if isMagnetic(chan):                                                                # 根据通道类型设置标题和单位
            plt.title("Amplitude {} [nT]".format(chan), fontsize=plotfonts["title"])        # 磁场通道
        else:
            plt.title("Amplitude {} [mV/km]".format(chan), fontsize=plotfonts["title"])     # 电场通道
        ax.set_ylabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])                    # Y轴标签
        ax.set_xlabel("Time", fontsize=plotfonts["axisLabel"])                              # X轴标签
        # set tick sizes(设置刻度标签字体大小)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(plotfonts["axisTicks"])                                      # 设置刻度标签字体大小
        plt.grid(True)                                                                      # 添加网格线

    # plot format
    fig.autofmt_xdate(rotation=90, ha="center")                                     # 自动调整日期标签（避免重叠）,旋转90度，水平对齐居中
    fig.tight_layout(rect=[0.02, 0.02, 0.96, 0.92])                                 # 调整图形布局，为标题和边距留出空间（左2%，右4%，上8%留白）

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraSection_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig


def viewSpectraStack(
    projData: ProjectData, site: str, meas: str, **kwargs
) -> Union[Figure, None]:
    """View spectra stacks for a measurement
    查看频谱堆叠图
    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        The site to view
    meas: str
        The measurement of the site to view
    chans : List[str], optional
        Channels to plot
    declevel : int, optional
        Decimation level to plot
    numstacks : int, optional
        The number of windows to stack
    coherences : List[List[str]], optional
        A list of coherences to add, specified as [["Ex", "Hy"], ["Ey", "Hx"]] 
    specdir : str, optional
        String that specifies spectra directory for the measurement
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict, optional
        Dictionary of plot options
    
    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed. If no data was found, then None is returned.
    """
    from resistics.common.plot import savePlot, plotOptionsSpec, colorbarMultiline

    options = {}
    options["chans"] = ["Ex", "Ey", "Hx", "Hy", "Hz"]                                   # 默认绘制5个标准通道
    options["declevel"] = 0                                                             # 默认抽取级别0（原始采样率）
    options["numstacks"] = 10                                                           # 要堆叠的窗口组数
    options["coherences"] = []                                                          # 默认不计算相干度
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]             # 从配置获取频谱目录
    options["show"] = True
    options["save"] = False
    options["plotoptions"] = plotOptionsSpec()                                          # 获取频谱绘图的默认选项
    options = parseKeywords(options, kwargs)

    projectText(
        "Plotting spectra stack for measurement {} and site {}".format(meas, site)
    )
    specReader = getSpecReader(projData, site, meas, **options)
    if specReader is None:
        return None

    # channels
    dataChans = specReader.getChannels()
    if len(options["chans"]) > 0:
        dataChans = options["chans"]
    numChans = len(dataChans)

    # get windows
    numWindows = specReader.getNumWindows()                                         # 总时间窗口数
    sampleFreqDec = specReader.getSampleFreq()                                      # 当前抽取级别的采样频率
    f = specReader.getFrequencyArray()                                              # 频率点数组

    # calculate num of windows to stack in each set(计算每组堆叠包含的窗口数)
    stackSize = int(np.floor(1.0 * numWindows / options["numstacks"]))                      # 向下取整,计算每组堆叠包含的窗口数(424)
    if stackSize == 0:                                                                      # 如果计算出的组大小为0（窗口太少），调整参数
        projectWarning(
            "Too few windows for number of stacks {}".format(options["numstacks"])
        )
        options["numstacks"] = numWindows                                                   # 调整堆叠组数为总窗口数（每个窗口一组）
        stackSize = 1                                                                       # 每组1个窗口
        projectWarning("Number of stacks changed to {}".format(options["numstacks"]))       # 输出调整后的参数

    # calculate number of rows - in case interested in coherences too(计算行数——若同时关注相干性分析)
    nrows = (                                                                               # 计算图形总行数（考虑幅度、相位和相干性子图）
        2                                                                                   # 如果没有相干性，只有幅度和相位两行
        if len(options["coherences"]) == 0
        else 2 + int(np.ceil(1.0 * len(options["coherences"]) / numChans))                  # 有相干性时增加行.np.ceil()向上取整,如-2.7 -> -2
    )

    # setup the figure
    plotfonts = options["plotoptions"]["plotfonts"]                                         # 绘图字体配置
    cmap = colorbarMultiline()                                                              # 多线颜色映射（用于区分不同堆叠组）
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])                             # 创建图形对象，使用预设尺寸
    st = fig.suptitle(                                                                      # 创建图形对象，使用预设尺寸
        "Spectra stack, fs = {:.6f} [Hz], decimation level = {:2d}, windows in each set = {:d}".format(
            sampleFreqDec, options["declevel"], stackSize                             # 采样频率（6位小数）,抽取级别（2位整数）,每组窗口数（1位整数）
        ),
        fontsize=plotfonts["suptitle"],                                                     # 使用标题字体大小
    )
    st.set_y(0.98)                                                                          # 调整标题位置（靠近顶部）

    # do the stacking(堆叠处理主循环,共10组的堆叠)
    for iP in range(0, options["numstacks"]):                                               # 循环处理每个堆叠组
        stackStart = iP * stackSize                                                         # 起始窗口索引(计算当前组的窗口范围)
        stackStop = min(stackStart + stackSize, numWindows)                                 # 结束窗口索引（不超过总窗口）
        color = cmap(iP / options["numstacks"])                                             # 为当前堆叠组分配颜色（实现颜色渐变）
        # dictionaries to hold data for this section(创建用于保存此部分数据的字典)
        stackedData = {}                                                                    # 堆叠平均数据（复数）
        ampData = {}                                                                        # 幅度数据
        phaseData = {}                                                                      # 相位数据（度）
        powerData = {}                                                                      # 功率数据（用于相干度计算）

        # assign initial zeros(数据存储初始化数值0)
        for c in dataChans:                                                                 # 为每个通道["Ex", "Ey", "Hx", "Hy", "Hz"]初始化数据数组
            stackedData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="complex")    # 堆叠平均数据（复数，保留相位信息）
            ampData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="float")        # 幅度数据（实数）
            phaseData[c] = np.zeros(shape=(specReader.getDataSize()), dtype="float")      # 相位数据（度）
            for c2 in dataChans:
                powerData[c + c2] = np.zeros(                                               # 为相干性计算初始化功率数据(如ExHy,EyHx)
                    shape=(specReader.getDataSize()), dtype="complex"
                )

        # now stack the data and create nice plots(窗口循环和数据累加)
        for iW in range(stackStart, stackStop):                                             # 循环处理当前组内的每个窗口
            winData = specReader.readBinaryWindowLocal(iW)                                  # 读取当前窗口的二进制数据
            for c in dataChans:                                                             # 处理每个通道的数据(每个频率对应的数据直接相加)
                stackedData[c] += winData.data[c]                                           # 累加复数频谱数据（用于堆叠平均）
                ampData[c] += np.absolute(winData.data[c])                                  # 累加幅度数据
                phaseData[c] += np.angle(winData.data[c]) * (180.0 / np.pi)                 # 累加相位数据（转换为度）。np.angle()得到弧度
                # get coherency data(计算功率谱数据（用于相干性）)
                for c2 in dataChans:
                    powerData[c + c2] += winData.data[c] * np.conjugate(                    # 累加互功率谱：c * conj(c2)   如:Ex×Ex*
                        winData.data[c2]
                    )
            # 记录时间范围（用于图例）
            if iW == stackStart:                                                            # 第一个窗口
                startTime = winData.startTime
            if iW == stackStop - 1:                                                         # 最后一个窗口
                stopTime = winData.stopTime

        # scale powers and stacks(对功率值进行缩放并执行堆叠操作,数据平均和归一化)
        ampLim = options["plotoptions"]["amplim"]                                           # 获取幅度范围限制
        for idx, c in enumerate(dataChans):                                                 # 对每个通道进行平均处理
            stackedData[c] = stackedData[c] / (stackStop - stackStart)                      # 计算堆叠频谱数据平均（除以窗口数）
            ampData[c] = ampData[c] / (stackStop - stackStart)                              # 计算平均幅度
            phaseData[c] = phaseData[c] / (stackStop - stackStart)                          # 计算平均相位
            for c2 in dataChans:
                # normalisation
                powerData[c + c2] = 2 * powerData[c + c2] / (stackStop - stackStart)        # 功率数据归一化（乘以2（双边谱->单边谱）并除以窗口数）。单边谱幅度是双边谱对应频率点的2倍（直流分量除外）,两者表示同一物理信号。
                # normalisation                                                             # 直流分量没有对应的负频率。奈奎斯特频率（f_nyq）是特殊情况。
                powerData[c + c2][[0, -1]] = powerData[c + c2][[0, -1]] / 2                 # 端点修正（0直流分量和nyq频率）。选择第一个(0)和最后一个(-1)频率点，0 -> 直流分量(DC, 0Hz)，-1 -> 奈奎斯特频率(f_nyq)

            # plot(幅度子图绘制)
            ax1 = plt.subplot(nrows, numChans, idx + 1)                               # 第一行，第idx+1个位置
            plt.title("Amplitude {}".format(c), fontsize=plotfonts["title"])
            h = ax1.semilogy(                                                               # 绘制频谱幅值图（X线性，Y对数）
                f,                                                                          # X轴：频率
                ampData[c],                                                                 # Y轴：平均幅度
                color=color,                                                                # 当前堆叠组的颜色
                label="{} to {}".format(                                                    # 图例标签：显示时间范围
                    startTime.strftime("%m-%d %H:%M:%S"),                             # 开始时间
                    stopTime.strftime("%m-%d %H:%M:%S"),                                    # 结束时间
                ),
            )
            if len(ampLim) == 2:                                                            # 设置Y轴范围
                ax1.set_ylim(ampLim)                                                        # 如果指定了范围
            else:
                ax1.set_ylim(0.001, 1000000)                                                    # 使用默认范围
            ax1.set_xlim(0, sampleFreqDec / 2.0)                                            # 设置X轴范围（0到奈奎斯特频率）
            if isMagnetic(c):                                                               # 设置Y轴标签（根据通道类型）
                ax1.set_ylabel("Amplitude [nT]", fontsize=plotfonts["axisLabel"])           # 磁场通道
            else:
                ax1.set_ylabel("Amplitude [mV/km]", fontsize=plotfonts["axisLabel"])        # 电场通道
            ax1.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])               # 设置X轴标签
            plt.grid(True)                                                                  # 添加网格

            # set tick sizes(设置坐标轴刻度字体大小)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])
            # plot phase(绘制相位图（线性坐标）)
            ax2 = plt.subplot(nrows, numChans, numChans + idx + 1)                    # 第二行
            plt.title("Phase {}".format(c), fontsize=plotfonts["title"])
            ax2.plot(                                                                       # 绘制相位图（线性坐标）
                f,                                                                          # X轴：频率
                phaseData[c],                                                               # Y轴：平均相位（度）
                color=color,
                label="{} to {}".format(
                    startTime.strftime("%m-%d %H:%M:%S"),
                    stopTime.strftime("%m-%d %H:%M:%S"),
                ),
            )
            ax2.set_ylim(-180, 180)                                                         # 设置相位范围（-180°到180°）
            ax2.set_xlim(0, sampleFreqDec / 2.0)                                            # 设置频率范围
            ax2.set_ylabel("Phase [degrees]", fontsize=plotfonts["axisLabel"])              # 设置Y轴标签
            ax2.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])               # 设置X轴标签
            plt.grid(True)                                                                  # 添加网格
            # set tick sizes(设置坐标轴刻度字体大小)
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])

        # plot coherences(绘制相干性子图（如果指定了相干性对）)
        for idx, coh in enumerate(options["coherences"]):
            c = coh[0]                                                                      # 第一个通道(Ex)
            c2 = coh[1]                                                                     # 第二个通道(Hy)
            # 计算相干度：|Pxy|² / (Pxx * Pyy)
            cohNom = np.power(np.absolute(powerData[c + c2]), 2)                            # 分子：|Pxy|²
            cohDenom = np.real(powerData[c + c] * powerData[c2 + c2])                       # 分母：Pxx * Pyy
            coherence = cohNom / cohDenom                                                   # 相干度
            ax = plt.subplot(nrows, numChans, 2 * numChans + idx + 1)                 # 创建相干度子图
            plt.title("Coherence {} - {}".format(c, c2), fontsize=plotfonts["title"])
            ax.plot(                                                                        # 绘制相干性曲线
                f,                                                                          # X轴：频率
                coherence,                                                                  # Y轴：相干性值（0-1）
                color=color,
                label="{} to {}".format(
                    startTime.strftime("%m-%d %H:%M:%S"),
                    stopTime.strftime("%m-%d %H:%M:%S"),
                ),
            )
            ax.set_ylim(0, 1.1)                                                             # 设置相干性范围（0-1.1，留出上边距）
            ax.set_xlim(0, sampleFreqDec / 2)                                               # 设置频率范围
            ax.set_ylabel("Coherence", fontsize=plotfonts["axisLabel"])                     # 设置坐标轴标签
            ax.set_xlabel("Frequency [Hz]", fontsize=plotfonts["axisLabel"])
            plt.grid(True)
            # set tick sizes(设置坐标轴刻度字体大小)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(plotfonts["axisTicks"])

    # fig legend and layout(图例和布局)
    ax = plt.gca()                                                                          # 获取当前轴
    h, l = ax.get_legend_handles_labels()                                                   # 获取图例句柄和标签
    fig.tight_layout(rect=[0.01, 0.01, 0.98, 0.81])                                         # 为顶部图例留出19%空间
    # legend(创建专门的图例轴)
    legax = plt.axes(position=[0.01, 0.82, 0.98, 0.12], in_layout=False)                    # 顶部图例区域
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)           # 隐藏刻度
    plt.box(False)                                                                          # 隐藏边框
    legax.legend(h, l, ncol=4, loc="upper center", fontsize=plotfonts["legend"])            # 添加图例（4列布局）

    # plot show and save
    if options["save"]:
        impath = projData.imagePath
        filename = "spectraStack_{}_{}_dec{}_{}".format(
            site, meas, options["declevel"], options["specdir"]
        )
        savename = savePlot(impath, filename, fig)
        projectText("Image saved to file {}".format(savename))
    if options["show"]:
        plt.show(block=options["plotoptions"]["block"])
    if not options["show"] and options["save"]:
        plt.close(fig)
        return None
    return fig
