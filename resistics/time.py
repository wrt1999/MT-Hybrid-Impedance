import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Union, List

from resistics.project.data import ProjectData
from resistics.time.reader import TimeReader
from resistics.time.writer_internal import TimeWriterInternal
from resistics.common.checks import parseKeywords, isElectric
from resistics.project.utils import projectText, projectBlock, checkDateOptions


def getTimeReader(projData: ProjectData, site: str, meas: str) -> TimeReader:
    """Get a data reader object for a measurement

    Data readers will then return a timeData object when given a sample range of data to return for

    Parameters
    ----------
    projData : ProjectData
        A project data object
    site : str
        The site for which to get the data reader
    meas : str
        The measurement in the site for which to get the data reader

    Returns
    -------
    DataReader
        A data reader object which allows a user to get time data
    """
    siteData = projData.getSiteData(site)
    return siteData.getMeasurement(meas)


def preProcess(projData: ProjectData, **kwargs) -> None:
    """Pre-process project time data

    Preprocess the time data using filters, notch filters, resampling or interpolation. A new measurement folder is created under the site. The name of the new measurement folder is:
    prepend_[name of input measurement]_postpend. By default, prepend is "proc" and postpend is empty. 

    Processed time series data can be saved in a new site by using the outputsite option.

    Parameters
    ----------
    projData : ProjectData
        A project data object
    sites : str, List[str], optional
        Either a single site or a list of sites
    sampleFreqs : int, float, List[float], optional
        The frequencies to preprocess
    start : str, optional
        Start date of data to preprocess in format "%Y-%m-%d %H:%M:%S"
    stop : str, optional
        Stop date of data to process in format "%Y-%m-%d %H:%M:%S"
    outputsite : str, optional
        A site to output the preprocessed time data to. If this site does not exist, it will be created
    polreverse :  Dict[str, bool]
        Keys are channels and values are boolean flags for reversing 
    scale : Dict[str, float]
        Keys are channels and values are floats to multiply the channel data by   
    calibrate : bool, optional
        Boolean flag for calibrating the data. Default is false and setting to True will calibrate where files can be found.
    normalise : bool, optional
        Boolean flag for normalising the data. Default is False and setting to True will normalise each channel independently.
    filter : Dict, optional
        Filtering options in a dictionary
    notch : List[float], optional
        List of frequencies to notch in spectra given as a list of floats
    resamp : Dict, optional
        Resampling parameters in a dictionary with entries in the format: {sampleRateFrom: sampleRateTo}. All measurement directories of sampleRateFrom will be resampled to sampleRateTo
    interp : bool, optional
        Boolean flag for interpolating the data on to the second, so that sampling is coincident with seconds. This is not always the case. For example, SPAM data is not necessarily sampled on the second, whereas ATS data is. This function is useful when combining data of multiple formats. Interpolation does not change the sampling rate. Default is False.
    prepend : str, optional
        String to prepend to the output folder. Default is "proc".
    postpend : str, optional
        String to postpend to the output folder. Default is empty.
    """
    from resistics.project.shortcuts import getCalibrator
    from resistics.project.preprocess import (
        applyPolarisationReversalOptions,
        applyScaleOptions,
        applyCalibrationOptions,
        applyFilterOptions,
        applyInterpolationOptions,
        applyNormaliseOptions,
        applyNotchOptions,
        applyResampleOptions,
    )

    options: Dict = {}
    options["sites"]: List = projData.getSites()
    options["sampleFreqs"]: List[float] = projData.getSampleFreqs()
    options["start"]: Union[bool, str] = False
    options["stop"]: Union[bool, str] = False
    options["outputsite"]: str = ""
    options["polreverse"]: Union[bool, Dict[str, bool]] = False
    options["scale"]: Union[bool, Dict[str, float]] = False
    options["calibrate"]: bool = False
    options["normalise"]: bool = False
    options["filter"]: Dict = {}
    options["notch"]: List[float] = []
    options["resamp"]: Dict = {}
    options["interp"]: bool = False
    options["prepend"]: str = "proc"
    options["postpend"]: str = ""
    options = parseKeywords(options, kwargs)

    # print info
    text: List = ["Processing with options"]
    for op, val in options.items():
        text.append("\t{} = {}".format(op, val))
    projectBlock(text)

    if isinstance(options["sites"], str):
        options["sites"] = [options["sites"]]

    # outputting to another site
    if options["outputsite"] != "":
        projectText(
            "Preprocessed data will be saved to output site {}".format(
                options["outputsite"]
            )
        )
        # create the site
        projData.createSite(options["outputsite"])
        projData.refresh()
        outputSitePath = projData.getSiteData(options["outputsite"]).timePath

    # output naming
    outPre = options["prepend"] + "_" if options["prepend"] != "" else ""
    outPost = "_" + options["postpend"] if options["postpend"] != "" else ""
    if outPre == "" and outPost == "" and options["outputsite"] == "":
        outPre = "proc_"

    # create a data calibrator writer instance
    cal = getCalibrator(projData.calPath, projData.config)
    if options["calibrate"]:
        cal.printInfo()
    writer = TimeWriterInternal()

    # format dates
    if options["start"]:
        options["start"] = datetime.strptime(options["start"], "%Y-%m-%d %H:%M:%S")
    if options["stop"]:
        options["stop"] = datetime.strptime(options["stop"], "%Y-%m-%d %H:%M:%S")

    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        siteData.printInfo()
        # loop over frequencies
        for sampleFreq in options["sampleFreqs"]:
            measurements = siteData.getMeasurements(sampleFreq)
            if len(measurements) == 0:
                # no data files at this sample rate
                continue

            # otherwise, process
            for meas in measurements:
                # get the reader
                projectText("Processing site {}, measurement {}".format(site, meas))
                reader = siteData.getMeasurement(meas)
                startTime = reader.getStartDatetime()
                stopTime = reader.getStopDatetime()
                if (options["start"] or options["stop"]) and not checkDateOptions(
                    options, startTime, stopTime
                ):
                    continue
                # if the data contributes, copy in the data if relevant
                if options["start"]:
                    startTime = options["start"]
                if options["stop"]:
                    stopTime = options["stop"]

                # calculate the samples
                sampleStart, sampleEnd = reader.time2sample(startTime, stopTime)
                # now get the data
                timeData = reader.getPhysicalSamples(
                    startSample=sampleStart, endSample=sampleEnd
                )
                timeData.printInfo()
                headers = reader.getHeaders()
                chanHeaders, _ = reader.getChanHeaders()

                # apply options
                applyPolarisationReversalOptions(options, timeData)
                applyScaleOptions(options, timeData)
                applyCalibrationOptions(options, cal, timeData, reader)
                applyFilterOptions(options, timeData)
                applyNotchOptions(options, timeData)
                applyInterpolationOptions(options, timeData)
                applyResampleOptions(options, timeData)
                applyNormaliseOptions(options, timeData)

                # output dataset path
                if options["outputsite"] != "":
                    timePath = outputSitePath
                else:
                    timePath = siteData.timePath
                outPath = os.path.join(timePath, "{}{}{}".format(outPre, meas, outPost))
                # write time data - need to manually change some headers (hence the keywords)
                writer = TimeWriterInternal()
                writer.setOutPath(outPath)
                writer.writeData(
                    headers,
                    chanHeaders,
                    timeData,
                    start_time=timeData.startTime.strftime("%H:%M:%S.%f"),
                    start_date=timeData.startTime.strftime("%Y-%m-%d"),
                    stop_time=timeData.stopTime.strftime("%H:%M:%S.%f"),
                    stop_date=timeData.stopTime.strftime("%Y-%m-%d"),
                    numSamples=timeData.numSamples,
                    sample_freq=timeData.sampleFreq,
                    physical=True,
                )
                writer.printInfo()


def viewTime(
    projData: ProjectData, startDate: str, endDate: str, **kwargs
) -> Union[Figure, None]:
    """View timeseries in the project
    查看项目中的时间序列数据
    Parameters
    ----------
    projData : ProjectData
        The project data instance
    startDate : str
        The start of the data range to plot
    endDate : str
        The end of the date range to plot
    sites : List[str], optional
        List of sites 
    sampleFreqs : List[float], optional
        List of sample frequencies to plot
    chans : List[str], optional
        List of channels to plot
    polreverse :  Dict[str, bool]
        Keys are channels and values are boolean flags for reversing   
    scale : Dict[str, float]
        Keys are channels and values are floats to multiply the channel data by   
    calibrate : bool, optional
        Boolean flag to calibrate data
    normalise : bool, optional
        Boolean flag to normalise the data. Default is False and setting to True will normalise each channel independently.
    notch : List[float], optional
        List of frequencies to notch out
    filter : Dict, optional
        Filter parameters
    show : bool, optional
        Boolean flag to show the plot
    save : bool, optional
        Boolean flag to save the plot to images folder
    plotoptions : Dict
        Dictionary of plot options

    Returns
    -------
    matplotlib.pyplot.figure or None
        A matplotlib figure unless the plot is not shown and is saved, in which case None and the figure is closed.
    """
    from resistics.project.shortcuts import getCalibrator
    from resistics.project.preprocess import (
        applyPolarisationReversalOptions,
        applyScaleOptions,
        applyCalibrationOptions,
        applyFilterOptions,
        applyNormaliseOptions,
        applyNotchOptions,
    )
    from resistics.common.plot import savePlot, plotOptionsTime
    # 设置默认参数
    options = {}
    options["sites"]: List[str] = projData.sites                                                # 默认所有测站
    options["sampleFreqs"]: Union[List[float], List[str]] = projData.getSampleFreqs()           # 默认所有采样频率
    options["chans"]: List[str] = ["Ex", "Ey", "Hx", "Hy", "Hz"]                                # 默认所有5个通道
    options["polreverse"]: Union[bool, Dict[str, bool]] = False                                 # 默认不反转极化
    options["scale"]: Union[bool, Dict[str, float]] = False                                     # 默认不缩放
    options["calibrate"]: bool = False                                                          # 默认不标定
    options["normalise"]: bool = False                                                          # 默认不归一化
    options["filter"]: Dict = {}                                                                # 默认无滤波
    options["notch"]: List[float] = []                                                          # 默认无陷波
    options["show"]: bool = True                                                                # 默认显示图形
    options["save"]: bool = False                                                               # 默认不保存
    options["plotoptions"]: Dict = plotOptionsTime()                                            # 获取时间序列绘图默认选项
    options = parseKeywords(options, kwargs)                                                    # 使用用户参数覆盖默认值

    # prepare calibrator(准备标定)
    cal = getCalibrator(projData.calPath, projData.config)              # 准备标定器,用于数据标定
    if options["calibrate"]:                                            # 如果需要标定，打印标定器信息
        cal.printInfo()

    # format startDate and endDate(格式化开始和结束时间字符串为datetime对象)
    start = datetime.strptime("{}.000".format(startDate), "%Y-%m-%d %H:%M:%S.%f")
    stop = datetime.strptime("{}.000".format(endDate), "%Y-%m-%d %H:%M:%S.%f")
    # collect relevant data - dictionary to store timeData(相关数据采集——用于存储时间序列数据的字典结构)
    timeDataAll = {}                                    # 创建字典存储所有时间数据
    for site in options["sites"]:                       # 遍历所有指定的测站
        siteData = projData.getSiteData(site)           # 获取测站数据对象
        if isinstance(siteData, bool):                  # 如果测站不存在，跳过
            # site does not exist
            continue
        siteData.printInfo()                            # 打印测站信息
        measurements = siteData.getMeasurements()       # 获取该测站的所有测量记录
        timeDataAll[site] = {}                          # 为该测站创建数据存储字典

        # loop over measurements and save data for each one(遍历该测站的所有测量记录)
        for meas in measurements:
            sampleFreq = siteData.getMeasurementSampleFreq(meas)        # 采样频率
            if sampleFreq not in options["sampleFreqs"]:                # 如果频率不在指定列表中，跳过
                continue

            # check if data contributes to user defined time period(检查数据是否涵盖用户定义的时间段)
            siteStart = siteData.getMeasurementStart(meas)          # siteStart:(2023,10,21,5,40,14)
            siteStop = siteData.getMeasurementEnd(meas)
            if siteStop < start or siteStart > stop:
                continue

            reader = siteData.getMeasurement(meas)                      # 获取该测量的数据读取器
            # get the samples of the datetimes(获取日期时间样本集)
            sampleStart, sampleStop = reader.time2sample(start, stop)           # 将时间转换为样本索引（找到对应时间点的数据位置）,sampleStart:0  sampleStop:1086705
            # as the samples returned from time2sample are rounded use sample2time to get the appropriate start and end times for those samples
            readStart, readStop = reader.sample2time(sampleStart, sampleStop)   # 将样本索引转换回精确时间(避免舍入误差)  readStart:(2023,10,21,5,40,14)
            # get the data for any available channels meaning even those sites with missing channels can be plotted
            timeData = reader.getPhysicalData(readStart, readStop)              # 读取物理数据（电压值，可能已部分处理）

            projectText(
                "Plotting measurement {} of site {} between {} and {}".format(
                    meas, site, readStart, readStop
                )
            )

            # apply various options(# 应用数据预处理流水线（按顺序执行）)
            applyPolarisationReversalOptions(options, timeData)             # 极化反转(默认False)
            applyScaleOptions(options, timeData)                            # 数据缩放(默认False)
            applyCalibrationOptions(options, cal, timeData, reader)         # 仪器标定(默认False)
            applyFilterOptions(options, timeData)                           # 滤波处理(默认{})
            applyNotchOptions(options, timeData)                            # 陷波去噪(默认[])
            applyNormaliseOptions(options, timeData)                        # 数据归一化(默认False)
            timeDataAll[site][meas] = timeData                              # 将处理后的数据存储到字典中

    # plot all the data
    plotfonts = options["plotoptions"]["plotfonts"]                         # 获取绘图字体设置
    fig = plt.figure(figsize=options["plotoptions"]["figsize"])             # 创建图形对象，使用配置的图形尺寸
    for site in timeDataAll:                                                # 遍历所有收集到的数据的站点并进行可视化
        for meas in timeDataAll[site]:                                      # 遍历该测站的每个测量
            timeData = timeDataAll[site][meas]                              # 获取时间数据对象
            timeData.view(                                                  # 调用时间数据对象的view方法进行绘图
                sampleStop=timeDataAll[site][meas].numSamples - 1,          # 绘制所有样本
                fig=fig,                                                    # 在同一个图形上绘制
                chans=options["chans"],                                     # 指定要绘制的通道
                label="{} - {}".format(site, meas),                   # 图例标签：测站-测量
                xlim=[start, stop],                                         # X轴时间范围
                plotfonts=plotfonts,                                        # 字体设置
            )

    # add the suptitle(添加图形总标题)
    st = fig.suptitle(
        "Time data from {} to {}".format(
            start.strftime("%Y-%m-%d %H-%M-%S"), stop.strftime("%Y-%m-%d %H-%M-%S")     # 格式化的开始时间,格式化的结束时间
        ),
        fontsize=plotfonts["suptitle"],                 # 使用标题字体大小
    )
    st.set_y(0.98)                                      # 调整标题垂直位置（避免与子图重叠）

    # do the axis labels（设置Y轴标签和范围）
    numChans = len(options["chans"])                             # 通道数量
    # enumerate：同时获取通道名和索引（从0开始）                       # 第一次循环：idx=0, chan="Ex"  第二次循环：idx=1, chan="Ey"
    for idx, chan in enumerate(options["chans"]):                # 遍历每个通道，设置对应的Y轴属性。功能：逐个处理每个通道的绘图设置
        plt.subplot(numChans, 1, idx + 1)                  # 选择当前通道的子图。创建垂直排列的子图(numChans：总行数（子图数量）,1：列数,idx + 1：当前子图位置（Matplotlib索引从1开始）)
        # do the yaxis(设置Y轴标签和范围)
        if isElectric(chan):                                                # 判断是否为电场通道
            plt.ylabel("mV/km", fontsize=plotfonts["axisLabel"])   # 电场单位。设置电场通道Y轴标签,设置Y轴标签为 "mV/km",使用 plotfonts["axisLabel"]指定的字体大小
            if len(options["plotoptions"]["Eylim"]) > 0:         # 如果设置了Y轴范围(这里没有)
                plt.ylim(options["plotoptions"]["Eylim"])        # 强制应用电场Y轴范围(保多幅图的Y轴范围一致，便于比较)
        else:                                                               # 磁场通道
            if options["calibrate"]:                                        # 如果进行了标定
                plt.ylabel("nT", fontsize=plotfonts["axisLabel"])     # 标定后单位
            else:
                plt.ylabel("mV", fontsize=plotfonts["axisLabel"])     # 原始电压单位
            if len(options["plotoptions"]["Hylim"]) > 0:                    # 如果设置了Y轴范围
                plt.ylim(options["plotoptions"]["Hylim"])                   # 应用磁场Y轴范围
        plt.legend(loc=1, fontsize=plotfonts["legend"])                     # 添加图例（右上角位置）

    # plot format（自动调整图形布局）
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])                           # 为标题预留空间
    fig.subplots_adjust(top=0.92)                                       # 进一步调整顶部间距

    # plot show and save
    if options["save"]:                                                 # 图形保存逻辑(如果需要保存)
        impath = projData.imagePath                                     # 获取图像保存路径
        filename = "timeData_{}_{}".format(                             # 生成文件名
            start.strftime("%Y-%m-%d_%H-%M-%S_"), stop.strftime("%Y-%m-%d_%H-%M-%S")    # 开始时间,结束时间
        )
        savename = savePlot(impath, filename, fig)                      # 保存图形
        projectText("Image saved to file {}".format(savename))          # 输出保存信息
    if options["show"]:                                                 # 图形显示逻辑(如果需要显示)
        plt.show(block=options["plotoptions"]["block"])                 # 显示图形（可阻塞或不阻塞）
    if not options["show"] and options["save"]:                         # 如果不显示只保存，关闭图形并返回None
        plt.close(fig)                                                  # 关闭图形释放内存
        return None                                                     # 返回None
    return fig                                                          # 正常情况返回图形对象
