import sys
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Union, List, Dict

from resistics.common.checks import parseKeywords, isElectric, isMagnetic
from resistics.common.io import checkFilepath, fileFormatSampleFreq
from resistics.common.print import arrayToString
from resistics.project.data import ProjectData
from resistics.project.utils import projectText, projectWarning, projectError
from resistics.site.data import SiteData
from resistics.transfunc.data import TransferFunctionData


def getTransferFunctionData(
    projData: ProjectData, site: str, sampleFreq: float, **kwargs
) -> TransferFunctionData:
    """Get transfer function data
    获取传输函数(阻抗)数据
    Parameters
    ----------
    projData : projecData
        The project data
    site : str
        Site to get the transfer functiond data for
    sampleFreq : int, float
        The sampling frequency for which to get the transfer function data
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    """
    from resistics.transfunc.io import TransferFunctionReader
    # 创建空字典存储配置选项
    options: Dict = dict()
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]    # 从项目配置中获取默认的频谱数据目录
    options["postpend"]: str = ""                   # 设置默认的后续标识为空字符串，用于区分不同处理阶段的文件，如"_remote"表示远程参考处理后
    options = parseKeywords(options, kwargs)        # 使用用户传入的参数覆盖默认选项

    # deal with the postpend(处理文件名的后续标识部分)
    if options["postpend"] != "":
        postpend = "_{}".format(options["postpend"])        # 如果有指定的postpend，在前面添加下划线
    else:
        postpend = options["postpend"]

    siteData = projData.getSiteData(site)                   # 获取指定测站的数据对象
    sampleFreqStr = fileFormatSampleFreq(sampleFreq)        # 将采样频率格式化为文件名字符串(15.0 -> 15_000)
    path = os.path.join(                                    # 构建完整的文件路径（关键步骤！）
        siteData.transFuncPath,                             # 测站的传递函数数据根目录
        "{:s}".format(sampleFreqStr),                       # 采样频率子目录（每种采样率一个文件夹,如:15_000）
        "{}_fs{:s}_{}{}".format(site, sampleFreqStr, options["specdir"], postpend),         # 具体的文件名,如：site1_fs15_000_spectra
    )
    # check path(检查构建的文件路径是否存在)(一般存在)
    if not checkFilepath(path):
        projectWarning("No transfer function file with name {}".format(path))
        return False
    # 如果文件存在，输出读取信息的日志
    projectText(
        "Reading transfer function for site {}, sample frequency {}, file {}".format(
            site, sampleFreq, path
        )
    )
    # 创建传递函数读取器对象
    tfReader = TransferFunctionReader(path)
    tfReader.printInfo()            # 打印数据信息
    return tfReader.tfData          # 返回读取的传递函数(阻抗(Z-ExHx、Z-ExHy、Z-EyHx、Z-EyHy)、freq、period、极化(ExHx、ExHy、EyHx、EyHy)，var)数据对象

    # **kwargs：可选参数，覆盖默认处理设置
def processProject(projData: ProjectData, **kwargs) -> None:
    """Process a project
    处理一个项目
    Parameters
    ----------
    projData : ProjectData
        The project data instance for the project    
    sites : List[str], optional
        List of sites 
    sampleFreqs : List[float], optional
        List of sample frequencies to process
    specdir : str, optional
        The spectra directories to use
    inchans : List[str], optional
        Channels to use as the input of the linear system
    inputsite : str, optional
        Site from which to take the input channels. The default is to use input and output channels from the same site        
    outchans : List[str], optional
        Channels to use as the output of the linear system
    remotesite : str, optional
        The site to use as the remote site
    remotechans : List[str], optional
        Channels to use from the remote reference site
    crosschannels : List[str], optional
        List of channels to use for cross powers
    masks : Dict, optional
        Masks dictionary for passing mask data. The key should be a site name and the value should either be a string for a single mask or a list of multiple masks.
    datetimes : List, optional
        List of datetime constraints, each one as a dictionary. For example [{"type": "datetime", "start": 2018-08-08 00:00:00, "end": 2018-08-08 16:00:00, "levels": [0,1]}]. Note that levels is optional.            
    postpend : str, optional
        String to postpend to the transfer function output
    ncores : int, optional
        The number of cores to run the transfer function calculations on        
    """
    options: Dict = dict()
    options["sites"]: List[str] = projData.getSites()
    options["sampleFreqs"]: List[float] = projData.getSampleFreqs()
    options["specdir"]: str = projData.config.configParams["Spectra"]["specdir"]
    options["inchans"]: List[str] = ["Hx", "Hy"]            # 输入通道：["Hx", "Hy"]（水平磁场）
    options["inputsite"]: str = ""
    options["outchans"]: List[str] = ["Ex", "Ey"]           # 输出通道：["Ex", "Ey"]（水平电场）
    options["remotesite"]: str = ""
    options["remotechans"]: List[str] = options["inchans"]
    options["crosschannels"]: List[str] = []
    options["masks"]: Dict = {}
    options["datetimes"]: List = []
    options["postpend"]: str = ""
    options["ncores"] = projData.config.getSolverCores()    # 并行核数：从项目配置读取 (getSolverCores())
    options = parseKeywords(options, kwargs)                # 功能：用于合并默认配置和用户自定义配置，返回值：合并后的选项字典（用户参数覆盖默认值）。(options：默认选项字典（预设的默认值），kwargs：用户提供的关键字参数（函数调用时的**kwargs）)

    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        siteFreqs = siteData.getSampleFreqs()
        for sampleFreq in siteFreqs:
            # check if not included
            if sampleFreq not in options["sampleFreqs"]:
                continue
            processSite(projData, site, sampleFreq, **options)              # 处理一个站点的单个采样频率


def processSite(
    projData: ProjectData, site: str, sampleFreq: Union[int, float], **kwargs
):
    """Process a single sampling frequency for a site
    处理一个站点的单个采样频率
    The site passed is assumed to be the output site (the output channels will come from this site). If channels from a different site are desired to be used as the input channels, this can be done by specifying the optional inputsite argument.
    所传入的站点默认被视为输出站点（输出通道将来源于该站点）。若需将其他站点的通道用作输入通道，可通过指定可选参数 inputsite 实现
    .. todo:: 
    
        Give a few different examples here
        这里给出几个不同的例子
    Parameters
    ----------
    projData : ProjectData
        The project data instance for the project
    site : str
        Site to process 
    sampleFreq : float, int
        Sample frequency to process
    specdir : str, optional
        The spectra directories to use
    inchans : List[str], optional
        Channels to use as the input of the linear system
    inputsite : str, optional
        Site from which to take the input channels. The default is to use input and output channels from the same site
    outchans : List[str], optional
        Channels to use as the output of the linear system
    remotesite : str, optional
        The site to use as the remote site
    remotechans : List[str], optional
        Channels to use from the remote reference site
    crosschannels : List[str], optional
        List of channels to use for cross powers
    masks : Dict, optional
        Masks dictionary for passing mask data. The key should be a site name and the value should either be a string for a single mask or a list of multiple masks.
    datetimes : List, optional
        List of datetime constraints, each one as a dictionary. For example [{"type": "datetime", "start": 2018-08-08 00:00:00, "end": 2018-08-08 16:00:00, "levels": [0,1]}]. Note that levels is optional.
    postpend : str, optional
        String to postpend to the transfer function output
    ncores : int, optional
        The number of cores to run the transfer function calculations on
    """
    from resistics.decimate.decimator import Decimator
    from resistics.window.selector import WindowSelector
    from resistics.project.shortcuts import (
        getDecimationParameters,
        getWindowParameters,
        getWindowSelector,
        getLocalRegressor,
        getRemoteRegressor,
    )

    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["inchans"] = ["Hx", "Hy"]
    options["inputsite"] = ""
    options["outchans"] = ["Ex", "Ey"]
    options["remotesite"] = ""
    options["remotechans"] = options["inchans"]
    options["crosschannels"] = []
    options["masks"] = {}
    options["datetimes"] = []
    options["postpend"] = ""
    options["ncores"] = projData.config.getSolverCores()
    options = parseKeywords(options, kwargs)
    if options["inputsite"] == "":
        options["inputsite"] = site

    projectText("Processing site {}, sampling frequency {}".format(site, sampleFreq))
    siteData = projData.getSiteData(site)

    # define decimation parameters(定义降采样参数)
    decParams = getDecimationParameters(sampleFreq, projData.config)            # decFactors，decFrequencies，evalFreq，evalFreqPerLevel
    decParams.printInfo()
    winParams = getWindowParameters(decParams, projData.config)                 # windows = [512 512 ... 512]  overlaps = [128 128 ... 128]
    # window selector(窗口选择器)
    winSelector = getWindowSelector(projData, decParams, winParams, options["specdir"])     # 根据 projectData、decimationParams 和 windowParams 创建一个 WindowSelector 对象(多为空字典和列表)

    # if two sites are duplicated (e.g. input site and output site), winSelector only uses distinct sites. Hence using site and inputSite is no problem even if they are the same(若两个站点存在重复（例如输入站点与输出站点），winSelector仅采用不同的站点。因此，即便site与inputSite相同，使用二者也无问题)
    processSites = []
    if options["remotesite"]:
        processSites = [site, options["inputsite"], options["remotesite"]]
        winSelector.setSites(processSites)
    else:
        # if no remote site, then single site processing(如果没有远程站点，则进行单站点处理)
        processSites = [site, options["inputsite"]]
        winSelector.setSites(processSites)              # 设置用于查找共享窗口的站点集合

    # add window masks(添加窗函数掩码，暂时没用到)
    if len(list(options["masks"].keys())) > 0:
        for maskSite in options["masks"]:
            if maskSite not in processSites:
                # there is a site in the masks dictionary which is of no interest
                continue
            if isinstance(options["masks"][maskSite], str):
                # a single mask
                winSelector.addWindowMask(maskSite, options["masks"][maskSite])
                continue
            if all(isinstance(item, str) for item in options["masks"][maskSite]):
                # list of masks for the site
                for mask in options["masks"][maskSite]:
                    winSelector.addWindowMask(maskSite, mask)

    # add datetime constraints(添加日期时间约束，没用到)
    for dC in options["datetimes"]:
        levels = None
        if "levels" in dC:
            levels = dC["levels"]

        if dC["type"] == "datetime":
            winSelector.addDatetimeConstraint(dC["start"], dC["stop"], levels)
        if dC["type"] == "time":
            winSelector.addTimeConstraint(dC["start"], dC["stop"], levels)
        if dC["type"] == "date":
            winSelector.addDateConstraint(dC["date"], levels)

    # calculate the shared windows and print info(计算共享窗函数区间并输出信息)
    winSelector.calcSharedWindows()                     # 计算各站点之间的共享窗口(暂时只有1个站点，所以全部窗口都是共享窗口)
    winSelector.printInfo()                             # WindowSelector:################
    winSelector.printDatetimeConstraints()              # WindowSelector::datetime constraints: ####################
    winSelector.printWindowMasks()                      # WindowSelector::window masks: ####################
    winSelector.printSharedWindows()                    # WindowSelector::shared windows: ####################
    winSelector.printWindowsForFrequency()              # WindowSelector::windows for frequency: ####################

    # now have the windows, pass the winSelector to processors(当前已获取目标窗口，将窗口选择器（winSelector）传递至处理器模块（processors）)
    outPath = siteData.transFuncPath
    if options["remotesite"]:
        projectText(
            "Remote reference processing with sites: in = {}, out = {}, reference = {}".format(
                options["inputsite"], site, options["remotesite"]
            )
        )
        processor = getRemoteRegressor(winSelector, outPath, projData.config)
        processor.setRemote(options["remotesite"], options["remotechans"])
    else:
        projectText(
            "Single site processing with sites: in = {}, out = {}".format(
                options["inputsite"], site
            )
        )
        processor = getLocalRegressor(winSelector, outPath, projData.config)        # 创建processor，用于保存各种参数

    # add the input and output site(添加输入和输出站点)
    processor.setInput(options["inputsite"], options["inchans"])        # ['Hx','Hy']
    processor.setOutput(site, options["outchans"])                      # ['Ex','Ey']
    if len(options["crosschannels"]) > 0:
        processor.crossChannels = options["crosschannels"]
    processor.postpend = options["postpend"]                            # postpend输出文件后缀添加操作
    processor.printInfo()
    projectText("Processing data using {} cores".format(options["ncores"]))
    processor.process(options["ncores"])                                # 处理每个窗口的频谱数据，并计算传输函数transferfunction


def viewImpedance(projData: ProjectData, **kwargs) -> List[Figure]:
    """View impedance tensor data
    查看张量阻抗数据
    Parameters
    ----------
    projData : projecData
        The project data
    sites : List[str], optional
        List of sites to plot transfer functions for
    sampleFreqs : List[float], optional 
        List of samples frequencies for which to plot transfer functions
    polarisations : List[str], optional 
        A list of polarisations to plot. For example, ["ExHx", "ExHy", "EyHx", "EyHy"]
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    oneplot : bool, optional
        Plot the polarisation on a single plot
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict
        A dictionary of plot options. For example, set the resistivity y limits using res_ylim, set the phase y limits using phase_ylim and set the xlimits using xlim
    """
    from resistics.common.plot import (
        savePlot,
        plotOptionsTransferFunction,
        getTransferFunctionFigSize,
        transferFunctionColours,
    )
    # 设置默认参数字典
    options = {}
    options["sites"] = projData.getSites()              # 从项目数据获取所有测站名称列表
    options["sampleFreqs"] = projData.getSampleFreqs()      # 采样频率列表
    options["polarisations"] = ["ExHx", "ExHy", "EyHx", "EyHy"]     # 标准张量阻抗的4个分量
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]     # 频谱数据存储目录
    options["postpend"] = ""            # 传输函数文件名后缀（用于区分不同处理版本）
    options["oneplot"] = True           # 控制是否将所有极化模式（如ExHx, EyHy等）的传递函数绘制在同一张图中
    options["save"] = False             # 不自动保存图片
    options["show"] = True              # 显示图形
    options["plotoptions"] = plotOptionsTransferFunction()          # 获取绘图默认配置（坐标轴范围、字体等）**************************************
    options = parseKeywords(options, kwargs)                        # 使用用户传入的参数覆盖默认值

    # loop over sites(遍历站点)
    figs = []
    for site in options["sites"]:                               # 站点遍历处理
        siteData = projData.getSiteData(site)                   # 获取当前站点的数据对象
        sampleFreqs = set(siteData.getSampleFreqs())            # 获取该站点实际存在的采样频率，并转换为集合便于交集运算{15.0}
        # find the intersection with the options["freqs"](找到与 options["freqs"] 的交集)
        sampleFreqs = sampleFreqs.intersection(options["sampleFreqs"])          # 计算实际可用的频率：站点频率与用户指定频率的交集
        sampleFreqs = sorted(list(sampleFreqs))                                 # 将集合转换回列表并排序（频率从低到高）,[15.0]

        # if prepend is a string, then make it a list(如果postpend是字符串，转换为列表，统一处理接口)       文件后缀处理
        if isinstance(options["postpend"], str):
            options["postpend"] = [options["postpend"]]     # 统一转为列表

        plotfonts = options["plotoptions"]["plotfonts"]     # 从绘图选项中获取字体设置
        # now loop over the postpend options(循环处理每个文件名后缀)
        for pp in options["postpend"]:
            # add an underscore if not empty(如果后缀非空，添加下划线前缀，否则保持空字符串)
            postpend = "_{}".format(pp) if pp != "" else pp
            # 自动计算或使用指定的图形尺寸
            if options["plotoptions"]["figsize"] is None:
                figsize = getTransferFunctionFigSize(
                    options["oneplot"], len(options["polarisations"])           # 根据是否单图模式和极化数量计算合适尺寸
                )
            else:
                figsize = options["plotoptions"]["figsize"]
            fig = plt.figure(figsize=figsize)                       # 创建图形对象(大小为figsize的空图)
            mks = ["o", "*", "d", "^", "h"]                         # 标记符号循环使用，区分不同频率的数据点(圆圈、星形、菱形、三角形、六边形)
            lstyles = ["solid", "dashed", "dashdot", "dotted"]      # 线型循环使用，增强可读性(实线、虚线、点划线、点线)
            colours = transferFunctionColours()                     # 获取标准颜色映射，确保颜色一致性("ExHx": "orange","EyHy": "green",...)

            # loop over sampling frequencies(记录实际找到数据的频率点)
            includedFreqs = []
            for idx, sampleFreq in enumerate(sampleFreqs):          # 循环处理每个采样频率
                # 加载当前站点、当前采样频率15Hz的传输函数数据(阻抗(Z-ExHx、Z-ExHy、Z-EyHx、Z-EyHy)、freq、period、极化方向(ExHx、ExHy、EyHx、EyHy)，var)
                tfData = getTransferFunctionData(
                    projData, site, sampleFreq, specdir=options["specdir"], postpend=pp
                )
                if not tfData:          # 如果数据加载失败，跳过该频率
                    continue
                # 记录成功加载的频率(15.0)
                includedFreqs.append(sampleFreq)
                projectText(
                    "Plotting transfer function for site {}, sample frequency {}".format(
                        site, sampleFreq
                    )
                )

                # plot(循环选择标记和线型（避免样式重复）)
                mk = mks[idx % len(mks)]            # 循环选择标记符号
                ls = lstyles[idx % len(lstyles)]    # 循环选择线型
                tfData.viewImpedance(               # 绘制图形(计算各个张量阻抗的视电阻率和相位，并绘制图形)，调用阻抗可视化方法（核心绘图功能）
                    fig=fig,                                                # 要绘制的图形对象（空图）
                    polarisations=options["polarisations"],                 # 要绘制的分量(ExHx、ExHy、EyHx、EyHy)
                    mk=mk,                                                  # 数据点标记符号
                    ls=ls,                                                  # 连线线型
                    colours=colours,                                        # 颜色映射
                    oneplot=options["oneplot"],                             # 是否单图显示
                    res_ylim=options["plotoptions"]["res_ylim"],            # 视电阻率Y轴范围
                    phase_ylim=options["plotoptions"]["phase_ylim"],        # 相位Y轴范围
                    xlim=options["plotoptions"]["xlim"],                    # X轴（频率）范围
                    label="{}".format(sampleFreq),                          # 图例标签（频率值）
                    plotfonts=options["plotoptions"]["plotfonts"],          # 字体设置
                )

            # check if any files found(检查是否找到任何有效数据)
            if len(includedFreqs) == 0:
                continue                    # 无数据时跳过该站点的后续处理

            # sup title(设置图形总标题)
            # ---------------------------------------------------------
            # 步骤 1: 将自适应布局引擎提前运算，锁定物理结构空间
            # ---------------------------------------------------------
            fig.tight_layout()  # 计算所有文本包围盒，重组留白空间边界
            fig.align_ylabels()  # 在上下双子图中对齐 Y 轴物理基线
            fig.subplots_adjust(top=0.93)  # 全局手动下压子图上缘边界，预留顶部标题安全区

            # ---------------------------------------------------------
            # 步骤 2: 在布局拓扑已锁固后提取坐标，动态注射全局超级标题
            # ---------------------------------------------------------
            sub = " "
            # sub = "(d) Fast and Robust Bootstrap"
            # 显式设定 x 锚点与对齐法则
            st = fig.suptitle(
                sub,
                fontsize=plotfonts["suptitle"],
                x=fig.subplotpars.left,  # 获取布局算法重置后最终生效的子图矩阵最左侧浮点坐标值
                horizontalalignment="left"  # 取消默认的 center 锚固，强制文本左边界钉入坐标标尺 x
            )
            st.set_y(0.99)  # 恢复极顶部的垂直高度位置

            figs.append(fig)  # 图表状态机封闭，向后传递

            # sub = "Site {}: {}".format(site, options["specdir"] + postpend)               # 构建主标题字符串的第一行。将站点名称（site）和正在使用的数据谱目录名及后缀（specdir + postpend）格式化填入字符串中。
            # sub = "{}\nfs = {}".format(sub, arrayToString(includedFreqs, decimals=3))     # 在原有的标题字符串（sub）后加上换行符 "\n"，并追加实际包含的采样频率（includedFreqs）。其中调用了 arrayToString 函数将频率数组转为字符串，并保留3位小数。
            # st = fig.suptitle(sub, fontsize=plotfonts["suptitle"])                          # 使用 Matplotlib 的 suptitle 方法为整个画布（Figure）添加居中的总标题，标题的字体大小由 plotfonts["suptitle"] 动态控制。
            # st.set_y(0.99)                      # 将总标题在画布 Y 轴上的相对位置设定为 0.99（即非常靠近画布最顶部），为了避免标题和下方图表元素挤在一起
            # fig.tight_layout()                  # 调用 Matplotlib 的 tight_layout 方法，系统会自动计算并调整各子图的边距，防止坐标轴标签、刻度或图例相互遮挡。
            # fig.align_ylabels()               # <--- 新增此行：自动对齐同一列中所有子图的 Y 轴标签
            # fig.subplots_adjust(top=0.93)       # 手动微调子图布局，将所有子图区域的顶部边界限制在画布 92% 的高度处，从而在顶部专门为刚才添加的总标题留出空白空间。
            # figs.append(fig)                    # 将图形添加到返回列表

            # 如果设置了保存选项，保存图形到文件
            if options["save"]:
                impath = projData.imagePath     # 获取图像保存路径
                filename = "transFunction_{}_{}{}".format(
                    site, options["specdir"], postpend
                )
                savename = savePlot(impath, filename, fig)                  # 调用保存函数
                projectText("Image saved to file {}".format(savename))      # 输出保存信息
        # 显示控制：如果不显示则关闭所有图形释放内存
        if not options["show"]:
            plt.close("all")
        else:
            plt.show(block=options["plotoptions"]["block"])                 # 显示图形，block参数控制是否阻塞程序

    return figs             # 返回所有生成的图形对象


def viewTipper(projData: ProjectData, **kwargs) -> List[Figure]:
    """View transfer function data
    查看传递函数数据（主要是Tipper数据）
    Parameters
    ----------
    projData : projecData
        The project data
    sites : List[str], optional
        List of sites to plot transfer functions for
    sampleFreqs : List[float], optional 
        List of samples frequencies for which to plot transfer functions
    specdir : str, optional
        The spectra directories used
    postpend : str, optional
        The postpend on the transfer function files
    cols : bool, optional
        Boolean flag, True to arrange tipper plot as 1 row with 3 columns
    show : bool, optional
        Show the spectra plot
    save : bool, optional
        Save the plot to the images directory
    plotoptions : Dict
        A dictionary of plot options. For example, set the resistivity y limits using res_ylim, set the phase y limits using phase_ylim and set the xlimits using xlim
    """
    from resistics.common.plot import (
        savePlot,
        plotOptionsTipper,
        getTransferFunctionFigSize,
        transferFunctionColours,
    )

    options = {}
    options["sites"] = projData.getSites()
    options["sampleFreqs"] = projData.getSampleFreqs()
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options["postpend"] = ""
    options["cols"] = True                              # 默认使用列布局（1行3列）
    options["save"] = False
    options["show"] = True
    options["plotoptions"] = plotOptionsTipper()        # 默认使用列布局（1行3列）
    options = parseKeywords(options, kwargs)

    # loop over sites
    figs = []
    for site in options["sites"]:
        siteData = projData.getSiteData(site)
        sampleFreqs = set(siteData.getSampleFreqs())
        # find the intersection with the options["freqs"](与用户指定的频率求交集（过滤不存在的频率）)
        sampleFreqs = sampleFreqs.intersection(options["sampleFreqs"])
        sampleFreqs = sorted(list(sampleFreqs))

        # if prepend is a string, then make it a list(将prepend字符串转换成列表)
        if isinstance(options["postpend"], str):
            options["postpend"] = [options["postpend"]]

        plotfonts = options["plotoptions"]["plotfonts"]             # 从绘图选项中获取字体设置
        # now loop over the postpend options(循环处理每个后缀配置（支持同一站点多个处理版本对比）)
        for pp in options["postpend"]:
            # add an underscore if not empty(添加下划线前缀（如果后缀非空）)
            postpend = "_{}".format(pp) if pp != "" else pp                 # _with_Hz

            fig = plt.figure(figsize=options["plotoptions"]["figsize"])     # 创建图形对象，使用预设的图形尺寸
            mks = ["o", "*", "d", "^", "h"]             # 定义标记样式列表（用于区分不同采样频率）。圆形、星形、菱形、三角形、六边形
            lstyles = ["solid", "dashed", "dashdot", "dotted"]              # 定义线型列表。实线、虚线、点划线、点线

            # loop over sampling frequencies(循环每个采样频率)
            includedFreqs = []                                      # 记录实际找到数据的频率
            for idx, sampleFreq in enumerate(sampleFreqs):          # 循环处理每个采样频率

                tfData = getTransferFunctionData(                   # 获取传递函数数据（Tipper数据）
                    projData, site, sampleFreq, specdir=options["specdir"], postpend=pp         # specdir:频谱目录,postpend:当前后缀
                )
                if not tfData:                                      # 如果未找到数据，跳过该频率
                    continue

                includedFreqs.append(sampleFreq)                    # 记录成功找到数据的频率[15.0]
                projectText(
                    "Plotting tipper for site {}, sample frequency {}".format(
                        site, sampleFreq
                    )
                )
                # 选择标记样式（循环使用预定义标记样式）
                mk = mks[idx % len(mks)]
                ls = lstyles[idx % len(lstyles)]
                tfData.viewTipper(                                          # 绘制Tipper数据的三个分量(幅度、实部角度、虚部角度)
                    fig=fig,                                                # 要绘制的图形对象
                    rows=options["cols"],                                   # 布局方向（True为1行3列）
                    mk=mk,                                                  # 数据点标记样式
                    ls=ls,                                                  # 连线线型样式
                    label="{}Hz".format(sampleFreq),                         # 图例标签（显示频率）
                    xlim=options["plotoptions"]["xlim"],                    # X轴范围
                    length_ylim=options["plotoptions"]["length_ylim"],      # 幅度Y轴范围
                    angle_ylim=options["plotoptions"]["angle_ylim"],        # 相位Y轴范围
                    plotfonts=options["plotoptions"]["plotfonts"],          # 字体设置
                )

            # check if any files found
            if len(includedFreqs) == 0:
                continue

            # sup title(设置图形总标题)
            sub = "Site {} tipper: {}".format(site, options["specdir"] + postpend)
            sub = "{}\nfs = {}".format(sub, arrayToString(includedFreqs, decimals=3))
            st = fig.suptitle(sub, fontsize=plotfonts["suptitle"])      # 设置图形顶部标题
            st.set_y(0.99)                          # 调整标题位置（靠近顶部）
            fig.tight_layout()                      # 自动调整图形布局
            fig.subplots_adjust(top=0.85)           # 为顶部标题留出空间
            figs.append(fig)                        # 将图形添加到返回列表

            if options["save"]:
                impath = projData.imagePath
                filename = "tipper_{}_{}{}".format(site, options["specdir"], postpend)
                savename = savePlot(impath, filename, fig)
                projectText("Image saved to file {}".format(savename))

        if not options["show"]:                     # 如果不显示图形，关闭所有图形释放内存
            plt.close("all")
        else:
            plt.show(block=options["plotoptions"]["block"])
    return figs
