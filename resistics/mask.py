import sys
import os
import numpy as np
from datetime import datetime
from typing import Union

from resistics.project.data import ProjectData
from resistics.mask.data import MaskData
from resistics.mask.io import MaskIO
from resistics.mask.calculator import MaskCalculator
from resistics.window.selector import WindowSelector
from resistics.common.checks import parseKeywords
from resistics.project.utils import projectText
from resistics.project.shortcuts import getDecimationParameters, getWindowParameters


def newMaskData(projData: ProjectData, sampleFreq: float) -> MaskData:
    """Get a mask data object that can then be passed onto calculateMask
    获取一个掩码数据对象，该对象可以传递给calculateMask函数
    Parameters
    ----------
    projData : ProjectData
        A ProjectData instance
    sampleFreq : float
        The sampling frequency to mask
    
    Returns
    -------
    MaskData
        A mask data object with parameters set
    """
    decParams = getDecimationParameters(sampleFreq, projData.config)                        # 获取分频参数
    decParams.printInfo()
    return MaskData(                                                                        # 创建并返回MaskData对象
        decParams.sampleFreq, decParams.numLevels, decParams.evalFreqPerLevel               # 采样频率,分频级数,每级评估频率
    )


def getMaskData(
    projData: ProjectData,
    site: str,
    maskName: str,
    sampleFreq: Union[float, int],
    **kwargs
) -> MaskData:
    """Get a mask data object

    Parameters
    ----------
    projData : projectData
        A project instance
    site : str
        The site for which to get the mask
    maskName : str
        The name of the mask
    sampleFreq : int, float
        The sampling frequency for which the mask was created
    specdir : str
        The spectra directory for which the mask was created  

    Returns
    -------
    MaskData
        A mask data object with the mask information
    """
    options = {}
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]
    options = parseKeywords(options, kwargs)
    siteData = projData.getSiteData(site)
    maskIO = MaskIO(siteData.getSpecdirMaskPath(options["specdir"]))                                    # 创建掩码IO对象，指定掩码数据的存储路径(如D:\resistics\tutorial\tutorialProject\maskData\site1\spectra)
    maskData = maskIO.read(maskName, sampleFreq)                                                        # 读取指定名称和采样频率的掩码数据(maskWindows:掩码(不满足)的窗口)
    return maskData


def calculateMask(projData: ProjectData, maskData: MaskData, **kwargs):
    """Calculate masks sites
    为站点计算掩码
    Parameters
    ----------
    projData : projectData
        A project instance
    maskData : MaskData
        A mask data instance
    sites : List[str], optional
        A list of sites to calculate masks for
    specdir : str, optional
        The spectra directory for which to calculate statistics
    """
    options = {}                                                                                    # 初始化选项字典
    options["sites"] = projData.getSites()                                                          # 从项目数据获取所有站点作为默认站点列表
    options["specdir"] = projData.config.configParams["Spectra"]["specdir"]                         # 从项目配置获取谱数据目录
    options = parseKeywords(options, kwargs)                                                        # 解析传入的关键字参数，更新选项
    # create a maskCalculator object(创建掩码计算器)
    maskCalc = MaskCalculator(projData, maskData, specdir=options["specdir"])                       # 创建掩码计算器对象
    maskIO = MaskIO()                                                                               # 创建掩码IO对象,负责文件读写
    sampleFreq = maskData.sampleFreq                                                                # 从掩码数据获取采样频率

    # loop over sites
    for site in options["sites"]:                                                                   # 遍历所有要处理的站点
        # see if there is a sample freq
        siteData = projData.getSiteData(site)                                                       # 获取站点数据
        siteSampleFreqs = siteData.getSampleFreqs()                                                 # 获取站点支持的采样频率列表
        if sampleFreq not in siteSampleFreqs:                                                       # 检查采样频率是否在站点的支持列表中
            continue                                                                                # 如果不支持，跳过此站点

        # decimation and window parameters
        decParams = getDecimationParameters(sampleFreq, projData.config)                            # 获取分频参数
        decParams.printInfo()
        winParams = getWindowParameters(decParams, projData.config)                                 # 获取窗口参数

        # clear previous windows from maskCalc
        maskCalc.clearMaskWindows()                                                                 # 清空掩码计算器中的先前窗口(清除前一个站点的计算结果,确保每个站点独立处理，避免数据污染)
        # calculate masked windows
        maskCalc.applyConstraints(site)                                                             # 计算掩码窗口(基于maskData中的约束条件，筛选有效数据窗口,maskCalc.maskData.maskWindows里是不在[0.7, 1.0]的窗口)
        maskCalc.maskData.printInfo()                                                               # 打印掩码数据信息

        # write maskIO file
        maskIO.datapath = os.path.join(siteData.getSpecdirMaskPath(options["specdir"]))             # 设置掩码IO的数据路径
        maskIO.write(maskCalc.maskData)                                                             # 写入掩码文件

        # test with the window selector(使用窗口选择器进行测试)
        winSelector = WindowSelector(
            projData, sampleFreq, decParams, winParams, specdir=options["specdir"]                  # 创建窗口选择器
        )
        winSelector.setSites([site])                                                                # 设置要处理的站点
        winSelector.addWindowMask(site, maskData.maskName)                                          # 添加窗口掩码
        winSelector.calcSharedWindows()                                                             # 计算共享窗口
        winSelector.printInfo()
        winSelector.printDatetimeConstraints()
        winSelector.printWindowMasks()
        winSelector.printSharedWindows()
        winSelector.printWindowsForFrequency()
