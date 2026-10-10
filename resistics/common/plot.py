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
from datetime import datetime, timedelta
from typing import Dict, Tuple

from resistics.common.checks import parseKeywords


def savePlot(path: str, filename: str, fig: plt.figure) -> str:
    """Save plot whilst checking for overwriting
    
    Parameters
    ----------
    path : str
        The path to save the file
    filename : str
        Filename to save to
    fig : plt.figure   
        Matplotlig figure to save
    
    Returns
    -------
    str
        The filename written out to
    """
    import os

    filenameForWriting = filename
    fileExists = True
    counter = 1
    while fileExists:
        filepath = os.path.join(path, "{}.png".format(filenameForWriting))
        if os.path.exists(filepath):
            filenameForWriting = "{}.{}".format(filename, counter)
            counter += 1
        else:
            fileExists = False
            break
    filepath = os.path.join(path, "{}.png".format(filenameForWriting))
    fig.savefig(filepath)
    return filenameForWriting


def getPlotRowsAndCols(maxCols: int, numPlots: int = 0) -> Tuple[int, int]:
    """Get the numbers of rows and columns for plots

    Parameters
    ----------
    maxCols : int
        The maximum number of columns
    numPlots : int
        The of subplots
    """
    if numPlots <= maxCols:
        ncols = numPlots
        nrows = 1
    else:
        ncols = maxCols
        nrows = int(np.ceil(1.0 * numPlots / maxCols))
    return nrows, ncols


def plotOptionsStandard(**kwargs) -> Dict:
    """Get a set of standard plot options
    获取一套标准绘图参数配置
    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown

    Returns
    -------
    Dict
        Dictionary of standard plot options
    """
    default: Dict = {}                          # 创建一个空字典，用于存储默认的绘图配置参数
    default["figsize"] = (20, 12)               # 设置默认图形尺寸为20英寸宽、12英寸高
    default["plotfonts"] = getViewFonts()       # 获取默认的视图字体设置
    default["block"] = True                     # 设置默认的阻塞标志为True(当显示图形时阻塞程序执行，用户关闭图形窗口后程序才继续)
    default = parseKeywords(default, kwargs)    # 使用传入的关键字参数覆盖默认配置(parseKeywords函数负责合并默认值和用户提供的参数)
    return default                              # 返回最终的配置字典


def plotOptionsTime(**kwargs) -> Dict:
    """Get default plot options for plotting time data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    Eylim : List[float], optional
        y limits for electric data
    Hylim : List[float], optional
        y limits for magnetic data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting time data
    """
    default = plotOptionsStandard()
    default["Eylim"] = []
    default["Hylim"] = []
    default = parseKeywords(default, kwargs)
    return default


def plotOptionsSpec(**kwargs) -> Dict:
    """Get default plot options for plotting spectra data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    amplim : List[float], optional
        Amplitude limits for plotting spectra

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """

    default = plotOptionsStandard()
    default["amplim"] = []
    default = parseKeywords(default, kwargs)
    return default


def plotOptionsTransferFunction(**kwargs) -> Dict:
    """Get default plot options for plotting transfer function data 
    获取用于绘制传递函数数据的默认绘图选项
    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    res_ylim : List[float], optional
        y limits for resistivity data
    phase_ylim : List[float], optional
        y limits for phase data
    xlim : List[float], optional
        x limits for transfer function data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """
    default = plotOptionsStandard()             # 获取标准绘图配置作为基础
    default["figsize"] = None                   # 自动根据屏幕DPI调整，将图形尺寸设置为None，让绘图函数根据实际情况自动计算
    default["res_ylim"] = [0.1, 10000]          # 视电阻率范围（Ω·m），设置视电阻率的Y轴范围：[最小值, 最大值]
    default["phase_ylim"] = [-180, 90]          # 相位范围（度），设置相位的Y轴范围：[-180°, 180°]
    default["xlim"] = [0.1, 10000]              # 频率范围（Hz），设置频率X轴范围：[最小值, 最大值]
    default = parseKeywords(default, kwargs)    # 使用用户传入的参数覆盖默认配置
    return default                              # 返回最终配置字典


def getTransferFunctionFigSize(oneplot: bool, npolarisations: int) -> Tuple[int, int]:
    """Get the plot size for a transfer function plot
    获取传递函数图的绘图尺寸
    Notes
    -----
    This is awkward because of maintaining aspect ratio for the apparent resistivity. A better solution might be required at some point.
    这一问题的棘手之处在于需维持视电阻率的纵横比，未来可能需要更优的解决方案
    Parameters
    ----------
    oneplot : bool
        All polarisations in one plot
    npolarisations : int
        Number of polarisations to plot
    
    Returns
    -------
    figsize : Tuple
        The figure size
    """
    if oneplot:
        return (6, 10)
    else:
        return (5 * npolarisations, 9)


def plotOptionsTipper(**kwargs) -> Dict:
    """Get default plot options for plotting transfer function data 

    Parameters
    ----------
    figsize : Tuple, optional
        Set the figure size
    plotfonts : Dict, optional
        Font sizes to use for plotting fonts
    block : bool, optional
        Boolean flag for blocking execution when plot is shown
    length_ylim : List[float], optional
        y limits for length data
    angle_ylim : List[float], optional
        y limits for angle data
    xlim : List[float], optional
        x limits for transfer function data

    Returns
    -------
    out : Dict
        Dictionary of default plot options for plotting spectra data
    """
    default = plotOptionsStandard()
    default["figsize"] = (16, 5)
    default["length_ylim"] = [0.00001, 100000]
    default["angle_ylim"] = [-10, 10]
    default["xlim"] = [0.1, 10]
    default = parseKeywords(default, kwargs)
    return default


def getViewFonts() -> Dict:
    """Get default plot font options for viewing plots   
    获取用于查看图形的默认绘图字体选项
    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}                # 创建一个空的字典，用于存储各种字体大小设置
    plotFonts["suptitle"] = 18          # 设置图形总标题（suptitle）的字体大小为14磅
    plotFonts["title"] = 18             # 置子图标题（title）的字体大小为12磅
    plotFonts["axisLabel"] = 18         # 设置坐标轴标签（axisLabel）的字体大小为12磅(如"Frequency (Hz)"、"Apparent Resistivity (Ω·m)"等)
    plotFonts["axisTicks"] = 18         # 设置坐标轴刻度标签（axisTicks）的字体大小为12磅(坐标轴上的刻度数值，如频率值0.001, 0.01, 0.1, 1, 10, 100等)
    plotFonts["legend"] = 18            # 设置图例（legend）的字体大小为12磅（区分不同频率的采样数据）
    return plotFonts

def getViewFonts1() -> Dict:
    """Get default plot font options for viewing plots
    获取用于查看图形的默认绘图字体选项
    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}                # 创建一个空的字典，用于存储各种字体大小设置
    plotFonts["suptitle"] = 15          # 设置图形总标题（suptitle）的字体大小为14磅
    plotFonts["title"] = 15             # 置子图标题（title）的字体大小为12磅
    plotFonts["axisLabel"] = 18         # 设置坐标轴标签（axisLabel）的字体大小为12磅(如"Frequency (Hz)"、"Apparent Resistivity (Ω·m)"等)
    plotFonts["axisTicks"] = 16         # 设置坐标轴刻度标签（axisTicks）的字体大小为12磅(坐标轴上的刻度数值，如频率值0.001, 0.01, 0.1, 1, 10, 100等)
    plotFonts["legend"] = 18            # 设置图例（legend）的字体大小为12磅（区分不同频率的采样数据）
    return plotFonts

def getPlotFonts() -> Dict:
    """Get default plot font options for saving plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 18
    plotFonts["title"] = 16
    plotFonts["axisLabel"] = 16
    plotFonts["axisTicks"] = 14
    plotFonts["legend"] = 14
    return plotFonts


def getPresentationFonts() -> Dict:
    """Get default plot font options for presentation plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 22
    plotFonts["title"] = 20
    plotFonts["axisLabel"] = 20
    plotFonts["axisTicks"] = 20
    plotFonts["legend"] = 20
    return plotFonts


def getPaperFonts() -> Dict:
    """Get default plot font options for paper plots   

    Returns
    -------
    out : Dict
        Dictionary of default plot font options
    """
    plotFonts: Dict = {}
    plotFonts["suptitle"] = 18
    plotFonts["title"] = 17
    plotFonts["axisLabel"] = 16
    plotFonts["axisTicks"] = 15
    plotFonts["legend"] = 15
    return plotFonts


def transferFunctionColours() -> Dict[str, str]:
    """Get colours from the different components of standard magnetotelluric transfer functions
    从标准大地电磁传递函数的各分量中提取颜色映射依据
    Returns
    -------
    Dict[str, str]
        Dictionary mapping magnetotelluric transfer funciton components to colours
    """
    colours = {
        "ExHx": "orange",
        "EyHy": "green",
        "ExHy": "red",
        "EyHx": "blue",
        "HzHx": "gray",
        "HzHy": "magenta",
    }
    return colours


def colorbar2dTime():
    """Get default colormap for 2d plots with time   

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.viridis


def colorbar2dSpectra():
    """Return colorbar for plotting spectra sections

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.magma


def colorbar2dOther():
    """Alternative colormap for 2d data 

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.plasma


def colorbarMultiline():
    """Alternative colormap for plotting multiple line plots 

    Returns
    -------
    out : plt.cm
        Colormap
    """
    return plt.cm.cividis


def addLegends(fig: plt.figure, pos: int = 1) -> None:
    """Add legends to all subplots in figure
    
    Parameters
    ----------
    fig : magplotlib.pyplot.figure
        A matplotlib figure
    pos : int
        Location of legend  
    """
    axList = fig.axes
    for ax in axList:
        ax.legend(loc=pos)
