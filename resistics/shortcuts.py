from typing import Union

from resistics.config.data import ConfigData
from resistics.calibrate.calibrator import Calibrator
from resistics.decimate.parameters import DecimationParameters
from resistics.window.parameters import WindowParameters
from resistics.window.selector import WindowSelector
from resistics.time.reader import TimeReader
from resistics.time.data import TimeData
from resistics.regression.local import LocalRegressor
from resistics.regression.remote import RemoteRegressor
from resistics.project.data import ProjectData


def getCalibrator(calPath: str, config: Union[ConfigData, None] = None) -> Calibrator:
    """Create a Calibrator object from calibration path and configuration information
    基于校准路径和配置信息创建一个 Calibrator 对象方法
    Parameters
    ----------
    calPath : str
        Path to calibration directory
    config : ConfigData
        Configuration data
    
    Returns
    -------
    Calibrator
        A calibrator object
    """
    cal = Calibrator(calPath)
    if config is None:
        return cal
    cal.useTheoretical = config.configParams["Calibration"]["usetheoretical"]
    return cal


def getDecimationParameters(
    sampleFreq: float, config: Union[ConfigData, None] = None
) -> DecimationParameters:
    """Create a DecimationParams object from sampling frequency and configuration information
    根据采样频率和配置信息创建一个 DecimationParams 对象
    Parameters
    ----------
    sampleFreq : float
        Sampling frequency of the data
    config : ConfigData
        Configuration data

    Returns
    -------
    DecimationParams
        A decimation parameters object        
    """
    decParams = DecimationParameters(sampleFreq)        # deciamtion参数
    if config is None:
        return decParams

    if config.flags["customfrequencies"]:               # 根据配置标志，选择两种不同的降采样参数设置方式
        decParams.setFrequencyParams(                                       # （自定义频率）：当用户提供了自定义频率列表时，使用这些特定频率进行降采样
            config.configParams["Frequencies"]["frequencies"],
            config.configParams["Decimation"]["numlevels"],
            config.configParams["Frequencies"]["perlevel"],
        )
    else:
        decParams.setDecimationParams(                                      # 自动频率）：当用户没有提供自定义频率时，自动生成合适的频率序列
            config.configParams["Decimation"]["numlevels"],
            config.configParams["Frequencies"]["perlevel"],
        )
    return decParams


def getWindowParameters(
    decParams: DecimationParameters, config: Union[ConfigData, None] = None
) -> WindowParameters:
    """Create a WindowParams object from decimationParams and configuration data
    根据降采样参数和配置数据创建一个 WindowParams 对象
    Parameters
    ----------
    decParams : DecimationParams
        DecimationParams object to hold the decimation parameters
    config : ConfigData
        Configuration data
    
    Returns
    -------
    WindowParams
        A window parameters object        
    """
    winParams = WindowParameters(decParams)
    if config is None:
        return winParams

    # set window factor and overlap fraction(设置窗口因子和重叠比例)
    winParams.windowFactor = config.configParams["Window"]["windowfactor"]
    winParams.overlapFraction = config.configParams["Window"]["overlapfraction"]

    if config.flags["customwindows"]:                 # 是否使用自定义窗口(默认不使用)
        winParams.setWindowParameters(
            config.configParams["Window"]["windowsizes"],
            config.configParams["Window"]["overlapsizes"],
        )
    else:                                             # 使用配置文件里的默认数据
        winParams.setMinParams(
            config.configParams["Window"]["minwindowsize"],
            config.configParams["Window"]["minoverlapsize"],
        )
    return winParams


def getWindowSelector(
    projData: ProjectData,
    decParams: DecimationParameters,
    winParams: WindowParameters,
    specdir: Union[str, None] = None,
) -> WindowSelector:
    """Create a WindowSelector object from projectData, decimationParams, windowParams
    根据 projectData、decimationParams 和 windowParams 创建一个 WindowSelector 对象
    Parameters
    ----------
    projData : ProjectData
        A project data object
    decParams : DecimationParams
        DecimationParams object to hold the decimation parameters
    winParams : Windowparams
        WindowParams object to hold the windowing parameters
    specdir : str
        The spectra directory
    
    Returns
    -------
    WindowSelector
        A window selector object        
    """
    if specdir is None:
        specdir = projData.config.configParams["Spectra"]["specdir"]

    selector = WindowSelector(
        projData, decParams.sampleFreq, decParams, winParams, specdir=specdir           # 初始化窗口选择器
    )
    return selector


def getLocalRegressor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
) -> LocalRegressor:
    """Create a ProcessorSingleSite object from a windowSelector object, outPath and config data
    基于窗口选择器（windowSelector）、输出路径和配置数据创建一个 ProcessorSingleSite 对象
    Parameters
    ----------
    winSelector : WindowSelector
        Window selector with sites, masks etc already specified
    tfPath : str
        Path to output transfer function data
    config : ConfigData, optional
        Configuration data
    
    Returns
    -------
    LocalRegressor
        local, single site regression        
    """
    processor = LocalRegressor(winSelector, outPath)    # 初始化processor，大地电磁传输函数transferfunction处理器的初始化，专门用于站点阻抗张量计算。
    if config is None:
        return processor
    return processor


def getRemoteRegressor(
    winSelector: WindowSelector, outPath: str, config: Union[ConfigData, None] = None
) -> RemoteRegressor:
    """Create a ProcessorRemoteReference object from a windowSelector object, outPath and config data

    Parameters
    ----------
    winSelector : WindowSelector
        Window selector with sites, masks etc already specified
    outPath : str
        Path to output transfer function data
    config : ConfigData, optional
        Configuration data
    
    Returns
    -------
    RemoteRegression
        Regression including a remote reference        
    """
    processor = RemoteRegressor(winSelector, outPath)
    if config is None:
        return processor
    return processor
