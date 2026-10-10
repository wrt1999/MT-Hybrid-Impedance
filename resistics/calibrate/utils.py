import numpy as np
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple


def getKnownCalibrationFormats() -> Tuple[List, List]:
    """Return list of supported calibration formats and their extensions
    返回支持的校准格式及其扩展名列表
    Returns
    -------
    extensions : List
        List of calibration file extensions
    formats : List
        List of calibration file formats 
    """
    calExt = ["TXT", "TXT", "RSP", "RSPX"]
    calFormats = ["induction", "metronix", "rsp", "rspx"]
    return calExt, calFormats


def getCalName(format, ext: str, sensor: str, serial: int, chopper) -> str:
    """Get the calibration file name
    
    Parameters
    ----------
    format : str
        Calibration format
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """
    if format == "induction":
        return inductionName(ext, sensor, serial, chopper)
    elif format == "metronix":
        return metronixName(ext, sensor, serial, chopper)
    elif format == "rsp":
        return rspName(ext, sensor, serial, chopper)
    elif format == "rspx":
        return rspxName(ext, sensor, serial, chopper)
    else:
        return metronixName(ext, sensor, serial, chopper)


def inductionName(ext: str, sensor: str, serial: int, chopper: bool) -> str:
    """Get internal format induction coil calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """
    return "IC_{}.{}".format(serial, ext)


def metronixName(ext: str, sensor: str, serial: int, chopper: bool) -> str:
    """Get Metronix calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : str
        Name of calibration file
    """
    if sensor == "" and not serial > 0:
        return None

    return "{}{}.{}".format(sensor, serial, ext)


def rspName(ext: str, sensor: str, serial: int, chopper: bool) -> List[str]:
    """Get RSP calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : List[str]
        Name of calibration files
    """
    if len(sensor) < 5:
        # not possible to get a sensor number
        return None

    board = "HF"
    if chopper:
        board = "LF"
    # try to turn the sensor into an integer serial number
    try:
        sensorNum = int(sensor[3:])
    except:
        sensorNum = 0

    names = []
    names.append("TYPE-{:03d}_{}-ID-{:06d}.{}".format(sensorNum, board, serial, ext))
    names.append("TYPE-{:03d}_BB-ID-{:06d}.{}".format(sensorNum, serial, ext))
    return names


def rspxName(ext: str, sensor: str, serial: int, chopper: bool) -> List[str]:
    """Get RSPX calibration file name
    
    Parameters
    ----------
    ext : str
        Calibration file extension
    sensor : str
        Sensor name
    serial : int
        The sensor serial number
    chopper : bool
        Boolean flag for chopper on or off

    Returns
    -------
    out : List[str]
        Name of calibration files
    """
    if len(sensor) < 6:
        # not possible to get a sensor number
        return None

    board = "HF"              # 默认 "HF"（高频板，斩波器关闭）
    if chopper:               # 若 chopper=True 则设为 "LF"（低频板）
        board = "LF"
    sensorNum = int(sensor[3:5])        # 从传感器型号提取第4-5字符并转为整数, sensor="MFS06" → sensor[3:5]="06" → sensorNum=6
    names = []
    names.append("TYPE-{:03d}_{}-ID-{:06d}.{}".format(sensorNum, board, serial, ext))        # 生成文件名，{:03d},3位数字，前导零补全,示例值6,输出效果006
    names.append("TYPE-{:03d}_BB-ID-{:06d}.{}".format(sensorNum, serial, ext))               # {:06d}, 6位数字，前导零补全,示例值307,输出效果"000307"
    return names   # 带电路板标识：TYPE-006_LF-ID-000307.txt  # 宽带模式：TYPE-006_BB-ID-000307.txt    # {}, 字符串插入,示例值"LF", 输出效果"LF"


def defaultCalibration():
    """Default calibration data

    Returns
    -------
    data : np.ndarray
        Data lines converted to a float array
    staticGain : float
        Static gain    
    """
    return [1] * 10, 1
