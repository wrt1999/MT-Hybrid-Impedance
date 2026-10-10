import os
import glob
import numpy as np
import scipy.interpolate as interp
from typing import Dict, List

from resistics.common.base import ResisticsBase
from resistics.common.checks import isMagnetic
from resistics.common.math import (
    forwardFFT,
    inverseFFT,
    getFrequencyArray,
    padNextPower2,
)
from resistics.common.print import generalPrint, warningPrint, blockPrint
from resistics.calibrate.data import CalibrationData
from resistics.calibrate.io import CalibrationIO
from resistics.calibrate.utils import getKnownCalibrationFormats, getCalName
from resistics.config.io import loadConfig
from resistics.time.data import TimeData


class Calibrator(ResisticsBase):
    """Class for calibrating a time dataset

    Attributes
    ----------
    calExt : list
        Accepted calibration extensions
    calFormats : list
        The calibration file formats of the respective extensions
    calDir : str
        Directory path of calibration files
    calFiles : list[str]
        List of calibration files found in calibration directory
    numCalFiles : int
        Number of found calibration files in calibration directory
    useTheoretical : bool
        Flag to use theoretical calibration function

    Methods
    -------
    __init__(calDirectory)
        Initialise with calibration directory
    setCalDir(calDirectory)
        Set the calibration directory path
    findCalFiles()
        Find calibration files in calibration directory
    calibrate(timeData, sensor, serial, chopper)
        Calibrate time data
    getCalFile(sensor, serial, chopper)
        Get index of matching calibration flile in calFiles list
    calibrateChan(data, fs, calData)
        Calibrate an individual channel
    interpolateCalData(calData, f, spline)
        Interpolate calibration data to same frequencies as channel data
    getTheoreticalCalData(sensor)
        Get theoretical calibration data
    printList()
        Class status returned as list of strings
    """

    def __init__(self, calDirectory: str) -> None:
        """Set the calibration directory and initialise
        设置标定目录并进行初始化
        Calibrator will automatically find calibration files in the provided directory
        标定器会自动在提供的目录中查找标定文件
        Parameters
        ----------
        calDirectory : str
            Path of directory containing calibration files
        """
        self.calExt, self.calFormats = getKnownCalibrationFormats()             # 获取已知的标定文件扩展名和格式
        self.calFiles: List[str] = []                                           # 初始化标定文件列表
        self.numCalFiles = 0                                                    # 标定文件数量计数（初始为0）
        self.calDir: str = calDirectory                                         # 存储标定目录路径（便于后续使用
        self.findCalFiles()                                                     # 调用方法在指定目录中查找标定文件
        # set whether to use theoretical calibration functions(设置是否使用理论校标定函数)
        conf = loadConfig()                                                     # 加载全局配置文件（通常从外部文件读取），返回配置字典
        self.extend: bool = conf["Calibration"]["extend"]
        self.useTheoretical: bool = conf["Calibration"]["usetheoretical"]

    def setCalDir(self, calDirectory: str) -> None:
        """Set the calibration directory and find files

        Calibrator will automatically find calibration files in the provided directory
    
        Parameters
        ----------
        calDirectory : str
            Path of directory containing calibration files
        """
        self.calDir = calDirectory
        self.findCalFiles()

    def findCalFiles(self) -> None:
        """Find calibration files in calibration directory"""
        self.calFiles.clear()
        # get unique extensions
        extensions = list(set(self.calExt))
        for calExt in extensions:
            # get all files of format cE
            self.calFiles = self.calFiles + glob.glob(
                os.path.join(self.calDir, "*.{}".format(calExt))
            )
        self.numCalFiles = len(self.calFiles)

    def calibrate(
        self,
        timeData: TimeData,
        sensor: Dict[str, str],
        serial: Dict[str, int],
        chopper: Dict[str, bool],
    ) -> TimeData:
        """Calibrate time data
        校准时间数据
        For each channel in timeData, searches for a matching calibration file based on sensor type, serial number and chopper. If a calibration file is found, the channel is calibrated using the data in the file. If useTheoretical is False and no file is found, the data is not calibrated
        对于 timeData 中的每个通道，根据sensor type, serial number and chopper来查找匹配的校准文件。如果找到校准文件，则使用文件中的数据对该通道进行校准。如果 useTheoretical 为 False 且未找到文件，则数据不进行校准
        todo:
        If no calibration file is found and the channel is a magnetic data channel, a theoretical function can be used
        如果未找到校准文件且通道为磁数据通道，则可以使用理论函数
        Parameters
        ----------
        timeData : TimeData
            TimeData object
        sensor : Dict
            Dictionary of sensor information with channels as the key and sensor as the value (sensor is a string)
        serial :
            Dictionary of serial information with channels as the key and sensor as the value (serial is a number)
        chopper :
            Dictionary of chopper information with channels as the key and sensor as the value (chopper is a bool)

        Returns
        -------
        timeData : TimeData
            Calibration TimeData object
        """
        calIO = CalibrationIO()     # 初始化calIO，chopper：False，datafile=''，extend=False，fileformat='induction'
        #############################################################
        for channel in timeData.chans:
            self.printText("Calibrating channel {}".format(channel))
            calFile, calFormat = self.getCalFile(
                str('Phoenix'), 4884, chopper[channel]
            )
            if calFile == "":
                # no file found
                if self.useTheoretical and isMagnetic(chan):
                    # use theoretical
                    calData = self.getTheoreticalCalData(sensor[chan])
                    timeData.data[chan] = self.calibrateChan(
                        timeData.data[chan], timeData.sampleFreq, calData
                    )
                    timeData.addComment(
                        "Channel {} calibrated with theoretical calibration function".format(
                            chan
                        )
                    )
                    continue
                else:
                    self.printText(
                        "No Calibration data found - channel will not be calibrated"
                    )
                    timeData.addComment("Channel {} not calibrated".format(chan))
                    continue  # nothing to do
            calIO.refreshing(calFile, calFormat, chopper=chopper[channel], extend=self.extend)
            calData = calIO.read()
            self.printText(
                "Calibration file found for channel {}, serial number {}, chopper {}: {}".format(
                    channel, 4884, chopper[channel], calFile
                )
            )
            self.printText("Format: {}".format(calFormat))
            self.printText(
                "Static gain correction of {} applied to calibration data".format(
                    calData.staticGain
                )
            )
            timeData.data[channel] = self.calibrateChan1(
                timeData.data[channel], timeData.sampleFreq, calData
                # 对通道数据进行傅里叶变换，乘以传感器脉冲响应transferFunc（即反褶积），然后进行傅里叶逆变换
            )
            timeData.addComment(
                "Channel {} calibrated with calibration data from file {}".format(
                    channel, calFile
                )
            )
        ###########################################################
        # iterate over data(遍历数据)
        for chan in timeData.chans:             # 从Ex开始
            # output some info
            self.printText("Calibrating channel {}".format(chan))
            # try and find calibration file  (calFile:txt   calFormat:induction)
            calFile, calFormat = self.getCalFile(
                sensor[chan], serial[chan], chopper[chan]
            )
            if calFile == "":
                # no file found
                if self.useTheoretical and isMagnetic(chan):
                    # use theoretical
                    calData = self.getTheoreticalCalData(sensor[chan])
                    timeData.data[chan] = self.calibrateChan(
                        timeData.data[chan], timeData.sampleFreq, calData
                    )
                    timeData.addComment(
                        "Channel {} calibrated with theoretical calibration function".format(
                            chan
                        )
                    )
                    continue
                else:
                    self.printText(
                        "No Calibration data found - channel will not be calibrated"
                    )
                    timeData.addComment("Channel {} not calibrated".format(chan))
                    continue  # nothing to do

            # else file found
            # no need to separately apply static gain, already included in cal data(无需单独应用静态增益，已包含在校准数据中)
            calIO.refresh(calFile, calFormat, chopper=chopper[chan], extend=self.extend)    # 更新extend为True（self.extend在前面已赋值为配置config中的True）
            calData = calIO.read()                  # 读取文件
            self.printText(
                "Calibration file found for sensor {}, serial number {}, chopper {}: {}".format(
                    sensor[chan], serial[chan], chopper[chan], calFile
                )
            )
            self.printText("Format: {}".format(calFormat))
            self.printText(
                "Static gain correction of {} applied to calibration data".format(
                    calData.staticGain
                )
            )
            # calibrate time data（快速傅里叶，将校准信息插值到数据中的各个频率点）***********
            timeData.data[chan] = self.calibrateChan(
                timeData.data[chan], timeData.sampleFreq, calData       # 对通道数据进行傅里叶变换，除以传感器脉冲响应transferFunc（即反褶积），然后进行傅里叶逆变换
            )
            timeData.addComment(
                "Channel {} calibrated with calibration data from file {}".format(
                    chan, calFile
                )
            )
        # return calibrated time data
        return timeData

    def getCalFile(self, sensor, serial, chopper) -> int:
        """Get calibration file for a sensor, serial and chopper combination
    
        Parameters
        ----------
        sensor : str
            Channel data
        serial : int
            Sampling frequency Hz
        chopper : bool
            Calibration data

        Returns
        -------
        calFile : str
            The mathing calibration file
        calFormat : str
            The calibration format
        """
        if self.numCalFiles == 0:
            # no calibration files to calibrate with
            return "", ""

        for extension, fileformat in zip(self.calExt, self.calFormats):         # extension:TXT    fileformat:induction
            # get the name for this format - there could be more than one acceptable name
            calNames = getCalName(fileformat, extension, sensor, serial, chopper)
            if calNames is None:
                continue
            # if a string rather than a list(如果是一个字符串而不是一个列表，本calNames是字符串，需转换)
            if isinstance(calNames, str):   # 如果是这种情况，转换为列表
                calNames = [calNames]
            # search to find a calibration file with that name and take the first encountered(进行检索，查找具有该名称的校准文件，并选取首个检索到的文件)
            for calName in calNames:
                for calFile in self.calFiles:
                    calFileExt = os.path.splitext(calFile)[1].lower()
                    if (
                        calFileExt == "." + extension.lower()
                        and calName.lower() in calFile.lower()         # 校准文件名（如"IC_7491"）是否存在于实际文件名中
                    ):
                        return calFile, fileformat                     # calFile：找到的校准文件完整路径/名称,fileformat：匹配到的文件格式（如"induction"）
        # else return empty strings
        return "", ""

    def calibrateChan1(
        self, data: np.ndarray, sampleFreq: float, calData: CalibrationData
    ) -> np.ndarray:
        """Calibrate a channel
        CLB校准一个通道
        Perform fourier transform of channel data, deconvolve (division) sensor impulse response and inverse fourier transform.
        对通道数据进行傅里叶变换，除以传感器脉冲响应transferFunc（即反褶积），然后进行傅里叶逆变换。
        Parameters
        ----------
        data : np.ndarray
            Channel data
        sampleFreq : float
            Sampling frequency Hz
        calData : CalibrationData
            Calibration data

        Returns
        -------
        out : np.ndarray
            Calibrated channel data
        """
        # do the forward transform(正向变换)
        dataSize = data.size
        # pad end of array(对数组末尾进行填充，在原始数组的末尾（右边）添加若干个0，使其总长度变为2的幂次方)
        data = np.pad(data, (0, padNextPower2(dataSize)), "constant")       # 目的：将数据长度填充到下一个2的幂次方（如 1000 → 1024），以提高FFT计算效率，在数据末尾添加零，不改变频谱特性
        fftData = forwardFFT(data)              # 执行正向傅里叶变换，将时域信号转换为频域复数频谱。输出：fftData是复数数组，包含幅值和相位信息。实数FFT的结果具有共轭对称性(只保留一半+直流)。频率范围：从0Hz到奈奎斯特频率（sampleFreq/2）
        f = getFrequencyArray(sampleFreq, fftData.size)         # 根据采样频率和FFT数据长度计算每个频率点的值，生成与FFT结果对应的频率点（单位：Hz），在 0 Hz 到nyquist频率之间生成fftData.size个等间隔的点，fftData.size：填充后的长度
        # do calibration in frequency domain(在频域中进行校准)
        # interpolate calibration info to frequencies in data(将校准信息插值到数据中的各个频率点)
        transferFunc = self.interpolateCalData(calData, f)      # 将标定数据插值到数据的频率点上。interpolateCalData()：将离散频率点上的传感器校准数据(calData)插值到连续频率轴上，生成完整的频率响应函数 transferFunc（复数，频率轴上的每个频率点对应，包含增益和相位修正）。
        # recall, zero element excluded, so start from 1        # 需注意，排除零元素，故从1开始计数
        # fft zero element should really be zero because average is removed from data       # 由于数据已去除均值，快速傅里叶变换（FFT）中的零元素实际上应为零
        fftData[1:] = fftData[1:] * transferFunc        # 对FFT结果（除零频外）进行复数除法，修正传感器的频率响应，幅度相除，相位相减
        # return the non padded part of the array(返回数组的未填充部分)
        return inverseFFT(fftData, data.size)[:dataSize]        # 逆FFT：频域(复数)→时域(实数)，返回时域校准数据。截断：[:dataSize]去除填充的零，恢复原始长度

    def calibrateChan(
        self, data: np.ndarray, sampleFreq: float, calData: CalibrationData
    ) -> np.ndarray:
        """Calibrate a channel
        CLC校准一个通道
        Perform fourier transform of channel data, deconvolve (division) sensor impulse response and inverse fourier transform.
        对通道数据进行傅里叶变换，除以传感器脉冲响应transferFunc（即反褶积），然后进行傅里叶逆变换。
        Parameters
        ----------
        data : np.ndarray
            Channel data
        sampleFreq : float
            Sampling frequency Hz
        calData : CalibrationData
            Calibration data

        Returns
        -------
        out : np.ndarray
            Calibrated channel data
        """
        # do the forward transform(正向变换)
        dataSize = data.size
        # pad end of array(对数组末尾进行填充，在原始数组的末尾（右边）添加若干个0，使其总长度变为2的幂次方)
        data = np.pad(data, (0, padNextPower2(dataSize)), "constant")       # 目的：将数据长度填充到下一个2的幂次方（如 1000 → 1024），以提高FFT计算效率，在数据末尾添加零，不改变频谱特性
        fftData = forwardFFT(data)              # 执行正向傅里叶变换，将时域信号转换为频域复数频谱。输出：fftData是复数数组，包含幅值和相位信息。实数FFT的结果具有共轭对称性(只保留一半+直流)。频率范围：从0Hz到奈奎斯特频率（sampleFreq/2）
        f = getFrequencyArray(sampleFreq, fftData.size)         # 根据采样频率和FFT数据长度计算每个频率点的值，生成与FFT结果对应的频率点（单位：Hz），在 0 Hz 到nyquist频率之间生成fftData.size个等间隔的点，fftData.size：填充后的长度
        # do calibration in frequency domain(在频域中进行校准)
        # interpolate calibration info to frequencies in data(将校准信息插值到数据中的各个频率点)
        transferFunc = self.interpolateCalData(calData, f)      # 将标定数据插值到数据的频率点上。interpolateCalData()：将离散频率点上的传感器校准数据(calData)插值到连续频率轴上，生成完整的频率响应函数 transferFunc（复数，频率轴上的每个频率点对应，包含增益和相位修正）。
        # recall, zero element excluded, so start from 1        # 需注意，排除零元素，故从1开始计数
        # fft zero element should really be zero because average is removed from data       # 由于数据已去除均值，快速傅里叶变换（FFT）中的零元素实际上应为零
        fftData[1:] = fftData[1:] / transferFunc        # 对FFT结果（除零频外）进行复数除法，修正传感器的频率响应，幅度相除，相位相减
        # return the non padded part of the array(返回数组的未填充部分)
        return inverseFFT(fftData, data.size)[:dataSize]        # 逆FFT：频域(复数)→时域(实数)，返回时域校准数据。截断：[:dataSize]去除填充的零，恢复原始长度

    def interpolateCalData(self, calData: CalibrationData, f: np.ndarray) -> np.ndarray:
        """Interpolation calibration data on to frequency points
        将校准数据插值到各频率点
        Parameters
        ----------
        calData : CalibrationData
            Calibration data
        f : np.ndarray
            Frequency array where calibration data is defined

        Returns
        -------
        out : np.ndarray
            Calibration data interpolated to the frequency array f
        """
        # linear interpolate(线性插值)
        interpFuncMag = interp.interp1d(calData.freqs, calData.magnitude)       # 创建幅度插值函数：从标定频率点到幅度值的线性映射。interpFuncMag是可调用函数，输入任意频率返回插值幅度。
        interpFuncPhase = interp.interp1d(calData.freqs, calData.phase)         # 创建相位插值函数：从标定频率点到相位值的线性映射。相位响应插值：类似。calData.phase：各频率点对应的相位响应（如相位延迟，单位可能是度或弧度）
        return interpFuncMag(f[1:]) * np.exp(1j * interpFuncPhase(f[1:]))       # 从索引1开始（跳过直流分量），将幅度和相位组合为复数传递函数。生成复数传递函数H(f) = |H(f)| × exp(j × φ(f))，f[1:]排除了直流分量（0Hz），因为传感器校准通常不定义零频响应

    def getTheoreticalCalData(self, sensor: str) -> np.ndarray:
        """Get theoretical calibration data for magnetic channels
    
        Parameters
        ----------
        sensor : str
            Sensor type

        Returns
        -------
        out : np.ndarray
            Theoretical calibration information
        """
        # should use the sensor to figure out what calibration func
        if "mfs06" in sensor or "MFS06" in sensor:
            return CalibrationIO().unitCalibration()

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : list
            List of strings with information
        """
        textLst = []
        textLst.append("Known extensions and calibration formats")
        textLst.append("Extensions:")
        textLst.append(", ".join(self.calExt))
        textLst.append("Associated formats")
        textLst.append(", ".join(self.calFormats))
        # now print the actual calibration files
        textLst.append("{} calibration files found:".format(self.numCalFiles))
        if self.numCalFiles == 0:
            textLst.append("\t\tNo calibration files found")
        else:
            for calFile in self.calFiles:
                textLst.append("\t\t{}".format(calFile))
        return textLst
