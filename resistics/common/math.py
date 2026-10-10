from typing import Dict, Union
import numpy as np


def intdiv(nom: Union[int, float], div: Union[int, float]) -> int:
    """Return an integer result of division
    返回整数除法结果
    The division is expected to be exact and ensures an integer return rather than float.该除法运算预计是精确的，并确保返回整数而非浮点数
    Code execution will exit if division is not exact
    若除法运算不精确，代码执行将终止
    Parameters
    ----------
    nom : int, float
        Nominator
    div : int, float
        Divisor    

    Returns
    -------
    out : int
        Result of division
    """
    from resistics.common.print import errorPrint

    if nom % div == 0:
        return nom // div
    else:
        errorPrint(
            "utilsMath::intdiv",
            "intdiv assumes exact division and exits upon having a remainder to make sure errors are not propagated through the code",
            quitRun=True,
        )
        return 0


def getFrequencyArray(fs: float, samples: int) -> np.ndarray:
    """Calculate the frequency array for a real fourier transform
    计算实数傅里叶变换对应的频率数组
    Frequency array goes from 0Hz to Nyquist. Nyquist = fs/2
    率数组从 0 赫兹到奈奎斯特。奈奎斯特 = 采样率 / 2
    Parameters
    ----------
    fs : float
        Sampling frequency
    samples : int
        Number of samples    

    Returns
    -------
    frequencies : np.ndarray
        Array of rfft frequencies
    """
    nyquist: float = 1.0 * fs / 2.0                  # 根据采样定理，采样频率fs必须至少是信号最高频率的2倍，否则会出现混叠。
    return np.linspace(0, nyquist, samples)     # np.linspace:在 0 Hz 到nyquist频率之间生成 samples个等间隔的点.例：0~7.5Hz之间生成fftData.size个频率点


def forwardFFT(data: np.ndarray, norm: bool = True):
    """Forward real fourier transform
    
    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed   
    norm : bool, optional
        Normalisation option

    Returns
    -------
    fourierData : np.ndarray
        Fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.rfft(data, axis=0)
    return fft.rfft(data, norm="ortho", axis=0)


def inverseFFT(data: np.ndarray, length: int, norm: bool = True):
    """Inverse real fourier transform
    实数傅里叶逆变换
    Parameters
    ----------
    data : np.ndarray
        Time array to be transformed
    length : int
        Length of output time data (to remove padding)
    norm : bool, optional
        Normalisation option

    Returns
    -------
    timeData : np.ndarray
        Inverse fourier transformed data
    """
    import numpy.fft as fft

    if not norm:
        return fft.irfft(data, n=length)
    return fft.irfft(data, n=length, norm="ortho")


def padNextPower2(size: int) -> int:
    """Calculate the amount of padding to next power of 2
    
    Parameters
    ----------
    size : float
        Size of array to be padded   

    Returns
    -------
    padSize : int
        Amout of extra padding required to increase to next power of 2
    """
    import math

    next2Power = math.ceil(math.log(size, 2))
    next2Size = math.pow(2, int(next2Power))
    return int(next2Size) - size
