import numpy as np
import math
from typing import List


def getEvaluationFreq(fs: float, minFreq: float) -> np.ndarray:
    """Calculate evaluation frequencies with mimum allowable frequency

    Highest frequency is Nyquist/2. Lowest allowable frequency is provided by user
    f_i = f_max / pow(2,(i-1)/2)
    
    Parameters
    ----------
    fs : float
        Sampling frequency
    minFreq : float
        Minimum allowable frequency    

    Returns
    -------
    out : np.ndarray
        Array of evaluation frequencies
    """
    fmax: float = fs / 4
    freq: List = []
    i: int = 1
    f: float = fmax
    while f > minFreq:
        f = fmax / math.pow(2, (i - 1.0) / 2.0)
        freq.append(f)
        i = i + 1
    return np.array(freq)


def getEvaluationFreqSize(fs: float, numFreq: int) -> np.ndarray:
    """Calculate evaluation frequencies with maximum size
    按最大规模计算评估频率
    Highest frequency is Nyquist/2 and number of evaluation frequencies is set by user(最高频率为奈奎斯特频率的二分之一，评估频率的数量由用户设定)
    f_i = f_max / pow(2,(i-1)/2)
    这个函数的核心功能是：生成一个对数间隔的评估频率数组
    Parameters
    ----------
    fs : float
        Sampling frequency      采样频率（单位：Hz）15
    numFreq : int
        Number of evaluation frequencies    需要生成的频率点数49

    Returns
    -------
    out : np.ndarray
        Array of evaluation frequencies     一个长度为 numFreq的NumPy数组，包含从高频到低频的对数间隔频率值
    """
    fmax: float = fs / 3.2          # 避免混叠(fs=15,fmax=3.75)    奈奎斯特频率=fs / 2      这里选择 fs / 4作为实际最高频率，提供了2倍的安全系数（原本为fmax: float = fs / 4）
    freq: List = []               # 创建空列表用于存储生成的频率
    i: int = 1
    f: float = fmax
    while i <= numFreq:
        f = fmax / math.pow(2, (i - 1.0) / 2.0)             # pow(x, y)计算x的y次幂,生成半倍频程对数间隔的频率序列
        freq.append(f)
        i = i + 1
    return np.array(freq)