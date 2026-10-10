import numpy as np
import scipy.signal as signal
from typing import List

from resistics.common.math import intdiv


def smooth1d(x: np.ndarray, winLen: int = 11, window: str = "hann"):
    """Smooth in 1 dimension
    一维平滑
    Parameters
    ----------
    x : np.ndarray
        The data array to be smoothed
    winLen : int
        The number of samples to smooth across
    window : str
        The window function. Default is "hann".
	
    Notes
    ----- 
    The data is padded before the smoothing to ensure that the output data is the same size as the input. 
    平滑处理前对数据进行填充操作，以保证输出数据与输入数据的维度一致
    todo:
    The window parameter could be the window itself if an array instead of a string 若window参数为数组（而非字符串），则可以直接将其作为窗口函数使用
	"""
    # 输入验证
    if x.ndim != 1:                                                                                     # 检查输入数组是否为一维数组。ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < winLen:                                                                                 # 检查输入数组的长度是否大于等于窗口长度
        raise ValueError("Input vector needs to be bigger than window size.")
    if winLen < 3:
        return x        # 窗口太小，不进行平滑

    s = np.pad(x, (winLen, winLen), mode="edge")                                               # 对输入数组进行边缘填充，填充长度为窗口长度，填充方式为边缘值填充
    if window == "flat":  # moving average                                                              # 根据窗口类型选择窗口函数
        w = np.ones(winLen, "d")                                                                  # 如果窗口类型是 "flat"，即移动平均，创建一个全为1的窗口数组
    else:
        w = eval("signal." + window + "(winLen)")                                                       # 获取指定类型的窗口函数。使用 eval 动态构建窗口函数，例如 signal.hann(winLen)

    # assume winLen is odd(假设 winLen 是奇数)
    off = winLen + intdiv((winLen - 1), 2)                                                          # 计算偏移量，假设窗口长度为奇数，以保证对称性(计算卷积结果的起始偏移量)
    if winLen % 2 == 0:  # check if even and recalc off(检查（winLen）是否为偶数并重新计算偏移量)              # 如果窗口长度为偶数，重新计算偏移量
        off = winLen + intdiv(winLen, 2)                                                            # 重新计算偏移量以保证对称性

    y = np.convolve(s, w / w.sum(), mode="full")                                                        # 进行卷积操作，计算加权平均,得到平滑后的数据
    return y[off : off + x.size]                                                                        # 返回平滑后的数据，截取与输入数组长度相同的片段(返回卷积结果的中心部分，保证输出长度与输入相同)


def smooth2d(x: np.ndarray, winLen: List[int] = [5, 5], window: str = "hann"):
    """Smooth in 2-D

    This smooths in a window and across windows too

    Parameters
    ----------
    x : np.ndarray
        A 2-D array to be smoothed
    winLen : List[int]
        A two element list with winLen[0] being smoothing across windows and winLen[1] the smoothing within a window
    window : str
        The window function to use. Default is hann.
    """

    w1 = eval("signal." + window + "(winLen[0])")
    w2 = eval("signal." + window + "(winLen[1])")
    # calculate the 2d smoothing kernel
    kernel = np.outer(w1, w2)
    # pad to help the boundaries
    padded = np.pad(x, ((winLen[0], winLen[0]), (winLen[1], winLen[1])), mode="edge")
    # 2d smoothing
    blurred = signal.fftconvolve(padded, kernel, mode="same")
    return blurred[
        winLen[0] : winLen[0] + x.shape[0], winLen[1] : winLen[1] + x.shape[1]
    ]
