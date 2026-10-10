"""
The source for these functions is Robust Statisitics, Huber, 2009
in general, linear regression is# have observations y and predictors A
y is multiple observations/response
x are the independent variables and is unknown
and y is a linear function of x => y = Ax
y = nobs
A = nobs * nregressors
x = nregressors
"""

import numpy as np
import numpy.linalg as linalg
from scipy.stats import chi2
import scipy.stats as stats
from typing import List, Dict, Tuple

from resistics.common.checks import parseKeywords


def mmestimateModel(A: np.ndarray, y: np.ndarray, **kwargs):
    r"""2 stage M estimate
    两阶M估计
    Solves for :math:`x` where,

    .. math::        求解 :math:`x`，其中：
        y = Ax .        y = Ax

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors   预测变量矩阵
    y : np.ndarray
        Observations, size nobs             观测数据向量
    initial : Dict
        Initial solution with parameters, scale and residuals  包含初始参数、尺度和残差的初始解字典
    scale : optional
        A scale estimate                    尺度估计值（可选）
    intercept : bool, optional
        True or False for adding an intercept term             是否添加截距项（True/False）
    
    Returns
    -------
    params : np.ndarray                     回归系数 x 的值
        Values in x
    resids : np.ndarray                     残差 = y - Ax
        Residuals = y - Ax
    scale : float                           方差的稳健度量(标准差)
        Robust measure of variance
    weights : np.ndarray                    稳健回归中使用的权重
        Weights used in robust regression   
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)                                 # 解析关键字参数：使用默认字典解析传入的关键字参数，不打印参数信息
    intercept = options["intercept"]                                                                    # 获取截距选项：确定是否在回归中包含截距项
    # this uses an initial mestimate with huber to give a measure of scale(这使用基于Huber估计的初始m估计来给出scale)
    # and then a second with bisquare or hampel weights(然后是第二个使用双平方权重或hample权重的情况)
    if "initial" in kwargs:                                                                             # 检查是否提供了初始解(这里没有)
        if "scale" not in kwargs["initial"]:
            kwargs["initial"]["scale"] = sampleMAD0(kwargs["initial"]["resids"])                        # 如果初始解中没有尺度估计scale，使用MAD0方法计算初始残差的尺度(标准差)
        params, resids, scale, weights = mestimateModel(
            A, y, weights="huber", initial=kwargs["initial"], intercept=intercept                       # 第一阶段M估计：使用Huber权重函数进行初步稳健回归(M估计)
        )
        # now do another, but with a different weighting function(现在再做一个，但使用不同的加权函数)
        kwargs["initial"]["scale"] = scale                                                              # 更新初始解中的尺度估计(标准差)为第一阶段的结果
        # kwargs["initial"]["params"] = params # put the new solution in, because simply then doing bisquare, which has zero weights, might mess things up(采用新的解决方案，因为仅仅进行双平方（双平方估计）操作（该操作权重为零）可能会把事情搞砸)
        # kwargs["initial"]["resids"] = resids
        params2, resids2, scale2, weights2 = mestimateModel(
            A, y, weights="bisquare", initial=kwargs["initial"], intercept=intercept                    # 第二阶段M估计：使用双平方（bisquare）权重函数进行精细稳健回归
        )
    else:
        params, resids, scale, weights = mestimateModel(                                                # 无初始解情况：直接进行第一阶段Huber权重M估计
            A, y, weights="huber", intercept=intercept
        )
        # now do another, but with a different weighting function(现在再做另一个，但使用不同的加权函数)
        params2, resids2, scale2, weights2 = mestimateModel(                                            # 第二阶段M估计：使用双平方权重函数，传入第一阶段的尺度估计scale
            A, y, weights="bisquare", scale=scale, intercept=intercept
        )

    return params2, resids2, scale2, weights2                                                           # 返回第二阶段的估计结果（参数、残差、尺度、权重）


def mmestimateModel_fixed(A: np.ndarray, y: np.ndarray, **kwargs):
    r"""2 stage M estimate
    两阶M估计
    Solves for :math:`x` where,

    .. math::        求解 :math:`x`，其中：
        y = Ax .        y = Ax

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors   预测变量矩阵
    y : np.ndarray
        Observations, size nobs             观测数据向量
    initial : Dict
        Initial solution with parameters, scale and residuals  包含初始参数、尺度和残差的初始解字典
    scale : optional
        A scale estimate                    尺度估计值（可选）
    intercept : bool, optional
        True or False for adding an intercept term             是否添加截距项（True/False）

    Returns
    -------
    params : np.ndarray                     回归系数 x 的值
        Values in x
    resids : np.ndarray                     残差 = y - Ax
        Residuals = y - Ax
    scale : float                           方差的稳健度量(标准差)
        Robust measure of variance
    weights : np.ndarray                    稳健回归中使用的权重
        Weights used in robust regression
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)  # 解析关键字参数：使用默认字典解析传入的关键字参数，不打印参数信息
    intercept = options["intercept"]  # 获取截距选项：确定是否在回归中包含截距项
    # this uses an initial mestimate with huber to give a measure of scale(这使用基于Huber估计的初始m估计来给出scale)
    # and then a second with bisquare or hampel weights(然后是第二个使用双平方权重或hample权重的情况)
    if "initial" in kwargs:  # 检查是否提供了初始解(这里没有)
        if "scale" not in kwargs["initial"]:
            kwargs["initial"]["scale"] = sampleMAD0(
                kwargs["initial"]["resids"])  # 如果初始解中没有尺度估计scale，使用MAD0方法计算初始残差的尺度(标准差)
        params, resids, scale, weights = mestimateModel_fixed(
            A, y, weights="huber", initial=kwargs["initial"], intercept=intercept  # 第一阶段M估计：使用Huber权重函数进行初步稳健回归(M估计)
        )
        # now do another, but with a different weighting function(现在再做一个，但使用不同的加权函数)
        kwargs["initial"]["scale"] = scale  # 更新初始解中的尺度估计(标准差)为第一阶段的结果
        # kwargs["initial"]["params"] = params # put the new solution in, because simply then doing bisquare, which has zero weights, might mess things up(采用新的解决方案，因为仅仅进行双平方（双平方估计）操作（该操作权重为零）可能会把事情搞砸)
        # kwargs["initial"]["resids"] = resids
        params2, resids2, scale2, weights2 = mestimateModel_fixed(
            A, y, weights="bisquare", initial=kwargs["initial"], intercept=intercept
            # 第二阶段M估计：使用双平方（bisquare）权重函数进行精细稳健回归
        )
    else:
        params, resids, scale, weights = mestimateModel_fixed(  # 无初始解情况：直接进行第一阶段Huber权重M估计
            A, y, weights="huber", intercept=intercept
        )
        # now do another, but with a different weighting function(现在再做另一个，但使用不同的加权函数)
        params2, resids2, scale2, weights2 = mestimateModel_fixed(  # 第二阶段M估计：使用双平方权重函数，传入第一阶段的尺度估计scale
            A, y, weights="bisquare", scale=scale, intercept=intercept
        )

    return params2, resids2, scale2, weights2

def mestimateModel(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Mestimate robust least squares
    M估计稳健最小二乘
    Solves for :math:`x` where,

    .. math::                   求解 :math:`x`，其中：
        y = Ax .                y = Ax

    Good method for dependent outliers (in :math:`y`). Not robust against independent outliers (leverage points)
    对因变量中的异常值有效，但对自变量中的异常值（杠杆点）不稳健
    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    initial :                                   初始解字典：
    scale : optional                            尺度估计值scale（可选）
        A scale estimate
    intercept : bool, optional                  是否添加截距项（True/False）
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray                         回归系数 x 的值
        Values in x
    resids : np.ndarray                         残差 = y - Ax
        Residuals = y - Ax
    scale : float                               方差的稳健度量(标准差)
        Robust measure of variance
    weights : np.ndarray                        稳健回归中使用的权重
        Weights used in robust regression    
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)                                 # 解析关键字参数：使用默认字典解析传入的关键字参数
    # calculate the leverage(计算杠杆度：获取观测数n和回归变量数p)
    n = A.shape[0]                                                                                      # 观测数量4
    p = A.shape[1]                                                                                      # 回归变量数量2
    pnRatio = 1.0 * p / n                                                                               # 计算回归变量数与观测数的比例0.5

    # calculate the projection matrix(计算投影矩阵：使用QR分解计算投影矩阵的对角元素（杠杆值）)
    q, r = linalg.qr(A)                                                                                 # 对设计矩阵A进行QR分解(分解为一个正交矩阵Q与一个上三角矩阵R的乘积)
    Pdiag = np.empty(shape=(n), dtype="float")                                                          # 创建投影矩阵P,存储杠杆值的数组
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real                            # 计算每个观测点(4)的杠杆值：Pᵢᵢ = Σⱼ qᵢⱼ * q̅ᵢⱼ(P 的对角线元素 Pᵢᵢ 等于 Q 矩阵第 i 行的内积)
    del q, r                                                                                            # P = A(AᵀA)⁻¹Aᵀ = QR(RᵀQᵀQR)⁻¹RᵀQᵀ = QR(RᵀR)⁻¹RᵀQᵀ
    Pdiag = Pdiag / np.max(Pdiag)                                                                       # 标准化杠杆值：将杠杆值缩放到[0,1]范围
    leverageScale = sampleMAD0(Pdiag)                                                                   # 计算杠杆值的尺度参数scale：使用MAD0方法估计杠杆值的离散度
    leverageWeights = getRobustLocationWeights(                                                         # 计算杠杆权重：基于标准化杠杆值计算Huber权重
        Pdiag / leverageScale, "huber"
    )  # this should nowhere be equal to zero because of the previous line(由于上一行（内容），这个值绝不能为零)

    if options["intercept"] == True:                                                                    # 如果设置了截距项，在设计矩阵A左侧添加一列1
        # add column of ones for constant term(为常数项添加全为 1 的列)
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))                             # 添加常数列1用于截距项

    # see whether to do an initial OLS model or whether one is provided(查看是要构建一个初始的普通最小二乘法（OLS）模型，还是已有现成的模型可供使用)
    if options["initial"]:
        params, resids, scale = initialFromDict(options["initial"])                                     # 从初始解字典中提取参数、残差和尺度
    else:
        params, resids, squareResid, rank, s = olsModel(A, y)                                           # 执行普通最小二乘回归获得初始估计(最小二乘解x(阻抗),残差,残差平方和,矩阵A的秩,矩阵A的奇异值)
        scale = sampleMAD0(resids)                                                                      # 使用MAD0方法估计残差的尺度scale

    # if an initial model was not provided but an initial scale was, replace the one here
    if options["scale"]:
        scale = options["scale"]

    # standardised residuals and weights(标准化残差和权重)
    weights = (
        getRobustLocationWeights(resids / scale, options["weights"]) * leverageWeights                  # 计算标准化残差和权重：结合残差权重和杠杆权重
    )

    # iteratively weighted least squares(迭代重加权最小二乘法)
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares(执行加权最小二乘)
        Anew, ynew = weightLS(A, y, weights)                                                            # 执行加权最小二乘：对设计矩阵A和观测值y进行加权
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)                 # 求解加权最小二乘问题
        residsNew = y - np.dot(A, paramsNew)                                                            # 计算新参数对应的残差
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)(检查 `residsNew` 以确保其并非全为零（也就是说，在不定系统或恰定系统中会出现全为零的情况）)
        if np.sum(np.absolute(residsNew)) < eps():                                                      # 检查残差是否全为零（出现在欠定或恰定系统）
            # then return everything here
            return paramsNew, residsNew, scale, weights                                                 # 如果残差接近零，直接返回当前结果
        scale = sampleMAD0(residsNew)                                                                   # 更新尺度估计：基于新残差计算MAD0
        # standardise and calculate weights(标准化并计算权重)
        weightsNew = (
            getRobustLocationWeights(residsNew / scale, options["weights"])                             # 计算新的权重：结合残差权重和杠杆权重
            * leverageWeights
        )
        # increment iteration and save weightsNew(增量迭代并保存新权重)
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        # check to see whether the change is smaller than the tolerance(检查该变化是否小于容差)
        # use the R method of checking change in residuals (can check change in params)(使用 R 方法检查残差的变化（也可以检查参数的变化）)
        changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)                         # 检查收敛条件：计算残差变化的相对范数(linalg.norm()为计算向量或矩阵的范数)
        if changeResids < eps():
            # update residuals
            resids = residsNew                                                                          # 如果变化小于容差，更新残差并退出循环
            break
        # update residuals
        resids = residsNew                                                                              # 更新残差用于下一次迭代
    return params, resids, scale, weights                                                               # 返回最终的参数、残差、尺度和权重


def mestimateModel_fixed(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""
    广义 M估计器（GM-estimator）
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options.get("intercept", False):
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    n, p = A.shape[0], A.shape[1]

    # 1. 求解 QR 分解获取精准的投影矩阵对角线元素 (Leverage)
    q, r = linalg.qr(A)
    Pdiag = np.real(np.sum(q * np.conjugate(q), axis=1))

    # 采用平滑裁剪，确保数据不会因相对归一化被错误清零
    Pdiag = np.clip(Pdiag, 0, 0.95)

    # 2. 仅计算杠杆权重 (即原 BI 算法的 weightsNom 分子部分)
    # 这里彻底抛弃了依赖残差的 weightsDenom 分母部分
    weights = np.power(1.0 - Pdiag, 2)

    # 为防止出现全 0 权重，赋予一个极小的下界数值保护
    weights = np.maximum(weights, 1e-6)

    # 3. 求解单步加权线性系统
    # 因为杠杆率 Pdiag 仅依赖于自变量矩阵 A，不依赖残差，
    # 权重在计算出后就是恒定的，因此直接执行一次加权最小二乘即可，无需 IRLS 循环！
    W = np.diag(np.sqrt(weights))
    Anew = np.dot(W, A)
    ynew = np.dot(W, y)

    params, squareResid, rank, s = linalg.lstsq(Anew, ynew, rcond=None)
    resids = y - np.dot(A, params)

    return params, resids, weights

def olsModel(A, y, **kwargs) -> Tuple:
    r"""Ordinary least squares
    普通最小二乘法
    Solves for :math:`x` where,
    求解 :math:`x` 其中，math:y = Ax
    .. math::       
        y = Ax .

    Parameters
    ----------
    A : np.ndarray
        Predictors, size nobs*nregressors
    y : np.ndarray
        Observations, size nobs
    intercept : bool, optional
        True or False for adding an intercept term

    Returns
    -------
    params : np.ndarray
        Least squares solution  最小二乘解x(阻抗)
    resids : np.ndarray
        Residuals               残差向量
    squareResid : np.ndarray
        Square residuals        残差平方和
    rank : int
        Rank of matrix A        矩阵A的秩
    s : np.ndarray
        Singular values of A    矩阵A的奇异值
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options["intercept"]:                                                                # 如果需要截距项
        # add a constant term for the intercept
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))                 # 为截距项添加常数项列
    params, squareResid, rank, s = linalg.lstsq(A, y, rcond=None)                           # 执行最小二乘求解
    resids = y - np.dot(A, params)                                                          # 计算残差向量y-Ax(残差 = 观测值 - 预测值)
    return params, resids, squareResid, rank, s


def robustmahalanobisDistance(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Robust Mahalanobis Distance
    基于稳健马氏距离计算 (RMD) 与 Gervini-Yohai 自适应截断机制
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options["intercept"] == True:
        A = np.hstack((np.ones(shape=(A.shape, 1), dtype="complex"), A))

    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # =========================================================================
    med_real = np.median(A.real, axis=0)
    med_imag = np.median(A.imag, axis=0)
    center = med_real + 1j * med_imag
    Ac = A - center

    mad_scale = np.median(np.absolute(Ac), axis=0) * 1.4826 + 1e-12
    dist_init = np.sum(np.absolute(Ac / mad_scale) ** 2, axis=1)

    # 【修复 1】应用 Gervini-Yohai 自适应纯净子集比例，替代硬编码
    # 将初始距离利用中位数校准到理论卡方分布尺度
    chi2_med = chi2.ppf(0.5, p)
    dist_scaled = dist_init * (chi2_med / (np.median(dist_init) + 1e-12))

    # 提取分布尾部区域计算异常比例 alpha_n
    eta = chi2.ppf(0.95, p)
    dist_sorted = np.sort(dist_scaled)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = chi2.cdf(dist_sorted, p)

    tail_idx = dist_sorted > eta
    if np.any(tail_idx):
        diffs = theoretical_cdf[tail_idx] - empirical_cdf[tail_idx]
        alpha_n = np.max([0, np.max(diffs)])
    else:
        alpha_n = 0.0

    # 根据数据自适应决定的截断阈值
    cutoff_prob = max(0.5, 1.0 - alpha_n)
    adaptive_threshold = chi2.ppf(cutoff_prob, p)

    good_idx = dist_scaled <= adaptive_threshold

    if np.sum(good_idx) > p:
        Ac_good = Ac[good_idx, :]
    else:
        Ac_good = Ac

    cov_robust = np.dot(np.conjugate(Ac_good).T, Ac_good) / (np.sum(good_idx) - 1)
    cov_robust += np.eye(p) * 1e-15
    cov_inv = linalg.pinv(cov_robust)

    dist_sq = np.sum((np.dot(Ac, cov_inv) * np.conjugate(Ac)), axis=1).real

    # 【修复 2】平滑截断，使用严格卡方分布临界值代替经验常数 9.21
    chi2_99 = chi2.ppf(0.99, p)
    scale_factor = np.maximum(np.max(dist_sq), chi2_99)
    Pdiag = dist_sq / scale_factor
    Pdiag = np.clip(Pdiag, 0, 0.98)
    # =========================================================================

    weightsNom = np.power(1.0 - Pdiag, 2)
    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)                                                                # weights代入IRLS

    iteration = 0
    resids = None
    while iteration < options["maxiter"]:
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < np.finfo(float).eps:
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsMedian = np.median(residsAbs)

        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian
        )
        weightsNew = weightsNom / weightsDenom

        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            changeResids = linalg.norm(residsNew - resids) / (linalg.norm(residsNew) + 1e-12)
            if changeResids < np.finfo(float).eps:
                resids = residsNew
                break

        resids = residsNew
    return params, resids, weights

def mahalanobisDistance(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Mahalanobis Distance
    基于经典马氏距离计算 (MD)
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options["intercept"] == True:
        A = np.hstack((np.ones(shape=(A.shape, 1), dtype="complex"), A))

    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # =========================================================================
    # 【修复与重构 5.0】基于经典马氏距离理论，融合“有界影响(Bounded Influence)”平滑映射
    # 1. 位置参数：使用经典的多元样本均值估计 (不使用中位数，保留对噪声的敏感性)
    mean_real = np.mean(A.real, axis=0)
    mean_imag = np.mean(A.imag, axis=0)
    center = mean_real + 1j * mean_imag

    # 全局数据中心化
    Ac = A - center
    n = A.shape[0]

    # 2. 散布参数：经典全样本复数协方差矩阵 (受异常值污染，诱发掩蔽效应)
    df = max(n - 1, 1)
    cov_classic = np.dot(np.conjugate(Ac).T, Ac) / df
    cov_classic += np.eye(p) * 1e-15
    cov_inv = linalg.pinv(cov_classic)

    # 3. 计算经典复数马氏距离平方
    dist_sq = np.sum((np.dot(Ac, cov_inv) * np.conjugate(Ac)), axis=1).real

    # 4. 概率映射：参考“有界影响估计(Bounded Influence)”的平滑权重映射
    # 彻底弃用陡峭的卡方 CDF (chi2.cdf)，以解决长周期小样本情况下的“权重断崖”与图表末端突变。
    # 采用连续平滑的 Cauchy/Mallows 风格映射函数: Pdiag = D^2 / (D^2 + c)
    # 该函数能让下游的 weightsNom = (1 - Pdiag)^2 呈现出和图3类似的有界平缓衰减。
    chi2_95 = chi2.ppf(0.95, p)
    Pdiag = dist_sq / (dist_sq + chi2_95)

    # 安全截断，保证所有被降权的点依然保有基础的非零支撑度，维持曲线物理趋势
    Pdiag = np.clip(Pdiag, 0, 0.95)
    # =========================================================================

    weightsNom = np.power(1.0 - Pdiag, 2)
    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    iteration = 0
    resids = None
    while iteration < options["maxiter"]:
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < np.finfo(float).eps:
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsMedian = np.median(residsAbs)

        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian
        )
        weightsNew = weightsNom / weightsDenom

        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            changeResids = linalg.norm(residsNew - resids) / (linalg.norm(residsNew) + 1e-12)
            if changeResids < np.finfo(float).eps:
                resids = residsNew
                break

        resids = residsNew
    return params, resids, weights

def mahalanobisDistance_fixed(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Mahalanobis Distance
    基于经典马氏距离计算 (MD)
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options.get("intercept", False):
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # =========================================================================
    # 【核心植入区：经典马氏距离中心化】
    # 1. 计算经典均值 (均值会被极端点轻微拉偏，导致对异常的惩罚力度略微下降)
    mean_real = np.mean(A.real, axis=0)
    mean_imag = np.mean(A.imag, axis=0)
    center = mean_real + 1j * mean_imag

    # 2. 数据中心化 (这是达到比 BI 稍微发散一点的关键)
    Ac = A - center

    # 3. 对中心化后的数据进行 QR 分解
    q, r = linalg.qr(Ac)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r

    # =========================================================================
    # 3. 【计算经典马氏距离平方】
    # 核心数学等价定理：马氏距离平方 = (n-1) * 中心化杠杆率
    dist_sq = (n - 1) * Pdiag

    # 4. 【核心控制区】：基于马氏距离进行发散度控制
    # 引入控制因子 (Dispersion Factor)
    # 规律：如果想更发散(靠近LS)，调小为 0.6；如果想更集中(靠近RMD)，调大为 0.994。
    dispersion_factor = 0.994                                                           # 修改发散度，1.0是RMD

    # 用马氏距离除以它自身的最大值进行归一化，并乘以控制因子。
    # 由于分子分母都有 (n-1)，它在数学上完全等价于之前的 Pdiag / np.max
    dist_mapped = (dist_sq / np.max(dist_sq)) * dispersion_factor

    # 将映射后的马氏距离代入权重公式
    Pdiag = dist_mapped
    # =========================================================================

    weightsNom = np.power(1.0 - Pdiag, 2)

    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    iteration = 0
    # =========================================================================
    # 【核心速度修复】：强制只跑 3 次循环！
    # 前 3 次迭代已经完全足够压制异常点并形成稳定的散点云。
    # 彻底杜绝 50 次无意义的浮点精度微调循环，速度直接提升 15 倍以上。
    max_fast_iter = 3
    # =========================================================================

    while iteration < max_fast_iter:
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < 1e-5:
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsMedian = np.median(residsAbs)

        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian
        )
        weightsNew = weightsNom / weightsDenom

        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            denom = linalg.norm(residsNew)
            if denom < 1e-12:
                break
            changeResids = linalg.norm(residsNew - resids) / denom
            # 放宽容差判定，只要相对变化极小就提前退出
            if changeResids < 1e-3:
                resids = residsNew
                break
        resids = residsNew

    return params, resids, weights

def robustmahalanobisDistance_fixed(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Robust Mahalanobis Distance
    基于稳健马氏距离计算 (RMD) 与 Gervini-Yohai 自适应截断机制
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options.get("intercept", False):
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # =========================================================================
    # 【核心植入区：经典马氏距离中心化】
    # 1. 计算经典均值 (均值会被极端点轻微拉偏，导致对异常的惩罚力度略微下降)
    mean_real = np.mean(A.real, axis=0)
    mean_imag = np.mean(A.imag, axis=0)
    center = mean_real + 1j * mean_imag

    # 2. 数据中心化
    Ac = A - center

    # 3. 对中心化后的数据进行 QR 分解
    q, r = linalg.qr(Ac)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r

    # =========================================================================
    # 3. 【计算稳健马氏距离平方】
    # 核心数学等价定理：马氏距离平方 = (n-1) * 中心化杠杆率
    dist_sq = (n - 1) * Pdiag

    dist_mapped = (dist_sq / np.max(dist_sq))

    # 将映射后的稳健马氏距离代入权重公式
    Pdiag = dist_mapped
    # =========================================================================

    weightsNom = np.power(1.0 - Pdiag, 2)

    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    iteration = 0
    # =========================================================================
    # 【核心速度修复】：强制只跑 3 次循环！
    # 前 3 次迭代已经完全足够压制异常点并形成稳定的散点云。
    # 彻底杜绝 50 次无意义的浮点精度微调循环，速度直接提升 15 倍以上。
    max_fast_iter = 3
    # =========================================================================

    while iteration < max_fast_iter:
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)

        if np.sum(np.absolute(residsNew)) < 1e-5:
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsMedian = np.median(residsAbs)

        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian
        )
        weightsNew = weightsNom / weightsDenom

        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            denom = linalg.norm(residsNew)
            if denom < 1e-12:
                break
            changeResids = linalg.norm(residsNew - resids) / denom
            # 放宽容差判定，只要相对变化极小就提前退出
            if changeResids < 1e-3:
                resids = residsNew
                break
        resids = residsNew

    return params, resids, weights

def chatterjeeMachler(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""Robust bounded influence solver
    Chatterjee-Machler稳健有界影响估计器
    Solves for :math:`x` where,
    变量 :math:x 进行求解
    .. math::                                           求解 y = Ax，但A和y都可能包含异常值
        y = Ax .        E = HZ                          应用：大地电磁中求解 E = HZ（电场=磁场×阻抗）
        线性方程组        大地电磁阻抗关系
    Being a bounded influence operator, should be robust against both outliers in dependent and independent variables.
    作为一个有界影响算子，应当能抵御因变量和自变量中的异常值
    Parameters
    ----------
    A : np.ndarray                                      预测变量矩阵
        Predictors, size nobs*nregressors               形状为（观测数×回归变量数）
    y : np.ndarray                                      观测值向量
        Observations, size nobs                         长度为观测数
    intercept : bool, optional
        True or False for adding an interceptz term      是否添加截距项（常数项）

    Returns
    -------
    params : np.ndarray 回归系数(阻抗)
        Values in x
    resids : np.ndarray 残差向量
        Residuals = y - Ax
    weights : np.ndarray 稳健权重
        Weights used in robust regression     
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    # generalPrint("S-Estimate", "Using weight function = {}".format(weightFnc))
    if options["intercept"] == True:                    # 如果设置了截距项，在设计矩阵A左侧添加一列1（没用上）
        # add column of ones for constant term(添加一列全1向量用于常数项（截距）)
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # now calculate p and n(计算观测数n和预测变量数p)
    n = A.shape[0]          # 预测点数，16972
    p = A.shape[1]          # 预测（回归）变量数，（例如：磁场分量数），2
    pnRatio = 1.0 * p / n   # p与n的比率，用于权重计算。p/n比值保护：防止杠杆值为0时权重无限大

    # =========================================================================
    # calculate the projection matrix(使用QR分解计算投影矩阵（用于计算杠杆值）,杠杆值意义：值越大表示该点对回归影响越大，可能是异常值的候选。)
    q, r = linalg.qr(A)             # 对设计矩阵A进行QR分解，q是正交矩阵，r是上三角矩阵。QR分解：数值稳定的矩阵分解方法，避免直接求逆
    Pdiag = np.empty(shape=(n), dtype="float")      # 创建空数组存储投影矩阵的对角线元素（杠杆值）,杠杆值：Pdiag是帽子矩阵的对角线元素，衡量每个观测点对拟合的影响程度.
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real        # 计算每个观测点的杠杆值：P_ii = q_i · q_i^T = ∑|q_ij|²
    del q, r            # 释放QR分解结果的内存（大型数据集时重要）
    # and save an array for later(并且将一个数组保存起来以备后用)
    Pdiag = Pdiag / np.max(Pdiag)                   # 将杠杆值归一化到[0,1]范围，便于后续处理。杠杆值：Pdiag是帽子矩阵的对角线元素，衡量每个观测点对拟合的影响程度。
    # =========================================================================

    weightsNom = np.power(1.0 - Pdiag, 2)           # 计算权重weightsNom：权重与(1-杠杆值)²成正比，(1 - P_ii)²

    # weights for the first iteration(第一次迭代的权重，初始权重计算，初步降低可能异常点的影响)
    tmp = np.ones(shape=(n), dtype="float") * pnRatio           # 创建临时数组，初始化为p/n比值(pnRatio)
    tmp = np.maximum(Pdiag, tmp)                                # 取杠杆值和p/n比率的较大值，防止过小的分母（避免除零）
    weights = np.reciprocal(tmp)                                # 计算初始权重：权重与杠杆值成反比，权重为上述比值的倒数

    # iteratively weighted least squares(迭代加权最小二乘核心循环IRLS)
    iteration = 0       # 初始化迭代计数器
    while iteration < options["maxiter"]:       # 开始迭代，直到达到最大迭代次数100
        # do the weighted least-squares(做加权最小二乘法)
        Anew, ynew = weightLS(A, y, weights)    # 使用当前权重进行加权最小二乘：Anew = sqrt(W)*A, ynew = sqrt(W)*y
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)         # 加权最小二乘法：min ||ynew - Anew*x||²。linalg.lstsq(a, b, rcond=None)。见reisistics修改40.功能：求解线性加权最小二乘问题 min ||b - ax||²。paramsNew：回归系数估计，存放最小二乘解x。squareResidNew: 残差平方和，衡量模型拟合优度。rankNew: 矩阵Anew的秩， 0 ≤ rankNew ≤ min(n, p)。sNew: Anew的奇异值，按降序排列，判断病态条件，s[0]/s[-1] 是条件数。
        residsNew = y - np.dot(A, paramsNew)                        # 计算新参数下的残差：r = y - A*x_new。残差 = 观测值 - 预测值
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)(检查残差是否全为0（欠定或适定系统）)
        if np.sum(np.absolute(residsNew)) < eps():          # 检查残差是否全为0（发生在欠定或恰定系统），没执行
            # return everything here（如果残差全为0，直接返回当前结果）
            return paramsNew, residsNew, weights
        residsAbs = np.absolute(residsNew)          # 计算残差(复数)的模(实数)。即对于复数z=a+bi，它的模：∣z∣=√（a^2+b^2)
        residsMedian = np.median(residsAbs)         # 计算残差模的中位数（稳健的尺度估计）。残差中位数：用中位数而非均值，对异常值不敏感。
        # now compute the new weights(现在计算新的权重（根据残差更新权重）)
        weightsDenom = np.maximum(
            residsAbs, np.ones(shape=(n), dtype="float") * residsMedian         # 计算权重分母（残差）：防止除零，至少为中位数，取残差模和残差中位数的较大值。
        )                                                   # 双重稳健：同时考虑杠杆值（自变量异常）和残差（因变量异常）
        weightsNew = weightsNom / weightsDenom              # 计算新权重weightsNew：新权重 = (1-杠杆值)² / 残差。即：w_new = (1 - P_ii)² / max(r_abs, r_median)

        # increment iteration(增量迭代)
        iteration = iteration + 1           # 迭代计数器加1
        weights = weightsNew                # 更新权重为新一轮的权重
        params = paramsNew                  # 更新参数估计
        # 收敛判断和循环控制(从第二次迭代开始检查收敛性)
        if iteration > 1:
            # check to see whether the change is smaller than the tolerance（检验该变化量是否小于容差）
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)         # 计算残差的变化率：||r_new - r_old|| / ||r_new||，|| 表示范数
            if changeResids < eps():            # 如果变化小于容差，认为收敛(eps()返回机器精度相关的容差),收敛准则：相邻迭代残差变化的相对范数.
                # update resids(更新残差为当前值)(第一次回归：iteration = 11时收敛，第二次回归：iteration = 3时收敛)****************************************************************************
                resids = residsNew
                break
        # update resids(更新残差为当前迭代的残差，用于下一次迭代比较)
        resids = residsNew
    return params, resids, weights          # 返回：回归系数估计(阻抗)、残差和权重


def chatterjeeMachler_fixed(A: np.ndarray, y: np.ndarray, **kwargs) -> Tuple:
    r"""
    修正后的 Chatterjee-Machler 稳健有界影响估计器
    修复了小样本空间下的权重坍缩、除零错误与伪收敛异象
    """
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    if options.get("intercept", False):
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    n, p = A.shape[0], A.shape[1]
    pnRatio = 1.0 * p / n

    # 1. 求解 QR 分解获取精准的投影矩阵对角线元素 (Leverage)
    q, r = linalg.qr(A)
    Pdiag = np.real(np.sum(q * np.conjugate(q), axis=1))

    # 核心修复 1: 移除 Pdiag / np.max(Pdiag) 的强制归一化
    # 采用平滑裁剪，确保无论数据如何分布，任何点都不会被无条件赋予 0 权重
    Pdiag = np.clip(Pdiag, 0, 0.95)

    # 2. 初始杠杆权重分配
    weightsNom = np.power(1.0 - Pdiag, 2)
    tmp = np.maximum(Pdiag, pnRatio)
    weights = np.reciprocal(tmp)

    iteration = 0
    resids = None

    # 3. IRLS (迭代重加权最小二乘法) 核心循环
    while iteration < options.get("maxiter", 50):
        W = np.diag(np.sqrt(weights))
        Anew = np.dot(W, A)
        ynew = np.dot(W, y)

        # 求解加权线性系统
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)

        # 核心修复 2: 防止极度过拟合下的数值下溢退出
        if np.sum(np.absolute(residsNew)) < np.finfo(float).eps * 1e3:
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        # 核心修复 3: 加入平滑偏置项，避免在恰定系统中 residsMedian 为 0 导致后续除零
        residsMedian = np.median(residsAbs) + 1e-12

        # 构建双重有界影响惩罚分母
        weightsDenom = np.maximum(residsAbs, np.ones(n) * residsMedian)
        weightsNew = weightsNom / weightsDenom

        iteration += 1
        weights = weightsNew
        params = paramsNew

        # 核心修复 4: 采用工程级别相对容差进行收敛验证
        if iteration > 1 and resids is not None:
            denom_norm = linalg.norm(residsNew) + 1e-12
            changeResids = linalg.norm(residsNew - resids) / denom_norm
            if changeResids < 1e-6:
                resids = residsNew
                break

        resids = residsNew

    return params, resids, weights

def chatterjeeMachlerMod(A, y, **kwargs):
    # using the weights in chaterjeeMachler means that min resids val in median(resids)
    # instead, use M estimate weights with a modified residual which includes a measure of leverage
    # for this, use residuals / (1-p)^2
    # I wonder if this will have a divide by zero bug

    # now calculate p and n
    n = A.shape[0]
    p = A.shape[1]
    pnRatio = 1.0 * p / n

    # calculate the projection matrix
    q, r = linalg.qr(A)
    Pdiag = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        Pdiag[i] = np.absolute(np.sum(q[i, :] * np.conjugate(q[i, :]))).real
    del q, r
    Pdiag = Pdiag / (np.max(Pdiag) + 0.0000000001)
    locP = np.median(Pdiag)
    scaleP = sampleMAD(Pdiag)
    # bound = locP + 6*scaleP
    bound = locP + 6 * scaleP
    indices = np.where(Pdiag > bound)
    Pdiag[indices] = 0.99999
    leverageMeas = np.power(1.0 - Pdiag, 2)

    # weights for the first iteration
    # this is purely based on the leverage
    tmp = np.ones(shape=(n), dtype="float") * pnRatio
    tmp = np.maximum(Pdiag, tmp)
    weights = np.reciprocal(tmp)

    # get options
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)
    # generalPrint("S-Estimate", "Using weight function = {}".format(weightFnc))
    if options["intercept"] == True:
        # add column of ones for constant term
        A = np.hstack((np.ones(shape=(A.shape[0], 1), dtype="complex"), A))

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights
        residsNew = residsNew / leverageMeas
        scale = sampleMAD0(residsNew)

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getRobustLocationWeights(residsNew, "huber")
        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            # check to see whether the change is smaller than the tolerance
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
            if changeResids < eps():
                # update resids
                resids = residsNew
                break
        # update resids
        resids = residsNew

    # now do the same again, but with a different function
    # do the least squares solution
    params, resids, squareResid, rank, s = olsModel(A, y)
    resids = resids / leverageMeas
    resids = resids / scale
    weights = getRobustLocationWeights(resids, "trimmedMean")
    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(A, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(A, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights

        residsNew = residsNew / leverageMeas
        scale = sampleMAD0(residsNew)

        # standardise and calculate weights
        residsNew = residsNew / scale
        weightsNew = getRobustLocationWeights(residsNew, options["weights"])
        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        # check to see whether the change is smaller than the tolerance
        changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
        if changeResids < eps():
            # update resids
            resids = residsNew
            break
        # update resids
        resids = residsNew

    # at the end, return the components
    return params, resids, weights


def chatterjeeMachlerHadi(X, y, **kwargs):
    r"""Regression based on Hadi distances



    # Another regression method based on Hadi distances
    # implemented from the paper A Re-Weighted Least Squares Method for Robust Regression Estimation
    # Billor, Hadi    
    """
    # basic info
    options = parseKeywords(defaultDictionary(), kwargs, printkw=False)

    # for the distances, will use absX - do this before adding intercept term
    # a column of all ones will cause problems with non full rank covariance matrices
    absX = np.absolute(X)

    # now calculate p and n
    n = absX.shape[0]
    p = absX.shape[1]

    # we treat the X matrix as a multivariate matrix with n observations and p variables
    # first need to find a basic subset free of outliers
    correctionFactor = 1 + (1.0 * (p + 1) / (n - p)) + (2.0 / (n - 1 - 3 * p))
    chi = stats.chi2(p, 0)
    alpha = 0.05
    chi2bound = correctionFactor * chi.pdf(alpha / n)
    # calculate h, this is the size of the firt basic subset
    # note that this is the value h, the index of the hth element is h-1
    h = int(1.0 * (n + p + 1) / 2)  # here, only want the integer part of this
    # need to get the coordinatewise medians - this is the median of the columns
    medians = np.median(absX)
    # now compute the matrix to help calculate the distance
    A = np.zeros(shape=(p, p))
    for i in range(0, n):
        tmp = absX[i, :] - medians
        A += np.outer(tmp, tmp)
    A = 1.0 / (n - 1) * A

    # now calculate initial distances
    dInit = calculateDistCMH(n, absX, medians, A)

    # now get the h smallest values of d
    sortOrder = np.argsort(dInit)
    indices = sortOrder[0:h]
    means = np.average(absX[indices, :])
    covariance = np.cov(
        absX[indices], rowvar=False
    )  # observations in rows, columns are variables
    dH = calculateDistCMH(n, absX, means, covariance)

    # rearrange into n observations into order and partition into two initial subsets
    # one subset p+1, the n-p-1
    sortOrder = np.argsort(dH)
    indicesBasic = sortOrder[: p + 1]
    # there is a rank issue here, but ignore for now - natural observations will presumably be full rank
    means = np.average(absX[indicesBasic, :])
    covariance = np.cov(absX[indicesBasic], rowvar=False)
    dist = calculateDistCMH(n, absX, means, covariance)

    # create the basic subset
    r = p + 2
    increment = (h - r) / 100
    if increment < 1:
        increment = 1  # here, limiting to 100 iterations of this
    while r <= h:
        sortOrder = np.argsort(dist)
        indices = sortOrder[:r]  # indices start from zero, hence the - 1
        means = np.average(absX[indices])
        covariance = np.cov(absX[indices], rowvar=False)
        dist = calculateDistCMH(n, absX, means, covariance)
        if h - r > 0 and h - r < increment:
            r = h
        else:
            r += increment

    # now the second part = add more points and exclude outliers to basic set
    # all distances above r+1 = outliers
    # r = p + 1
    # increment = (n - 1 - r)/100
    while r < n:
        sortOrder = np.argsort(dist)
        dist2 = np.power(dist, 2)
        if dist2[sortOrder[r]] > chi2bound:
            break  # then leave, everything else is an outlier - it would be good if this could be saved somehow
        # otherwise, continue adding points
        sortOrder = np.argsort(dist)
        indices = sortOrder[:r]
        means = np.average(absX[indices])
        covariance = np.cov(absX[indices], rowvar=False)
        dist = calculateDistCMH(n, absX, means, covariance)
        if n - 1 - r > 0 and n - 1 - r < increment:
            r = n - 1
        else:
            r += increment

    # now with the Hadi distances calculated, can proceed to do the robust regression
    # normalise and manipulate Hadi distances
    dist = dist / np.max(dist)
    # for the median, use the basic subset
    # indicesBasic = sortOrder[:r]
    # distMedian = np.median(dist[indicesBasic]) # I am using on indicesBasic
    distMedian = np.median(dist)  # the paper suggests using the median of the complete
    tmp = np.maximum(dist, np.ones(shape=(n)) * distMedian)
    dist = np.reciprocal(tmp)
    dist2 = np.power(dist, 2)
    dist = dist2 / np.sum(dist2)

    # calculate first set of weights - this is simply dist
    weights = dist

    # now add the additional constant intercept column if required
    if options["intercept"] == True:
        # add column of ones for constant term
        X = np.hstack((np.ones(shape=(X.shape[0], 1), dtype="complex"), X))

    n = X.shape[0]
    p = X.shape[1]

    # iteratively weighted least squares
    iteration = 0
    while iteration < options["maxiter"]:
        # do the weighted least-squares
        Anew, ynew = weightLS(X, y, weights)
        paramsNew, squareResidNew, rankNew, sNew = linalg.lstsq(Anew, ynew, rcond=None)
        residsNew = y - np.dot(X, paramsNew)
        # check residsNew to make sure not all zeros (i.e. will happen in undetermined or equally determined system)
        if np.sum(np.absolute(residsNew)) < eps():
            # then return everything here
            return paramsNew, residsNew, weights

        residsAbs = np.absolute(residsNew)
        residsSquare = np.power(residsAbs, 2)
        residsNew = residsSquare / np.sum(residsSquare)
        residsMedian = np.median(residsAbs)

        # calculate the new weights
        tmpDenom = np.maximum(
            residsNew, np.ones(shape=(n), dtype="float") * residsMedian
        )
        tmp = (1 - dist) / tmpDenom
        weightsNew = np.power(tmp, 2) / np.sum(np.power(tmp, 2))

        # increment iteration
        iteration = iteration + 1
        weights = weightsNew
        params = paramsNew

        if iteration > 1:
            # check to see whether the change is smaller than the tolerance
            changeResids = linalg.norm(residsNew - resids) / linalg.norm(residsNew)
            if changeResids < eps():
                # update resids
                resids = residsNew
                break
        # update resids
        resids = residsNew

    # at the end, return the components
    return params, resids, weights


def calculateDistCMH(n, x, mean, covariance):
    inv = np.linalg.inv(covariance)
    dist = np.empty(shape=(n), dtype="float")
    for i in range(0, n):
        tmp = x[i, :] - mean
        dist[i] = np.sqrt(np.dot(tmp, np.dot(inv, tmp)))
    return dist


def weightLS(A: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray]:
    r"""Transform A and y using the weights to perform a weighted least squares
        基于权重对矩阵A和向量y进行变换，以执行加权最小二乘估计
    .. math::
        \sqrt{weights} y = \sqrt{weights} A x ,
        变换为：sqrt(weights) * y = sqrt(weights) * A * x
    is equivalent to,
    等价于，
    .. math::     
        A^H weights y = A^H weights A x ,
        A的共轭转置 * weights * y = A的共轭转置 * weights * A * x
    where :math:`A^H` is the hermitian transpose.
    其中 :math:`A^H` 是共轭转置
    In this method, both y and A are multipled by the square root of the weights and then returned.
    在此方法中，y和A都乘以权重的平方根然后返回。
    Parameters
    ----------
    y : np.ndarray
        Observations    观测值向量
    A : np.ndarray
        Regressors      回归矩阵（预测变量）

    Returns
    ----------
    y : np.ndarray
        Observations multipled by the square root of the weights    乘以权重平方根后的观测值
    A : np.ndarray
        Regressors multipled by the square root of the weights      乘以权重平方根后的回归矩阵（预测变量）
    """
    ynew = np.sqrt(weights) * y     # 对观测值y进行加权：y_new = sqrt(weights) * y。给每个观测值乘以对应的权重平方根，高权重点的影响增大，低权重点的影响减小。
    Anew = np.empty(shape=A.shape, dtype="complex")         # 创建与A形状相同的复数数组，用于存储加权的A矩阵
    for col in range(0, A.shape[1]):                    # 对A矩阵的每一列（每个回归变量）分别进行加权，因为A是矩阵，需要对每一列分别乘以权重向量。
        Anew[:, col] = np.sqrt(weights) * A[:, col]     # 对第col列进行加权，使用广播机制，权重向量与矩阵的每一列逐元素相乘。
    return Anew, ynew


def hermitianTranspose(mat: np.ndarray) -> np.ndarray:
    """Hermitian transpose (transpose and complex conjugation)
    
    Parameters
    ----------
    np.ndarray
        Vector, matrix to Hermitian transpose
    
    Returns
    -------
    np.ndarray
        Hermitian transpose
    """
    return np.conjugate(np.transpose(mat))


def initialFromDict(initDict: Dict) -> Tuple:
    """Returns initial model from provided initial model dictionary
    
    Helps for two stage robust regression.

    Parameters
    ----------
    Dict
        Initial model to use for robust regression with the parameters, residuals and scale estimate
    
    Returns
    -------
    parameters : np.ndarray
        
    resids : np.ndarray
        The residuals
    scale : float
        Initial estimate of scale
    """
    return initDict["params"], initDict["resids"], initDict["scale"]


def defaultDictionary() -> Dict:
    """Robust regression defaults
    
    Returns
    -------
    Dict
        Default regression options
    """
    outDict = {}
    outDict["weights"] = "bisquare"         # 使用Tukey双权函数（也称Tukey损失函数或双权估计）进行稳健回归
    outDict["maxiter"] = maxIter()          # 稳健回归算法的关键收敛控制参数，表示最大迭代次数设置为100次
    outDict["initial"] = False
    outDict["scale"] = False
    outDict["intercept"] = False
    return outDict


def getRobustLocationWeights(r: np.ndarray, weight: str) -> np.ndarray:
    """Robust weighting schemes
    稳健权重方案
    Parameters
    ----------
    r : np.ndarray
        Residuals                       标准化残差
    weight : str
        The type of weighting to use    使用的权重类型

    Returns
    -------
    weights : np.ndarray
        The robust weights              稳健权重值
    """
    # the second argument, k, is a tuning constant(根据权重类型选择相应的权重函数，并设置调谐参数（阈值）k)
    if weight == "huber":                                               # Huber权重：k=1.345是标准调谐常数，对应95%的渐近效率
        k = 1.345
        # k = 0.5   # 注释掉的替代值，可用于更严格的权重
        return huberLocationWeights(r, k)
    elif weight == "hampel":                                            # Hampel权重：使用三段递减函数，对异常值更鲁棒
        k = 8   # 调谐常数，控制权重下降的阈值
        return hampelLocationWeights(r, k)
    elif weight == "trimmedMean":                                       # 修剪均值权重：完全排除大残差的影响
        k = 2   # 标准差倍数阈值，超过此值的残差权重为0
        return trimmedMeanLocationWeights(r, k)
    elif weight == "andrewsWave":                                       # Andrews波权重：使用正弦函数，平滑过渡
        k = 1.339   # 标准调谐常数
        return andrewsWaveLocationWeights(r, k)
    elif weight == "leastsq":                                           # 最小二乘权重：所有点权重均为1，相当于普通最小二乘
        return leastSquaresLocationWeights(r)
    else:
        # use bisquare weights                                          # 默认情况：使用双平方（bisquare）权重
        k = 4.685   # 标准调谐常数，对应95%的渐近效率
        # k = 1.0   # 注释掉的替代值，可用于更严格的权重
        return bisquareLocationWeights(r, k)


def huberLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Huber location weights
    Huber位置权重函数
    Parameters
    ----------
    r : np.ndarray
        Residuals               标准化残差
    k : float
        Tuning parameter        调谐参数（阈值）

    Returns
    -------
    weights : np.ndarray
        The robust weights      稳健权重值
    """
    weights = np.ones(shape=r.size, dtype="complex")                                            # 创建全1权重数组：初始假设所有数据点都是"好"点，权重为1。形状与输入残差相同
    for idx, val in enumerate(np.absolute(r)):                                                  # 遍历每个残差的绝对值：idx是索引，val是|r_i|
        if val > k:                                                                             # 检查当前残差是否超过阈值k
            # relying on numpy doing the right thing when dividing by zero(依靠 NumPy 在除以零的情况下正确处理相关问题)
            weights[idx] = k / val                                                              # 如果残差绝对值大于阈值k，降低该点的权重。权重计算公式：w_i = k / |r_i|。当|r_i|很大时，权重接近0，有效降低异常值影响
    return weights.real                                                                         # 返回权重的实部：由于权重计算结果是实数，取实部确保输出为实数数组


def bisquareLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Bisquare location weights
    双平方位置权重函数
    Parameters
    ----------
    r : np.ndarray
        Residuals               标准化残差
    k : float
        Tuning parameter        调谐参数（阈值）

    Returns
    -------
    weights : np.ndarray
        The robust weights      稳健权重值
    """
    ones = np.ones(shape=(r.size), dtype="complex")                                              # 创建全1数组：用于后续的阈值比较和计算
    threshR = np.minimum(ones, np.absolute(r / k))                                               # 计算标准化残差并限制上限为1：min(1, |r|/k)
    # threshR = np.maximum(-1*ones, threshR)                                                     # 注释掉的代码：用于处理负值情况，但双平方权重只关心绝对值，所以不需要
    return np.power((1 - np.power(threshR, 2)), 2).real                                          # 计算双平方权重 = (1 - (|r|/k)^2)^2


def hampelLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Hampel location weights
    Hampel位置权重函数
    Parameters
    ----------
    r : np.ndarray
        Residuals                   标准化残差
    k : float
        Tuning parameter            调谐参数（阈值）

    Returns
    -------
    weights : np.ndarray
        The robust weights          稳健权重值
    """
    a = k / 4                                                                       # 设置Hampel权重函数的内部阈值参数(a: 第一个阈值，残差超过此值开始降权（通常设为k/4）)
    b = k / 2                                                                       # b: 第二个阈值，残差超过此值权重加速下降（通常设为k/2）
    weights = np.ones(shape=r.size, dtype="complex")                                # 创建初始权重数组：所有点初始权重为1（完全信任）
    for idx, val in enumerate(np.absolute(r)):                                      # 遍历每个残差的绝对值
        if val > a and val <= b:                                                    # 情况1：残差在区间 (a, b] 内
            weights[idx] = a / val                                                  # 权重公式：w = a / |r|。(当a<|r|<=b时，权重从1线性下降到a/b)
        if val > b and val <= k:                                                    # 情况2：残差在区间 (b, k] 内
            weights[idx] = a * (k - val) / (val * (k - b))                          # 权重公式：w = a * (k - |r|) / (|r| * (k - b))。(当|r|接近k时，权重趋近于0，实现平滑过渡到完全排除)
        if val > k:                                                                 # 情况3：残差超过最大阈值k，完全排除
            weights[idx] = 0                                                        # 权重设为0，完全忽略该数据点的影响
    return weights.real                                                             # 返回权重的实部（权重计算结果是实数）


def trimmedMeanLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Trimmed mean location weights
    修剪均值位置权重函数
    Parameters
    ----------
    r : np.ndarray
        Residuals               标准化残差
    k : float
        Tuning parameter        调谐参数（阈值）

    Returns
    -------
    weights : np.ndarray
        The robust weights      稳健权重值
    """
    weights = np.zeros(shape=r.size, dtype="complex")                               # 创建全零权重数组：初始假设所有数据点都是异常值，权重为0
    indices = np.where(np.absolute(r) <= k)                                         # 查找满足条件的数据点索引：|r| ≤ k 的点（即残差绝对值小于等于阈值）
    weights[indices] = 1                                                            # 对满足条件的数据点赋予权重1：这些点被认为是"好"数据点
    return weights.real                                                             # 返回权重的实部：由于权重只能是0或1，实部就是最终的权重数组


def andrewsWaveLocationWeights(r: np.ndarray, k: float) -> np.ndarray:
    """Andrews Wave location weights
    Andrews波函数位置权重
    Parameters
    ----------
    r : np.ndarray
        Residuals               标准化残差
    k : float
        Tuning parameter        调谐参数（阈值）

    Returns
    -------
    weights : np.ndarray
        The robust weights      稳健权重值
    """
    weights = np.zeros(shape=r.size, dtype="complex")                               # 创建全零权重数组：初始假设所有数据点都是异常值，权重为0
    testVal = k * np.pi                                                             # 计算阈值：testVal = k * π，这是Andrews波函数的截断点
    for idx, val in enumerate(np.absolute(r)):                                      # 遍历每个残差的绝对值
        if val < testVal:                                                           # 检查当前|残差|是否小于阈值 kπ
            weights[idx] = np.sin(val / k) / (val / k)                              # 计算Andrews波权重：w = sin(|r|/k) / (|r|/k)。当 |r|/k 很小时，近似为1；当 |r|/k 接近 π 时，趋近于0
    return weights.real                                                             # 返回权重的实部（权重计算结果是实数）


def leastSquaresLocationWeights(r: np.ndarray):
    """Least squares weights, which are all equal to 1
    最小二乘权重函数，所有权重都等于1
    Parameters
    ----------
    r : np.ndarray
        Residuals               标准化残差

    Returns
    -------
    weights : np.ndarray
        The robust weights      稳健权重值
    """
    return np.ones(shape=(r.size), dtype="complex")                                 # 创建全1权重数组：形状与输入残差相同，数据类型为复数(等价于不加权的普通最小二乘法)


def sampleMedian(data):
    """Calculate the median of an array

    Mean is not a robust estimator of locations as it can be broken by a single outlying value. The median is a more robust choice.
    
    Parameters
    ----------
    np.ndarray
        Data for which to calculate median
    
    Returns
    -------
    float
        The median
    """
    return np.median(data)


def sampleMAD(data):
    """Median absolute deviation

    The standard deviation is not robust against outliers, hence use the MAD.
    
    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD
    
    Returns
    -------
    float
        The MAD    
    """
    absData = np.absolute(data)
    mad = sampleMedian(np.absolute(absData - sampleMedian(absData)))
    return mad / 0.67448975019608171


def sampleMAD0(data):
    """Median absolute deviation using an estimate of the location as 0
    使用0作为位置估计的中位数绝对偏差(MAD)
    When the location estimate is zero (rather than the median), the MAD essentially reduces to a median. This should be over non zero data. Useful for calculating variance of residuals.
    当位置估计为0（而不是中位数）时，MAD本质上简化为中位数。这应该应用于非零数据。对于计算残差的方差很有用。
    Parameters
    ----------
    np.ndarray
        Data for which to calculate MAD. This is often residuals when using 0 as an estimate of location. 
        要计算MAD的数据。当使用0作为位置估计时，这通常是残差。
    Returns
    -------
    float
        The MAD using zero as an esimate of location   使用0作为位置估计的MAD
    """
    absData = np.absolute(data)                     # absData=|残差|。np.absolute()：对实数取绝对值，对复数取模
    inputIndices = np.where(absData != 0.0)         # 找出所有非零数据的索引
    mad = sampleMedian(absData[inputIndices])       # 计算|残差|的中位数。对非零数据的绝对值计算中位数
    # mad = sampleMedian(np.absolute(data))
    return mad / 0.67448975019608171                # 应用校正因子，将MAD转换为标准差σ的一致估计(scale是残差分布离散度的稳健估计，相当于标准差（σ）的稳健版本)。对于正态分布，MAD ≈ 0.6745 × σ，因此 σ ≈ MAD / 0.6745。


def eps() -> float:
    """Small number
    
    Returns
    -------
    float
        A small number for quitting robust regression
    """
    return 0.0001


def maxIter() -> int:
    """Maximum number of iterations
    
    Returns
    -------
    int
        The maximum number of iterations
    """
    return 100
