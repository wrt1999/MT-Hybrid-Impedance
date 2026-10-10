import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import math
from matplotlib.figure import Figure
from typing import List, Dict, Tuple

from resistics.common.base import ResisticsBase
from resistics.common.print import listToString, arrayToString
from resistics.common.plot import (
    getViewFonts,
    getTransferFunctionFigSize,
    transferFunctionColours,
)


class TransferFunctionData(ResisticsBase):
    """Class for holding information time data

    Attributes
    ----------
    freq : np.ndarray
        The evaluation frequencies for which transfer functions were estimated
    period : np.ndarray
        The evaluation periods for which the transfer functions were estimated (1/freq)
    data : Dict
        The transfer function data keyed by polarisations. Data is in the units E = mV, H = nT
    polarisations : List[str]
        List of polarisations in the data
    variances : Dict
        The variances of the estimates to give an idea of uncertainty

    Methods
    -------
    __init__(freq, data, variances)
        Initialise the transfer function data
    getComponent(component)
        Set data with parameters
    getVariance(component)
        Add a comment to the dataset
    getResAndPhase(component) 
        A datetime array of the sample times
    getResAndPhaseErrors(component)
        View the spectra data 
    view()
        View the impedance tensor components
    viewTipper()
        View the tipper
    printList()
        Class status returned as list of strings        

    Notes
    -----
    Information about data units
    Magnetic permeability in nT . m / A
    Electric (E) data is in mV/m 
    Magnetic (H) data is in nT
    Z = E/H is in mV / m . nT
    Units of resistance = Ohm = V / A        
    T = H/H is dimensionless              
    """

    def __init__(self, freq: np.ndarray, data: Dict, variances: Dict):
        """Initialise and set object parameters

        Parameters
        ----------
        freq : np.ndarray
            Evaluation frequency array
        data : Dict
            Dictionary of data keyed by polarisation and with array values. Data units expected to be E = mV, H = nT
        variances : Dict
            Dictionary of uncertainties keyed by polarisation and with array values 
        """
        self.freq: np.ndarray = freq
        self.period: np.ndarray = np.reciprocal(self.freq)
        self.data: Dict = data
        self.polarisations: List[str] = sorted(self.data.keys())
        self.variances: Dict = variances

    def getComponent(self, component: str) -> np.ndarray:
        """Get data for a polarisation

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        np.ndarray
            The data for the polarisation component
        """
        return self.data[component]

    def getVariance(self, component: str) -> np.ndarray:
        """Get the variance for a component

        Parameters
        ----------
        component : str
            The polarisation to return the data for
        
        Returns
        -------
        np.ndarray
            The data for the polarisation component
        """
        return self.variances[component]

    def getResAndPhase(self, component: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the resistivity and phase for a component
        返回阻抗分量对应的视电阻率与相位参数
        The raw data is expected to be units E = mV, H = nT                 原始数据的预期单位：电场E = 毫伏(mV)，磁场H = 纳特斯拉(nT)
        Resisitivity is calculated as 0.2 * period * (abs(data))^2          视电阻率计算公式：0.2 × 周期 × |阻抗|²
        Phase is calculated as the complex angle, unwrapped and then shifted for components ExHx and ExHy to put them in 0-90 degree quadrant
        相位计算流程为：先求解复角，再对相位进行解缠处理，最后对ExHx与ExHy分量实施相位偏移，使其相位值限定于0-90°象限内
        Parameters
        ----------
        component : str
            The polarisation to return the data for
            要返回数据的极化分量，如"ExHx"、"ExHy"等
        Returns
        -------
        res : np.ndarray
            The resistivity for the evaluation frequencies 各评估频率点的视电阻率值
        phase : np.ndarray
            The phase for the evaluation frequencies 各评估频率点的相位值
        """
        data = self.getComponent(component)                             # 获取指定阻抗分量的复数数据(component如"ExHx"对应Zxx，这里是复数数组，实部虚部分别代表阻抗的实部和虚部)
        res = 0.2 * self.period * np.power(np.absolute(data), 2)        # 计算视电阻率：0.2 × 周期 × |阻抗|²(这是Cagniard公式的简化形式，将阻抗幅值转换为视电阻率)
        phase = np.angle(data)                                          # 计算复数阻抗的相位角（弧度制）（np.angle返回复数在复平面中的角度，范围[-π, π]）
        # check, can we unwrap into specific quadrant(验证：相位解缠能否限定于特定象限内执行)(相位解缠绕：消除2π跳变，使相位曲线连续)
        phase = np.unwrap(phase)                                        # 当np.unwrap检测到相邻相位变化超过π（180°）的地方，自动加减2π(如：未解卷绕的相位：[-180°, 180°]范围内的跳变original = [170°, 175°, -179°, -174°, -169°]  解卷绕后：连续的角度变化unwrapped = [170°, 175°, 181°, 186°, 191°])
        # convert to degrees(将弧度转换为角度：乘以180/π)
        phase = phase * 180 / np.pi
        # if component == "ExHx" or component == "ExHy":                  # 对特定的阻抗分量进行相位象限调整，仅对Z-ExHx(Zxx)和Z-ExHy(Zxy)进行处理
        #     # do modulo 360, see if this helps
        #     phase = np.mod(phase, 360)              # 取模360°：确保相位在[0, 360°)范围内,处理解卷绕后可能产生的大于360°的值
        #     phase = phase - 180                     # 减去180°，将相位平移到[-180°, 180°]范围,然后将负相位视为正相位（大地电磁相位的物理约定）
        #####################################################上面4行为原来的代码
        phase = np.mod(phase, 360)
        phase = phase - 180
        ####################################################
        return res, phase

    def getResAndPhaseErrors(self, component: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the resistivity and phase errors for a component
        返回指定阻抗分量的视电阻率和相位误差
        The raw data is expected to be units E = mV, H = nT                             原始数据的预期单位：电场E = 毫伏(mV)，磁场H = 纳特斯拉(nT)
        Errors are calculated parameterically (for confidence intervals)                误差采用参数化方法计算（用于置信区间）
        Error in resistivity is given by 1.96 * sqrt(2 * period * res * var / 5.0)      视电阻率误差公式：1.96 × √(2 × 周期 × 电阻率 × 方差 / 5.0)
        Error in phase is given by 1.96 * (180 / pi) * (sqrt(var / 2.0) / abs(data))    相位误差公式：1.96 × (180/π) × (√(方差/2.0) / |阻抗数据|)

        .. todo:: 
        
            Is there are better way to calculate errors     是否有更好的误差计算方法？（有待完善）

        Parameters
        ----------
        component : str
            The polarisation to return the data for
            要返回数据的极化分量，如"ExHx"、"ExHy"等
        Returns
        -------
        resError : np.ndarray
            The errors in resistivity data for the evaluation frequencies 各评估频率点的视电阻率误差值
        phaseError : np.ndarray
            The errors in phase data for the evaluation frequencies 各评估频率点的视电阻率误差值
        """
        data = self.getComponent(component)                                             # 获取指定阻抗分量的复数数据data
        res, phase = self.getResAndPhase(component)                                     # 获取已经计算好的视电阻率和相位值
        var = self.getVariance(component)                                               # 获取该阻抗分量的方差估计
        resError = 1.96 * np.sqrt(2 * self.period * res * var / 5.0)                    # 计算视电阻率的误差（基于误差传播理论），公式：1.96 × √(2 × 周期 × 电阻率 × 方差 / 5.0)
        phaseError = 1.96 * (180 / np.pi) * (np.sqrt(var / 2.0) / np.absolute(data))    # 计算相位的误差（基于误差传播和小角度近似），公式：1.96 × (180/π) × (√(方差/2.0) / |阻抗数据|)
        # calculate uncertainties on apparent resistivity and phase(计算视电阻率与相位参数的不确定度)
        return resError, phaseError         # 返回视电阻率和相位的误差棒数据

    def getTipper(self):
        """Return tipper length and angle
        返回Tipper的幅度和角度
        The tipper are the Tx = HzHx and Ty = HzHy components       Tipper是Tx = HzHx和Ty = HzHy分量
        The tipper length is sqrt(Re(Tx)^2 + Re(Ty)^2)              Tipper幅度计算公式：sqrt(Re(Tx)^2 + Re(Ty)^2)
        The tipper angle is arctan (Re(Ty)/Re(Tx))                  Tipper角度计算公式：arctan(Re(Ty)/Re(Tx))

        This needs HzHx and HzHy to be components of the transfer function      这需要HzHx和HzHy存在于传递函数中
        
        Returns                                                                         实部(Re)：反映同相响应，与电流同相位
        -------                                                                         虚部(Im)：反映异相响应，相位差90°
        tipperLength : np.ndarray
            The tipper length
        tipperAngle : np.ndarray
            The tipper angle
        """
        if "HzHy" not in self.polarisations or "HzHx" not in self.polarisations:        # 检查是否包含必要的极化分量
            return False, False                                                         # 如果缺少任一分量，返回False
        txRe = np.real(self.getComponent("HzHx"))                                       # Tx的实部
        txIm = np.imag(self.getComponent("HzHx"))                                       # Tx的虚部
        tyRe = np.real(self.getComponent("HzHy"))                                       # Ty的虚部
        tyIm = np.imag(self.getComponent("HzHy"))                                       # Ty的虚部
        tipperLength = np.sqrt(np.power(txRe, 2) + np.power(tyRe, 2))                   # 计算Tipper幅度（矢量模长）：|T| = √(Re(Tx)² + Re(Ty)²)。值越大表示三维性越强，表示垂直与水平磁场的比值。
        tipperAngleRe = np.arctan(tyRe / txRe)                                          # 计算Tipper实部角度（矢量方向）:arctan(Re(Ty)/Re(Tx))。范围是(-π/2, π/2)，需注意象限处理
        tipperAngleIm = np.arctan(tyIm / txIm)                                          # 计算Tipper虚部角度:arctan(Im(Ty)/Im(Tx))
        return tipperLength, tipperAngleRe, tipperAngleIm

    def viewImpedance(self, **kwargs) -> Figure:
        """Plots the transfer function data
        绘制传递函数数据
        For resistivity data, both axes are log scale (period and resistivity). For phase data, period is in log scale and phase is linear scale.
        对于电阻率数据，横纵坐标轴均采用对数刻度（分别对应周期与电阻率）；对于相位数据，周期轴为对数刻度，相位轴为线性刻度
        Units, x axis is seconds, resistivity is Ohm m and phase is degrees.
        单位说明：x轴为秒（周期单位），电阻率单位为欧姆·米（Ω·m），相位单位为度（°）
        Parameters
        ----------
        polarisations : List[str], optional
            Polarisations to plot
        fig : matplotlib.pyplot.figure, optional
            A figure object
        oneplot : bool, optional
            Boolean flag for plotting all polarisations on one plot rather than separate plots
        colours : Dict[str, str], optional
            Colours dictionary for plotting impedance components
        mk : str, optional
            Plot marker type
        ls : str, optional
            Line style
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        res_ylim : List, optional
            Limits for the resistivity y axis
        phase_ylim : List, optional
            Limits for the phase y axis

        Returns
        -------
        plt.figure
            Matplotlib figure object
        """
        polarisations = (               # 获取要绘制的极化分量列表，默认为所有可用分量
            kwargs["polarisations"] if "polarisations" in kwargs else self.polarisations
        )

        # limits(设置坐标轴范围限制)
        xlim = kwargs["xlim"] if "xlim" in kwargs else [1e-4, 1e4]                      # 周期范围：0.000001到1000000秒,如果kwargs没有则设置默认值
        res_ylim = kwargs["res_ylim"] if "res_ylim" in kwargs else [1e-2, 1e3]          # 视电阻率范围
        phase_ylim = kwargs["phase_ylim"] if "phase_ylim" in kwargs else [-180, 90]     # 相位范围

        # markers(设置绘图样式)
        colours = (
            kwargs["colours"] if "colours" in kwargs else transferFunctionColours()     # 默认颜色映射
        )
        mk = kwargs["mk"] if "mk" in kwargs else "o"                                    # 图形标记
        ls = kwargs["ls"] if "ls" in kwargs else "none"                                 # 线形
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()    # 默认字体

        # calculate number of rows and columns(计算行数和列数)
        oneplot = False
        if "oneplot" in kwargs and kwargs["oneplot"]:
            oneplot = True              # 是否将所有分量绘制在同一图中(这里是True)
        nrows = 2                                           # 固定2行：电阻率图 + 相位图
        ncols = 1 if oneplot else len(polarisations)        # 列数：单图模式为1列，分图模式为分量数量(这里为4列)
        # a multiplier to make sure all the components end up on the right plot(子图索引乘数，确保正确的位置计算)
        plotNumMult = 0 if ncols > 1 else 1                 # 一个乘数，以确保所有组件最终都出现在正确的图表上

        # plot(使用现有图形)
        if "fig" in kwargs:
            fig = plt.figure(kwargs["fig"].number)          # 使用现有图形
        else:
            figsize = getTransferFunctionFigSize(oneplot, len(polarisations))       # 自动计算合适尺寸
            fig = plt.figure(figsize=figsize)               # 创建新图形

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 核心修复：无条件裁剪画布，防止由于外部传入 fig 导致尺寸不更新
        # 利用 hasattr 防止多次调用此函数时导致画布尺寸被反复缩小
        if not hasattr(fig, '_orig_figsize_for_mt'):
            fig._orig_figsize_for_mt = fig.get_size_inches()

        orig_w, orig_h = fig._orig_figsize_for_mt

        # 1. 计算对齐比例 (相位 90 度 等于 视电阻率 1 个十倍程)
        res_decades = math.log10(res_ylim[1]) - math.log10(res_ylim[0])
        phase_span = abs(phase_ylim[1] - phase_ylim[0])
        bottom_ratio = (phase_span / 90.0) / res_decades

        # 2. 精确计算并强制设置新高度，物理切除原本属于底部空白的部分
        new_h = orig_h * (0.6112 + 0.3888 * bottom_ratio)
        fig.set_size_inches(orig_w, new_h)

        gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=[1.0, bottom_ratio])

        ax1_specs = [gs[0, i] for i in range(ncols)]
        ax2_specs = [gs[1, i] for i in range(ncols)]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        st = fig.suptitle(
            "Impedance tensor apparent resistivity and phase",      # 设置图形总标题"张量阻抗视电阻率和相位"
            fontsize=plotfonts["suptitle"],
        )
        st.set_y(0.98)              # 调整标题位置，避免与子图重叠
        # 核心：循环处理每个阻抗极化分量（Z-ExHx, Z-ExHy, Z-EyHx, Z-EyHy）
        for idx, pol in enumerate(polarisations):
            res, phase = self.getResAndPhase(pol)                       # 获取当前阻抗分量的视电阻率和相位数据(对特定的阻抗分量（Z-ExHx(Zxx)和Z-ExHy(Zxy)）需多一步进行相位象限调整)
            resError, phaseError = self.getResAndPhaseErrors(pol)       # 获取视电阻率和相位的误差棒数据
            res = res[:-1]
            phase = phase[:-1]
            resError = resError[:-1]
            phaseError = phaseError[:-1]
            # label = kwargs["label"] + " - {}".format(pol) if "label" in kwargs else pol    # 构建图例标签
            label = f"Z{pol[1]}{pol[3]}" if "label" in kwargs else pol

            # plot resistivity(绘制视电阻率子图，上图)
            # ax1 = plt.subplot(nrows, ncols, idx + 1 - plotNumMult * idx)           # 计算子图位置
            # +++++ 替换旧的 plt.subplot 为使用 GridSpec 定位器 +++++
            col_idx = 0 if oneplot else idx
            ax1 = plt.subplot(ax1_specs[col_idx])
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # the title(设置子图标题)
            if not oneplot:
                plt.title("Polarisation {}".format(pol), fontsize=plotfonts["title"], loc="left")   # 分图模式：显示分量名
            else:
                plt.title(
                    " ",
                    # "(d) Fast and Robust Bootstrap",             # 单图模式：显示所有分量名
                    # "NRMS:0.95",
                    fontsize=plotfonts["title"],
                    loc="left",
                )
            # plot the data(使用errorbar绘制视电阻率数据图（带误差棒）)
            ax1.errorbar(
                self.period[:-1],                       # X轴数据：周期数组
                res,                                    # Y轴数据：视电阻率数组
                yerr=resError,                          # Y轴误差：电阻率误差
                ls=ls,                                  # 线条样式
                marker=mk,                              # 数据点标记样式
                markersize=7,                           # 数据点标记样式大小
                markerfacecolor="white",                # 数据点标记填充色：白色
                markeredgecolor=colours[pol],           # 标记边框色：分量专属颜色
                mew=1.1,                                # 标记边框宽度
                color=colours[pol],                     # 线条颜色：分量专属颜色
                ecolor=colours[pol],                    # 误差棒颜色：与数据一致
                elinewidth=1.0,                         # 误差棒线宽
                capsize=4,                              # 误差棒端帽大小
                barsabove=False,                        # 误差棒在数据点下方(False)
                label=label,                            # 图例标签
            )
            ax1.set_xscale("log")                       # 设置坐标轴属性,X轴对数刻度
            ax1.set_yscale("log")                       # Y轴对数刻度（视电阻率）
            #####################################################################上下两行代码原本会执行，但影响成图感官。
            # ax1.set_aspect("equal", adjustable="box")   # 保持纵横比
            #######################################################################
            # axis options(坐标轴标签和范围)##############################################添加了(, labelpad=10)在下行代码的末尾)
            plt.ylabel("Apparent Res. [Ohm m]", fontsize=plotfonts["axisLabel"], labelpad=0)    # Y轴标签"Apparent Res. [Ohm m]"
            plt.xlim(xlim)                 # 设置X轴范围
            plt.ylim(res_ylim)             # 设置Y轴范围
            # ++++++ 新增此行代码 ++++++
            ax1.tick_params(axis='x', labelbottom=False)  # 隐藏上图(视电阻率图)的 X 轴数字
            # +++++++++++++++++++++++++
            # set tick sizes(设置刻度标签字体大小,横纵坐标轴上数字大小)
            for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
                lab.set_fontsize(plotfonts["axisTicks"])
                # lab.set_color("none")                   # 新增：将 ax1 的 X 轴和 Y 轴数字的颜色均设为透明

            # plot phase(绘制相位子图，下图)
            # ax2 = plt.subplot(nrows, ncols, ncols + idx + 1 - plotNumMult * idx)        # 计算相位图位置
            # +++++ 替换旧的 plt.subplot 为使用 GridSpec 定位器 +++++
            ax2 = plt.subplot(ax2_specs[col_idx])
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # plot the data(绘制相位数据（带误差棒）)
            ax2.errorbar(
                self.period[:-1],                       # X轴数据：周期数组（与电阻率相同）
                phase,                                  # Y轴数据：相位数组
                yerr=phaseError,                        # Y轴误差：相位误差
                ls="none",                              # 相位图通常不连线
                marker=mk,                              # 数据点标记样式
                markersize=7,                           # 数据点标记样式大小
                markerfacecolor="white",                # 数据点标记填充色：白色
                markeredgecolor=colours[pol],           # 标记边框色：分量专属颜色
                mew=1.1,                                # 标记边框宽度
                color=colours[pol],                     # 线条颜色：分量专属颜色
                ecolor=colours[pol],                    # 误差棒颜色：与数据一致
                elinewidth=1.0,                         # 误差棒线宽
                capsize=4,                              # 误差棒端帽大小
                barsabove=False,                        # 误差棒在数据点下方(False)
                label=label,                            # 图例标签
            )
            ax2.set_xscale("log")                       # 设置坐标轴属性,X轴对数刻度
            # axis options
            plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])           # X轴标签（只在相位图显示）
            plt.ylabel("Phase [degrees]", fontsize=plotfonts["axisLabel"], labelpad=0)      # Y轴标签#######################################################
            plt.xlim(xlim)                          # 相同X轴范围
            plt.ylim(phase_ylim)                    # 相同Y轴范围
            # ++++++ 新增代码：强制Y轴主刻度以 30 为步长 ++++++
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(90))
            # +++++++++++++++++++++++++++++++++++++++++++++++
            # set tick sizes(设置刻度标签字体大小,横纵坐标轴上数字大小)
            for lab in ax2.get_xticklabels() + ax2.get_yticklabels():
                lab.set_fontsize(plotfonts["axisTicks"])
            # 分开处理 X 轴和 Y 轴的数字
            # for lab in ax2.get_xticklabels():
            #     lab.set_fontsize(plotfonts["axisTicks"])
            #     # X 轴数字正常显示，不需要变透明
            #
            # for lab in ax2.get_yticklabels():
            #     lab.set_fontsize(plotfonts["axisTicks"])
            #     lab.set_color("none")  # 新增：仅将 ax2 的 Y 轴数字设为透明

        # add the legend(为每个子图添加图例和网格)
        for idx, pol in enumerate(polarisations):
            ax1 = plt.subplot(nrows, ncols, idx + 1 - plotNumMult * idx)            # 定位到当前分量的视电阻率子图（上图），plt.subplot(总行数，列数，位置编号)函数指定子图位置
            # leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])            # 在当前子图（电阻率图）中添加图例，loc="upper right"：图例放置在右上角，loc="upper right"：图例放置在右上角
            # leg.get_frame().set_linewidth(0.0)                                          # 设置图例框的边框线宽为0，实现无边框效果
            # leg.get_frame().set_facecolor("w")                                          # 设置图例框的背景颜色为白色（"w"）
            # plt.grid(True, ls="--")                                             # 为电阻率子图添加网格线，True：启用网格，ls="--"：网格线使用虚线样式（更适合对数坐标）
            # ax2 = plt.subplot(nrows, ncols, ncols + idx + 1 - plotNumMult * idx)    # 定位到当前分量的相位子图（下图），同上，但位置编号处：ncols是列数，加上后定位到第二行的对应位置
            # # leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])           # 同上
            # # leg.get_frame().set_linewidth(0.0)                                          # 同上
            # # leg.get_frame().set_facecolor("w")                                          # 同上
            # plt.grid(True, ls="--")                                               # 同上
            # +++++ 替换结尾添加图例网格处的 plt.subplot +++++
            col_idx = 0 if oneplot else idx

            ax1 = plt.subplot(ax1_specs[col_idx])
            plt.grid(True, ls="--")

            ax2 = plt.subplot(ax2_specs[col_idx])
            plt.grid(True, ls="--")
            # ++++++++++++++++++++++++++++++++++++++++++++++++

        # show if the figure is not in keywords(如果是新创建的图形，进行布局优化并显示)
        if "fig" not in kwargs:
            # layout options
            # plt.tight_layout()                  # 自动调整布局
            # # +++++ 新增这行代码：强制锁定上下子图的垂直间距 +++++
            # fig.subplots_adjust(hspace=0.08)  # hspace 控制垂直间隙，数值代表子图高度的比例
            # # +++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++ 核心修复：直接使用 gs.update 覆盖 tight_layout 创建的过大间隙 +++++
            # 利用绝对公式将间距和顶部比例强行锁定在最原始状态，防止由于画布大小改变产生的放大效应
            adjusted_hspace = 0.16 / (1.0 + bottom_ratio)
            new_top = 1.0 - (0.08 * orig_h / new_h)

            gs.update(hspace=adjusted_hspace, top=new_top)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # fig.subplots_adjust(top=0.92)       # 为总标题预留顶部空间
            plt.show()                          # 显示图形

        return fig

    def viewTipper(self, **kwargs) -> Figure:
        """Plots tipper data where available
        绘制可用的Tipper（倾子）数据
        For length data, both axes are log scale (period and length). For angle data, period is in log scale and phase is linear scale.
        Units, x axis is seconds, length is dimensionless and angle is degrees.
        对于幅度数据，两个轴都是对数刻度（周期和幅度）。对于角度数据，周期是对数刻度，相位是线性刻度。单位：X轴是秒，幅度是无量纲，角度是度。
        Parameters
        ----------
        fig : matplotlib.pyplot.figure, optional
            A figure object
        cols : bool, optional       有三个Tipper图：Tipper幅度、Tipper实部角度、Tipper虚部角度。可以按行或列排列。设为True表示1行3列布局。
            There are three tipper plots: tipper length, tipper real angle, tipper imaginary angle. These can either be plotted in rows or columns. Set to True to plot in with one row and three columns.
        colours : str, optional
            Colour of plot
        mk : str, optional
            Plot marker type
        ls : str, optional
            Line style  
        plotfonts : Dict, optional
            A dictionary of plot fonts
        label : str, optional
            Label for the plots
        xlim : List, optional
            Limits for the x axis
        length_ylim : List, optional
            Limits for the length y axis
        angle_ylim : List, optional
            Limits for the angle y axis            
        """
        if "HzHx" not in self.polarisations or "HzHy" not in self.polarisations:        # 检查是否包含必要的极化分量数据
            return False                                                                # 如果没有垂直磁场分量数据，返回False

        # rows or columns(行或列)
        cols = kwargs["cols"] if "cols" in kwargs else True                             # 设置行或列布局(默认为列布局（1行3列）)
        if cols:
            nrows = 1
            ncols = 3
        else:
            nrows = 3
            ncols = 1
        # plot options(绘图选项设置)
        mk = kwargs["mk"] if "mk" in kwargs else "o"                                    # 默认标记：圆形
        ls = kwargs["ls"] if "ls" in kwargs else "none"                                 # 默认线型：无线（只有标记点）
        plotfonts = kwargs["plotfonts"] if "plotfonts" in kwargs else getViewFonts()    # 默认字体

        # limits(坐标轴范围设置,首选用户指定配置，否则使用默认)
        xlim = kwargs["xlim"] if "xlim" in kwargs else [1e-3, 1e4]                      # 默认X轴范围：0.001-10000秒
        length_ylim = kwargs["length_ylim"] if "length_ylim" in kwargs else [1e-2, 1e3] # 幅度范围：0.01-1000
        angle_ylim = kwargs["angle_ylim"] if "angle_ylim" in kwargs else [-30, 30]      # 角度范围：-30°到30°

        fig = (                                                                         # 创建或获取图形对象
            plt.figure(kwargs["fig"].number)                                            # 如果提供了现有图形，使用它
            if "fig" in kwargs
            else plt.figure(figsize=(16, 7))                                            # 否则创建新图形，尺寸16x7英寸
        )
        st = fig.suptitle("Tipper")                                                     # 设置图形总标题
        st.set_y(0.98)                                                                  # 调整标题位置（靠近顶部）

        # plot
        tipperLength, tipperAngleRe, tipperAngleIm = self.getTipper()                   # 获取Tipper数据(幅度、实部角、虚部角)
        # lengthError, angleError = self.getTipperErrors()                              # 注释:误差棒获取（当前未使用）
        label = kwargs["label"] if "label" in kwargs else "tipper"                      # 图例标签
        # plot resistivity
        ax1 = plt.subplot(nrows, ncols, 1)                                        # 绘制第一个子图：Tipper幅度
        plt.title("Tipper Length")                                                      # 子图标题
        ax1.errorbar(                                                                   # 绘制误差棒图
            self.period,                                                                # X轴：period
            tipperLength,                                                               # Y轴：Tipper幅度
            yerr=tipperLength / 50,                                                     # Y轴误差：幅度的2%（固定比例）
            ls=ls,
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax1.set_xscale("log")                                                           # X轴对数刻度
        ax1.set_yscale("log")                                                           # Y轴对数刻度（幅度）
        # axis options(轴标签)
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Length [dimensionless]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(length_ylim)
        # set tick sizes and legend(设置刻度字体大小)
        for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])               # 图例设置
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")                                                   # 虚线网格

        # plot real tipper angles
        ax2 = plt.subplot(nrows, ncols, 2)
        plt.title("Tipper Angle Real")
        ax2.errorbar(
            self.period,
            tipperAngleRe,
            yerr=tipperAngleRe / 50,
            ls="none",
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax2.set_xscale("log")
        # axis options
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Angle [degrees]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(angle_ylim)
        # set tick sizes
        for lab in ax2.get_xticklabels() + ax2.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")

        # plot imaginary tipper angles
        ax3 = plt.subplot(nrows, ncols, 3)
        plt.title("Tipper Angle Imaginary")
        ax3.errorbar(
            self.period,
            tipperAngleIm,
            yerr=tipperAngleIm / 50,
            ls="none",
            marker=mk,
            markerfacecolor="white",
            markersize=9,
            color="red",
            ecolor="red",
            mew=1.1,
            elinewidth=1.0,
            capsize=4,
            barsabove=False,
            label=label,
        )
        ax3.set_xscale("log")
        # axis options
        plt.xlabel("Period [s]", fontsize=plotfonts["axisLabel"])
        plt.ylabel("Angle [degrees]", fontsize=plotfonts["axisLabel"])
        plt.xlim(xlim)
        plt.ylim(angle_ylim)
        # set tick sizes
        for lab in ax3.get_xticklabels() + ax3.get_yticklabels():
            lab.set_fontsize(plotfonts["axisTicks"])
        leg = plt.legend(loc="upper right", fontsize=plotfonts["legend"])
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("w")
        plt.grid(True, ls="--")

        # show if the figure is not in keywords
        if "fig" not in kwargs:
            # layout options
            plt.tight_layout()
            fig.subplots_adjust(top=0.90)
            plt.show()

        return fig

    def printList(self) -> List[str]:
        """Class information as a list of strings

        Returns
        -------
        out : List[str]
            List of strings with information
        """
        textLst = []
        textLst.append("Frequencies [Hz] = {}".format(arrayToString(self.freq)))
        textLst.append("Periods [s] = {}".format(arrayToString(self.period)))
        textLst.append("Polarisations = {}".format(self.polarisations))
        return textLst
