from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData)          # 时域数据归一化，标定(时->频->时)，降采样(分层)，加窗(当前层)，对每个窗口fft(时->频)，频域数据写入.bin(创建注释文件.info)
projData.refresh()

# process the spectra
from resistics.project.transfunc import processProject

processProject(projData)            # 功率谱矩阵，obs reg，robust处理，阻抗元素(Zxx,Zxy,Zyx,Zyy)估计out，方差var。（阻抗：self.impedances、方差：self.variances），保存文件transFuncData     Ex=ZxxHx + ZxyHy         Ey=ZyxHx + ZyyHy

# plot transfer function and save the plot
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(projData, sites=["site1"], polarisations=["ExHy", "EyHx"], show=False, save=False)            # 读取传输函数数据((阻抗(Z-ExHx、Z-ExHy、Z-EyHx、Z-EyHy)、freq、period、极化(ExHx、ExHy、EyHx、EyHy)，var))
figs[0].savefig(imagePath / "simpleRun_viewimp_default")                                          # 计算并绘制各张量阻抗的视电阻率（双对数坐标）-相位（X轴对数，Y轴线性）图













