from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# calculate spectrum using standard options
from resistics.project.spectra import calculateSpectra

calculateSpectra(projData)          
projData.refresh()

# process the spectra
from resistics.project.transfunc import processProject

processProject(projData)            

# plot transfer function and save the plot
from resistics.project.transfunc import viewImpedance

figs = viewImpedance(projData, sites=["site1"], polarisations=["ExHy", "EyHx"], show=False, save=False)            
figs[0].savefig(imagePath / "simpleRun_viewimp_default")                                          













