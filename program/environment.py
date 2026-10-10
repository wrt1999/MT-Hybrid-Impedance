from datapaths import projectPath, imagePath
from resistics.project.io import newProject

# define reference time for project
referenceTime = "2022-01-01 00:00:00"
# create a new project and print information
projData = newProject(projectPath, referenceTime)
projData.printInfo()

# create a new site
projData.createSite("site1")
projData.printInfo()

from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project and print information
projData = loadProject(projectPath)
projData.printInfo()

# view the project timeline
fig = projData.view()
fig.savefig(imagePath / "TimeSeries_ranges")

# get site data
siteData = projData.getSiteData("site1")
siteData.printInfo()



















