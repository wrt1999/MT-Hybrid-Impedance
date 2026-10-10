from datapaths import projectPath, imagePath
from resistics.project.io import loadProject

# load the project
projData = loadProject(projectPath)

# load the viewing method
from resistics.project.time import viewTime

# view data between certain date range(project.time279行)
fig = viewTime(projData, "2022-01-01 00:00:00", "2022-01-01 00:08:00", sites=["site1"], show=False, save=False)
fig.savefig(imagePath / "TimeSeries_view_15")






















