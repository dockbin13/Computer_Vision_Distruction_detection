from roboflow import Roboflow

rf = Roboflow(api_key="YWC5EBUH5K3h3XAEeHeU")
project = rf.workspace("senior-project-r4kzm").project("rubble-and-person-detection")
version = project.version(4)
dataset = version.download("yolov11")