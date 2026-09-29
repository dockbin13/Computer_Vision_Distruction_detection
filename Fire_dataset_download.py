from roboflow import Roboflow

rf = Roboflow(api_key="YWC5EBUH5K3h3XAEeHeU")
project = rf.workspace("sean-cftrp").project("fire-z2n21")
version = project.version(1)
dataset = version.download("yolo26")
                