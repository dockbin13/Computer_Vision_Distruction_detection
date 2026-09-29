from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"D:\Computer_Vision_Combined\Fire--1\data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        optimizer="AdamW",
        lr0=0.001
    )

if __name__ == "__main__":
    freeze_support()
    main()