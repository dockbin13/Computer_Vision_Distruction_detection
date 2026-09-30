from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO("yolo11n.pt")

    results = model.train(

        data=r"D:\Computer Vision\Fire--1\data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project="disaster_cv",
        name="disaster_detection",
        patience=20
    )

if __name__ == "__main__":
    freeze_support()
    main()
