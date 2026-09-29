from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=r"D:\Computer_Vision_Combined\Rubble-and-Person-Detection-4\data.yaml",
        epochs=100,
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