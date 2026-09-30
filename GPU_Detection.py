
from ultralytics import YOLO
import os
import cv2 as cv
import time

YOLOn26 = YOLO("yolon26.pt")
Rubble_Model = YOLO("Rubble.pt")
thermal_Cam = YOLO("Thermal.pt")

cap = cv.VideoCapture(0)

# Create folder for saved frames
save_path = "Detected_Persons"
os.makedirs(save_path, exist_ok=True)

# 5-second cooldown
last_saved_time = 0
SAVE_DELAY = 5


def Detection():
    global last_saved_time

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv.flip(frame, 1)

        classes = [0, 7, 10, 61]

        Yolon26_result = YOLOn26(
            frame, conf=0.7, classes=classes, device=0
        )
        Rubble_Result = Rubble_Model(
            frame, conf=0.8, device=0
        )
        Thermal_Result = thermal_Cam(
            frame, conf=0.8, device=0
        )

        # Check whether class 0 (person) is detected
        person_detected = False

        boxes = Yolon26_result[0].boxes

        if boxes is not None and boxes.cls is not None:
            detected_classes = boxes.cls.cpu().tolist()
            person_detected = 0 in [int(c) for c in detected_classes]

        # Draw all model detections
        frame = Yolon26_result[0].plot(img=frame)
        frame = Rubble_Result[0].plot(img=frame)
        frame = Thermal_Result[0].plot(img=frame)

        # Save one frame every 5 seconds while a person is detected
        current_time = time.time()

        if person_detected and current_time - last_saved_time >= SAVE_DELAY:
            filename = os.path.join(
                save_path, f"person_{int(current_time)}.jpg"
            )

            cv.imwrite(filename, frame)
            last_saved_time = current_time

            print(f"Person detected! Frame saved: {filename}")

        cv.imshow("Detection", frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    Detection()
