
from ultralytics import YOLO
import cv2 as cv
import os
import time
import threading

# ---------------- CONFIGURATION ----------------

NORMAL_CAM_ID = 0
THERMAL_CAM_ID = 1

NORMAL_MODEL_PATH = "yolo11n.pt"  # Change to your person model
THERMAL_MODEL_PATH = "Thermal.pt"

CONFIDENCE = 0.7
SAVE_DELAY = 5  # Seconds between saves per detection type

NORMAL_SAVE_DIR = "Normal_Person"
THERMAL_SAVE_DIR = "Thermal_Alerts"

os.makedirs(NORMAL_SAVE_DIR, exist_ok=True)
os.makedirs(THERMAL_SAVE_DIR, exist_ok=True)

# Shared latest normal camera frame
latest_normal_frame = None
normal_frame_lock = threading.Lock()

# Stop both threads safely
stop_event = threading.Event()


# ---------------- CAMERA SETUP ----------------

def open_camera(camera_id):
    cap = cv.VideoCapture(camera_id, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    return cap


# ---------------- NORMAL CAMERA ----------------

def normal_camera_worker():
    global latest_normal_frame

    model = YOLO(NORMAL_MODEL_PATH)
    cap = open_camera(NORMAL_CAM_ID)

    last_saved = 0

    try:
        while not stop_event.is_set():
            success, frame = cap.read()

            if not success:
                print("Normal camera frame error")
                time.sleep(0.1)
                continue

            frame = cv.flip(frame, 1)

            # Store the latest raw normal camera frame
            with normal_frame_lock:
                latest_normal_frame = frame.copy()

            # Person detection (class 0 for COCO models)
            results = model.predict(
                frame,
                conf=CONFIDENCE,
                classes=[0],
                device="cpu",
                imgsz=416,
                verbose=False
            )

            detected = (
                results[0].boxes is not None
                and len(results[0].boxes) > 0
            )

            display_frame = results[0].plot()

            # Save normal frame when a person is detected
            current_time = time.monotonic()

            if detected and current_time - last_saved >= SAVE_DELAY:
                filename = os.path.join(
                    NORMAL_SAVE_DIR,
                    f"person_{time.time_ns()}.jpg"
                )

                cv.imwrite(filename, display_frame)
                last_saved = current_time

                print(f"Person detected. Saved: {filename}")

            cv.imshow("Normal Camera", display_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    finally:
        cap.release()
        cv.destroyWindow("Normal Camera")


# ---------------- THERMAL CAMERA ----------------

def thermal_camera_worker():
    global latest_normal_frame

    model = YOLO(THERMAL_MODEL_PATH)
    cap = open_camera(THERMAL_CAM_ID)

    last_saved = 0

    try:
        while not stop_event.is_set():
            success, thermal_frame = cap.read()

            if not success:
                print("Thermal camera frame error")
                time.sleep(0.1)
                continue

            # Flip if required by your camera orientation
            thermal_frame = cv.flip(thermal_frame, 1)

            # Thermal detection: all model classes
            results = model.predict(
                thermal_frame,
                conf=CONFIDENCE,
                device="cpu",
                imgsz=416,
                verbose=False
            )

            detected = (
                results[0].boxes is not None
                and len(results[0].boxes) > 0
            )

            display_frame = results[0].plot()

            current_time = time.monotonic()

            # If thermal detection occurs, save the normal camera frame
            if detected and current_time - last_saved >= SAVE_DELAY:

                with normal_frame_lock:
                    if latest_normal_frame is not None:
                        normal_snapshot = latest_normal_frame.copy()
                    else:
                        normal_snapshot = None

                if normal_snapshot is not None:
                    filename = os.path.join(
                        THERMAL_SAVE_DIR,
                        f"thermal_alert_{time.time_ns()}.jpg"
                    )

                    cv.imwrite(filename, normal_snapshot)
                    last_saved = current_time

                    print(
                        "Thermal detection! "
                        f"Normal camera frame saved: {filename}"
                    )

            cv.imshow("Thermal Camera", display_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    finally:
        cap.release()
        cv.destroyWindow("Thermal Camera")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    normal_thread = threading.Thread(
        target=normal_camera_worker
    )

    thermal_thread = threading.Thread(
        target=thermal_camera_worker
    )

    try:
        normal_thread.start()
        thermal_thread.start()

        normal_thread.join()
        thermal_thread.join()

    except KeyboardInterrupt:
        print("Stopping cameras...")
        stop_event.set()

    finally:
        stop_event.set()
        normal_thread.join()
        thermal_thread.join()
        cv.destroyAllWindows()
        print("Detection stopped.")
