import os 
import cv2 as cv 
from ultralytics import YOLO
import time

debris_model = YOLO("debris_det_renamed.pt")
smoke_model = YOLO("smoke.pt")
person_Object_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo26n-pose.pt")
Fire_smoke = YOLO("Fire_Smoke.pt")
rubble_model = YOLO("best.pt")


cap = cv.VideoCapture(0)

while True :

    success, frame = cap.read()

    if not success :
        print("Camera Not Found")
        break

    frame = cv.flip(frame, 1)
    
    debries_result = debris_model(frame, conf = 0.6, device = 0)
    
    
    excluded_classes = [14, 1, 25, 26, 27, 28, 55]
    excluded_classes_pose = [0]
    classes = [i for i in range(80) if i not in excluded_classes]
    person_Object_result = person_Object_model(frame, conf = 0.6, classes = classes, device = 0)
    pose_result = pose_model(frame, conf = 0.6, device = 0)
    smoke_result = smoke_model(frame, conf = 0.6, device = 0)
    SmokeFire = Fire_smoke(frame, conf = 0.6, device = 0)

    rubble_result = rubble_model(frame, conf = 0.4, device = 0)

    debris_model.names[1] = "debris / Garbage (0.5 Confidance)"
    
    frame = debries_result[0].plot()
    frame = person_Object_result[0].plot(img = frame)
    frame = pose_result[0].plot(img = frame)
    frame = smoke_result[0].plot(img = frame)
    frame = SmokeFire[0].plot(img = frame)
    frame = rubble_result[0].plot(img = frame)

    cv.imshow("Detection", frame)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()