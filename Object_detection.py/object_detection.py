import os 
import cv2 as cv 
from ultralytics import YOLO

debris_model = YOLO("debris_det_renamed.pt")
smoke_model = YOLO("smoke.pt")
person_Object_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo26n-pose.pt")
Fire_smoke = YOLO("Fire_Smoke.pt")
# Water_flood = YOLO("Flood_water.pt")


cap = cv.VideoCapture(0)

while True :

    success, frame = cap.read()

    if not success :
        print("Camera Not Found")
        break

    frame = cv.flip(frame, 1)
    
    debries_result = debris_model(frame, conf = 0.6, device = 0)
    
    
    excluded_classes = [14, 1, 25, 26, 27, 28, 55]
    classes = [i for i in range(80) if i not in excluded_classes]
    person_Object_result = person_Object_model(frame, conf = 0.6, classes = classes, device = 0)
    
    # In person if 14th class is not there then and then only run the model 
    # or else dont run the model 
    # Here for every class from 0 - 80 if there are these spedified class 
    # exclude the model or dont run the model
    pose_result = pose_model(frame, conf = 0.6, device = 0)
    smoke_result = smoke_model(frame, conf = 0.6, device = 0)
    SmokeFire = Fire_smoke(frame, conf = 0.6, device = 0)
    # FloodWater = Water_flood(frame, conf = 0.9, device = 0)

    # Changed the confidence percentage to 50% for debris or Garbage
    # Cuz this shit ahh model does not know the difference between Debri and Garbage
    debris_model.names[1] = "debris / Garbage (0.5 Confidance)" 

    frame = debries_result[0].plot()
    frame = person_Object_result[0].plot(img = frame)
    frame = pose_result[0].plot(img = frame)
    frame = smoke_result[0].plot(img = frame)
    frame = SmokeFire[0].plot(img = frame)
    # frame = FloodWater[0].plot(img = frame)

    cv.imshow("debris", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()