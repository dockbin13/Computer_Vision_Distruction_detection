import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def frame_capture():
    cap = cv.VideoCapture(0)

    plt.ion()
    _, ax = plt.subplots(1, 3, figsize=(12, 4))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY)

        kernel = np.ones((7, 7), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=1)

        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contour_image = frame.copy()
        cv.drawContours(contour_image, contours, -1, (0, 0, 255), 3)

        hull = []

        for c in contours:
            hull.append(cv.convexHull(c))
        cv.drawContours(contour_image, hull, -1, (255, 0, 0), 3)

        for c in contours:
            x, y, w, h = cv.boundingRect(c)     # cv.rectangle(image, start_point, end_point, color, thickness)
            # So, here the start point in any image is x,y as define above. BoundingRect returns ---> topleft_x, topleft_y, width, height 
            # due of the rectangle due to the contour
            cv.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 3)

        ax[0].clear()
        ax[0].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        ax[0].set_title("Original")

        ax[1].clear()
        ax[1].imshow(thresh, cmap="gray")
        ax[1].set_title("Threshold")

        ax[2].clear()
        ax[2].imshow(cv.cvtColor(contour_image, cv.COLOR_BGR2RGB))
        ax[2].set_title("Contours")

        plt.pause(0.001)

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    frame_capture()