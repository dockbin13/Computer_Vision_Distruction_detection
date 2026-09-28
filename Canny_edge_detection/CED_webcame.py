import cv2 as cv
import time
import matplotlib.pyplot as plt

def callback(x):
    pass

def cannyEdge():
    cap = cv.VideoCapture(0)

    plt.ion()

    x = []
    y = []
    y_2 = []
    start = time.time()

    winname = "canny"
    cv.namedWindow(winname)

    cv.createTrackbar("minThresh", winname, 50, 255, callback)
    cv.createTrackbar("maxThresh", winname, 100, 255, callback)

    while True:
        success, image = cap.read()

        if not success:
            break

        image = cv.flip(image, 1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        minThresh = cv.getTrackbarPos("minThresh", winname)
        maxThresh = cv.getTrackbarPos("maxThresh", winname)

        edges = cv.Canny(gray, minThresh, maxThresh)

        cv.imshow(winname, edges)

        contours, _ = cv.findContours(
            edges,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        nContours = len(contours)

        total_area = 0
        for c in contours:
            total_area += cv.contourArea(c)

        x.append(time.time() - start)
        y.append(nContours)
        y_2.append(total_area)

        plt.cla()

        plt.subplot(121)
        plt.plot(x, y)
        plt.title("Number of Contours")
        plt.xlabel("Time")
        plt.ylabel("Contours")
        plt.xlim(max(0, x[-1] - 10), x[-1] + 1)

        plt.subplot(122)
        plt.plot(x, y_2)
        plt.title("Total Contour Area")
        plt.xlabel("Time")
        plt.ylabel("Area")
        plt.xlim(max(0, x[-1] - 10), x[-1] + 1)

        plt.pause(0.001)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    cannyEdge()