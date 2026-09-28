import os
import matplotlib.pyplot as plt 
import cv2 as cv 
import numpy as np

def readImage() :
    root = os.getcwd()
    path = os.path.join(root, "image.jpeg")
    image = cv.imread(path)

    return image

def image_plotter(RealImage, dilated_image) :
    plt.figure()
    plt.subplot(121)
    plt.imshow(RealImage, cmap = 'gray')

    plt.subplot(122)
    plt.imshow(dilated_image, cmap = 'gray')

    plt.show()

def kernal() :
    kernal = np.ones((5,5), np.uint8)
    return kernal

def dilate(image):
    dilated_image = cv.dilate(image, kernal(), iterations = 1)
    return dilated_image



if __name__ == '__main__' :
    img = readImage()
    D_image = dilate(img)
    image_plotter(img, D_image)
