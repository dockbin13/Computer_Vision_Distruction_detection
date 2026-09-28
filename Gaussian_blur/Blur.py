import cv2 as cv 
import os 
import matplotlib.pyplot as plt 

root = os.getcwd()
path = os.path.join(root, "image.jpeg")
image = cv.imread(path)

# blur = cv.GaussianBlur(image, (10,10), 10)   // this will not work because it needs a clear center pixel like (5,5)
blur = cv.GaussianBlur(image, (9,9), 100)
blur_2 = cv.GaussianBlur(image, (9,9), 5)
blur_3 = cv.GaussianBlur(image, (5,5), 10)

plt.figure()
plt.subplot(221)
plt.imshow(image)
plt.title("Orignal Image")

plt.subplot(222)
plt.imshow(blur)
plt.title("kernel = 9,9, SigmaX = 100")

plt.subplot(223)
plt.imshow(blur_2)
plt.title("kernel = 9,9, SimgaX = 5")

plt.subplot(224)
plt.imshow(blur_3)
plt.title("kernel = 5,5, SigmaX = 10")

plt.show()