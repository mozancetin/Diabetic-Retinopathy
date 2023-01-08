import cv2
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt

# Get Image Names
def get_image_names():
    all_files = dict()
    for a, _, files in os.walk("./images/"):
        for file in files:
            if str(file).endswith(".jpeg"):
                if a[-1] not in all_files:
                    all_files[a[-1]] = [file]
                else:
                    all_files[a[-1]].append(file)
    return all_files

all_images = get_image_names()
# print("Total images: " + str(len(all_images.values())))
# print(all_images.values())

# Canny Edge Detection
def canny(image, k):
    # #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #threshold = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]
    # #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imshow("TH", threshold)
    # #cv2.imshow("Gray", gray)
    for i in range(3, 5):
        if k != "4":
            canny = cv2.Canny(image, 50, 10 + (i * 10)) # 50 20 50 40 50 50
        else:
            canny = cv2.Canny(image, 50, 10 + (i * 10))
        cv2.imshow("Canny " + str(50) + " " + str(10 + (i * 10)), cv2.fastNlMeansDenoising(canny, None, 40, 7, 21))
    return canny

# Get Image
k = random.choice(list(all_images.keys()))
v = random.choice(all_images[k])
print("Image:" + " images/" + k + "/" + v)
image = cv2.imread("./images/" + k + "/" + v)
if k == "4":
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize Image
image = cv2.resize(image, (600, 400))
cv2.imshow("Resized", image)

# Canny Edge Detection
canny_img = canny(image, k)
# cv2.imshow("Canny", canny_img)
cv2.waitKey(0)