import cv2
import os
import random

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

# Canny Edge Detection
def canny(image, k):
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

# Resize Image
image = cv2.resize(image, (600, 400))
cv2.imshow("Resized", image)

# Canny Edge Detection
canny_img = canny(image, k)
cv2.waitKey(0)