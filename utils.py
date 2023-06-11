import shutil
import os
import random
import pandas as pd
import Augmentor

matches = {
    0: "0-NoDR",
    1: "1-Mild",
    2: "2-Moderate",
    3: "3-Severe",
    4: "4-ProliferativeDR"
}
# Function to move specific amount of random files from one folder to another
def move_random_files(source, destination, fromN, toN):
    print("Moving " + str(fromN - toN) + " files from " + source + " to " + destination)
    files = os.listdir(source)
    for i in range(fromN - toN):
        file = random.choice(files)
        shutil.move(os.path.join(source, file), destination)
        files.remove(file)
        if i % 1000 == 0:
            print("Moved " + str(i) + " files")

def move_files_from_csv(csv_path):
    veri = pd.read_csv(csv_path)

    for _, satır in veri.iterrows():
        dosya_adı = satır['image']
        klasör = satır['level']

        kaynak_yol = os.path.join(os.getcwd(), "resized_cropped", dosya_adı + ".jpeg")
        hedef_yol = os.path.join(os.getcwd(), "new", str(klasör), dosya_adı + ".jpeg")
        #print(kaynak_yol, hedef_yol)

        if os.path.exists(kaynak_yol):
            shutil.move(kaynak_yol, hedef_yol)  # Dosyayı ilgili klasöre taşı

    print("Dosyalar başarıyla taşındı.")

def augmentate():
    for i in range(5):
        print("./new_converted_imgs/" + str(i))
        p = Augmentor.Pipeline("./new_converted_imgs/" + str(i), r"C:\Users\MONSTER\Desktop\Proje\Diabetic-Retinopathy\augmented\\" + str(i))
        #p.skew_tilt(0.5)
        p.flip_random(0.5)
        p.random_distortion(0.5, 8, 8, 3)
        p.rotate(0.3, 10, 10)
        p.random_brightness(0.3, 0.8, 1.2)

        p.process()
        p.sample(9292)

def temp_augmentate():
    p = Augmentor.Pipeline("./temp", "output")
    p.flip_random(0.5)
    p.random_distortion(0.5, 8, 8, 3)
    p.rotate(0.3, 10, 10)
    p.random_brightness(0.3, 0.8, 1.2)

    p.process()
    p.sample(9)

def crop_image(image):
    # https://www.kaggle.com/code/tanlikesmath/diabetic-retinopathy-resnet50-binary-cropped/notebook
    import cv2
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        flag = 0
        return image, flag
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x); y = int(y); r = int(r)
    flag = 1
    #print(x,y,r)
    if r > 100:
        return output[0 + (y-r)*int(r<y):-1 + (y+r+1)*int(r<y),0 + (x-r)*int(r<x):-1 + (x+r+1)*int(r<x)], flag
    else:
        print('none!')
        flag = 0
        return image,flag
    
def crop_image_from_gray(img,tol=7):
    # https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    import cv2
    import numpy as np
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def circle_crop(img_array, sigmaX = 10):  
    # https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    """
    Create circular crop around image centre    
    """    
    import cv2
    import numpy as np
    IMG_SIZE = 512
    img = crop_image_from_gray(img_array)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

#augmentate()