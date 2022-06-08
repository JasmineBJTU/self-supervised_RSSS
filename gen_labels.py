import os
from PIL import Image
import numpy as np
import imageio
import cv2
img_path="/home/heshuyi/Desktop/potsdam/processed/train/color_gt/"
save_path="/home/heshuyi/Desktop/potsdam/processed/train/gt/"
fileList = os.listdir(img_path)
print('--------fileList--------')
for file in fileList:
    print('-------Producing ' + file + '-------')
    img = cv2.imread(img_path + str(file))
    print(np.max(img))
    label = -np.ones([600, 600])
    for i in range(0, 600):
        for j in range(0, 600):
            if (img[i, j, 0] == 0 and img[i, j, 1] == 255 and img[i, j, 2] == 255):
                label[i, j] = 0
            if (img[i, j, 0] == 0 and img[i, j, 1] == 255 and img[i, j, 2] == 0):
                label[i, j] = 1
            if (img[i, j, 0] == 255 and img[i, j, 1] == 255 and img[i, j, 2] == 255):
                label[i, j] = 2
            if (img[i, j, 0] == 255 and img[i, j, 1] == 0 and img[i, j, 2] == 0):
                label[i, j] = 3
            if (img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 255):
                label[i, j] = 4
            if (img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0):
                label[i, j] = 5
    label = np.uint8(label)
    if np.min(label) == -1:
        print("!!!!!!!!ERROR")
    cv2.imwrite(save_path + str(file), label)
