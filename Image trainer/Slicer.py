import cv2
import numpy as np
import sys

def decoupe(file):
    x, y = 0, 0
    img = cv2.imread(file)
    width = len(img[0])
    height = len(img)
    k = 1
    while x + 32 < width:
        while y + 32 < height:
            new_img = img[y:y+32, x:x+32]
            cv2.imwrite("img_" + str(k) + ".jpg", new_img)
            y = y + 8
            k = k + 1
        x = x + 8
        y = 0

decoupe("banquise.png")