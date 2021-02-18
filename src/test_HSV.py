from cv2 import cv2 
import numpy as np
img = cv2.imread('image/DJI_0016.png')
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(img_HSV[:,:,2])

img_HSV[:,:,2] = img_HSV[:,:,2] * 0.75

img_dark = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

cv2.imwrite('img_dark.png', img_dark)


img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(img_HSV[:,:,2])
img_HSV = img_HSV.astype(np.float64)
img_HSV[:,:,2] = img_HSV[:,:,2] // 5 * 8
img_HSV[:,:,2][img_HSV[:,:,2]>255] = 255
img_HSV = img_HSV.astype(np.uint8)
img_bright = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
# img_bright[img_bright>255] = 255
cv2.imwrite('img_bright.png', img_bright)
tar_img = cv2.imread('image/UAV/DJI_0004_dark.JPG')