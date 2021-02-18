from cv2 import cv2

warp_tar_img = cv2.imread('image/warped_target.png')
cv2.imwrite('warp_tar_img_gray.png', cv2.cvtColor(warp_tar_img, cv2.COLOR_BGR2GRAY))

warp_ref_img = cv2.imread('image/warped_reference.png')
cv2.imwrite('warp_ref_img_gray.png', cv2.cvtColor(warp_ref_img, cv2.COLOR_BGR2GRAY))