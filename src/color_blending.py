from cv2 import cv2
import numpy as np
from mask import Mask
from SLIC import MaskedSLIC
from numba import jit


warp_tar_img = cv2.imread('image/warped_target.png')#.astype(np.float) / 255
warp_ref_img = cv2.imread('image/warped_reference.png')#.astype(np.float) / 255
warp_tar_img_YUV = cv2.cvtColor(warp_tar_img, cv2.COLOR_BGR2YUV)
warp_ref_img_YUV = cv2.cvtColor(warp_ref_img, cv2.COLOR_BGR2YUV)
warp_tar_img_YUV = cv2.cvtColor(warp_tar_img_YUV, cv2.COLOR_YUV2BGR)
warp_ref_img_YUV = cv2.cvtColor(warp_ref_img_YUV, cv2.COLOR_YUV2BGR)
warp_tar_img_YUV_norm = warp_tar_img_YUV / 255
warp_ref_img_YUV_norm = warp_ref_img_YUV / 255

no_blending_result = cv2.imread('image/result_no_blenging.png')
h,w = warp_tar_img.shape[0:2]
# print(h,w)
mask = Mask(warp_tar_img, warp_ref_img)

seam_mask = cv2.imread('image/seam_mask.png',cv2.IMREAD_GRAYSCALE)
ref_region_mask =  cv2.imread('image/result_from_reference.png',cv2.IMREAD_GRAYSCALE)
tar_region_mask = cv2.bitwise_and( cv2.bitwise_not(ref_region_mask) , mask.tar )


seam_pixel_coordinates = np.nonzero(seam_mask)
a = cv2.bitwise_and(ref_region_mask, seam_mask)
seam_color_diff_lst = warp_ref_img_YUV_norm[seam_pixel_coordinates] - warp_tar_img_YUV_norm[seam_pixel_coordinates]


result_from_tar_pixel_coordinates = np.nonzero(tar_region_mask)



color_compensation_array = np.zeros((h,w), dtype=np.float)
color_compensation = np.zeros(3, dtype=np.float)
color_compensation_array = np.zeros(warp_tar_img.shape, dtype=np.float)
@jit(nopython=True)
def aaaaaaa(color_compensation, color_compensation_array, warp_tar_img_YUV_norm, ):
    sigma1 = 0.5
    sigma2 = 5
    M = max(h,w)

    for tar_pixel_y,tar_pixel_x in zip(result_from_tar_pixel_coordinates[0],result_from_tar_pixel_coordinates[1]):
        color_compensation[:] = 0
        tar_pixel_color = warp_tar_img_YUV[tar_pixel_y, tar_pixel_x]
        normalization_term = 0
        for seam_color_diff, seam_pixel_y, seam_pixel_x in zip(seam_color_diff_lst, seam_pixel_coordinates[0], seam_pixel_coordinates[1]):
            seam_tar_pixel_color = warp_tar_img_YUV[seam_pixel_y, seam_pixel_x]
            color_distance = (tar_pixel_color - seam_tar_pixel_color)[0]**2 + (tar_pixel_color - seam_tar_pixel_color)[1]**2 + (tar_pixel_color - seam_tar_pixel_color)[2]**2
            spatial_distance = (seam_pixel_y - tar_pixel_y)**2 + (seam_pixel_x - tar_pixel_x)**2

            weight = np.exp( -color_distance / (sigma1**2) ) * np.exp( -spatial_distance / (sigma2*M)**2 )
            normalization_term += weight
            color_compensation[:] += weight * seam_color_diff
        if normalization_term == 0:
            color_compensation_array[tar_pixel_y, tar_pixel_x, :] = 0
        else:
            color_compensation = color_compensation / normalization_term
            color_compensation_array[tar_pixel_y, tar_pixel_x, :] = color_compensation

    return color_compensation_array

color_compensation_array = aaaaaaa(color_compensation, color_compensation_array, warp_tar_img_YUV_norm)


no_blending_result_YUV = cv2.cvtColor( no_blending_result, cv2.COLOR_BGR2YUV)
no_blending_result_YUV = cv2.cvtColor( no_blending_result_YUV, cv2.COLOR_YUV2BGR)
blending_result_YUV = color_compensation_array*255 + no_blending_result_YUV


blending_result = cv2.cvtColor( blending_result_YUV.astype(np.uint8), cv2.COLOR_BGR2YUV)
blending_result = cv2.cvtColor( blending_result.astype(np.uint8), cv2.COLOR_YUV2BGR)

cv2.imwrite('result_with_blending.png', blending_result )



