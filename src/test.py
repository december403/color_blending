from cv2 import cv2 
import numpy as np
from SLIC import MaskedSLIC
from mask import Mask
from numba import jit
import time
import matplotlib.pyplot as plt




class CBlender():
    def __init__(self,tar_img, ref_img, seam_mask, mask):
        self.seam_pixel_coordi_lst = np.array( np.nonzero(seam_mask) ).T
        self.tar_img = tar_img
        self.ref_img = ref_img
        self.mask = mask


    def get_color_diff_lst(self, pixel_coordi_lst):
         return self.ref_img[pixel_coordi_lst[:,0], pixel_coordi_lst[:,1]] / 255 - self.tar_img[pixel_coordi_lst[:,0], pixel_coordi_lst[:,1]] / 255 

    def  __calc_color_compensation(self, sampling_pixels_coordi_lst, refered_pixel_coordi_lst, sigma1, sigma2):
        @jit(nopython=True)
        def numba_calc_color_compensation(color_compensation, color_compensation_lst, warp_tar_img_norm, seam_pixel_coordi_lst,\
             seam_color_diff_lst, sampling_pixel_coordi_lst, M, sigma1=0.5, sigma2=5):
            for idx, (tar_pixel_y,tar_pixel_x) in enumerate(sampling_pixel_coordi_lst):
                color_compensation[:] = 0
                tar_pixel_color = warp_tar_img_norm[tar_pixel_y, tar_pixel_x]
                normalization_term = 0
                for seam_color_diff, (seam_pixel_y, seam_pixel_x) in zip(seam_color_diff_lst, seam_pixel_coordi_lst):
                    seam_tar_pixel_color = warp_tar_img_norm[seam_pixel_y, seam_pixel_x]
                    tar_color_diff = tar_pixel_color - seam_tar_pixel_color
                    color_distance = tar_color_diff[0]**2 + tar_color_diff[1]**2 + tar_color_diff[2]**2
                    spatial_distance = (seam_pixel_y - tar_pixel_y)**2 + (seam_pixel_x - tar_pixel_x)**2

                    weight = np.exp( -color_distance / (sigma1**2) ) * np.exp( -spatial_distance / (sigma2*M)**2 )
                    normalization_term += weight
                    color_compensation[:] += weight * seam_color_diff
                if normalization_term == 0:
                    color_compensation_lst[idx,:] = 0
                else:
                    color_compensation = color_compensation / normalization_term
                    color_compensation_lst[idx] = color_compensation

            return color_compensation_lst 

        seam_color_diff_lst = self.get_color_diff_lst(refered_pixel_coordi_lst)
        color_compensation = np.zeros(3)
        color_compensation_lst = np.zeros( (len(sampling_pixels_coordi_lst), 3) )
        warp_tar_img_norm = self.tar_img / 255
        M = self.tar_img.shape[0]
        color_compensation_lst = numba_calc_color_compensation( color_compensation, color_compensation_lst, warp_tar_img_norm,\
             refered_pixel_coordi_lst, seam_color_diff_lst, sampling_pixels_coordi_lst, M, sigma1, sigma2 )
        
        return color_compensation_lst

    def blend_color(self, refered_pixel_coordi_lst, blending_area_mask, sigma1, sigma2):
        # overlap_tar_mask = np.bitwise_and(self.mask.overlap, self.mask.tar_result )
        overlap_slic = MaskedSLIC(self.tar_img, blending_area_mask,region_size=20, compactness=3)
        sampling_pixel_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(overlap_slic.labels_position) if idx != 0] )
        color_compensation_lst =  self.__calc_color_compensation(sampling_pixel_coordi_lst, refered_pixel_coordi_lst, sigma1, sigma2)

        ref_img = np.copy(self.ref_img)
        blended_img = np.zeros(self.ref_img.shape, dtype=np.float64)
        tar_img_norm = self.tar_img / 255

        for idx , color_compensation in enumerate(color_compensation_lst):
            super_pixel_coordis = overlap_slic.labels_position[idx+1]
            blended_img[super_pixel_coordis[0], super_pixel_coordis[1]] =  (( tar_img_norm[super_pixel_coordis[0], super_pixel_coordis[1]] + color_compensation )*255)

        blended_img[blended_img<0] = 0
        blended_img[blended_img>255] = 255
        ref_img[blended_img>0] = blended_img[blended_img>0].astype(np.uint8)

        return ref_img

    

    

start = time.time()

warp_tar_img = cv2.imread('image/warped_target.png')

warp_ref_img = cv2.imread('image/warped_reference.png')
mask = Mask(warp_tar_img, warp_ref_img)
seam_mask = cv2.imread('image/seam_mask.png',cv2.IMREAD_GRAYSCALE)
ref_region_mask =  cv2.imread('image/result_from_reference.png',cv2.IMREAD_GRAYSCALE)
tar_region_mask = cv2.bitwise_and( cv2.bitwise_not(ref_region_mask) , mask.tar )
mask.tar_result = tar_region_mask
mask.ref_result = ref_region_mask
slic = MaskedSLIC(warp_tar_img, np.bitwise_and(mask.tar_result, mask.overlap) ,region_size=20, compactness=3)


CB = CBlender(warp_tar_img, warp_ref_img,seam_mask, mask)

refered_pixel_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(slic.labels_position) if idx != 0] )
sigma1 = 0.1
sigma2 = 0.25
# print(refered_pixel_cooordi_lst)
a = CB.blend_color(refered_pixel_coordi_lst, mask.tar_result, sigma1, sigma2)
# a[refered_pixel_cooordi_lst[0], refered_pixel_cooordi_lst[1],:] = (255,255,255)
# a[slic.labels_position[1][0], slic.labels_position[1][1]] = (0,0,255)
# a[slic.contour_mask>0] = (0,255,0)

# for i in range(len(refered_pixel_cooordi_lst)):

#     cv2.circle(a, tuple(refered_pixel_cooordi_lst[i][::-1]), 3, (0,0,255), -1)
cv2.imwrite('supa_img.png', a)

print(f'time: {time.time() - start}')

# warp_ref_img[slic.contour_mask>0] = (0,255,0)
# warp_tar_img[slic.contour_mask>0] = (0,255,0)

# cv2.imwrite('1.png', warp_tar_img)
# cv2.imwrite('2.png', warp_ref_img)

# plt.hist( np.linalg.norm(CB.get_color_diff_lst(refered_pixel_coordi_lst), axis=1), bins=1000 )
# plt.show()

for i in range(len(slic.labels_position)):
    if i != 0:
        rows, cols = slic.labels_position[i]
        warp_tar_img[rows, cols] = np.median(warp_tar_img[rows, cols], axis=0)

cv2.imwrite('3.png', warp_tar_img)

for i in range(len(slic.labels_position)):
    if i != 0:
        rows, cols = slic.labels_position[i]
        warp_ref_img[rows, cols] = np.median(warp_ref_img[rows, cols], axis=0)

cv2.imwrite('4.png', warp_ref_img)
