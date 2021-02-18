from SLIC import MaskedSLIC
import numpy as  np
from cv2 import cv2
from mask import Mask
from numba import jit
import matplotlib.pyplot as plt
import time

class CBlender():
    def __init__(self,tar_img, ref_img, maskSLIC, seam_mask, tar_mask):
        self.seam_pixel_coordi_lst = np.array( np.nonzero(seam_mask) ).T
        self.maskSLIC = maskSLIC
        self.tar_img = tar_img
        self.ref_img = ref_img
        self.tar_mask = tar_mask
        self.seam_color_diff_lst = self.__get_seam_color_diff_lst(tar_img, ref_img)
        self.sampling_pixels_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(self.maskSLIC.labels_position) if idx != 0] )
        self.color_compensation_lst = None


    def __get_seam_color_diff_lst(self,tar_img, ref_img):
         return ref_img[self.seam_pixel_coordi_lst[:,0], self.seam_pixel_coordi_lst[:,1]] / 255 - tar_img[self.seam_pixel_coordi_lst[:,0], self.seam_pixel_coordi_lst[:,1]] / 255 

    def  __calc_color_compensation(self):
        
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

        # self.sampling_pixels_coordi_lst = np.array( [ (rows[0],cols[0]) for idx, (rows, cols) in enumerate(self.maskSLIC.labels_position) if idx != 0] )
        color_compensation = np.zeros(3)
        color_compensation_lst = np.zeros( (len(self.sampling_pixels_coordi_lst), 3) )
        warp_tar_img_norm = self.tar_img / 255
        M = self.tar_img.shape[0]
        self.color_compensation_lst = numba_calc_color_compensation( color_compensation, color_compensation_lst, warp_tar_img_norm,\
             self.seam_pixel_coordi_lst, self.seam_color_diff_lst, self.sampling_pixels_coordi_lst, M, sigma1=0.5, sigma2=0.05 )

    def blend_color(self):
        self.__calc_color_compensation()
        ref_img = np.copy(self.ref_img)
        blended_img = np.zeros(self.ref_img.shape, dtype=np.float64)
        tar_img_norm = self.tar_img / 255

        for idx , color_compensation in enumerate(self.color_compensation_lst):
            # if idx == 50:
            #     break
            super_pixel_coordis = self.maskSLIC.labels_position[idx+1]
            blended_img[super_pixel_coordis[0], super_pixel_coordis[1]] =  (( tar_img_norm[super_pixel_coordis[0], super_pixel_coordis[1]] + color_compensation )*255)#.astype(np.uint8)

        blended_img[blended_img<0] = 0
        blended_img[blended_img>255] = 255
        ref_img[blended_img>0] = blended_img[blended_img>0].astype(np.uint8)

        return ref_img


start = time.time()

warp_tar_img = cv2.imread('image/warped_target.png')
warp_ref_img = cv2.imread('image/warped_reference.png')
seam_mask = cv2.imread('image/seam_mask.png',cv2.IMREAD_GRAYSCALE)

mask = Mask(warp_tar_img, warp_ref_img)

ref_region_mask =  cv2.imread('image/result_from_reference.png',cv2.IMREAD_GRAYSCALE)
tar_region_mask = cv2.bitwise_and( cv2.bitwise_not(ref_region_mask) , mask.tar )


maskSLIC = MaskedSLIC(warp_tar_img, tar_region_mask, region_size=20)
CB = CBlender(warp_tar_img, warp_ref_img, maskSLIC, seam_mask, tar_region_mask)
blended_img = CB.blend_color()

print(CB.maskSLIC.numOfPixel)
print(np.sum(CB.tar_mask)/255)

print(f'time: {time.time()-start}')
cv2.imwrite('blended_img.png', blended_img)
blended_img[maskSLIC.contour_mask>0] = (0,255,0)
cv2.imwrite('pixel_blended_img.png', blended_img)
blended_img[seam_mask>0] = (0,0,255)
cv2.imwrite('seam_pixel_blended_img.png', blended_img)
# cv2.imwrite('mask.png', mask.tar_nonoverlap)
# print(CB.seam_color_diff_lst.shape)

# plt.hist( np.linalg.norm(CB.seam_color_diff_lst, axis=1), bins=100 )
# plt.show()
