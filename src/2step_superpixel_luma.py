from SLIC import MaskedSLIC
import numpy as  np
from cv2 import cv2
from mask import Mask
from numba import jit
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from numba import jit
import time


class CBlender():
    def __init__(self,tar_img, ref_img, seam_mask, tar_4_corners, ref_4_corners, homoMat, shiftMat, mask):
        self.seam_pixel_coordi_lst = np.array( np.nonzero(seam_mask) ).T
        self.tar_img = tar_img
        self.ref_img = ref_img
        self.mask = mask


    def __get_color_diff_lst(self, pixel_coordi_lst):
        return self.ref_img[pixel_coordi_lst[:,0], pixel_coordi_lst[:,1]] / 255 - self.tar_img[pixel_coordi_lst[:,0], pixel_coordi_lst[:,1]] / 255 

    def  __calc_color_compensation(self, sampling_pixels_coordi_lst, refered_pixel_coordi_lst, sigma1=0.5, sigma2=0.5):
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
                    color_distance = tar_color_diff[0]**2 #+ tar_color_diff[1]**2 + tar_color_diff[2]**2
                    spatial_distance = (seam_pixel_y - tar_pixel_y)**2 + (seam_pixel_x - tar_pixel_x)**2

                    weight = np.exp( -color_distance / (sigma1**2) ) * np.exp( -spatial_distance / (sigma2*M)**2 )
                    normalization_term += weight
                    color_compensation[:] += weight * seam_color_diff
                if normalization_term == 0:
                    color_compensation_lst[idx,:] = 0
                else:
                    color_compensation = color_compensation / normalization_term
                    color_compensation_lst[idx] = color_compensation
            # color_compensation_lst[:,2] = 0
            # color_compensation_lst[:,1] = 0 
            return color_compensation_lst 
        seam_color_diff_lst = self.__get_color_diff_lst(refered_pixel_coordi_lst)
        color_compensation = np.zeros(3)
        color_compensation_lst = np.zeros( (len(sampling_pixels_coordi_lst), 3) )
        warp_tar_img_norm = self.tar_img / 255
        M = self.tar_img.shape[0]
        color_compensation_lst = numba_calc_color_compensation( color_compensation, color_compensation_lst, warp_tar_img_norm,\
             refered_pixel_coordi_lst, seam_color_diff_lst, sampling_pixels_coordi_lst, M, sigma1, sigma2)
        
        return color_compensation_lst

    def blend_color(self, refered_pixel_coordi_lst, blending_area_mask, sigma1=0.5, sigma2=0.5):
        # overlap_slic = MaskedSLIC(cv2.cvtColor(self.tar_img, cv2.COLOR_YUV2RGB), blending_area_mask,region_size=20)
        overlap_slic = MaskedSLIC(self.tar_img, blending_area_mask,region_size=20)
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


    def __get_ref_corner_in_overlap_and_nearest_tar_corner(self, ref_4_corners, tar_4_corners, shiftMat, homoMat):
        tar_mask = self.mask.tar
        homo_coordi = np.ones(3)
        picked_ref_corner = np.zeros(2, dtype=np.int)
        picked_tar_corner = np.zeros(2, dtype=np.int)

        for i in range(4):
            ref_x,ref_y = ref_4_corners[i]
            homo_coordi[0] = ref_x
            homo_coordi[1] = ref_y
            new_homo_coordi = shiftMat @ homo_coordi
            new_homo_coordi = new_homo_coordi / new_homo_coordi[2]

            ref_x = int(new_homo_coordi[0])
            ref_y = int(new_homo_coordi[1])

            if tar_mask[ref_y, ref_x] == 255:
                picked_ref_corner[0] = int(ref_x)
                picked_ref_corner[1] = int(ref_y)

        min_dist = np.Inf
        for i in range(4):
            tar_x,tar_y = tar_4_corners[i]
            homo_coordi[0] = tar_x
            homo_coordi[1] = tar_y
            new_homo_coordi = shiftMat @ homoMat @ homo_coordi
            new_homo_coordi = new_homo_coordi / new_homo_coordi[2]

            tar_x = int(new_homo_coordi[0])
            tar_y = int(new_homo_coordi[1])

            if np.linalg.norm( ( tar_x-picked_ref_corner[0], tar_y-picked_ref_corner[1] ) ) <  min_dist:
                picked_tar_corner[0] = int(tar_x)
                picked_tar_corner[1] = int(tar_y)
                min_dist = np.linalg.norm( ( tar_x-picked_ref_corner[0], tar_y-picked_ref_corner[1] ) )

        return picked_ref_corner, picked_tar_corner


    def split_tar_nonoverlap_area_and_edge(self, ref_4_corners, tar_4_corners, shiftMat, homoMat):
        @jit(nopython=True)
        def numba_divide_region(picked_ref_corner, picked_tar_corner, mask, tar_nonoverlap_coordi_lst):
            dx, dy =  picked_ref_corner - picked_tar_corner
            x1, y1 = picked_ref_corner
            x2, y2 = picked_tar_corner
            a = dy / dx
            b = y1 - a * x1
            for y,x in zip(tar_nonoverlap_coordi_lst[0], tar_nonoverlap_coordi_lst[1]):
                ans = a * x + b - y
                if ans>0:
                    mask[y,x] = 255
                else:
                    mask[y,x] = 127
            return mask


        picked_ref_corner, picked_tar_corner = self.__get_ref_corner_in_overlap_and_nearest_tar_corner(ref_4_corners ,tar_4_corners, shiftMat, homoMat)
        
        devided_tar_nonoverlap_mask = np.zeros(self.mask.tar_nonoverlap.shape, dtype=np.uint8)
        tar_nonoverlap_coordi_lst = np.nonzero(self.mask.tar_nonoverlap)
        devided_tar_nonoverlap_mask = numba_divide_region(picked_ref_corner, picked_tar_corner, devided_tar_nonoverlap_mask, tar_nonoverlap_coordi_lst)
        cv2.imwrite('devided_tar_nonoverlap_mask.png', devided_tar_nonoverlap_mask)

        devided_tar_edge_mask = np.zeros(self.mask.tar_overlap_edge.shape, dtype=np.uint8)
        tar_nonoverlap_edge_coordi_lst = np.nonzero(self.mask.tar_overlap_edge)
        devided_tar_edge_mask = numba_divide_region(picked_ref_corner, picked_tar_corner, devided_tar_edge_mask, tar_nonoverlap_edge_coordi_lst)
        # cv2.imwrite('devided_tar_edge_mask.png', devided_tar_edge_mask)
        
        return devided_tar_nonoverlap_mask, devided_tar_edge_mask



def main():
    '''
    Initialization
    '''
    start = time.time()

    tar_4_corners_xy = None
    ref_4_corners_xy = None
    homoMat = np.load('image/save_H.npy')
    shiftMat = np.load('image/save_Shift.npy')
    ref_4_corners_xy = np.load('image/save_ref_4_corners_xy.npy')
    tar_4_corners_xy = np.load('image/save_tar_4_corners_xy.npy')
    warp_tar_img = cv2.cvtColor(cv2.imread('image/warped_target.png'), cv2.COLOR_BGR2YUV)
    warp_ref_img = cv2.cvtColor(cv2.imread('image/warped_reference.png'), cv2.COLOR_BGR2YUV)
    seam_mask = cv2.imread('image/seam_mask.png',cv2.IMREAD_GRAYSCALE)

    mask = Mask(cv2.cvtColor(warp_tar_img, cv2.COLOR_YUV2BGR), cv2.cvtColor(warp_ref_img, cv2.COLOR_YUV2BGR))
    ref_region_mask =  cv2.imread('image/result_from_reference.png',cv2.IMREAD_GRAYSCALE)
    tar_region_mask = cv2.bitwise_and( cv2.bitwise_not(ref_region_mask) , mask.tar )

    mask.tar_result = tar_region_mask
    mask.ref_result = ref_region_mask






    '''
    Blend color
    '''

    CB = CBlender(warp_tar_img, warp_ref_img, seam_mask, tar_4_corners_xy, ref_4_corners_xy, homoMat, shiftMat, mask)

    devided_tar_nonoverlap_mask, devided_tar_edge_mask = CB.split_tar_nonoverlap_area_and_edge(ref_4_corners_xy, tar_4_corners_xy,shiftMat, homoMat)
    devided_tar_edge_mask_1 = np.copy(devided_tar_edge_mask)
    devided_tar_edge_mask_1[devided_tar_edge_mask_1==127] = 0
    devided_tar_edge_mask_2 = np.copy(devided_tar_edge_mask)
    devided_tar_edge_mask_2[devided_tar_edge_mask_1==255] = 0
    cv2.imwrite('aa.png', devided_tar_edge_mask)


    # refered_pixel_coordi_lst = np.vstack( (\
    #     np.array( np.nonzero(seam_mask) ).T, \
    #     np.array( np.where(devided_tar_edge_mask==255) ).T, \
    #     np.array( np.where(devided_tar_edge_mask==127) ).T \
    #     ))

    slic = MaskedSLIC(cv2.cvtColor(warp_tar_img, cv2.COLOR_YUV2BGR), np.bitwise_and(mask.tar_result, mask.overlap) ,region_size=20, compactness=5)

    seam_superpixel_idx_lst = np.array([idx for idx, (rows, cols) in enumerate(slic.labels_position) if np.sum(seam_mask[rows, cols]) > 0])
    tar_edge_1_superpixel_idx_lst = np.array([idx for idx, (rows, cols) in enumerate(slic.labels_position) if np.sum(devided_tar_edge_mask_1[rows, cols]) > 0])
    tar_edge_2_superpixel_idx_lst = np.array([idx for idx, (rows, cols) in enumerate(slic.labels_position) if np.sum(devided_tar_edge_mask_2[rows, cols]) > 0])

    refered_idx_lst = np.hstack((seam_superpixel_idx_lst,tar_edge_1_superpixel_idx_lst,tar_edge_2_superpixel_idx_lst))

    refered_pixel_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(slic.labels_position) if (idx != 0) and (np.isin(idx, refered_idx_lst))] )
    # print(refered_pixel_coordi_lst.shape)
    fuck_this_shit = CB.blend_color( refered_pixel_coordi_lst, np.bitwise_and(CB.mask.overlap, CB.mask.tar_result ), sigma1=0.3, sigma2=0.2 )

    cv2.imwrite('overlap_blending_only.png', cv2.cvtColor(fuck_this_shit,cv2.COLOR_YUV2BGR) )

    CB.ref_img = fuck_this_shit

    refered_pixel_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(slic.labels_position) if (idx != 0) and (np.isin(idx, tar_edge_1_superpixel_idx_lst))] )
    mask = np.zeros(devided_tar_nonoverlap_mask.shape)
    mask[devided_tar_nonoverlap_mask==255] = 255
    fuck_this_shit  = CB.blend_color(refered_pixel_coordi_lst, mask, sigma1=0.1, sigma2=0.05)
    cv2.imwrite('nonoverlap_blending_only1.png', cv2.cvtColor(fuck_this_shit,cv2.COLOR_YUV2BGR))
    CB.ref_img = fuck_this_shit
    refered_pixel_coordi_lst = np.array( [ (rows[len(rows)//2],cols[len(rows)//2]) for idx, (rows, cols) in enumerate(slic.labels_position) if (idx != 0) and (np.isin(idx, tar_edge_2_superpixel_idx_lst))] )
    mask = np.zeros(devided_tar_nonoverlap_mask.shape)
    mask[devided_tar_nonoverlap_mask==127] = 255
    fuck_this_shit  = CB.blend_color(refered_pixel_coordi_lst, mask, sigma1=0.1, sigma2=0.05)
    cv2.imwrite('nonoverlap_blending_only2.png', cv2.cvtColor(fuck_this_shit,cv2.COLOR_YUV2BGR))
    for idx in tar_edge_2_superpixel_idx_lst:
        fuck_this_shit[slic.labels_position[idx][0], slic.labels_position[idx][1]] = np.random.randint(0,255,3)
    cv2.imwrite('araara.png', cv2.cvtColor(fuck_this_shit,cv2.COLOR_YUV2BGR))



    print(f'time: {time.time() - start}')
if __name__ == '__main__':
    main()
    