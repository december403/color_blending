from cv2 import cv2
import numpy as np
from grids import Grids
from homomatrix import HomoMatrix

class APAP_Stitcher():
    def __init__(self, tar_img, ref_img, src_pts, dst_pts, grid_size=100, scale_factor=15):
        '''

        The image stitching engine usings APAP algorithm.

        tar_img : Target image that will be warped to be stitched to reference image.
        ref_img : Reference image  that will not be warped.
        src_pts : Paired key points' coordinates in target image.
        dst_pts : Paired key points' coordinates in reference image.
        grid_size : The grid size is grid_size by grid_size.
        scale_factor : Scale factor used in matching pairs' weight adjustment.
        '''


        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.scale_factor = scale_factor
        self.grids = Grids(grid_size, tar_img.shape[0], tar_img.shape[1])
        self.mask = None
        self.homoMat = HomoMatrix()

    def warp(self):
        pass




    def find_stitched_img_size_and_shift_amount(self, tar_img, ref_img):
        '''
        This method finds the height and width of final stitched image.
        '''
        if self.homoMat.globalHomoMat is None:
            raise ValueError(' Run constructGlobalMat() before stitch the image!')

        

        tar_h, tar_w = tar_img.shape[0:2]
        tar_four_corners = np.array( [ (0,0), (0,tar_h), (tar_w,0), (tar_w,tar_h)] )

        ref_h, ref_w = ref_img.shape[0:2]
        ref_four_coeners = np.array( [ (0,0), (0,ref_h), (ref_w,0), (ref_w,ref_h)] )

        H = self.homoMat.globalHomoMat
        warped_tar_four_corners = np.zeros((4,2))
        for idx, (x, y) in enumerate(tar_four_corners):
            temp = H @ np.array([x,y,1]).reshape((3,1))
            temp = temp/temp[2,0]
            warped_tar_four_corners[idx] = temp[0:2,0]

        eight_corners = np.vstack((warped_tar_four_corners, ref_four_coeners))

        min_x, min_y = np.min(eight_corners, axis=0).astype(np.int16)
        max_x, max_y = np.max(eight_corners, axis=0).astype(np.int16)
        shift_amount = np.array( (-min_x, -min_y)).reshape((2,1)).astype(np.int16)
        stitched_img_size = np.array((int(max_x-min_x+1), int(max_y-min_y+1)))
        return stitched_img_size, shift_amount

        