from cv2 import cv2
import numpy as np

class Mask():
    def __init__(self, warp_tar_img, shift_ref_img):
        self.overlap = None
        self.tar = None
        self.ref = None
        self.tar_nonoverlap = None
        self.ref_nonoverlap = None
        self.tar_overlap_edge = None
        self.ref_overlap_edge = None
        self.overlap_edge = None
        self.__constructMask(warp_tar_img,shift_ref_img)
        self.tar_result = None
        self.ref_result = None

    def __constructMask(self, warped_tar_img, warped_ref_img):
        tar_img_gray = cv2.cvtColor(warped_tar_img,cv2.COLOR_BGR2GRAY)
        ref_img_gray = cv2.cvtColor(warped_ref_img,cv2.COLOR_BGR2GRAY)
        _, binary_tar_img = cv2.threshold(tar_img_gray,1,255,cv2.THRESH_BINARY)
        _, binary_ref_img = cv2.threshold(ref_img_gray,1,255,cv2.THRESH_BINARY)

        kernal = np.ones((5,5), np.int8)

        binary_tar_img = cv2.morphologyEx(binary_tar_img, cv2.MORPH_CLOSE, kernal)
        binary_ref_img = cv2.morphologyEx(binary_ref_img, cv2.MORPH_CLOSE, kernal)
        overlap = cv2.bitwise_and(binary_tar_img, binary_ref_img)

        tar_nonoverlap = binary_tar_img - overlap
        ref_nonoverlap = binary_ref_img - overlap

        kernal = np.ones((3,3), np.int8)
        overlap_erode = cv2.morphologyEx(overlap, cv2.MORPH_ERODE, kernal)
        overlap_edge = overlap - overlap_erode
        
        tar_overlap_edge = cv2.bitwise_and(overlap_edge, cv2.morphologyEx(tar_nonoverlap, cv2.MORPH_DILATE, kernal))
        ref_overlap_edge = cv2.bitwise_and(overlap_edge, cv2.morphologyEx(ref_nonoverlap, cv2.MORPH_DILATE, kernal))


        self.overlap = overlap
        self.tar = binary_tar_img
        self.ref = binary_ref_img
        self.tar_nonoverlap = tar_nonoverlap
        self.ref_nonoverlap = ref_nonoverlap
        self.overlap_edge = overlap_edge
        self.tar_overlap_edge = tar_overlap_edge
        self.ref_overlap_edge = ref_overlap_edge


        # cv2.imwrite('./edge.png',overlap_edge)
        # cv2.imwrite('./tar_edge.png',self.tar_overlap_edge)
        # cv2.imwrite('./ref_edge.png',self.ref_overlap_edge)
        

