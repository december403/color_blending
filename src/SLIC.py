import numpy as np
from cv2 import cv2
from mask import Mask
import pandas as pd
from skimage.segmentation import find_boundaries#, slic
from cuda_slic.slic import slic 
from numba import jit



class MaskedSLIC():
    def __init__(self, img, ROI, region_size=50, compactness=5):

        h,w = img.shape[0:2]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img = cv2.GaussianBlur(img,(3,3),0)
        n_segment = int((h*w)/(region_size**2))
        labels = slic(img,n_segment, compactness=compactness)
        contour_mask = find_boundaries(labels)
        contour_mask[ROI==0] = 0
        
        labels[ROI==0] = -1
        self.labels_position = self.__get_labels_position(labels)

        self.__remap_labels(labels)

        self.labels = labels
        self.numOfPixel = np.max(labels) + 1
        self.contour_mask = contour_mask
        # self.adjacent_pairs = self.__construct_adjacency(labels)



    def __get_labels_position(self,labels):
        data = labels.ravel()
        f = lambda x: np.unravel_index(x.index, labels.shape)
        temp = pd.Series(data).groupby(data).apply(f)
        temp = temp.reset_index(drop=True)
        return temp

    def __remap_labels(self, labels):
        for idx, (rows, cols) in enumerate(self.labels_position):
            labels[rows, cols] = idx


    def __construct_adjacency(self, labels):
        @jit(nopython=True)
        def get_adjacency_matrix(adjacent_pairs,adj_mat):
            for (a,b) in adjacent_pairs:
                adj_mat[a,b] = True
                adj_mat[b,a] = True
            return adj_mat


        h,w = labels.shape[0:2]
        right_adjacent = np.zeros( ( (h-1),(w-1), 2), dtype=np.int32  )
        right_adjacent[:,:,0] = labels[0:h-1, 0:w-1]
        right_adjacent[:,:,1] = labels[0:h-1, 1:w]
        right_adjacent = right_adjacent.reshape((-1,2))

        bottom_adjacent = np.zeros( ( (h-1),(w-1), 2), dtype=np.int32   )
        bottom_adjacent[:,:,0] = labels[0:h-1, 0:w-1]
        bottom_adjacent[:,:,1] = labels[1:h, 0:w-1]
        bottom_adjacent = bottom_adjacent.reshape((-1,2))   
        adjacent_pairs = np.vstack((right_adjacent,bottom_adjacent))

        adj_mat = np.zeros((self.numOfPixel,self.numOfPixel),dtype=np.bool) 
        adj_mat = get_adjacency_matrix(adjacent_pairs,adj_mat)
        adj_mat = np.tril(adj_mat,k=1)

        adjacent_pairs = np.nonzero(adj_mat)
        adjacent_pairs = np.vstack((adjacent_pairs[0],adjacent_pairs[1])).T


        return adjacent_pairs

    


















