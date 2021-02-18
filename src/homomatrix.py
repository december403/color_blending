import numpy as np
import cupy as cp


class HomoMatrix():
    def __init__(self):
        self.globalHomoMat = None
        self.localHomoMat_lst = None
        self.C1 = None
        self.C2 = None
        self.A = None
        self.non_global_homo_mat_lst = None

    def constructGlobalMat(self, src_pts, dst_pts):
        src_mean = np.mean(src_pts, axis=0)
        src_std = max(np.std(src_pts, axis=0))
        C1 = np.array([[1/src_std,         0, -src_mean[0]/src_std],
                    [0, 1/src_std, -src_mean[1]/src_std],
                    [0,         0,                    1]])
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        src_pts = src_pts @ C1.T

        dst_mean = np.mean(dst_pts, axis=0)
        dst_std = max(np.std(dst_pts, axis=0))
        C2 = np.array([[1/dst_std,         0, -dst_mean[0]/dst_std],
                    [0, 1/dst_std, -dst_mean[1]/dst_std],
                    [0,         0,                    1]])
        dst_pts = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
        dst_pts = dst_pts @ C2.T

        A = np.zeros((2*len(src_pts), 9))
        for i in range(len(src_pts)):
            x1, y1, _ = src_pts[i]
            x2, y2, _ = dst_pts[i]
            A[i*2, :] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
            A[i*2+1, :] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]


        u, s, v = np.linalg.svd(A)
        H = v[-1, :].reshape((3, 3))
        H = np.linalg.inv(C2) @ H @ C1
        H = H/H[-1, -1]

        self.globalHomoMat = H
        self.A = A
        self.C1 = C1
        self.C2 = C2


    def constructLocalMat(self, src_pts, grids, scale_factor):
        '''
        This function 
        src_pts : A N by 2 matrix. N is number of matching pairs.
        grids : A instance of Grids class. 
        '''

        gamma = 0.0025
        src_pts = cp.asarray(src_pts)
        grids_center_coordi = cp.asarray(grids.center_lst) # A M by 2 matrix, M is number of grids.
        grid_num = len(grids.center_lst)
        A = cp.asarray(self.A)
        C1 = cp.asarray(self.C1)
        C2 = cp.asarray(self.C2)
        matchingPairNum = src_pts.shape[0]
        skip = 0
        global_H = cp.asarray( np.copy(self.globalHomoMat ) )
        local_homo_mat_lst = cp.zeros((grid_num,3,3))

        change_mask = []
        for idx in range( grid_num):
            grid_coordi = grids_center_coordi[idx]

            weight = cp.exp(
                (-1) * cp.sum((src_pts - grid_coordi)**2, axis=1) / scale_factor**2)

            print(f'SVD {idx+1:8d}/{grid_num}({(idx+1)/(grid_num)*100:8.1f}%)  Current skip {skip} times. Current Skip rate is {skip/grid_num:5.3%}', end='\r')
            

            if cp.amax(weight) < gamma:
                skip += 1
                local_homo_mat_lst[idx, :, :] = global_H
                continue

            weight = cp.repeat(weight, 2)
            weight[weight < gamma] = gamma
            weight = weight.reshape((2*matchingPairNum, 1))
            weighted_A = cp.multiply(weight, A)
            u, s, v = cp.linalg.svd(weighted_A)
            H = v[-1, :].reshape((3, 3))
            H = cp.linalg.inv(C2) @ H @ C1
            H = H/H[-1, -1]
            local_homo_mat_lst[idx, :, :] = H
            change_mask.append(idx)
        print()

        self.non_global_homo_mat_lst = change_mask
        self.localHomoMat_lst = cp.asnumpy( local_homo_mat_lst )
        