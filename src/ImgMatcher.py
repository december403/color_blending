from cv2 import cv2
import numpy as np

class ImgMatcher():
    def __init__(self, tar_img, ref_img):
        self.__tar_img = tar_img
        self.__ref_img = ref_img
        self.matches = None
        self.ref_des = None
        self.tar_des = None
        self.ref_kps = None
        self.tar_kps = None
        self.mathcNum = None
        self.mask = None




    def detectORB(self,ptsNum=10000):
        orb = cv2.ORB_create(ptsNum)
        self.ref_kps, self.ref_des = orb.detectAndCompute(self.__ref_img, None)
        self.tar_kps, self.tar_des = orb.detectAndCompute(self.__tar_img, None)

    def matchORB(self, projErr=5, crossCheck=True):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        matches = matcher.match(self.tar_des, self.ref_des)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for match in matches])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for match in matches])
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projErr)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        self.mask = mask
        self.matches = matches
        self.matchNum = len(mask==1)
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    def detectAKAZE(self):
        akaze = cv2.AKAZE_create()
        self.ref_kps, self.ref_des = akaze.detectAndCompute(self.__ref_img, None)
        self.tar_kps, self.tar_des = akaze.detectAndCompute(self.__tar_img, None)

    def matchAKAZE(self, projErr=5, crossCheck=True):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        matches = matcher.match(self.tar_des, self.ref_des)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for match in matches])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for match in matches])
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projErr)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        self.mask = mask
        self.matches = matches
        self.matchNum = len(mask==1)
        self.src_pts = src_pts
        self.dst_pts = dst_pts




    def detectSIFT(self, ptsNum):
        sift = cv2.SIFT_create(ptsNum)
        self.ref_kps, self.ref_des = sift.detectAndCompute(self.__ref_img, None)
        self.tar_kps, self.tar_des = sift.detectAndCompute(self.__tar_img, None)

    def matchSIFT(self, projErr=5, crossCheck=True):
        matcher = cv2.BFMatcher(crossCheck=crossCheck)
        matches = matcher.match(self.tar_des, self.ref_des)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for match in matches])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for match in matches])
        
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projErr)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        self.mask = mask
        self.matches = matches
        self.matchNum = len(mask[mask==1])
        self.src_pts = src_pts
        self.dst_pts = dst_pts


    def KNNmatchSIFT(self,projErr=5):
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(self.tar_des, self.ref_des, k=2)
        print(len(matches))
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        print(len(good_matches))
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for match in good_matches])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for match in good_matches])
        
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projErr)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for idx,
                            match in enumerate(good_matches) if mask[idx] == 1])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for idx,
                            match in enumerate(good_matches) if mask[idx] == 1])
        self.mask = mask
        self.matches = good_matches
        self.matchNum = len(mask[mask==1])
        self.src_pts = src_pts
        self.dst_pts = dst_pts


    def KNNmatchAKAZE(self,projErr=5):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(self.tar_des, self.ref_des, k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for match in good_matches])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for match in good_matches])
        
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projErr)
        src_pts = np.array([self.tar_kps[match.queryIdx].pt for idx,
                            match in enumerate(good_matches) if mask[idx] == 1])
        dst_pts = np.array([self.ref_kps[match.trainIdx].pt for idx,
                            match in enumerate(good_matches) if mask[idx] == 1])
        self.mask = mask
        self.matches = good_matches
        self.matchNum = len(mask[mask==1])
        self.src_pts = src_pts
        self.dst_pts = dst_pts

        




# ref_img = cv2.imread('./image/UAV/DJI_0001.png')
# tar_img = cv2.imread('./image/UAV/DJI_0002.JPG')

# imgRegister = ImgRegister(tar_img, ref_img)
# start_time = time.time()
# imgRegister.detectAKAZE()
# print(f"Process finished --- {(time.time() - start_time)} seconds ---")
# imgRegister.KNNmatchAKAZE(projErr=5)
# print(f"Process finished --- {(time.time() - start_time)} seconds ---")
# img4 = cv2.drawMatches(tar_img, imgRegister.tar_kps, ref_img, imgRegister.ref_kps, imgRegister.matches, None,
#                        flags=cv2.DrawMatchesFlags_DEFAULT, matchesMask=imgRegister.mask)
# cv2.imwrite('match.jpg', img4)
# print(len(imgRegister.mask[imgRegister.mask==1]))
