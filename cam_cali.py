#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
import cv2
import os
import glob

class CameraCali(object):
    def __init__(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        wc = 8 ## 체스 보드 가로 패턴 개수 - 1
        hc = 6  ## 체스 보드 세로 패턴 개수 - 1
        objp = np.zeros((wc * hc, 3), np.float32)
        objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(os.path.dirname(__file__) + '/images/*.jpg')

        for frame in images:
            img = cv2.imread(frame)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## gray scale로 바꾸기

            ret, corners = cv2.findChessboardCorners(self.gray, (wc, hc), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)  ## 체스 보드 찾기
            print(ret, end=' ')
            ## 만약 ret값이 False라면, 체스 보드 이미지의 패턴 개수를 맞게 했는지 확인하거나 (wc, hc)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(self.gray, corners, (10, 10), (-1, -1), criteria) ## Canny86 알고리즘으로
                imgpoints.append(corners2)

                ## 찾은 코너 점들을 이용해 체스 보드 이미지에 그려넣는다
                img = cv2.drawChessboardCorners(img, (wc, hc), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey()

                ## mtx = getOptimalNewCameraMatrix parameter alpha 
                ## dist = Free scaling parameter 
                ## 4번째 인자 = between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image)
                ## 1에 가까울수록 왜곡을 펼 때 잘라낸 부분들을 더 보여준다
                ## 전체를 보고 싶다면 1, 펴진 부분만 보고 싶다면 0에 가깝게 인자 값을 주면 된다

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, self.gray.shape[::-1], None, None)

        #2차원 영상좌표
        points_2D = np.array([
                                (5, 450),  #좌 하단 
                                (600, 450),  #우 하단
                                (250, 250),  #좌 상단
                                (430, 250),  #우 상단
                            ], dtype="double")
        
        # (5, 471),  #좌 하단 
        # (600, 471),  #우 하단
        # (257, 261),  #좌 상단
        # (373, 261),  #우 상단
        # 어안 카메라

        # (11, 330),  #좌 하단 
        # (637, 340),  #우 하단
        # (277, 190),  #좌 상단
        # (427, 190),  #우 상단 
        # 기본 카메라
                            
        #3차원 월드좌표
        points_3D = np.array([
                            (3.3, 1.75, 0),       #좌 하단
                            (3.65, -1.75, 0),        #우 하단
                            (15.3, 1.75, 0),        #좌 상단
                            (15.3, -1.75, 0)          #우 상단
                            ], dtype="double") # 세로, 가로, 높이

        cameraMatrix = mtx

        dist_coeffs = np.array([0,0,0,0,0])

        # # 어안 카메라 내부 파라미터
        fx = 345.727618
        fy = 346.002121
        cx = 320.000000
        cy = 240.000000
        
        cameraMatrix[0][2] = cx
        cameraMatrix[1][2] = cy
        cameraMatrix[0][0] = fx
        cameraMatrix[1][1] = fy

        retval, rvec, tvec = cv2.solvePnP(points_3D, points_2D, cameraMatrix, dist_coeffs, rvec=None, tvec=None, useExtrinsicGuess=None, flags=None)

        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        cameraMatrix = np.append(cameraMatrix, [[0,0,0]], axis=0)
        cameraMatrix = np.append(cameraMatrix, [[0],[0],[0],[1]], axis=1)

        self.intrinsic = cameraMatrix # 내부 파라미터
        extrinsic = np.append(R, t, axis = 1) # 외부 파라미터
        self.extrinsic = np.append(extrinsic, [[0,0,0,1]], axis=0)
        
        # 다크 프로그래머로 찍은 내부 파라미터 # Logitech cam
        # fx = 599.112989
        # fy = 599.112989
        # cx = 320.000000
        # cy = 240.000000
        
        
        print()
        print("intrinsic: ", end='\n')
        print(self.intrinsic)
        print("extrinsic: ", end='\n')
        print(self.extrinsic)


# extrinsic: 
# [[ 0.02085655 -0.99942966  0.02655859 -0.07606992]
#  [-0.08873191 -0.02830993 -0.99565315  1.52154723]
# [ 0.99583716  0.01840929 -0.08927175 -0.59290113]
# [ 0.          0.          0.          1.        ]]


# extrinsic: 
# [[-0.21672598 -0.96008707 -0.1768125  -0.12273296]
#  [-0.15972832  0.21355007 -0.96378588  1.92370343]
#  [ 0.96307669 -0.18063548 -0.19963495 -3.75339273]
#  [ 0.          0.          0.          1.        ]]




"""
Camera matrix : 

[[670.58082454   0.         299.62265463]
 [  0.         672.3836304  222.97593079]
 [  0.           0.           1.        ]]
 

dist : 

[[ 0.04901945  0.43784568 -0.00677837 -0.00950096 -1.96108272]]

""" 
