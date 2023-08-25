import numpy as np
import cv2 
import matplotlib.pyplot as plt
from configs.clrnet.clr_resnet34_tusimple import world_x_interval, world_y_interval

# 이차 곡선을 만드는 함수
def curve(blue, green, ploty):

    left_fitx, right_fitx, left_fit, right_fit = 0, 0, [0], [0] # 초기화
    
    detect_one, detect_two = False, False

    if blue.any(): # 차선 하나 검출
        left_fit = np.polyfit(blue[:][0], blue[:][1], 2) #(lefty, leftx)에 대한 2차 방정식의 계수를 반환함

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 왼쪽 차선에 대해서 2차 함수 생성
        
        detect_one = True # 검출 완료

    if green.any(): # 차선 두개 검출
        right_fit = np.polyfit(green[:][0], green[:][1], 2) #(lefty, leftx)에 대한 2차 방정식의 계수를 반환함
   
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # 왼쪽 차선에 대해서 2차 함수 생성
        
        detect_two = True # 검출 완료
    
    
    return left_fitx, right_fitx, detect_one, detect_two

# 그림을 그리는 함수
def draw_lanes(img, left_fit, right_fit): # img랑 각각 차선의 2차 방정식

    try:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0]) 
        color_img = np.zeros_like(img) # img 사이즈에 맞게 0으로 채워짐 
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        #flipud : 상하반전, vstack : 상하로 합치기, hstack : 좌우로 합치기 [[]]기준임

        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))]) # 오른쪽 차선도 똑같이 진행
        points = np.hstack((left, right)) # 차선끼리의 배열이 합쳐짐

        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255)) # 그림 그리기

        add_image = cv2.addWeighted(img, 1, color_img, 0.7, 0) # img와 inv_perspective 사진을 하나로 합치기
        

    except: # 그리지 못하는 경우 -> left나 right 둘 중 하나가 0
        print("Draw Failed!")
        
        cv2.putText(color_img, "draw_failed", org=(90, 200),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6, color=(255 ,255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False) 
        
        add_image = cv2.addWeighted(img, 1, color_img, 0.7, 0)
        

    return add_image
       
# 대강적인 steer값을 구하는 함수
def get_steer(img, leftx, rightx, detect_one, detect_two):

    center_length = 0
    
    car_pos = img.shape[1]/2 # 중앙은 이미지의 중앙값

    xm_per_pix =  world_y_interval # width
    # -> configs에서 수정가능

    try:
        if detect_two: # 양쪽 다 구해진 경우
            lane_center_position = (leftx[len(leftx)//2] + rightx[len(rightx)//2]) / 2 # 차선의 중앙 픽셀값
        
            center_length = (lane_center_position - car_pos) * xm_per_pix * 100 # 중앙에서부터 떨어진 거리(cm단위)
            
            print("Detect both!")

        
        elif detect_one: # 왼쪽만 구해진 경우

            sub_first_last = leftx[0] - leftx[-1]
            
            if sub_first_last > 0: # 양수면 오른쪽으로
                direction = 1
            else: # 음수면 왼쪽으로
                direction = -1

            center_length = (leftx[len(leftx)//2] - car_pos) * xm_per_pix * 100 # 중앙에서부터 떨어진 거리(cm단위)
        
            center_length = abs(center_length) * direction

            print("Only one!")
        
        else:
            print("Detect False!")
    
        return center_length
   
    except:
        print("error!")

        return 0