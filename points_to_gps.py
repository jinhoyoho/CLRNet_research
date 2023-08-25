import numpy as np
from math import sin,cos
from numpy import transpose,dot
from configs.clrnet.clr_resnet34_tusimple import world_x_interval, world_y_interval

def points2no_z_point(img, leftx, rightx, ploty, detect_one, detect_two):

    car_center = img.shape[1]/2 # 차량 중심

    world_x = world_x_interval # height
    world_y = world_y_interval # width 작을수록 map이 커짐

    x_point = ploty * world_x # real world의 앞+ 뒤-
    z_point = ploty * 0

    if detect_two: # 둘 다 검출된 경우
        y_point1 = ((car_center - leftx[::-1]) * world_y) # real world의 좌+ 우-
        y_point2 = ((car_center - rightx[::-1]) * world_y) # real world의 좌+ 우-

        point1 = np.column_stack((x_point, y_point1)) # x, y 합치기
        point1 = np.column_stack((point1, z_point)) # z축 합치기
        point2 = np.column_stack((x_point, y_point2)) # x, y s합치기
        point2 = np.column_stack((point2, z_point)) # z축 합치기

        point = np.row_stack((point1, point2))

        return point # 1열이 height, 2열이 width
        
    elif detect_one:
        y_point1 = (car_center - leftx[::-1]) * world_y # real world의 좌+ 우-

        point = np.column_stack((x_point, y_point1)) # x, y 합치기
        point = np.column_stack((point, z_point)) # z축 합치기

        return point
    
    else:
        return []
    

def tf2tm(points, x, y, heading):
    obs_tm = [] # 빈 리스트 생성
    
    T = [[cos(heading), -1*sin(heading), x], \
            [sin(heading),  cos(heading), y], \
            [      0     ,      0       , 1]] 
    
    for point in points:
        obs_tm.append(dot(T,transpose([point[0]+1, point[1], 1]))) # 열 방향으로 결합

    obs_tm = np.array(obs_tm)

    return obs_tm