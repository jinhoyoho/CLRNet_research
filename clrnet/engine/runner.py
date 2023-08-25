import cv2
import torch
import numpy as np
import random

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


from cam_cali import CameraCali
from inverse_birdeyeview import IBE
from clrnet.models.registry import build_net
from clrnet.utils.net_utils import load_network
from mmcv.parallel import MMDataParallel
from get_curve import curve, get_steer, draw_lanes
from configs.clrnet.clr_resnet34_tusimple import img_h, img_w, cut_height

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.resume()
        self.steer = 0
        self.CameraCali = CameraCali()
        self.IBE = IBE(self.CameraCali.extrinsic, self.CameraCali.intrinsic)
        self.previous = 0 # 이전 steer 값

        self.plot = [] # steer 그림을 그리기 위한 plot list
        
        self.cap = cv2.VideoCapture(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +\
                                      "/drive_video.mp4")

        print("device: ", device)


    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from)
        

    def test(self): # test만 실행
        # try:
            ret, ori_img = self.cap.read() # 이미지 불러오기
            
            if ret: # 이미지가 있다면
                data = ori_img # 받아온 이미지
                data = data[cut_height:, :, :] # cut_height
                data = cv2.resize(data, (img_w, img_h), interpolation=cv2.INTER_CUBIC) # resize(img_h, img_w)
                data = data.astype(np.float32) / 255.0 # normalize
                data = to_tensor(data) # tensor
                data = data.permute(2, 0, 1) # BGR -> RGB
                data = data.unsqueeze(dim = 0) # Batch size 지정
                data = data.to(device) # cuda로 올리기
                
                self.net.eval()

                with torch.no_grad():
                    output = self.net(data) # 에측
                    output = self.net.module.heads.get_lanes(output) # 예측
                    imshow_lanes(ori_img, output, cut_height) # 시각화

                # Inverse perspective birdeyeview 진행
                birdeyeview = self.IBE.birdeyeview(ori_img)

                lane_blue, lane_green = [], []

                dst1 = cv2.inRange(birdeyeview, (255, 0, 0), (255, 0, 0)) # 파란색 차선 검출
                dst2 = cv2.inRange(birdeyeview, (0, 255, 0), (0, 255, 0)) # 초록색 차선 검출

                lane_blue = np.argwhere(dst1) # 왼쪽 차선 좌표
                lane_blue = lane_blue.T
                
                lane_green = np.argwhere(dst2) # 오른쪽 차선 좌표
                lane_green = lane_green.T

                # 사진을 기준으로 위에서부터 아래로 점 생성
                # lane_blue[0][:] -> y좌표
                # lane_blue[1][:] -> x좌표

                # cv2.imshow('binary_image', binary_image)
                cv2.imshow('ori_image', ori_img) # 원래 이미지

                ploty = np.linspace(0, birdeyeview.shape[0]-1, birdeyeview.shape[0]) # 0부터 img.shape[0]-1까지 img.shape[0]개의 점을 생성 -> 그냥 0부터 정수로 179까지 1씩 증가함

                left_x, right_x, detect_one, detect_two = curve(lane_blue, lane_green, ploty) # 왼쪽 차선과 오른쪽 차선 2차 함수
                # 왼쪽 차선, 오른쪽 차선 이차함수, 차선 이미지, 왼쪽 오른쪽 차선 검출 여부, steer 방향
                
                add_image = draw_lanes(birdeyeview, left_x, right_x) # 그림 그리기

                center_length = get_steer(add_image, left_x, right_x, detect_one, detect_two)

                # steer값 설정           
                self.steer = int(np.round(center_length)) # cm단위

                if self.steer == 0: # 0은 사실상 검출 안됨
                    self.steer = self.previous # 이전 steer값으로 설정

                if self.steer < 0: # 음수면 왼쪽으로
                    cv2.putText(add_image, "GO LEFT!", org=(90, 160),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(255 ,255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False) 
            
                elif self.steer > 0: # 양수면 오른쪽으로
                    cv2.putText(add_image, "GO RIGHT!", org=(90, 160),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(255 ,255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False) 

                cv2.imshow('birdeyeview', add_image)

                self.previous = self.steer # 이전 steer 저장

                print("steer: ", self.steer)
                print("center_length: ", center_length)

                self.plot.append(self.steer) # steer값 저장

                ###### 차선 gps 좌표로 변환 ######

                # 아래는 차선의 pixel 좌표를 카메라 및 gps 좌표계로 변환하는 과정이다.
                # 이를 위해서는 ros를 통해서 gps좌표값을 입력해주어야 한다.
                # 하지만 누구나 ros 없이 실행하기 위해서 삭제해주었다.
                
                # 카메라 기준으로 real world 좌표로 변경 
                # point = points2no_z_point(add_image, left_x, right_x, ploty, detect_one, detect_two)

                # gps 좌표로 변경
                # gps_lane = tf2tm(point, self.x, self.y, self.z) 

                # rviz로 visualization하기 위한 코드이나 
                # ros를 없애서 누구나 실행할 수 있도록 하기 위해서 코드 삭제
                # self.visual_lane(gps_lane)

            else:
                print("Video 없음!")
                return
        
        # except:
        #     print("error!")

        
# 차선 시각화
COLORS = [
    (255, 0, 0), # B
    (0, 255, 0), # G
    (0, 0, 255), # R
]


def imshow_lanes(img, prediction, cut_height, width=4):
    for lanes in prediction:
        lanes = [lane.to_array(img.shape[1], img.shape[0]) for lane in lanes] # width, height        

    lanes_xys = [] #초기화
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)

    try:
        if lanes_xys:
            lanes_xys.sort(key=lambda xys : xys[0][0])

        for idx, xys in enumerate(lanes_xys):
            for i in range(1, len(xys)):
                cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)

    except:
        print("Detect Fail!")

    return lanes_xys, len(lanes_xys)



def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
    
