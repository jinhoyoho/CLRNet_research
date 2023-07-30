import cv2
import torch
import numpy as np
import random
import rospy
import os
import sys

from clrnet.models.registry import build_net
from clrnet.utils.net_utils import load_network
from mmcv.parallel import MMDataParallel
from std_msgs.msg import Int16

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/src/sensor/CLRNet_research")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Runner(object):
    def __init__(self, cfg):
        self.steer_pub = rospy.Publisher("lane", Int16, queue_size=1)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.resume() # 이거 안하면 검출 안됨
        self.right_lane = []
        self.left_lane = []
        self.center = [0,0]
        self.steer = 0
        
        self.cap = cv2.VideoCapture(0) # 카메라 구동
        # self.cap = cv2.VideoCapture('/home/macaron/바탕화면/lane_detection_ljh/test_dataset/test_video.mp4')
        self.cap = cv2.VideoCapture(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +\
        "/FMTC_drive_video_lane3.mp4")
        #self.cap = cv2.VideoCapture('/home/macaron/Desktop/FMTC_drive_video_lane3.mp4')
        #self.cap = cv2.VideoCapture('./FMTC_drive_video_lane3.mp4')

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from)

    def test(self): # test만 실행
    
        ret, image = self.cap.read()
        
        if ret:
            ori_img = image
            # self.center = [ori_img.shape[1] // 2, ori_img.shape[0] // 2] # w, h

            # p1 =  [215, 115]  # 좌상
            # p2 =  [405, 115] # 우상
            # p3 =  [640, 460] # 우하
            # p4 =  [0, 460]  # 좌하
            # # corners_point_arr는 변환 이전 이미지 좌표 4개 
            # corner_points_arr = np.float32([p1, p2, p3, p4])
            # height, width = image.shape[0], image.shape[1]

            # image_p1 = [20, 0] # 좌상
            # image_p2 = [width-20, 0] # 우상
            # image_p3 = [width - width//4, height] # 우하
            # image_p4 = [width//4, height] # 좌하

            # image_params = np.float32([image_p1, image_p2, image_p3, image_p4])
            # mat = cv2.getPerspectiveTransform(corner_points_arr, image_params) # mat = 변환행렬(3*3 행렬) 반
            # data = cv2.warpPerspective(image, mat, (width, height))
            
            # ori_img = data

            data = cv2.resize(ori_img, (800, 320), interpolation=cv2.INTER_CUBIC) # resize
            data = data.astype(np.float32) / 255.0 # normalize
            data = to_tensor(data) # tensor
            data = data.permute(2, 0, 1) # BGR -> RGB
            data = data.unsqueeze(dim = 0) # batch size
            data = data.to(device) # cuda
            
            self.net.eval()
            predictions = []

            with torch.no_grad():
                output = self.net(data) # inference
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
                lanes, count = imshow_lanes(ori_img, output) # 차선과 나온 개수
            
            try:
                if count:
                    self.lanes = [] # 차선 저장
                    
                    for idx in range(count):
                        lane = lanes[idx]
                        x, y = int(np.mean(lane, axis=0)[0]), int(np.mean(lane, axis=0)[1]) # 점의 x, y의 평균값
                        cv2.line(ori_img, (x, y), (x, y), (255,0,0), 20)
                        self.lanes.append((x, y)) # 차선의 평균 좌표 구하기
                    
                    self.center = (int(np.mean(self.lanes, axis=0)[0]), int(np.mean(self.lanes, axis=0)[1]))
                    cv2.line(ori_img, (self.center[0], self.center[1]), (self.center[0], self.center[1]), (0,0,255), 20)
                    print("Center: ", self.center)
                    
                    print("Difference with center:", self.center[0] - ori_img.shape[1] // 2)
                    
                    self.steer = int(self.center[0] - ori_img.shape[1] // 2)
                
            except:
                print("ERROR!")
                self.steer = 0
            
            
            cv2.imshow('image', ori_img)
            self.steer_pub.publish(self.steer)

        else:
            print("Video 없음!")
            return

    def extendLine(self, pt1, pt2):
        if pt1[0] - pt2[0] != 0:
            dx = pt1[0] - pt2[0]    
            dy = pt1[1] - pt2[1]
            
            k = dy / dx
            
        else:
            k = 0
            return k
        
        return k
        
# 차선 시각화
COLORS = [
    (255, 0, 0), # B
    (0, 255, 0), # G
    (0, 0, 255), # R
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_lanes(img, prediction, width=4):
    for lanes in prediction:
        lanes = [lane.to_array(640, 480) for lane in lanes] # ori_w_img, ori_h_img
    
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