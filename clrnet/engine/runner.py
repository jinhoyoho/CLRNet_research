import cv2
import torch
import numpy as np
import random
import rospy

from clrnet.models.registry import build_net
from clrnet.utils.net_utils import load_network
from mmcv.parallel import MMDataParallel
from std_msgs.msg import Float64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Runner(object):
    def __init__(self, cfg):
        self.steer = rospy.Publisher("stop", Float64, queue_size=1)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.resume()
        self.right_lane = []
        self.left_lane = []
        self.center = [0,0]
        
        #self.cap = cv2.VideoCapture('/home/macaron/바탕화면/lane_detection_ljh/test_dataset/test_video.mp4')
        self.cap = cv2.VideoCapture('./FMTC_drive_video_lane3.mp4')

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from)

    def test(self): # test만 실행
    
        _, image = self.cap.read()
        #image = cv2.imread('/home/macaron/바탕화면/CLRNet/data/tusimple/clips/0530/1492626760788443246_0/20.jpg') # 데이터 이미지 불러오기
        ori_img = image
        img_center = (ori_img.shape[1] // 2, ori_img.shape[0] // 2) # w, h
        #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (820, 320), interpolation=cv2.INTER_CUBIC)
        data = image
        # img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])    
        data = data.astype(np.float32) / 255.0
        data = to_tensor(data)
        # data = self.img2tensor(data) # tensor로 변환
        data = data.permute(2, 0, 1)
        data = data.unsqueeze(dim = 0)
        data = data.to(device)
        
        self.net.eval()
        predictions = []

        with torch.no_grad():
            output = self.net(data)
            output = self.net.module.heads.get_lanes(output)
            predictions.extend(output)
            lanes, count = imshow_lanes(ori_img, output) # 차선과 나온 개수
        
        try:
            for idx in range(count): # 차선 탐색
                lane = lanes[idx]
                if lane[0][0] < img_center[0]: # 중심보다 왼쪽에 있다면
                    self.left_lane = [[item[0] for item in lane], [item[1] for item in lane]]
                else: 
                    self.right_lane = [[item[0] for item in lane], [item[1] for item in lane]] # x와 y를 저장

            
            lk, rk = [0, 0]

            lpt1 = (min(self.left_lane[0]), max(self.left_lane[1]))
            lpt2 = (max(self.left_lane[0]), min(self.left_lane[1]))
            lk = self.extendLine(lpt1, lpt2)
            
            rpt1 = (min(self.right_lane[0]), min(self.right_lane[1]))
            rpt2 = (max(self.right_lane[0]), max(self.right_lane[1]))
            rk = self.extendLine(rpt1, rpt2)

            print("lk, rk:", lk, rk)


            # K = [lk, rk]
            # 보정 계수 조정
            a = 20
            b = 7 
            w = img_center[1]

            
            if lk == 0 and rk == 0:
                #########################
                #print("Detected Nothing") 
                #########################
                self.currentDirection = 0
                self.center[0] += 0
            elif lk == 0:
                self.currentDirection = -1
                self.center[0] -= int(a * (abs(rk) - abs(lk)))
            elif rk == 0:
                self.currentDirection = 1
                self.center[0] += int(a * (abs(lk) - abs(rk)))
            else:
                if abs(lk) < abs(rk):
                    self.currentDirection = 1
                    self.center[0] += int(b * abs(abs(lk) - abs(rk)))
                else:
                    self.currentDirection = -1
                    self.center[0] -= int(b * abs(abs(rk) - abs(lk)))

            steer = np.deg2rad(-(self.center[0] - w // 2))

            print("steer:" , steer)
            
        except:
            print("ERROR!")
            steer = 0
          
        
        cv2.imshow('image', ori_img)
        self.steer.publish(steer)

        # print("error!")
        # return #종료

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