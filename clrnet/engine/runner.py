import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random

from clrnet.models.registry import build_net
from clrnet.utils.net_utils import load_network
from mmcv.parallel import MMDataParallel

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
            imshow_lanes(ori_img, output)
        
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


def imshow_lanes(img, prediction, show=True, width=4):
    for lanes in prediction:
        lanes = [lane.to_array(640, 480) for lane in lanes] # ori_w_img, ori_h_img
    
    lanes_xys = []
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


    if show:
        cv2.imshow('view', img)



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