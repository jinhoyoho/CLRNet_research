import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        # self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None
        self.cap = cv2.VideoCapture('/home/macaron/바탕화면/lane_detection_ljh/test_dataset/test_video.mp4')

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from)
    
    def img2tensor(self, img):
    #the format of img needs to be bgr format

        img = img[..., ::-1]  #bgr2rgb
        img = img.transpose(2, 0, 1)  #(H, W, CH) -> (CH, H, W)
        img = np.ascontiguousarray(img)
        
        tensor = torch.tensor(img, dtype=torch.float32)
        return tensor

    def test(self): # test만 실행
        # if not self.test_loader:
        #     self.test_loader = build_dataloader(self.cfg.dataset.test,
        #                                         self.cfg,
        #                                         is_train=False)

        _, image = self.cap.read()
        data = image

        #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data, (820, 320), interpolation=cv2.INTER_CUBIC)
        # img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])    
        tf_toTensor = ToTensor() 
        data = tf_toTensor(data)
        # data = self.img2tensor(data) # tensor로 변환
        data = data.unsqueeze(dim = 0)
        data = data.to(device)
        print(data)
        print(data.size())
        
        self.net.eval()
        predictions = []

        with torch.no_grad():
            output = self.net(data)
            print("output1:", output)
            output = self.net.module.heads.get_lanes(output)
            print("lane:", output)
            predictions.extend(output)

            if output:
                imshow_lanes(image, output)

        # if self.cfg.view:
        #     self.test_loader.dataset.view(output, data['meta'])

        # metric = self.test_loader.dataset.evaluate(predictions,
        #                                            self.cfg.work_dir)
        # if metric is not None:
        #     self.recorder.logger.info('metric: ' + str(metric))

    # def validate(self):
    #     if not self.val_loader:
    #         self.val_loader = build_dataloader(self.cfg.dataset.val,
    #                                            self.cfg,
    #                                            is_train=False)
    #     self.net.eval()
    #     predictions = []
    #     for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
    #         data = self.to_cuda(data)
    #         with torch.no_grad():
    #             output = self.net(data)
    #             output = self.net.module.heads.get_lanes(output)
    #             predictions.extend(output)
    #         if self.cfg.view:
    #             self.val_loader.dataset.view(output, data['meta'])

    #     metric = self.val_loader.dataset.evaluate(predictions,
    #                                               self.cfg.work_dir)
    #     self.recorder.logger.info('metric: ' + str(metric))
        
# 차선 시각화
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
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
        lanes = [lane.to_array(1280, 720) for lane in lanes] # ori_w_img, ori_h_img

    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)

    if lanes_xys:
        lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)


    if show:
        cv2.imshow('view', img)
        # cv2.waitKey(0)
