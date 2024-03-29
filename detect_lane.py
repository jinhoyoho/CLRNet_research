#!/usr/bin/env python
# -- coding: utf-8 --
import os
import cv2
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

from clrnet.utils.config import Config
from clrnet.engine.runner import Runner

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/src/sensor/CLRNet_research")

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 사용하고자 하는 특정 gpu 
    cfg = Config.fromfile((os.path.dirname(os.path.abspath(__file__))) +\
        "/configs/clrnet/clr_resnet34_tusimple.py") # 모델 아키텍처 지정
    
    cfg.gpus = 1 # gpu 개수 지정
    
    cfg.load_from = 'pth파일 경로 입력' # pt파일 경로
    
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view # 시각화
    cfg.seed = args.seed
    
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    mean_fps = np.array([]) # 평균 프레임 구하기

    runner = Runner(cfg)

    prevTime = time.time()		# previous time

    while True:
        runner.test() # 실행

        curTime = time.time()	# current time
        fps = 1 / (curTime - prevTime)
        mean_fps = np.append(mean_fps, fps)
        prevTime = curTime
        print("FPS : ", fps) # 프레임 수 문자열에 저장

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            break
    

    #steer값 그림 그리기
    x = range(len(runner.plot))
    plt.figure()
    plt.scatter(x, runner.plot)
    plt.ylabel('steer')
    plt.xlabel('frame')
    plt.show()
    

    print('평균 프레임: ', mean_fps.mean())

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_false', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
