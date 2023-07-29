import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

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

cap= cv2.VideoCapture('./FMTC_drive_video_lane3.mp4')

while True:
        
    _, img = cap.read()
    image = img
    img = cv2.imread('/home/macaron/바탕화면/CULane/driver_23_30frame/05151643_0420.MP4/00000.jpg')
    cv2.imshow('ori', img)

    print(img.shape) # (height, width, channel) = (1668, 1720, 3)
    resized_img_1 = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img", resized_img_1)
    # cv2.waitKey()
    print(img)
    print("img_shape:", img.shape)
    img = to_tensor(img)
    img = img.permute(2, 0, 1)
    img = img / 255.
    img = img.unsqueeze(dim = 0)
    print(img)
    print("img_shape:", img.shape)

    p1 =  [220, 110]  # 좌상
    p2 =  [400, 110] # 우상
    p3 =  [640, 460] # 우하
    p4 =  [0, 460]  # 좌하
    # corners_point_arr는 변환 이전 이미지 좌표 4개 
    corner_points_arr = np.float32([p1, p2, p3, p4])
    height, width = image.shape[0], image.shape[1]

    image_p1 = [20, 0] # 좌상
    image_p2 = [width-20, 0] # 우상
    image_p3 = [width - width//4, height] # 우하
    image_p4 = [width//4, height] # 좌하

    image_params = np.float32([image_p1, image_p2, image_p3, image_p4])
    mat = cv2.getPerspectiveTransform(corner_points_arr, image_params) # mat = 변환행렬(3*3 행렬) 반
    data = cv2.warpPerspective(image, mat, (width, height))
    cv2.imshow('birdeye', data)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break


# # 중심 구하는 것
# try:
#     if count:
#         self.lanes = [] # 차선 저장
        
#         for idx in range(count):
#             lane = lanes[idx]
#             x, y = int(np.mean(lane, axis=0)[0]), int(np.mean(lane, axis=0)[1]) # 점의 x, y의 평균값
#             cv2.line(ori_img, (x, y), (x, y), (255,0,0), 20)
#             self.lanes.append((x, y)) # 차선의 평균 좌표 구하기
        
#         self.center = (int(np.mean(self.lanes, axis=0)[0]), int(np.mean(self.lanes, axis=0)[1]))
#         cv2.line(ori_img, (self.center[0], self.center[1]), (self.center[0], self.center[1]), (0,0,255), 20)
#         print("Center: ", self.center)
        
#         print("Difference with center:", self.center[1] - img_center[1])

        

# except:
#     print("Detect Fail 2!")


# lane = np.array(lane)
# x = list()
# y = list()
# for i in range(len(lane)):
#     x.append(lane[i][0]*1280) 
#     y.append((1-lane[i][1])*720) 

# plt.figure()
# plt.xlim(0, 1280)
# plt.ylim(0, 720)

# plt.plot(x, y)
# plt.show()

