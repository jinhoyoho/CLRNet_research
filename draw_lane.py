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

img = cv2.imread('/home/macaron/바탕화면/CLRNet/data/tusimple/clips/0530/1492626760788443246_0/20.jpg')
print(img.shape) # (height, width, channel) = (1668, 1720, 3)
resized_img_1 = cv2.resize(img, dsize=(820, 320), interpolation=cv2.INTER_CUBIC)
print(resized_img_1.shape) 
cv2.imshow("img", resized_img_1)
cv2.waitKey()
print(img)
print("img_shape:", img.shape)
img = to_tensor(img)
img = img.permute(2, 0, 1)
img = img / 255.
img = img.unsqueeze(dim = 0)
print(img)
print("img_shape:", img.shape)

lane = [[0.46868697, 0.66388159],
 [0.45909098, 0.67152064],
 [0.44948155, 0.6791597 ],
 [0.43989509, 0.68679874],
 [0.43029407, 0.6944378 ],
 [0.42068687, 0.70207686],
 [0.41108149, 0.70971592],
 [0.40147701, 0.71735498],
 [0.39186737, 0.72499405],
 [0.38225064, 0.73263311],
 [0.37265649, 0.74027213],
 [0.36305571, 0.74791119],
 [0.35345021, 0.75555026],
 [0.34384626, 0.76318928],
 [0.33424947, 0.77082834],
 [0.3246505 , 0.7784674 ],
 [0.31504837, 0.78610647],
 [0.30544642, 0.79374553],
 [0.29584524, 0.80138455],
 [0.28624958, 0.80902361],
 [0.27665943, 0.81666268],
 [0.26706043, 0.82430174],
 [0.25746113, 0.8319408 ],
 [0.24786855, 0.83957986],
 [0.23827922, 0.84721892],
 [0.22868539, 0.85485798],
 [0.2190966 , 0.86249701],
 [0.20950776, 0.87013607],
 [0.19992663, 0.87777513],
 [0.19033977, 0.88541419],
 [0.18075863, 0.89305325],
 [0.1711913 , 0.90069231],
 [0.16164447, 0.90833137],
 [0.15210019, 0.91597043],
 [0.14255787, 0.92360946],
 [0.13303299, 0.93124852],
 [0.1235332 , 0.93888758],
 [0.11405496, 0.94652661],
 [0.10457928, 0.95416567],
 [0.09510852, 0.96180473],
 [0.08564913, 0.96944379],
 [0.07621777, 0.97708285],
 [0.06686679, 0.98472188],
 [0.05764965, 0.99236094],
 [0.04866111, 1.        ]]


lane = np.array(lane)
x = list()
y = list()
for i in range(len(lane)):
    x.append(lane[i][0]*1280) 
    y.append((1-lane[i][1])*720) 

plt.figure()
plt.xlim(0, 1280)
plt.ylim(0, 720)

plt.plot(x, y)
plt.show()

