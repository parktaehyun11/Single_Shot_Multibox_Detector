import cv2
import numpy as np
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt
import os
from copy import copy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def xyxy2xywh(xyxy):
    """
    Description:
    x1 y1 x2 y2 좌표를 cx cy w h 좌표로 변환합니다.
    :param xyxy: shape (..., 4), 2차원 이상의 array 가 들어와야 함
    :return: xywh shape(... , 4), 2차원 이상의 array 가 들어와야 함
    """
    w = xyxy[..., 2:3] - xyxy[..., 0:1]
    h = xyxy[..., 3:4] - xyxy[..., 1:2]
    cx = xyxy[..., 0:1] + w * 0.5
    cy = xyxy[..., 1:2] + h * 0.5
    xywh = np.concatenate([cx, cy, w, h], axis=1)
    return xywh


def xywh2xyxy(xywh):
    """
    Description:
    cx cy w h 좌표를 x1 y1 x2 y2 좌표로 변환합니다.
    center x, center y, w, h 좌표계를 가진 ndarray 을 x1, y1, x2, y2 좌표계로 변경
    xywh : ndarary, shape, (..., 4), 마지막 차원만 4 이면 작동.
    """
    cx = xywh[..., 0]
    cy = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]

    x1 = cx - (w * 0.5)
    x2 = cx + (w * 0.5)
    y1 = cy - (h * 0.5)
    y2 = cy + (h * 0.5)

    return np.stack([x1, y1, x2, y2], axis=-1)


def plot_images(imgs, names=None, random_order=False, savepath=None):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        #
        if not names is None:
            ax.set_title(str(names[ind]))
    if not savepath is None:
        plt.savefig(savepath)
    plt.tight_layout()
    plt.show()


def draw_rectangle(img, coordinate, color=(255, 0, 0)):
    """
    Description:
    img 에 하나의 bounding box 을 그리는 함수.
    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinate: tuple or list(iterable), shape=(4,) x1 ,y1, x2, y2
    :param color: tuple(iterable), shape = (3,)
    :return:
    """

    # opencv 에 입력값으로 넣기 위해 반드시 정수형으로 변경해야함
    coordinate = coordinate.astype('int')
    x_min = coordinate[0]
    x_max = coordinate[2]
    y_min = coordinate[1]
    y_max = coordinate[3]

    img = img.astype('uint8')

    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=1)


def draw_rectangles(img, coordinates, color=(255, 0, 0)):
    """
    Description:
    하나의 img 에 복수개의  bounding box 을 그리는 함수.
    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinates: tuple or list(iterable), ((x y, x, y), (x y, x, y) .. (x y, x, y))
    내부적으로는 x y x y 좌표계를 가지고 있어야 함
    :return: img: ndarray, 3d array, shape = (H, W, CH)
    """
    for coord in coordinates:
        img = draw_rectangle(img, coord, color)
    return np.array(img)


def show_image(array, title=None, cmap='jet'):
    plt.figure(figsize=(20, 5))
    im = plt.imshow(array, cmap=cmap)

    if title:
        plt.title(title)
    plt.colorbar(im)
    plt.show()


def str2bool(str_):
    if str_ == 'False':
        return False
    elif str_ == 'True':
        return True
    else:
        raise NotImplementedError


def generate_tmp_folder(folder_name):
    """
    Description:
    입력된 인자의 경로에 폴더가 존재한다면 뒤에 숫자를 추가합니다.
    foldername -> foldername0
    Args:
        folder_name:
    Returns:
    """
    tmp_count = 0
    new_folder_name = copy(folder_name)
    while (True):
        if os.path.isdir(new_folder_name):
            # 폴더가 존재하면 count 을 1올립니다.
            tmp_count += 1
            new_folder_name = '{}_{}'.format(folder_name, tmp_count)
        else:
            # 폴더가 존재하지 않으면 새롭게 만들어진 폴더 name 을 반환하니다.
            return new_folder_name


def set_optimizer(optmizer_name, lr):
    """
    Description:
     keras optimizer 을 결정합니다.
    :param optmizer_name:
     - adam
     - sgd
     - rmsprop
    :param lr: float
    :return: tensorflow.keras.optimizer
    """
    if optmizer_name == 'adam':
        opt = Adam(lr=lr)

    elif optmizer_name == 'sgd':
        opt = SGD(lr=lr)

    elif optmizer_name == 'rmsprop':
        opt = RMSprop(lr=lr)

    elif optmizer_name == 'momentum':
        opt = SGD(lr=lr, momentum=0.9, nesterov=True)

    else:
        raise ValueError
    return opt
