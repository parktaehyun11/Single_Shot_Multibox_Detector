import numpy as np
from tensorflow.python.keras.applications.resnet import ResNet50
from tqdm import tqdm
from iou import calculate_iou


def tiling_default_boxes(center_xy, sizes):
    """
    Description:
        original image 와 좌표 위치가 매칭된 feature map의 모든 cell에 적용된 default boxes의 좌표값을 반환합니다.
    Args:
        :param center_xy: shape=(N_center, 2), feature map의 각 cell과 original image에 매칭되는 좌표
        :param sizes: tuple, shape=(n_scales, n_ratios, 2), ratio에 scale 이 곱해진 결과값을 반환
            example)
                scales = (10, 100)
                ratios = ((1, 1) (0.5 ,1))
                return = [[[10. 10.], [5.  10.]],
                          [[100. 100.], [50.  100.]]]
    Returns:
        :return: ndarray, (N_center, N_size(=n_scales * n_ratios), 4=(cx cy w h)) 을 반환
    """

    # ((w, h), (w, h) ... (w, h)) 로 구성되어 있음
    sizes = sizes.reshape(-1, 2)

    # center_xy 을 sizes 의 개 수 만큼 중첩해 아래와 같은 shape 로 변경
    # shape: (N_sizes , N_center xy, 2)
    stacked_xy = np.stack([center_xy] * len(sizes), axis=1)

    # sizes 을 center_xy 의 개 수 만큼 중첩해 아래와 같은 shape 로 변경
    # shape (# center_xy, 2)=(N_sizes * N_center xy, 2)
    stacked_wh = np.stack([sizes] * len(center_xy), axis=0)

    # (N_sizes , N_center xy, 4 =(cx cy w h ))를 생성
    return np.concatenate([stacked_xy, stacked_wh], axis=-1)


def generate_default_boxes(scales, ratios):
    """
    Description:
        지정된 크기(scales)와 비율(ratio)에 대한 복수개의 bounding box 을 생성합니다.
    Args:
        :param scales: tuple or list, shape=(n_shape, ), (int, int, ... int )
            example) (3, 6, 9)
            ratio 가 1 일때 default 박스의 size 크기
        :param ratios: tuple or list, ((H_ratio, W_ratio), (H_ratio, W_ratio) ... (H_ratio, W_ratio)) , shape=(n_ratio, 2)
            default boxes의 h, w 정보가 순차적으로 들어있는 자료구조.
            example) ((1, 0.5), (1, 1), ... (0.5, 1))
    Returns:
        :return: tuple, shape = (n_scales, n_ratios, 2) scale 별 ratio 가 적용된 h, w 을  반환
            example)
                scales = (10, 100)
                ratios = ((1, 1) (0.5 ,1))
                return = [[[10. 10.], [5.  10.]],
                          [[100. 100.], [50.  100.]]]
    """
    # shape (n_ratios, 2) -> (n_ratios, 1, 2)
    ratios = np.expand_dims(ratios, axis=1)

    # shape (n_scales) -> (n_scales, 1)
    scales = np.expand_dims(scales, axis=1)

    # shape (n_ratios, n_scales, 2) -> (n_scales, n_ratios, 2)
    scale_per_ratios = np.transpose(ratios * scales, axes=[1, 0, 2])

    return scale_per_ratios


def original_rectangle_coords(fmap_size, kernel_sizes, strides, paddings):
    """
    Description:
        주어진 Feature map의 center x, center y 좌표를 Original Image center x, center y에 맵핑합니다.
        아래 코드에서 사용된 공식은 "A guide to convolution arithmetic for deep learning" 에서 가져옴
    Args:
        :param fmap_size: tuple or list or ndarray, shape (2=(height, width)), 최종 출력된 feature map 의 크기
            example) (4, 4)
        :param kernel_sizes: tuple or list, 각 Layer 에 적용된 filter 크기,
            example) [3, 3]
        :param strides: tuple or list or ndarray, shape (N_layer), [int, int, ... int],
            tuple or list, 각 Layer 에 적용된 stride 크기(pooling layer 포함)
            example) [2, 4, ... 3 ]
        :param paddings: tuple or list, [str, str, ..., str],
            feature map을 생성하기 까지의 적용된 padding,
            paddings 의 element 개 수는 layer 의 깊이와 같아야 합니다.(pooling layer 포함)
            List 의 Element 가 반드시 'SAME' 또는 'VALID' 로 구성되어 있어야 함
            example) ['SAME', 'SAME', ... 'VALID']
    Returns:
        :return: feature map의 각 cell과 original image에 매칭되는 좌표를 반환,
        example)
           feature map
            +---+---+
            | a | b |
            +---+---+
            | c | d |
            +---+---+
                    a               b               c               d
            [[cx, cy, w, h], [cx, cy, w, h] [cx, cy, w, h], [cx, cy, w, h]]
    """

    rf = 1  # receptive field
    jump = 1  # 점과 점사이의 거리
    start_out = 0.5
    assert len(kernel_sizes) == len(strides) == len(paddings), 'kernel sizes, strides, paddings 의 크기가 같아야 합니다.'

    for stride, kernel_size, padding in zip(strides, kernel_sizes, paddings):
        # padding 의 크기를 계산합니다.
        if padding == 'SAME':
            padding = (kernel_size - 1) / 2
        elif padding == 'VALID':
            padding = 0
        else:
            print('padding 값은 SAME 또는 VALID로 구성 되어야 합니다')
            raise ValueError

        # 시작점을 계산합니다.
        start_out += ((kernel_size - 1) * 0.5 - padding) * jump

        # receptive field 을 계산합니다.
        rf += (kernel_size - 1) * jump

        # 점과 점사이의 거리를 계산합니다.
        jump *= stride

    xs, ys = np.meshgrid(range(fmap_size[1]), range(fmap_size[0]))
    xs = xs * jump + start_out
    ys = ys * jump + start_out
    ys = ys.ravel()
    xs = xs.ravel()
    n_samples = len(xs)

    # coords = ((cx, cy, w, h), (cx, cy, w, h) ... (cx, cy, w, h))
    coords = np.stack([xs, ys, [rf] * n_samples, [rf] * n_samples], axis=-1)
    return coords


def generate_tiling_default_boxes(**kwargs):
    """
    Description:
        위 함수는 아래 순서로 작동 합니다.
        1.  모든 feature map cell 을 original image  의 원 좌표(cx, cy, w, h)로변환
            주어진 Feature map의 center x, center y 좌표를 Original Image center x, center y에 맵핑합니다.
            (아래 코드에서 사용된 공식은 "A guide to convolution arithmetic for deep learning" 에서 참조함)
        2.  default boxes 생성
            지정된 크기(scales)와 비율(ratio)에 대한 복수개의 bounding box 을 생성합니다.
        3.  default boxes 을 1번에서 변환된 좌표에 적용
            각 파트의 자세한 설명은 아래 주석에 나와 있습니다.
    :key fmap_size: tuple or list or ndarray, shape (2=(height, width)), 최종 출력된 feature map 의 크기
        example) (4, 4)
    :key paddings: tuple or list, [str, str, ..., str],
        feature map을 생성하기 까지의 적용된 padding,
        paddings 의 element 개 수는 layer 의 깊이와 같아야 합니다.(pooling layer 포함)
        List 의 Element 가 반드시 'SAME' 또는 'VALID' 로 구성되어 있어야 함
        example) ['SAME', 'SAME', ... 'VALID']
    :key strides: tuple or list or ndarray, shape (N_layer), [int, int, ... int]
        tuple or list, 각 Layer 에 적용된 stride 크기(pooling layer 포함)
        example) [2, 4, ... 3 ]
    :key kernel_sizes: tuple or list or ndarray, shape (N_layer), [int, int, ... int]
        feature map을 생성하기 까지의 적용된 kernel_size(pooling layer 포함)
        example) [3, 3, ... 3 ]
    :key scales: tuple or list, shape=(n_shape, ), (int, int, ... int )
        example) (3, 6, 9)
        ratio 가 1 일때 default 박스의 size 크기
    :key ratios: tuple or list, ((H_ratio, W_ratio), (H_ratio, W_ratio) ... (H_ratio, W_ratio)) , shape=(n_ratio, 2)
        default boxes의 h, w 정보가 순차적으로 들어있는 자료구조.
        example) ((1, 0.5), (1, 1), ... (0.5, 1))
    Returns:
        :return: ndarray, (N_center, N_size(=n_scales * n_ratios), 4=(cx cy w h))
    """

    # 주어진 Feature map의 center x, center y 좌표를 Original Image center x, center y에 맵핑합니다.
    fmap_size = kwargs['fmap_size']
    kernel_sizes = kwargs['kernel_sizes']
    strides = kwargs['strides']
    paddings = kwargs['paddings']
    center_xy = original_rectangle_coords(fmap_size, kernel_sizes, strides, paddings)[:, :2]

    # 지정된 크기(scales)와 비율(ratio)에 대한 복수개의 bounding box 을 생성합니다.
    scales = kwargs['scales']
    ratios = kwargs['ratios']
    default_boxes_sizes = generate_default_boxes(scales, ratios)

    # original image 와 좌표 위치가 매칭된 feature map의 모든 cell에 적용된 default boxes의 좌표값을 반환합니다.
    # shape= (f_h * f_w, n_scale* n_ratio, 4) (f=feature_map)
    default_boxes = tiling_default_boxes(center_xy, default_boxes_sizes)
    return default_boxes


def inspect_default_boxes(default_boxes, canvas_size, object_size, iou_threshold, save_path):
    """
    Description:
    obj cx, cy와 canvas에 모든 점에 matching 한 후 실제로 특정 iou threshold 이하인 점을 찾아 반환합니다.
    이  method 을 통해 어느 영역에서 obj 을 찾지 못하는지 파악할 수 있습니다.
    단 obj 가 cropped 된 obj 는 고려하지 않습니다.
    :param: default_boxes: ndarray, (N_fmap, N_anchor, 4=(cx, cy ,w, h))
    :param: canvas_size: tuple, shape=(H, W)
    :param: obj_size: tuple, shape=(H, W)
    :param: iou_threshold: : float, shape=()
    :param: 결과를 저장하는 path
    Returns:
    """
    N_fmap, N_anchor, _ = np.array(default_boxes).shape
    # canvas 내 존재하는 obj가 온전히 위치 할 수 있는 공간
    start_x, start_y = (np.ceil(np.array(object_size) / 2) + 1)[::-1]
    end_x, end_y = (np.array(canvas_size) - np.ceil(np.array(object_size) / 2) + 1)[::-1]

    # canvas 내 obj 가 matching 될 수 있는 center x, center y, width, height 을 찾아 생성
    center_xs, center_ys = np.meshgrid(np.arange(start_x, end_x), np.arange(start_y, end_y))
    center_xy = np.reshape(np.stack([center_xs, center_ys], axis=-1), newshape=(-1, 2))
    center_wh = center_xy.copy()
    center_wh[:, :] = object_size
    center_xywh = np.concatenate([center_xy, center_wh], axis=-1)

    # iou 계산
    ious_bucket = []
    for i in tqdm(range(len(default_boxes))):
        ious = calculate_iou(center_xywh, default_boxes[i])
        ious_bucket.append(ious)

    # shape (N_pixels, N_anchor, N_fmap)
    ious_bucket = np.stack(ious_bucket, axis=-1)
    newshape = list(center_xs.shape) + [-1]
    ious_bucket = np.reshape(ious_bucket, newshape)

    mask = np.any(ious_bucket > iou_threshold, axis=-1).astype(int)
    n_tot = np.prod(mask.shape)
    n_pos = np.sum(mask == 1, axis=None)

    pos_idx = np.where(mask == 1)
    pos_loc = center_xywh.reshape(*mask.shape, -1)[pos_idx]

    print('전체 center 개 수 : {}'.format(n_tot))
    print('positive anchor 개 수 : {}'.format(n_pos))
    print('total anchor 개 수 : {}'.format(n_pos / n_tot))

    # save result
    print('mask shape : {}'.format(mask.shape))
    if save_path:
        f = open(save_path, 'w')
        for str_ in mask:
            for c in str_:
                f.write(str(c))
            f.write('\n')
        f.close()

    return ious_bucket, pos_loc


if __name__ == '__main__':
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    h, w = model.output[1:3]

    fmap_size = (h, w)
    n_layer = 7
    paddings = ['SAME'] * n_layer
    kernel_sizes = [3] * n_layer
    strides = [2, 2, 2, 1, 2, 1, 2]
    scales = [50]
    ratios = [(1, 1), (1.5, 0.5), (1.2, 0.8), (0.8, 1.2), (1.4, 1.4)]

    # Get default boxes over feature map
    default_boxes_ = generate_tiling_default_boxes(fmap_size=(4, 4),
                                                   paddings=paddings,
                                                   strides=strides,
                                                   kernel_sizes=kernel_sizes,
                                                   scales=scales,
                                                   ratios=ratios)

    ious, pos_loc = inspect_default_boxes(default_boxes_, canvas_size=(100, 100), object_size=(50, 50),
                                          iou_threshold=0.96,
                                          save_path='./mask.txt')
