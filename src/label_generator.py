from delta import calculate_delta
from default_boxes import *
from iou import calculate_iou
import numpy as np


def label_generator(default_bboxes, gt_bboxes, gt_classes, n_classes):
    """
    Desscription:
        이미지 한장에 대해 Detection 을 위한 라벨을 생성합니다.
        라벨 데이터는 값은 default bbox 에 대한 이미지 내 obj 의 상대적 위치 값(delta)로 이루어져 있습니다.
         - 아래 순서로 진행됩니다.
            1. default bboxes 와 ground truth 간의 iou 계산
            2. 학습할 default bbox 선택 (Matching policy)
            3. delta 계산
    Args:
        default_bboxes: ndarray, shape=(N_default_bbox, 4=(cx cy ,w, h))
        gt_bboxes: ndarray, shape=(N_gt, 4=(cx, cy, w, h))
        gt_classes: ndarray, shape=(N_gt), ground truth 에 들어 있는 정답값
            ⚠️ 배경 클래스는 마지막 숫자로 지정되어 있어야 합니다.
            가령 n_classes 가 4 라면 배경 클래스는 3이 되어야 합니다.
    Returns:
        true_delta: ndarray, shape = N_default_boxes, 4=(dx, dy, dw, dh)
        true_cls: ndarray, (N_default_boxes),
            ⚠️️ 단 배경 클래스는 -1 로 표기되어 있음.
    """
    if gt_bboxes.shape[0] == 0:
        ious = np.zeros((default_bboxes.shape[0], 1))
    else:
        # iou 계산
        ious = calculate_iou(default_bboxes, gt_bboxes)

    # iou 중 가장 overlay 비율이 큰 Index 선택합니다.
    # shape = (N_default_boxes, )
    iou_max_index = np.argmax(ious, axis=-1)

    # 모든 obj 에 대해 iou 가 0.6 이하이면 background class, -1로 지정합니다.
    threshold = 0.5
    background_mask = np.all(ious < threshold, axis=-1)
    iou_max_index[background_mask] = -1

    # 기존의 class 에 배경 class 을 추가합니다.
    gt_classes = np.concatenate([gt_classes, np.array([n_classes - 1])])

    # ground truths 의 index을 class 로 변경합니다.
    true_cls = gt_classes[iou_max_index]

    # 기존의 정답 데이터에 [0, 0, 0, 0] 을 추가합니다.
    # cx cy w h
    gt_with_bg = np.concatenate([gt_bboxes, np.array([[0, 0, 0, 0]])], axis=0)

    # 각 default boxes에 해당하는 ground truth 의 좌표값을 가져옵니다.
    true_reg = gt_with_bg[iou_max_index]

    # boolean mask 을 생성합니다.
    pos_mask = (iou_max_index != -1)

    # positive 에 대해 delta 값을 계산합니다.
    pos_true_reg = true_reg[pos_mask]
    pos_default_bboxes = default_bboxes[pos_mask]
    pos_true_delta = calculate_delta(pos_default_bboxes, pos_true_reg)

    # 전체 delta 값에 positive delta 값을 넣어줍니다.
    true_delta = np.zeros_like(default_bboxes)
    true_delta[pos_mask] = pos_true_delta
    return true_delta, true_cls
