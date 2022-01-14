from iou import calculate_iou
import numpy as np
from utils import xyxy2xywh


def nms(onehots, bboxes, threshold):
    labels = np.argmax(onehots, axis=-1)
    probs = np.max(onehots, axis=-1)

    sorted_index = np.argsort(probs)[::-1]

    bboxes = bboxes[sorted_index]
    labels = labels[sorted_index]

    bboxes = xyxy2xywh(bboxes)

    final_bboxes = []
    final_labels = []
    final_onehots = []

    # bounding boxes에 아무것도 없을때까지 수행
    while bboxes.tolist():
        # prediction 값이 가장 높은 bounding box 후보군을 선택하고 final bboxes 집어 넣음.
        trgt_bbox = bboxes[0]
        final_bboxes.append(trgt_bbox)
        trgt_labels = labels[0]
        final_labels.append(trgt_labels)
        trgt_onehots = onehots[0]
        final_onehots.append(trgt_onehots)

        # 후보 bounding box을 bboxes 에서 제거함
        bboxes = np.delete(bboxes, 0, axis=0)
        labels = np.delete(labels, 0, axis=0)
        onehots = np.delete(onehots, 0, axis=0)

        # 후보 bbox 와 bboxes 와의 iou 을 계산함.
        ious = calculate_iou(trgt_bbox[None], bboxes)

        # 후보 bbox 와 bboxes 와의 iou 을 계산해 특정 threshold 이상 겹치는 bbox 는 bboxes 후보에서 제거
        overlay_index = np.where(np.squeeze(ious > threshold))
        bboxes = np.delete(bboxes, overlay_index, axis=0)
        onehots = np.delete(onehots, overlay_index, axis=0)
        labels = np.delete(labels, overlay_index, axis=0)

    return final_bboxes, final_labels, final_onehots
