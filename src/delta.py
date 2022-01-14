import numpy as np


def calculate_delta(default_boxes, ground_truth):
    """
    Description:
    default boxes 에 대한 ground truth 의 상대적인 위치 좌표를 얻어 옵니다.
    상대적 좌표를 얻어 오는 방식은 아래와 같습니다.
    delta_x = (cx_gt - cx_default) / w_default
    delta_y = (cy_gt - cy_default) / h_default
    log(delta_w) = log(w_gt) -np.log(w_default) = log(w_gt/w_default)
    log(delta_h) = log(h_gt) -np.log(h_default) = log(h_gt/h_default)
    :param default_boxes: ndarray, (N_default_boxes, 4=(x y w h))
    :param ground_truth: ndarray, (N_default_boxes, 4=(x y w h))
    :return: delta, ndarray, (N_default_boxes, 4=(Δx Δy Δw Δh))
    """

    # coordinates -> delta
    dx = (ground_truth[:, 0] - default_boxes[:, 0]) / default_boxes[:, 2]
    dy = (ground_truth[:, 1] - default_boxes[:, 1]) / default_boxes[:, 3]
    dw = np.log(ground_truth[:, 2] / default_boxes[:, 2])
    dh = np.log(ground_truth[:, 3] / default_boxes[:, 3])
    delta = np.stack([dx, dy, dw, dh], axis=-1)
    return delta


def calculate_gt(default_boxes, delta_hat):
    """
    Description:
    주어진 delta와 default_boxes을 활영하여 ground truth bbox 의 좌표를 복원합니다.
    복원 공식은 아래와 같습니다.
    cx_gt = delta_x * w_default + cx_default
    cy_gt = delta_y * h_default + cy_default
    w_gt = exp(log(delta_w) + np.log(w_default)) = exp(log(delta_w * w_default))
    h_gt = exp(log(delta_h) + np.log(h_default)) = exp(log(delta_h * h_default))
    :param default_boxes: ndarray, (N_default_boxes, 4=(x y w h))
    :param delta_hat: ndarray, (N_default_boxes, 4=(Δx̂ Δŷ Δŵ Δĥ)), 모델을 통해 예측된 상대적 객체 위치 및 크기
    :return: gt_hat, ndarray, (N_default_boxes, 4=(x̂ ŷ ŵ ĥ))
    """
    x_hat = (delta_hat[..., 0] * default_boxes[..., 2]) + default_boxes[..., 0]
    y_hat = (delta_hat[..., 1] * default_boxes[..., 3]) + default_boxes[..., 1]
    w_hat = np.exp(np.log(default_boxes[..., 2]) + delta_hat[..., 2])
    h_hat = np.exp(np.log(default_boxes[..., 3]) + delta_hat[..., 3])
    gt_hat = np.stack([x_hat, y_hat, w_hat, h_hat], axis=-1)
    return gt_hat
