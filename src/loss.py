from tensorflow.keras.losses import MSE, CategoricalCrossentropy
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from resnet50_model import resnet50_detection_network, resnet_kernel_info


def detection_loss(y_true, y_pred):
    """
    Description:
        아래 코드는 keras model.fit 에서 작동하기 위해 설계 및 구현되어 있습니다.
        Tensorflow Graph 모드에서 작동하는것을 기반으로 생각하고 있습니다.
        일번적인 detection model 의 loss 을 구합니다.
        구해야 할 loss는 아래와 같습니다.
        loss = Classification loss + Regression loss
        Classification loss:
            모든 데이터에 대해 classification loss 을 계산합니다.
        Regression loss:
            iou 가 50 이상인 positive 데이터 셋에 대해서만 loss 을 구합니다.
    :param y_true: (N data, N default_boxes, 4 + 1)
    :param y_pred: (N data, N default_boxes, 4 + n_classes(with background))
    (※ docs class 은 -1 로 지정되어 있음)
    """

    # classification error
    true_reg = y_true[..., :4]
    true_cls = y_true[..., 4:]
    pred_reg = y_pred[..., :4]
    pred_cls = y_pred[..., 4:]

    # get positive, negative mask
    # shape = (N_img, N_default_boxes)
    pos_mask = (true_cls[..., -1] != 1)
    neg_mask = (true_cls[..., -1] == 1)

    # get positive negative index for tensor
    # shape = (N_pos, 2=(axis0, axis=1)) or (N_neg, 2=(axis0, axis=1))
    pos_index_tf = tf.where(pos_mask)
    neg_index_tf = tf.where(neg_mask)

    # Extract positive dataset
    pos_true_cls = tf.gather_nd(true_cls, pos_index_tf)
    pos_pred_cls = tf.gather_nd(pred_cls, pos_index_tf)
    neg_true_cls = tf.gather_nd(true_cls, neg_index_tf)
    neg_pred_cls = tf.gather_nd(pred_cls, neg_index_tf)

    # Negative 데이터을 positive 3배 비율로 추출합니다.
    n_pos = len(pos_index_tf)
    n_neg = len(neg_index_tf)
    neg_rand_index = tf.range(n_neg)
    neg_rand_index = tf.random.shuffle(neg_rand_index)
    neg_true_cls = tf.gather(neg_true_cls, neg_rand_index[:n_pos * 3])
    neg_pred_cls = tf.gather(neg_pred_cls, neg_rand_index[:n_pos * 3])

    #
    trgt_pred_cls = tf.concat([neg_pred_cls, pos_pred_cls], axis=0)
    trgt_true_cls = tf.concat([neg_true_cls, pos_true_cls], axis=0)

    # Classification loss
    cls_loss = CategoricalCrossentropy()(trgt_true_cls, trgt_pred_cls)

    # extract positive localization
    pos_true_reg = tf.gather_nd(true_reg, pos_index_tf)
    pos_pred_reg = tf.gather_nd(pred_reg, pos_index_tf)

    # Regression loss
    reg_loss = tf.reduce_mean(MSE(y_true=pos_true_reg, y_pred=pos_pred_reg))

    loss = cls_loss + reg_loss
    return loss


def detection_classification_loss(y_true, y_pred):
    """
    Description:
        아래 코드는 keras model.fit 에서 작동하기 위해 설계 및 구현되어 있습니다.
        Tensorflow Graph 모드에서 작동하는것을 기반으로 생각하고 있습니다.
        일번적인 detection model 의 loss 을 구합니다.
        구해야 할 loss는 아래와 같습니다.
        loss = Classification loss + Regression loss
        Classification loss:
            모든 데이터에 대해 classification loss 을 계산합니다.
        Regression loss:
            iou 가 50 이상인 positive 데이터 셋에 대해서만 loss 을 구합니다.
    :param y_true: (N data, N default_boxes, 4 + 1)
    :param y_pred: (N data, N default_boxes, 4 + n_classes)
    (※ docs class 은 -1 로 지정되어 있음)
    """

    # classification error
    true_cls = y_true[..., 4:]
    pred_cls = y_pred[..., 4:]

    # get positive, negative mask
    # shape = (N_img, N_default_boxes)
    pos_mask = (true_cls[..., -1] != 1)
    neg_mask = (true_cls[..., -1] == 1)

    # get positive negative index for tensor
    # shape = (N_pos, 2=(axis0, axis=1)) or (N_neg, 2=(axis0, axis=1))
    pos_index_tf = tf.where(pos_mask)
    neg_index_tf = tf.where(neg_mask)

    # Extract positive dataset
    pos_true_cls = tf.gather_nd(true_cls, pos_index_tf)
    pos_pred_cls = tf.gather_nd(pred_cls, pos_index_tf)
    neg_true_cls = tf.gather_nd(true_cls, neg_index_tf)
    neg_pred_cls = tf.gather_nd(pred_cls, neg_index_tf)

    # Negative 데이터을 positive 3배 비율로 추출합니다.
    n_pos = len(pos_index_tf)
    n_neg = len(neg_index_tf)
    neg_rand_index = tf.range(n_neg)
    neg_rand_index = tf.random.shuffle(neg_rand_index)
    neg_true_cls = tf.gather(neg_true_cls, neg_rand_index[:n_pos * 3])
    neg_pred_cls = tf.gather(neg_pred_cls, neg_rand_index[:n_pos * 3])
    #
    trgt_pred_cls = tf.concat([neg_pred_cls, pos_pred_cls], axis=0)
    trgt_true_cls = tf.concat([neg_true_cls, pos_true_cls], axis=0)

    # Classification loss
    cls_loss = CategoricalCrossentropy()(trgt_true_cls, trgt_pred_cls)
    return cls_loss


def detection_localization_loss(y_true, y_pred):
    """
    Description:
        아래 코드는 keras model.fit 에서 작동하기 위해 설계 및 구현되어 있습니다.
        Tensorflow Graph 모드에서 작동하는것을 기반으로 생각하고 있습니다.
        일번적인 detection model 의 loss 을 구합니다.
        구해야 할 loss는 아래와 같습니다.
        loss = Classification loss + Regression loss
        Classification loss:
            모든 데이터에 대해 classification loss 을 계산합니다.
        Regression loss:
            iou 가 50 이상인 positive 데이터 셋에 대해서만 loss 을 구합니다.
    :param y_true: (N data, N default_boxes, 4 + 1)
    :param y_pred: (N data, N default_boxes, 4 + n_classes)
    (※ docs class 은 -1 로 지정되어 있음)
    """

    # classification error
    true_reg = y_true[..., :4]
    true_cls = y_true[..., 4:]
    pred_reg = y_pred[..., :4]
    pred_cls = y_pred[..., 4:]

    # get positive, negative mask
    # shape = (N_img, N_default_boxes)
    pos_mask = (true_cls[..., -1] != 1)
    neg_mask = (true_cls[..., -1] == 1)

    # get positive negative index for tensor
    # shape = (N_pos, 2=(axis0, axis=1)) or (N_neg, 2=(axis0, axis=1))
    pos_index_tf = tf.where(pos_mask)

    # extract positive localization
    pos_true_reg = tf.gather_nd(true_reg, pos_index_tf)
    pos_pred_reg = tf.gather_nd(pred_reg, pos_index_tf)

    # Regression loss
    reg_loss = tf.reduce_mean(MSE(y_true=pos_true_reg, y_pred=pos_pred_reg))

    loss = reg_loss
    return loss


def ssd_loss(y_true, y_pred):
    pos_neg_ratio = 3.
    num_classes = tf.shape(y_true)[2] - 4
    y_true = tf.reshape(y_true, [-1, num_classes + 4])
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.reshape(y_pred, [-1, num_classes + 4])
    y_pred = tf.cast(y_pred, tf.float64)
    eps = K.epsilon()

    # Split Classification and Localization output
    y_true_loc, y_true_clf = tf.split(y_true, [4, num_classes], axis=-1)
    y_pred_loc, y_pred_clf = tf.split(y_pred, [4, num_classes], axis=-1)

    # split foreground & background
    neg_mask = y_true_clf[:, -1]
    pos_mask = 1 - neg_mask
    num_pos = tf.reduce_sum(pos_mask)
    num_neg = tf.reduce_sum(neg_mask)
    num_neg = tf.minimum(pos_neg_ratio * num_pos, num_neg)

    # softmax loss
    y_pred_clf = K.clip(y_pred_clf, eps, 1. - eps)
    clf_loss = -tf.reduce_sum(y_true_clf * tf.math.log(y_pred_clf), axis=-1)
    pos_clf_loss = tf.reduce_sum(clf_loss * pos_mask) / (num_pos + eps)
    neg_clf_loss = clf_loss * neg_mask
    values, indices = tf.nn.top_k(neg_clf_loss, k=tf.cast(num_neg, tf.int32))
    neg_clf_loss = tf.reduce_sum(values) / (num_neg + eps)
    clf_loss = pos_clf_loss + neg_clf_loss

    # smooth l1 loss
    l1_loss = tf.abs(y_true_loc - y_pred_loc)
    l2_loss = 0.5 * (y_true_loc - y_pred_loc) ** 2
    loc_loss = tf.where(tf.less(l1_loss, 1.0),
                        l2_loss,
                        l1_loss - 0.5)
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = tf.reduce_sum(loc_loss * pos_mask) / (num_pos + eps)

    return clf_loss + loc_loss


def ssd_clf_loss(y_true, y_pred):
    pos_neg_ratio = 3.
    num_classes = tf.shape(y_true)[2] - 4
    y_true = tf.reshape(y_true, [-1, num_classes + 4])
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.reshape(y_pred, [-1, num_classes + 4])
    y_pred = tf.cast(y_pred, tf.float64)
    eps = K.epsilon()

    # Split Classification and Localization output
    y_true_loc, y_true_clf = tf.split(y_true, [4, num_classes], axis=-1)
    y_pred_loc, y_pred_clf = tf.split(y_pred, [4, num_classes], axis=-1)

    # split foreground & background
    neg_mask = y_true_clf[:, -1]
    pos_mask = 1 - neg_mask
    num_pos = tf.reduce_sum(pos_mask)
    num_neg = tf.reduce_sum(neg_mask)
    num_neg = tf.minimum(pos_neg_ratio * num_pos, num_neg)

    # softmax loss
    y_pred_clf = K.clip(y_pred_clf, eps, 1. - eps)
    clf_loss = -tf.reduce_sum(y_true_clf * tf.math.log(y_pred_clf), axis=-1)
    pos_clf_loss = tf.reduce_sum(clf_loss * pos_mask) / (num_pos + eps)
    neg_clf_loss = clf_loss * neg_mask
    values, indices = tf.nn.top_k(neg_clf_loss, k=tf.cast(num_neg, tf.int32))
    neg_clf_loss = tf.reduce_sum(values) / (num_neg + eps)
    clf_loss = pos_clf_loss + neg_clf_loss

    return clf_loss


def ssd_loc_loss(y_true, y_pred):
    pos_neg_ratio = 3.
    num_classes = tf.shape(y_true)[2] - 4
    y_true = tf.reshape(y_true, [-1, num_classes + 4])
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.reshape(y_pred, [-1, num_classes + 4])
    y_pred = tf.cast(y_pred, tf.float64)
    eps = K.epsilon()

    # Split Classification and Localization output
    y_true_loc, y_true_clf = tf.split(y_true, [4, num_classes], axis=-1)
    y_pred_loc, y_pred_clf = tf.split(y_pred, [4, num_classes], axis=-1)

    # split foreground & background
    neg_mask = y_true_clf[:, -1]
    pos_mask = 1 - neg_mask
    num_pos = tf.reduce_sum(pos_mask)
    num_neg = tf.reduce_sum(neg_mask)
    num_neg = tf.minimum(pos_neg_ratio * num_pos, num_neg)

    # smooth l1 loss
    l1_loss = tf.abs(y_true_loc - y_pred_loc)
    l2_loss = 0.5 * (y_true_loc - y_pred_loc) ** 2
    loc_loss = tf.where(tf.less(l1_loss, 1.0),
                        l2_loss,
                        l1_loss - 0.5)
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = tf.reduce_sum(loc_loss * pos_mask) / (num_pos + eps)

    return loc_loss


if __name__ == '__main__':
    # load model
    inputs_, outputs_ = resnet50_detection_network(input_shape=(None, None, 3), n_anchors=10, n_classes=10 + 1)
    model = Model(inputs_, outputs_)

    # compile model
    model.compile('adam', loss=ssd_loss, metrics=[ssd_clf_loss, ssd_loc_loss])
