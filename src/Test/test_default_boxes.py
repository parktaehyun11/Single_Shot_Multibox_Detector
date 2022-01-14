from unittest import TestCase
import numpy as np
# from dataprovider import mnist_dataprovider
from multi_dataprovider import mnist_dataprovider
from default_boxes import generate_tiling_default_boxes
from tensorflow.keras.models import Model
from tqdm import tqdm
from delta import calculate_gt
from resnet50_model import resnet50_detection_network, resnet_kernel_info
from utils import xyxy2xywh, xywh2xyxy, draw_rectangles, plot_images, show_image


class TestDefualtBoxes(TestCase):
    def setUp(self):
        self.n_classes = 10 + 1

        # generate SSD detection model
        inputs, outputs = resnet50_detection_network(input_shape=(300, 300, 3), n_anchors=10, n_classes=10 + 1)

        model = Model(inputs, outputs)
        # print(model.summary())
        # extract kernel sizes, strides, paddings, output sizes of resnet50 model
        kernel_sizes, strides, paddings, output_sizes = resnet_kernel_info(model)

        # default boxes bucket
        default_boxes_bucket = []

        # c3, c4, c5 시작 index
        c3_index = 11
        c4_index = 23
        c5_index = 41

        # (w_ratio, h_ratio)
        ratios = [(1.3, 0.7), (1.2, 0.8), (1.1, 0.9),
                  (1, 1),
                  (0.4, 1), (0.3, 1), (0.2, 1),
                  (0.9, 1.1), (0.8, 1.2), (0.7, 1.3)]

        block_indices = [c3_index, c4_index, c5_index]
        block_scales = [[50], [53], [56]]
        block_fmaps = [output_sizes[c3_index].get_shape()[1:3],
                       output_sizes[c4_index].get_shape()[1:3],
                       output_sizes[c5_index].get_shape()[1:3]]

        # Generate default boxes
        for c_index, scales, fmap in tqdm(list(zip(block_indices, block_scales, block_fmaps))[:]):
            default_boxes_ = generate_tiling_default_boxes(fmap_size=fmap,
                                                           paddings=paddings[:c_index + 1],
                                                           strides=strides[:c_index + 1],
                                                           kernel_sizes=kernel_sizes[:c_index + 1],
                                                           scales=scales,
                                                           ratios=ratios)
            default_boxes = default_boxes_.reshape(-1, 4)
            default_boxes_bucket.append(default_boxes)
        final_boxes_ = np.concatenate(default_boxes_bucket, axis=0)

        # print(final_boxes_.shape)
        self.default_boxes_ = final_boxes_

        self.mnist_dp = mnist_dataprovider(default_boxes_=final_boxes_, input_resize=300, batch_size=64, shuffle=True)

    def test_default_boxes_localization(self):
        """
        Test Description:
            default boxes 제대로 지정되어 있는지 확인합니다.
        Returns:
        """

        for n_batch in range(0, len(self.mnist_dp)):
            xs, ys = self.mnist_dp[n_batch]
            drawed_xs = []
            for i in range(self.mnist_dp.batch_size):
                sample_x = xs[i]
                sample_y = ys[i]

                sample_onehot = sample_y[..., 4:]
                sample_cls = np.argmax(sample_onehot, axis=-1)
                sample_pos_index = (sample_cls != (self.n_classes - 1))

                sample_pos_boxes = self.default_boxes_[sample_pos_index]

                sample_pos_boxes = xywh2xyxy(sample_pos_boxes)
                drawed_x = draw_rectangles(sample_x, sample_pos_boxes)
                drawed_xs.append(drawed_x)
            plot_images(drawed_xs)

    def test_ground_truth_boxes(self):
        """
        Test Description:
            ground truth box의 좌표를 복원해서, bbox가 제대로 설정되었지는 확인합니다.
        Returns:
        """
        for n_batch in range(0, len(self.mnist_dp)):
            xs, ys = self.mnist_dp[n_batch]
            drawed_xs = []
            for i in range(self.mnist_dp.batch_size):
                sample_x = xs[i]
                sample_y = ys[i]

                sample_delta = sample_y[..., :4]
                sample_onehot = sample_y[..., 4:]
                sample_cls = np.argmax(sample_onehot, axis=-1)
                sample_pos_index = (sample_cls != (self.n_classes - 1))

                sample_gt = calculate_gt(self.default_boxes_, sample_delta)
                pos_gt_hat = sample_gt[sample_pos_index]

                pos_gt_hat = xywh2xyxy(pos_gt_hat)

                drawed_x = draw_rectangles(sample_x, pos_gt_hat)
                drawed_xs.append(drawed_x)
            plot_images(drawed_xs)