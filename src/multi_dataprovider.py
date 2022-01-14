import cv2
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tqdm import tqdm
from utils import xyxy2xywh, plot_images
from label_generator import label_generator
from resnet50_model import resnet50_detection_network, resnet_kernel_info
from default_boxes import generate_tiling_default_boxes


class mnist_dataprovider(Sequence):
    def __init__(self, default_boxes_, input_resize=300, batch_size=64, shuffle=True):

        self.default_boxes_ = default_boxes_
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = 10
        self.input_resize = input_resize

        self.n_sample_train = 10000
        self.n_sample_test = 1000

        (self.train_images, self.train_reg_true, self.train_true), \
        (self.test_images, self.test_reg_true, self.test_true) = \
            self.mnist_localization_generator((120, 120, 3), (120, 120, 3))

    def search_mnist_bbox(self, image):
        """
        Description
            3차원 mnist 데이터가 들어왔을때, 숫자의 bounding box 좌표 정보 (x1, y1, x2, y2)를 찾아 반환
        Args:
            image: (28, 28, 3)

        Returns:
            (x_min, y_min) : (int, int), 이미지 내 mnist 객체의 왼쪽 상단 좌표
            (x_max, y_max) : (int, int), 이미지 내 mnist 객체의 오른쪽 하단 좌표
        """
        axis_0_ind, axis_1_ind = np.where(image[..., 0] > 0)

        y_min = axis_0_ind.min()
        y_max = axis_0_ind.max()
        x_min = axis_1_ind.min()
        x_max = axis_1_ind.max()

        return (x_min, y_min), (x_max, y_max)

    def multi_random_image_fetcher(self, xs, ys, shape=(120, 120, 3), type='train'):
        bg_bucket = []
        coord_bucket = []
        cls_bucket = []

        if type == 'train':
            num = self.n_sample_train
        elif type == 'test':
            num = self.n_sample_test

        for _ in tqdm(range(0, num)):
            n_mnist = int(np.random.uniform(1, 4))

            # split areas into number of n_mnist
            split_areas = np.linspace(0, shape[1], n_mnist + 1)

            # background
            bg = np.zeros(shape)

            # tmp list for coords, cls
            tmp_coords = []
            tmp_cls = []

            for i in range(0, n_mnist):
                if type == 'train':
                    rand_int = np.random.randint(self.n_sample_train)
                elif type == 'test':
                    rand_int = np.random.randint(self.n_sample_test)

                sample_img = xs[rand_int]
                sample_ys = ys[rand_int]

                (obj_x_min, obj_y_min), (obj_x_max, obj_y_max) = self.search_mnist_bbox(sample_img)

                # Generate random coords
                min_x = split_areas[i]
                max_x = split_areas[i + 1] - 28 - 1
                rand_x = np.random.randint(min_x, max_x)
                rand_y = np.random.randint(0, shape[0] - 28 - 1)

                # Patch image to background image
                sample_img = sample_img.astype('uint8')
                bg[rand_y: rand_y + 28, rand_x: rand_x + 28] += sample_img

                # add Offset to mnist object coordinate
                obj_x_min += rand_x
                obj_x_max += rand_x
                obj_y_min += rand_y
                obj_y_max += rand_y

                obj_x_min = obj_x_min * (self.input_resize / shape[0])
                obj_x_max = obj_x_max * (self.input_resize / shape[0])
                obj_y_min = obj_y_min * (self.input_resize / shape[0])
                obj_y_max = obj_y_max * (self.input_resize / shape[0])

                tmp_coords.append(np.array([obj_x_min, obj_y_min, obj_x_max, obj_y_max]))
                tmp_cls.append(sample_ys)

            # constraint pixel value from 0 to 255
            bg = np.clip(bg, 0, 255)

            # resize whole background : mnist 숫자 객체의 크기가 너무 작아서 default box를 제대로 잡지 못하는 이슈때문에
            bg = bg.astype('uint8')
            bg = cv2.resize(bg, (self.input_resize, self.input_resize))

            # append image and coord
            bg_bucket.append(bg)
            coord_bucket.append(np.array(tmp_coords))
            cls_bucket.append(np.array(tmp_cls))

        return np.array(bg_bucket), np.array(coord_bucket), np.array(cls_bucket)

    def mnist_localization_generator(self, train_shape, test_shape):
        """
        Description:
            mnist 숫자 데이터를 (84, 84)의 검정 배경화면에 random한 위치에 오려붙여 주는 함수
        Args:
            train_shape: shape = (h, w, c), 단 channel은 3
            test_shape: shape = (h, w, c), 단 channel은 3

        Returns:
            (train_images, train_true, train_reg_true) :
                train_images : shape = (5000, 84, 84, 3), train을 위한 학습 이미지
                train_true : shape = (5000, ), train 학습 이미지에 대한 label 값
                train_reg_true : shape = (5000, 4), train 학습 이미지에대한 bounding box 값, (cx, cy, w, h)
            (test_images, test_true, test_reg_true)
                test_images : shape = (1000, 84, 84, 3), test를 위한 이미지
                test_true : shape = (1000, ), test 이미지에 대한 label 값
                test_reg_true : shape = (1000, 4), test 학습
        """
        n_sample_train = self.n_sample_train
        n_sample_test = self.n_sample_test

        # mnist image, reg label generator
        (train_xs_, train_ys), (test_xs_, test_ys) = mnist.load_data()

        train_xs = []
        for train_x in train_xs_:
            train_x = np.where(train_x > 10, 255, 0)
            train_x = cv2.merge([train_x, train_x, train_x])
            train_xs.append(train_x)

        test_xs = []
        for test_x in test_xs_:
            test_x = np.where(test_x > 10, 255, 0)
            test_x = cv2.merge([test_x, test_x, test_x])
            test_xs.append(test_x)

        train_images, train_coords, train_true = self.multi_random_image_fetcher(train_xs[:n_sample_train],
                                                                                 train_ys[:n_sample_train], train_shape,
                                                                                 type='train')
        test_images, test_coords, test_true = self.multi_random_image_fetcher(test_xs[:n_sample_test],
                                                                              test_ys[:n_sample_test], test_shape,
                                                                              type='test')

        return (train_images, train_coords, train_true), (test_images, test_coords, test_true)

    def __len__(self):
        """
        Description:
            1 epoch 당 step의 수
        Returns:
        """
        n_steps = np.floor(len(self.train_images) / (self.batch_size)).astype(np.int)

        return n_steps

    def __getitem__(self, idx):

        batch_size = self.batch_size
        slice_ = slice(batch_size * idx, batch_size * (idx + 1))

        sliced_train_xs = self.train_images[slice_]
        sliced_train_ys = self.train_true[slice_]
        sliced_train_coords = self.train_reg_true[slice_]

        batch_xs = []
        batch_ys = []

        for train_x, train_y, train_coord in zip(sliced_train_xs, sliced_train_ys, sliced_train_coords):
            train_coord = xyxy2xywh(train_coord)
            batch_xs.append(train_x)

            # generate labels and calculate delta for each default boxes
            true_delta, true_cls = label_generator(default_bboxes=self.default_boxes_.reshape(-1, 4),
                                                   # gt_bboxes=np.expand_dims(train_coord, axis=0),
                                                   gt_bboxes=train_coord,
                                                   gt_classes=train_y,
                                                   n_classes=self.n_classes + 1
                                                   )

            # change ys to one-hot-encoding
            true_cls_onehot = to_categorical(true_cls, num_classes=self.n_classes + 1)

            # concatenate deltas and one-hot-encoding labels
            true = np.concatenate([true_delta, true_cls_onehot], axis=1)

            # append true(delta + labels) to batch_ys
            batch_ys.append(true)

        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)

        return batch_xs, batch_ys

    def shuffle_(self):
        indexes = np.arange(len(self.train_images))
        np.random.shuffle(indexes)
        self.train_images = self.train_images[indexes]
        self.train_true = self.train_true[indexes]
        self.train_reg_true = self.train_reg_true[indexes]

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_()


if __name__ == '__main__':
    # generate resnet50 model
    inputs, outputs = resnet50_detection_network(input_shape=(300, 300, 3), n_anchors=10, n_classes=10 + 1)
    model = Model(inputs, outputs)

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

    # mnist data provider for SSD
    dataprovider = mnist_dataprovider(default_boxes_=final_boxes_, input_resize=300, batch_size=64, shuffle=True)

    # visualize train datasets
    xs_, ys_ = dataprovider[0]
    pred = model.predict(xs_)
    print("xs_.shape : {}, ys_.shape : {}, pred.shape : {}".format(xs_.shape, ys_.shape, pred.shape))
    plot_images(xs_)
