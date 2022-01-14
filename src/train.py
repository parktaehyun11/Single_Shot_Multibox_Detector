import argparse
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from multi_dataprovider import mnist_dataprovider
import math
import os
import numpy as np
import multiprocessing
from utils import generate_tmp_folder, set_optimizer
from resnet50_model import resnet50_detection_network, resnet_kernel_info
from default_boxes import generate_tiling_default_boxes
from loss import ssd_loss, ssd_loc_loss, ssd_clf_loss
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='config 파일을 불러옵니다.')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--log_dir', type=str, help='log dir')
parser.add_argument('--model_dir', type=str, help='model dir')
parser.add_argument('--opt', type=str, help='optimizer')
parser.add_argument('--n_epochs', type=int, help='epoch 개 수')

args = parser.parse_args()
learning_rate = args.lr
log_folder = args.log_dir
model_folder = args.model_dir
optimizer_name = args.opt
n_epochs = args.n_epochs

# path for saved model and logs
model_folder = generate_tmp_folder(model_folder)
log_folder = generate_tmp_folder(log_folder)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# generate model
inputs, outputs = resnet50_detection_network(input_shape=(300, 300, 3), n_anchors=10, n_classes=10 + 1)
model = Model(inputs, outputs)
model.summary()

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
mnist_dp = mnist_dataprovider(default_boxes_=final_boxes_, input_resize=300, batch_size=64, shuffle=True)

# model compile
print('Learning rate : {}'.format(learning_rate))
print('Selected Optimizer : {}'.format(optimizer_name))
opt = set_optimizer(optmizer_name=optimizer_name, lr=learning_rate)
model.compile(optimizer=opt, loss=ssd_loss, metrics=[ssd_clf_loss, ssd_loc_loss])

xs_, ys_ = mnist_dp[0]
pred_ = model.predict(xs_)
print("xs_ shape : {}".format(xs_.shape))
print("ys_ shape : {}".format(ys_.shape))
print("pred_ shape : {}".format(pred_.shape))


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.0
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)

# 기록될 log
log_path = os.path.join(log_folder, "logs.txt")
f = open(log_path, 'w')
# f.write("{}\t{}\t".format("loss", ""))

best_loss = 100

n_worker = multiprocessing.cpu_count() * 2 + 1
queue_size = mnist_dp.batch_size * (n_worker * 2)
print("# of workers : {}".format(n_worker))
print("# of queue size : {}".format(queue_size))

try:
    for i in range(n_epochs):
        print('Step : {}\n'.format(i))
        hist = model.fit(mnist_dp, epochs=1, max_queue_size=queue_size, workers=n_worker)

        # hist = model.fit(mnist_dp, epochs=1, max_queue_size=queue_size, workers=n_worker, callbacks=[lrate])
        # loss = hist.history['loss'][0]
        # acc = hist.history['acc'][0]
        # lr = hist.history['lr'][0]

        model.save(os.path.join(model_folder, "mnist_ssd_{}.h5".format(i)))

except Exception as e:
    print(e)
    f.close()
f.close()
