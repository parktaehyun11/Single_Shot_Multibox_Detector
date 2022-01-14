import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Softmax, Concatenate
import re
from tensorflow.keras.models import Model


def resnet_kernel_info(model):
    """
    Description: Keras Resnet Model에서 Convolution, Pooling layer의 kernel의 정보를 추출합니다.
    추출할 정보는 'strides', kernel size, padding 정보 입니다.
    Convolution layer 이름는 마지막에 '_conv' 라는 문자열을 가지고 있습니다.
    Pooling layer 이름은 마지막에 '_pool' 라는 문자열을 가지고 있습니다.
    resnet model의 각 layer 이름 패턴은 아래와 같습니다.
    * plain layer : block 내 layer 을 지칭하는 숫자가 1로 변경되어 있습니다.
    example)
        conv2_block1_1_conv
    * short cut layer : block 내 layer을 지칭하는 숫자가 0으로 변경되어 있습니다.
    example)
        conv2_block1_0_conv
    해당 모듈에서는 convolution 과 pooling layer의 *plain kernel 정보를 추출해 반환합니다.
    :return: kernel_sizes, strides, paddings, (list, list, list)
    """

    kernel_sizes = [7, 2]
    strides = [2, 2]
    paddings = ['SAME', 'SAME']
    output_sizes = [model.get_layer('pool1_pad').output, model.get_layer('conv2_block3_add').output]

    # 각 plain conv, pooling layer 별 kernel_size, stride, padding을 가져옵니다.
    # layer 이름 문자열중 앞에 문자열이 conv 이고 뒤에 오는 숫자 2보다 커야 함.
    # resnet 에서 stem 부분에만 pooling 이 있고 이후에는 모두 convolution 으로 구성 되어 있음.
    conv_pattern = re.compile(pattern="^conv[2-9]{1,}.*[1-9]{1,}_conv$")

    for layer in model.layers:

        # resnet block 에서 plain layer 의 convolution layer 정보만 추출함.
        if conv_pattern.match(layer.name):

            # stride 의 계산합니다.
            stride = 1
            input_size = model.get_layer(layer.name).input
            output_size = model.get_layer(layer.name).output

            # resnet 에서 default padding 정책은 'SAME'입니다. 그래서 stride 가 1 일때 input 과 output 의 크기는 같습니다.
            # 이미지가 줄었들었다면 stride 가 1이 아니고 2 입니다.
            if input_size.get_shape()[1] != output_size.get_shape()[1]:
                stride = 2

            # resnet 의 모든 padding 은 same 으로 되어 있습니다.
            padding = 'SAME'

            # kernel size 을 추가 합니다.
            kernel_size = 3

            kernel_sizes.append(kernel_size)
            strides.append(stride)
            paddings.append(padding)
            output_sizes.append(output_size)

    return kernel_sizes, strides, paddings, output_sizes


def resnet50_detection_network(input_shape, n_anchors, n_classes):
    """
    Description:
        tensorflow keras resnet50 classification 모델의 header 을 제거하고 detection model을 할 수 있도록 header을 추가합니다.
        Detection 은 SSD 모델을 구성하기 위해서 multi-header 구조를 가지고 있습니다.
        feature map 의 크기가 입력 이미지의 크기의 2^3배 줄어든 block 의 최종 출력인 c3 layer을 multi header 첫번째 입력으로 사용합니다.
            __________________________________________________________________________________________________
            Layer (type)                    Output Shape         Param #     Connected to
            __________________________________________________________________________________________________
            conv3_block4_out (Activation)   (None, 28, 28, 512)  0           conv3_block4_add[0][0] <-
            __________________________________________________________________________________________________
            conv4_block1_1_conv (Conv2D)    (None, 14, 14, 256)  131328      conv3_block4_out[0][0]
        feature map 의 크기가 입력 이미지의 크기의 2^4배 줄어든 block 의 최종 출력인 c4 layer을 multi header 두번째 입력으로 사용합니다.
        layer name = conv4_block6_out
            __________________________________________________________________________________________________
            Layer (type)                    Output Shape         Param #     Connected to
            __________________________________________________________________________________________________
            conv4_block6_out (Activation)   (None, 14, 14, 1024) 0           conv4_block6_add[0][0] <-
            __________________________________________________________________________________________________
            conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block6_out[0][0]
            __________________________________________________________________________________________________
        feature map 의 크기가 입력 이미지의 크기의 2^5배 줄어든 block 의 최종 출력인 c5 layer을 multi header 세번째 입력으로 사용합니다.
        해당 레이어는 resnet model 50 의 가장 마지막 layer 입니다.
            __________________________________________________________________________________________________
            Layer (type)                    Output Shape         Param #     Connected to
            __________________________________________________________________________________________________
            conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]
                                                                             conv5_block3_3_bn[0][0]
            __________________________________________________________________________________________________
            conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0] <-
            ==================================================================================================
    Args:
        :param input_shape: tuple or list, shape=(3=(h, w, ch))
        :param n_anchors: int
        :param n_classes: int
    :return:
        inputs: keras tensor
        predicts: keras tensor
        (soft3_5, locz3_6): (keras tensor, keras tensor),
        (soft4_5, locz4_6): (keras tensor, keras tensor),
        (soft5_5, locz5_6): (keras tensor, keras tensor)
    """
    # download and load pretraiend resnet50 model
    model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=None,
                                           input_shape=input_shape, pooling=None, classes=1000)
    model.trainable = True

    # extract multi header layers from resnet 50 layer
    c3_layer_name, c4_layer_name, c5_layer_name = "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"
    header_layer_names = [c3_layer_name, c4_layer_name, c5_layer_name]
    header_layers = []
    for header_layer_name in header_layer_names:
        layer = model.get_layer(header_layer_name)
        header_layers.append(layer.output)
    # multi head for C3
    c3_layer = header_layers[0]
    clss3_3 = Conv2D(n_anchors * n_classes, (1, 1), padding='same', activation=None, name='clas3_3')(c3_layer)
    clss3_3 = Reshape((-1, n_classes))(clss3_3)
    soft3_5 = Softmax(axis=-1, name='soft3_5')(clss3_3)

    locz3_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz3_6')(c3_layer)
    locz3_6 = Reshape((-1, 4))(locz3_6)
    head_3 = Concatenate(axis=-1)([locz3_6, soft3_5])

    # multi head for C4
    c4_layer = header_layers[1]
    clss4_3 = Conv2D(n_anchors * n_classes, (1, 1), padding='same', activation=None, name='clas4_3')(c4_layer)
    clss4_3 = Reshape((-1, n_classes))(clss4_3)
    soft4_5 = Softmax(axis=-1, name='soft4_5')(clss4_3)

    locz4_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz4_6')(c4_layer)
    locz4_6 = Reshape((-1, 4))(locz4_6)
    head_4 = Concatenate(axis=-1)([locz4_6, soft4_5])

    # multi head for C5
    c5_layer = header_layers[2]
    clss5_3 = Conv2D(n_anchors * n_classes, (1, 1), padding='same', activation=None, name='clas5_3')(c5_layer)
    clss5_3 = Reshape((-1, n_classes))(clss5_3)
    soft5_5 = Softmax(axis=-1, name='soft5_5')(clss5_3)

    locz5_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz5_6')(c5_layer)
    locz5_6 = Reshape((-1, 4))(locz5_6)
    head_5 = Concatenate(axis=-1)([locz5_6, soft5_5])

    inputs = model.input
    predict = Concatenate(axis=1)([head_3, head_4, head_5])

    return inputs, predict


if __name__ == '__main__':
    # load moel
    inputs_, outputs_ = resnet50_detection_network(input_shape=(None, None, 3), n_anchors=10, n_classes=10 + 1)
    model = Model(inputs_, outputs_)

    # model sumamry
    model.summary()

    # extract model imformation
    kernel_sizes, strides, paddings, output_sizes = resnet_kernel_info(model)
