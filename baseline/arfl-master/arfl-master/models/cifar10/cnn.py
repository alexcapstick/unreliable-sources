import numpy as np
import ray
import sys
from tensorflow import keras

sys.path.append("..")
from model import FedModel
from utils.args import GPU_PER_ACTOR


@ray.remote(num_gpus=GPU_PER_ACTOR)
class ClientModel(FedModel):
    def __init__(self, seed, lr, num_classes, test_batch_size):
        FedModel.__init__(self, seed, lr, test_batch_size)
        self.num_classes = num_classes
        self.net = self.create_model()

    def create_model(self, lr=None):
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        from tensorflow.keras import layers, models
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import AveragePooling2D
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        def conv2d_bn(x, filters, kernel_size, weight_decay=0.0, strides=(1, 1)):
            layer = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
            )(x)
            layer = BatchNormalization()(layer)
            return layer

        def conv2d_bn_relu(x, filters, kernel_size, weight_decay=0.0, strides=(1, 1)):
            layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
            layer = Activation("relu")(layer)
            return layer

        def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
            if downsample:
                # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
                residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
                stride = 2
            else:
                residual_x = x
                stride = 1
            residual = conv2d_bn_relu(
                x,
                filters=filters,
                kernel_size=kernel_size,
                weight_decay=weight_decay,
                strides=stride,
            )
            residual = conv2d_bn(
                residual,
                filters=filters,
                kernel_size=kernel_size,
                weight_decay=weight_decay,
                strides=1,
            )
            out = layers.add([residual_x, residual])
            out = Activation("relu")(out)
            return out

        def ResNetForCIFAR10(
            classes, name, input_shape, block_layers_num, weight_decay
        ):
            input = Input(shape=input_shape)
            x = input
            x = conv2d_bn_relu(
                x,
                filters=16,
                kernel_size=(3, 3),
                weight_decay=weight_decay,
                strides=(1, 1),
            )

            # # conv 2
            for i in range(block_layers_num):
                x = ResidualBlock(
                    x,
                    filters=16,
                    kernel_size=(3, 3),
                    weight_decay=weight_decay,
                    downsample=False,
                )
            # # conv 3
            x = ResidualBlock(
                x,
                filters=32,
                kernel_size=(3, 3),
                weight_decay=weight_decay,
                downsample=True,
            )
            for i in range(block_layers_num - 1):
                x = ResidualBlock(
                    x,
                    filters=32,
                    kernel_size=(3, 3),
                    weight_decay=weight_decay,
                    downsample=False,
                )
            # # conv 4
            x = ResidualBlock(
                x,
                filters=64,
                kernel_size=(3, 3),
                weight_decay=weight_decay,
                downsample=True,
            )
            for i in range(block_layers_num - 1):
                x = ResidualBlock(
                    x,
                    filters=64,
                    kernel_size=(3, 3),
                    weight_decay=weight_decay,
                    downsample=False,
                )
            x = AveragePooling2D(pool_size=(8, 8), padding="valid")(x)
            x = Flatten()(x)
            x = Dense(classes, activation="softmax")(x)
            model = Model(input, x, name=name)
            return model

        def ResNet20ForCIFAR10(classes, input_shape, weight_decay):
            return ResNetForCIFAR10(classes, "resnet20", input_shape, 3, weight_decay)

        weight_decay = 1e-4
        num_classes = 100
        model = ResNet20ForCIFAR10(
            input_shape=(32, 32, 3), classes=num_classes, weight_decay=weight_decay
        )

        opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(5)],
        )
        print(model.summary())

        return model

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return keras.utils.to_categorical(raw_y_batch, 100)
