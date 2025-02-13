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

        def vgg(num_classes):

            input = keras.Input(shape=(32, 32, 3))

            x = layers.Conv2D(
                64,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(input)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                64,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.MaxPooling2D((2, 2), strides=2)(x)

            x = layers.Conv2D(
                128,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                128,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.MaxPooling2D((2, 2), strides=2)(x)

            x = layers.Conv2D(
                256,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                256,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                256,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.MaxPooling2D((2, 2), strides=2)(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.MaxPooling2D((2, 2), strides=2)(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(
                512,
                3,
                activation=None,
                padding="same",
                input_shape=(32, 32, 3),
                kernel_initializer="he_uniform",
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.MaxPooling2D((2, 2), strides=2)(x)

            x = layers.concatenate([x] * 7, axis=1)

            x = layers.concatenate([x] * 7, axis=2)

            # x = layers.Lambda(
            #     lambda x: tf.repeat(x, 7, axis=0),
            # )(x)

            # x.set_shape((None, 7, 1, 512))

            # x = layers.Lambda(
            #     lambda x: tf.repeat(x, 7, axis=1),
            # )(x)

            # x.set_shape((None, 7, 7, 512))

            x = layers.Flatten()(x)

            x = layers.Dense(4096, activation="relu")(x)
            x = layers.Dropout(0.5)(x)

            x = layers.Dense(4096, activation="relu")(x)
            x = layers.Dropout(0.5)(x)

            x = layers.Dense(1000)(x)

            x = layers.Dense(
                num_classes,
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                bias_regularizer=tf.keras.regularizers.l2(5e-4),
            )(x)

            model = models.Model(inputs=input, outputs=x)

            return model

        model = vgg(num_classes=self.num_classes)

        opt = tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9,
        )
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(5)],
        )

        return model

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return keras.utils.to_categorical(raw_y_batch, self.num_classes)
