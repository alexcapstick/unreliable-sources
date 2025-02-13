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

        model = models.Sequential()

        # layer 1
        model.add(
            layers.Dense(
                16,
                activation="relu",
                input_shape=(784,),
            )
        )
        model.add(layers.Dropout(0.2))

        # layer 2
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dropout(0.2))

        # layer 3
        model.add(
            layers.Dense(
                10,
            )
        )

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
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
        return keras.utils.to_categorical(raw_y_batch, 10)
