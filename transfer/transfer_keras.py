import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from tensorflow.keras.datasets import cifar10
import cv2

num_classes = 10
BATCH_SIZE = 32
IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def _data_preprocessing(x, value_dtype):
    x = x.astype(value_dtype)
    return (x / 127.5) - 1

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def transfer_learning():
    (x_train, _y_train), (x_test, _y_test) = cifar10.load_data()
    x_train = _data_preprocessing(x_train, "float32")
    x_test = _data_preprocessing(x_test, "float32")
    resize_x_train = np.zeros((5000, IMG_SIZE, IMG_SIZE, 3))
    y_train = np.ndarray((5000, 1), dtype=np.uint8)
    resize_x_test = np.zeros((1000, IMG_SIZE, IMG_SIZE, 3))
    y_test = np.ndarray((1000, 1), dtype=np.uint8)
    for a in range(5000):
        resize_x_train[a] = cv2.resize(x_train[a], dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        y_train[a] = _y_train[a]
    for a in range(1000):
        resize_x_test[a] = cv2.resize(x_test[a], dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        y_test[a] = _y_test[a]

    # splits = tfds.Split.TRAIN.subsplit(weighted=(8,2))
    # (raw_train, raw_test), metadata = tfds.load('cifar10', split=list(splits), with_info=True, as_supervised=True)
    # resize_x_train = raw_train.map(format_example)
    # resize_x_test = raw_test.map(format_example)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
                                                   # weights=None)
    print("Number of layers in the base model: ", len(base_model.layers)) #155

    base_model.trainable = False
    # base_model.trainable = True
    # fine_tune_at = 155

    # Freeze all the layers before the `fine_tune_at` layer
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("trainerble variables: ", model.trainable_variables)
    print(model.summary())

    scores = model.evaluate(resize_x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.fit(x=resize_x_train, y=y_train, batch_size=BATCH_SIZE, epochs=10,
              validation_data=(resize_x_test, y_test), shuffle=True)


if __name__ == '__main__':
    transfer_learning()