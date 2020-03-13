import os
import os.path

import cv2
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from ttictoc import TicToc
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt

BASE_PATH=""
AUGMENTED_VERTICAL_FLIP_FILE_NAME = BASE_PATH + 'augmented_vertical_flip.csv'
AUGMENTED_HORIZONTAL_FLIP_FILE_NAME = BASE_PATH + 'augmented_horizontal_flip.csv'
AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME = BASE_PATH + 'augmented_gaussian_noise.csv'
TRAIN_DATASET_FILE_NAME = 'sign_mnist_train.csv'
TEST_DATASET_FILE_NAME = 'sign_mnist_test.csv'
CLASSES_COUNT = 25

EPOCHS_NUMBER = 200

# without augmentations 512*8 and split 1 / 6 and divisor TPU_BATCH_SIZE
TPU_BATCH_SIZE = 128*8


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def augment_vertical_flip():
    if os.path.isfile(AUGMENTED_VERTICAL_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(TRAIN_DATASET_FILE_NAME)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(np.flipud(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(28, 28))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_VERTICAL_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')

def augment_horizontal_flip():
    if os.path.isfile(AUGMENTED_HORIZONTAL_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(TRAIN_DATASET_FILE_NAME)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(np.fliplr(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(28, 28))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_HORIZONTAL_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')


def augment_gaussian_noise():
    if os.path.isfile(AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(TRAIN_DATASET_FILE_NAME)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(with_gaussian_noise(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(28, 28))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')


def with_gaussian_noise(image):
    gaussian = np.random.randint(0, 25, (28, 28))
    return image + gaussian


def build_all_train_dataset(use_vertical_flip=False, use_horizontal_flip=False, use_gaussian_noise=False):
    original_dataset, original_labels = load_dataset(TRAIN_DATASET_FILE_NAME)
    all_train_dataset = original_dataset
    all_train_labels = original_labels
    if use_vertical_flip:
        vertical_flip_dataset, vertical_flip_labels = load_dataset(AUGMENTED_VERTICAL_FLIP_FILE_NAME)
        all_train_dataset = np.concatenate((all_train_dataset, vertical_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, vertical_flip_labels))
    if use_horizontal_flip:
        horizontal_flip_dataset, horizontal_flip_labels = load_dataset(AUGMENTED_HORIZONTAL_FLIP_FILE_NAME)
        all_train_dataset = np.concatenate((all_train_dataset, horizontal_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, horizontal_flip_labels))
    if use_gaussian_noise:
        gaussian_noise_dataset, gaussian_noise_labels = load_dataset(AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME)
        all_train_dataset = np.concatenate((all_train_dataset, gaussian_noise_dataset))
        all_train_labels = np.concatenate((all_train_labels, gaussian_noise_labels))
    print(all_train_dataset.shape)
    print(all_train_labels.shape)
    batch_remainder = all_train_dataset.shape[0] % (2*TPU_BATCH_SIZE)
    all_train_dataset = all_train_dataset[:-batch_remainder]
    all_train_labels = all_train_labels[:-batch_remainder]
    return all_train_dataset, all_train_labels


def load_csv(filename):
    csv = pd.read_csv(filename)
    dataset = csv.iloc[:, 1:].to_numpy()
    labels = csv.iloc[:, 0].to_numpy()
    return np.reshape(dataset, (len(dataset), 28, 28)), labels


def load_dataset(filename):
    csv = pd.read_csv(filename)
    dataset = csv.iloc[:, 1:].to_numpy() / 255
    labels = csv.iloc[:, 0].to_numpy()
    return np.reshape(dataset, (len(dataset), 28, 28, 1)).astype('float32'), np_utils.to_categorical(labels, CLASSES_COUNT)


def build_test_dataset():
    csv = pd.read_csv(TEST_DATASET_FILE_NAME)
    dataset = csv.iloc[:, 1:].to_numpy() / 255
    labels = csv.iloc[:, 0].to_numpy()
    return np.reshape(dataset, (len(dataset), 28, 28, 1)).astype('float32'), np_utils.to_categorical(labels, CLASSES_COUNT)


def add_conv_layer(model, kernel_size, dropout_rate):
    model.add(keras.layers.Conv2D(kernel_size, (5, 5), padding='same', kernel_initializer='he_normal', activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(padding='same'))
    model.add(keras.layers.Dropout(dropout_rate))


if __name__ == "__main__":
    augment_vertical_flip()
    augment_horizontal_flip()
    augment_gaussian_noise()

    all_train_dataset, all_train_labels = build_all_train_dataset(use_vertical_flip=True)
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(all_train_dataset,
                                                                                          all_train_labels,
                                                                                          test_size=1 / 4, shuffle=True)
    print(train_dataset.shape)
    print(validation_dataset.shape)
    test_dataset, test_labels = build_test_dataset()
    print(test_dataset.shape)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + '')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(28, 28, 1)))
        add_conv_layer(model, 48, 0.2)
        add_conv_layer(model, 64, 0.25)
        add_conv_layer(model, 128, 0.3)
        add_conv_layer(model, 160, 0.4)
        add_conv_layer(model, 192, 0.5)
        add_conv_layer(model, 192, 0.5)
        add_conv_layer(model, 192, 0.5)
        add_conv_layer(model, 192, 0.5)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, kernel_initializer='he_normal', activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(512, kernel_initializer='he_normal', activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(CLASSES_COUNT, activation='softmax'))

        t = TicToc('learning')
        t.tic()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, train_labels, batch_size=TPU_BATCH_SIZE, epochs=EPOCHS_NUMBER)
        t.toc()
        print("Elapsed: %f" % (t.elapsed / 60))
        loss, acc = model.evaluate(validation_dataset, validation_labels)
        print("Validation accuracy: %s" % acc)
        loss, acc = model.evaluate(test_dataset, test_labels)
        print("Test accuracy: %s" % acc)