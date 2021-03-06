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
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras

BASE_PATH=""
AUGMENTED_VERTICAL_FLIP_FILE_NAME = BASE_PATH + 'augmented_vertical_flip.csv'
AUGMENTED_HORIZONTAL_FLIP_FILE_NAME = BASE_PATH + 'augmented_horizontal_flip.csv'
AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME = BASE_PATH + 'augmented_gaussian_noise.csv'
TRAIN_DATASET_FILE_NAME = 'sign_mnist_train.csv'
TEST_DATASET_FILE_NAME = 'sign_mnist_test.csv'
UPSCALED_TRAIN_DATASET_FILE_NAME = 'upscaled_sign_mnist_train.csv'
UPSCALED_TEST_DATASET_FILE_NAME = 'upscaled_sign_mnist_test.csv'
CLASSES_COUNT = 25

EPOCHS_NUMBER = 200

# without augmentations 512*8 and split 1 / 6 and divisor 2*TPU_BATCH_SIZE
TPU_BATCH_SIZE = 128*8


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def upscale(dataset_filename, output_dataset_filename):
    if os.path.isfile(output_dataset_filename):
        return

    original_dataset, labels = load_csv(dataset_filename, 28)
    upscaled_dataset = []
    for image in original_dataset:
        upscaled_dataset.append(upscale_image(image).ravel())
    upscaled_dataset = np.array(upscaled_dataset)
    #show_image(augmented_dataset[0].reshape(28, 28))
    upscaled = np.column_stack([labels, upscaled_dataset])
    np.savetxt(output_dataset_filename, upscaled, delimiter=',', fmt='%d')


def upscale_image(image):
    return cv2.resize(image.astype('float32'), (32, 32), interpolation=cv2.INTER_AREA)


def augment_vertical_flip():
    if os.path.isfile(AUGMENTED_VERTICAL_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(UPSCALED_TRAIN_DATASET_FILE_NAME, 32)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(np.flipud(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(32, 32))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_VERTICAL_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')

def augment_horizontal_flip():
    if os.path.isfile(AUGMENTED_HORIZONTAL_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(UPSCALED_TRAIN_DATASET_FILE_NAME, 32)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(np.fliplr(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(32, 32))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_HORIZONTAL_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')


def augment_gaussian_noise():
    if os.path.isfile(AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME):
        return

    original_dataset, labels = load_csv(UPSCALED_TRAIN_DATASET_FILE_NAME, 32)
    augmented_dataset = []
    for image in original_dataset:
        augmented_dataset.append(with_gaussian_noise(image).ravel())
    augmented_dataset = np.array(augmented_dataset)
    #show_image(augmented_dataset[0].reshape(32, 32))
    augmented = np.column_stack([labels, augmented_dataset])
    np.savetxt(AUGMENTED_GUASSIAN_NOISE_FLIP_FILE_NAME, augmented, delimiter=',', fmt='%d')


def with_gaussian_noise(image):
    gaussian = np.random.randint(0, 25, (32, 32))
    return image + gaussian


def build_all_train_dataset(use_vertical_flip=False, use_horizontal_flip=False, use_gaussian_noise=False):
    original_dataset, original_labels = load_dataset(UPSCALED_TRAIN_DATASET_FILE_NAME)
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
    return to_fake_rgb(all_train_dataset), all_train_labels


def load_csv(filename, dim):
    csv = pd.read_csv(filename)
    dataset = csv.iloc[:, 1:].to_numpy()
    labels = csv.iloc[:, 0].to_numpy()
    return np.reshape(dataset, (len(dataset), dim, dim)), labels


def load_dataset(filename):
    csv = pd.read_csv(filename)
    dataset = csv.iloc[:, 1:].to_numpy() / 255
    labels = csv.iloc[:, 0].to_numpy()
    return np.reshape(dataset, (len(dataset), 32, 32, 1)).astype('float32'), np_utils.to_categorical(labels, CLASSES_COUNT)


def build_test_dataset():
    csv = pd.read_csv(UPSCALED_TEST_DATASET_FILE_NAME)
    dataset = csv.iloc[:, 1:].to_numpy() / 255
    labels = csv.iloc[:, 0].to_numpy()
    return to_fake_rgb(np.reshape(dataset, (len(dataset), 32, 32, 1)).astype('float32')), np_utils.to_categorical(labels, CLASSES_COUNT)


def to_fake_rgb(dataset):
    return np.repeat(dataset, 3, -1)


if __name__ == "__main__":
    upscale(TRAIN_DATASET_FILE_NAME, UPSCALED_TRAIN_DATASET_FILE_NAME)
    upscale(TEST_DATASET_FILE_NAME, UPSCALED_TEST_DATASET_FILE_NAME)

    augment_vertical_flip()
    augment_horizontal_flip()
    augment_gaussian_noise()

    all_train_dataset, all_train_labels = build_all_train_dataset(use_horizontal_flip=True)
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(all_train_dataset,
                                                                                          all_train_labels,
                                                                                          test_size=1 / 4, shuffle=True)
    print(train_dataset.shape)
    print(validation_dataset.shape)
    test_dataset, test_labels = build_test_dataset()
    print(test_dataset.shape)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        input_shape=(32, 32, 3)
        vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                          input_shape=input_shape)

        output = vgg.layers[-1].output
        output = Flatten()(output)
        vgg_model = Model(vgg.input, output)
        model = Sequential()
        model.add(vgg_model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(CLASSES_COUNT, activation='softmax'))

        t = TicToc('learning')
        t.tic()
        model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, train_labels, batch_size=TPU_BATCH_SIZE, epochs=EPOCHS_NUMBER)
        t.toc()
        print("Elapsed: %f" % (t.elapsed / 60))
        loss, acc = model.evaluate(validation_dataset, validation_labels)
        print("Accuracy: %s" % acc)
        loss, acc = model.evaluate(test_dataset, test_labels)
        print("Test accuracy: %s" % acc)