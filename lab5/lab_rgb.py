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

BASE_PATH=""
PROCESSED_DIRECTORY_PREFIX = BASE_PATH + 'processed_rgb_'
AUGMENTED_VERTICAL_FLIP_DIRECTORY = BASE_PATH + 'augmented_vertical_flip'
AUGMENTED_HORIZONTAL_FLIP_DIRECTORY = BASE_PATH + 'augmented_horizontal_flip'
AUGMENTED_BLUR_FLIP_DIRECTORY = BASE_PATH + 'augmented_blur'
TRAIN_DATASET_NAME = 'train'
TEST_DATASET_NAME = 'test'
CLASSES = [0, 1]

EPOCHS_NUMBER = 500

TPU_BATCH_SIZE = 512*8


def crop_and_resize_images(dataset_name):
    processed_dataset_directory = PROCESSED_DIRECTORY_PREFIX + dataset_name
    if os.path.isdir(processed_dataset_directory):
        return

    os.makedirs(processed_dataset_directory)

    for image_file_name in os.listdir(dataset_name):
        full_file_name = dataset_name + '/' + image_file_name
        image = cv2.imread(full_file_name)
        resized_image = cv2.resize(image, (64, 64))
        cv2.imwrite(processed_dataset_directory + '/' + image_file_name, resized_image)


def augment_vertical_flip():
    augmented_dataset_directory = AUGMENTED_VERTICAL_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    processed_dataset_directory = PROCESSED_DIRECTORY_PREFIX + TRAIN_DATASET_NAME

    for image_file_name in os.listdir(processed_dataset_directory):
        full_file_name = processed_dataset_directory + '/' + image_file_name
        image = cv2.imread(full_file_name)
        flipped = np.flipud(image)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, flipped)


def augment_horizontal_flip():
    augmented_dataset_directory = AUGMENTED_HORIZONTAL_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    processed_dataset_directory = PROCESSED_DIRECTORY_PREFIX + TRAIN_DATASET_NAME

    for image_file_name in os.listdir(processed_dataset_directory):
        full_file_name = processed_dataset_directory + '/' + image_file_name
        image = cv2.imread(full_file_name)
        flipped = np.fliplr(image)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, flipped)


def augment_blur():
    augmented_dataset_directory = AUGMENTED_BLUR_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    processed_dataset_directory = PROCESSED_DIRECTORY_PREFIX + TRAIN_DATASET_NAME

    for image_file_name in os.listdir(processed_dataset_directory):
        full_file_name = processed_dataset_directory + '/' + image_file_name
        image = cv2.imread(full_file_name)
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, blurred)


def build_all_train_dataset(use_vertical_flip=False, use_horizontal_flip=False, use_blur=False):
    original_dataset, original_labels = load_dataset(PROCESSED_DIRECTORY_PREFIX + TRAIN_DATASET_NAME)
    all_train_dataset = original_dataset
    all_train_labels = original_labels
    if use_vertical_flip:
        vertical_flip_dataset, vertical_flip_labels = load_dataset(AUGMENTED_VERTICAL_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, vertical_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, vertical_flip_labels))
        del vertical_flip_dataset
        del vertical_flip_labels
    if use_horizontal_flip:
        horizontal_flip_dataset, horizontal_flip_labels = load_dataset(AUGMENTED_HORIZONTAL_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, horizontal_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, horizontal_flip_labels))
        del horizontal_flip_dataset
        del horizontal_flip_labels
    if use_blur:
        blur_dataset, blur_labels = load_dataset(AUGMENTED_BLUR_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, blur_dataset))
        all_train_labels = np.concatenate((all_train_labels, blur_labels))
        del blur_dataset
        del blur_labels
    print(all_train_dataset.shape)
    print(all_train_labels.shape)
    batch_remainder = all_train_dataset.shape[0] % TPU_BATCH_SIZE
    all_train_dataset = all_train_dataset[:-batch_remainder]
    all_train_labels = all_train_labels[:-batch_remainder]
    return all_train_dataset, all_train_labels


def load_dataset(directory):
    dataset = []
    labels = []
    for image_file_name in os.listdir(directory):
        full_file_name = directory + '/' + image_file_name
        dataset.append(io.imread(full_file_name) / 255)
        if "dog" in image_file_name:
            labels.append(1)
        else:
            labels.append(0)
    print(np.array(dataset).shape)
    return np.reshape(np.array(dataset), (len(dataset), 64, 64, 3)).astype('float32'), np.array(labels)


def build_test_dataset():
    dataset = []
    processed_dataset_directory = 'processed_rgb_' + TEST_DATASET_NAME
    for image_file_name in os.listdir(processed_dataset_directory):
        full_file_name = processed_dataset_directory + '/' + image_file_name
        dataset.append(io.imread(full_file_name) / 255)
    return np.reshape(np.array(dataset), (len(dataset), 64, 64, 3)).astype('float32')


def add_conv_layer(model, kernel_size, dropout_rate):
    model.add(
        keras.layers.Conv2D(kernel_size, (5, 5), padding='same', kernel_initializer='he_normal', activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(padding='same'))
    model.add(keras.layers.Dropout(dropout_rate))


if __name__ == "__main__":
    crop_and_resize_images(TRAIN_DATASET_NAME)
    crop_and_resize_images(TEST_DATASET_NAME)

    augment_vertical_flip()
    augment_horizontal_flip()
    augment_blur()

    all_train_dataset, all_train_labels = build_all_train_dataset(use_blur=True)
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(all_train_dataset,
                                                                                          all_train_labels,
                                                                                          test_size=1 / 6, shuffle=True)
    del all_train_dataset
    del all_train_labels
    print(train_dataset.shape)
    print(train_labels.shape)
    print(validation_dataset.shape)
    print(validation_labels.shape)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(64, 64, 3)))
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
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        t = TicToc('learning')
        t.tic()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, train_labels, batch_size=TPU_BATCH_SIZE, epochs=EPOCHS_NUMBER)
        t.toc()
        print("Elapsed: %f" % (t.elapsed / 60))
        del train_dataset
        del train_labels
        loss, acc = model.evaluate(validation_dataset, validation_labels)
        del validation_dataset
        del validation_labels
        print("Accuracy: %s" % acc)
        test_dataset = build_test_dataset()
        print(test_dataset.shape)
        predicted = model.predict(test_dataset)
        submission = pd.DataFrame({'id': np.array(range(1, len(test_dataset) + 1)), 'label': predicted.ravel()})
        submission.to_csv('submission.csv', index=False)