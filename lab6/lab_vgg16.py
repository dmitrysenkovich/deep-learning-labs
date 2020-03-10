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
from keras.applications import vgg16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential, Model
import keras

BASE_PATH=""
AUGMENTED_VERTICAL_FLIP_DIRECTORY = BASE_PATH + 'augmented_vertical_flip'
AUGMENTED_HORIZONTAL_FLIP_DIRECTORY = BASE_PATH + 'augmented_horizontal_flip'
AUGMENTED_BLUR_FLIP_DIRECTORY = BASE_PATH + 'augmented_blur'
TRAIN_DATASET_NAME = 'train'
TEST_DATASET_NAME = 'test'
CLASSES = [0, 1]

EPOCHS_NUMBER = 100

TPU_BATCH_SIZE = 256

def augment_vertical_flip():
    augmented_dataset_directory = AUGMENTED_VERTICAL_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    for image_file_name in os.listdir(TRAIN_DATASET_NAME):
        full_file_name = TRAIN_DATASET_NAME + '/' + image_file_name
        image = cv2.imread(full_file_name)
        flipped = np.flipud(image)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, flipped)


def augment_horizontal_flip():
    augmented_dataset_directory = AUGMENTED_HORIZONTAL_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    for image_file_name in os.listdir(TRAIN_DATASET_NAME):
        full_file_name = TRAIN_DATASET_NAME + '/' + image_file_name
        image = cv2.imread(full_file_name)
        flipped = np.fliplr(image)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, flipped)


def augment_blur():
    augmented_dataset_directory = AUGMENTED_BLUR_FLIP_DIRECTORY
    if os.path.isdir(augmented_dataset_directory):
        return

    os.makedirs(augmented_dataset_directory)

    for image_file_name in os.listdir(TRAIN_DATASET_NAME):
        full_file_name = TRAIN_DATASET_NAME + '/' + image_file_name
        image = cv2.imread(full_file_name)
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        cv2.imwrite(augmented_dataset_directory + '/' + image_file_name, blurred)


def build_all_train_dataset(use_vertical_flip=False, use_horizontal_flip=False, use_blur=False):
    original_dataset, original_labels = load_dataset(TRAIN_DATASET_NAME)
    all_train_dataset = original_dataset
    all_train_labels = original_labels
    if use_vertical_flip:
        vertical_flip_dataset, vertical_flip_labels = load_dataset(AUGMENTED_VERTICAL_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, vertical_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, vertical_flip_labels))
    if use_horizontal_flip:
        horizontal_flip_dataset, horizontal_flip_labels = load_dataset(AUGMENTED_HORIZONTAL_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, horizontal_flip_dataset))
        all_train_labels = np.concatenate((all_train_labels, horizontal_flip_labels))
    if use_blur:
        blur_dataset, blur_labels = load_dataset(AUGMENTED_BLUR_FLIP_DIRECTORY)
        all_train_dataset = np.concatenate((all_train_dataset, blur_dataset))
        all_train_labels = np.concatenate((all_train_labels, blur_labels))
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
    return np.reshape(np.array(dataset), (len(dataset), 28, 28, 1)).astype('float32'), np_utils.to_categorical(np.array(labels), len(CLASSES))


def build_test_dataset():
    dataset = []
    for image_file_name in os.listdir(TEST_DATASET_NAME):
        full_file_name = TEST_DATASET_NAME + '/' + image_file_name
        dataset.append(io.imread(full_file_name) / 255)
    return np.reshape(np.array(dataset), (len(dataset), 28, 28, 1)).astype('float32')

if __name__ == "__main__":
    augment_vertical_flip()
    augment_horizontal_flip()
    augment_blur()

    all_train_dataset, all_train_labels = build_all_train_dataset()
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(all_train_dataset,
                                                                                          all_train_labels,
                                                                                          test_size=1 / 6, shuffle=True)
    print(train_dataset.shape)
    print(validation_dataset.shape)
    test_dataset = build_test_dataset()
    print(test_dataset.shape)

    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # strategy = tf.distribute.experimental.TPUStrategy(resolver)
    # with strategy.scope():

    input_shape=(28, 28, 1)
    vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                      input_shape=input_shape)

    output = vgg.layers[-1].output
    output = Flatten()(output)
    vgg_model = Model(vgg.input, output)

    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False

    def get_bottleneck_features(model, input_imgs):
        features = model.predict(input_imgs, verbose=0)
        return features

    train_features_vgg = get_bottleneck_features(vgg_model, train_dataset)
    validation_features_vgg = get_bottleneck_features(vgg_model, validation_dataset)
    test_features_vgg = get_bottleneck_features(vgg_model, test_dataset)

    #input_shape = vgg_model.output_shape[1]
    model = Sequential()
    model.add(vgg_model)
    #model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    t = TicToc('learning')
    t.tic()
    model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, train_labels, batch_size=TPU_BATCH_SIZE, epochs=EPOCHS_NUMBER)
    t.toc()
    print("Elapsed: %f" % (t.elapsed / 60))
    loss, acc = model.evaluate(validation_dataset, validation_labels)
    print("Accuracy: %s" % acc)
    predicted = model.predict(test_dataset)
    submission = pd.DataFrame({'id': np.array(range(1, len(test_dataset) + 1)), 'label': predicted.ravel()})
    submission.to_csv('submission_vgg16.csv', index=False)