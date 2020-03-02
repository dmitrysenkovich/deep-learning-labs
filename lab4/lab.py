import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn import linear_model
from sklearn import metrics
from ttictoc import TicToc
from tensorflow import keras
from keras import regularizers
from keras.utils import np_utils
from scipy.io import loadmat
import h5py
from numpy import savetxt
import os.path
import scipy.io
from PIL import Image
from skimage.color import rgb2gray

TRAIN_DATASET_NAME = 'train'
TEST_DATASET_NAME = 'test'
STRUCTURE_FILE_NAME = 'digitStruct.mat'
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NAN_CLASS = 10
MAX_LENGTH = 5

EPOCHS_NUMBER = 10

def get_labels_from_h5py_reference(file, ref):
    if len(ref) > 1:
        return [int(file[ref.value[j].item()].value[0][0]) for j in range(len(ref))]
    return [int(ref.value[0][0])]

def get_bboxes_from_h5py_reference(file, ref):
    if len(ref) > 1:
        return [file[ref.value[j].item()].value[0][0] for j in range(len(ref))]
    return [ref.value[0][0]]

def extract_labels(dataset_name):
    cached_labels_file_name = dataset_name + '_labels.mat'
    if os.path.isfile(cached_labels_file_name):
        return scipy.io.loadmat(cached_labels_file_name)['labels']

    file = h5py.File(dataset_name + '/' + STRUCTURE_FILE_NAME, 'r')
    digitStruct = file['digitStruct']
    bboxes = digitStruct['bbox']
    labels = []
    for i in range(len(bboxes)):
        bbox = bboxes[i].item()
        target = file[bbox]['label']
        number_digits = get_labels_from_h5py_reference(file, target)
        # because one of the numbers is 135458 > 5
        number_digits = [0 if digit == 10 else digit for digit in number_digits]
        length = len(number_digits)
        if length > MAX_LENGTH:
            length = length - 1
            del number_digits[MAX_LENGTH:]
        for _ in range(MAX_LENGTH - len(number_digits)):
            number_digits.append(10)
        number_digits = np.insert(number_digits, 0, length, axis=0)
        labels.append(np.array(number_digits))
    labels = np.array(labels)

    scipy.io.savemat(cached_labels_file_name, mdict={'labels': labels}, oned_as='row')

    return labels

def extract_bboxes(dataset_name):
    cached_bboxes_file_name = dataset_name + '_bboxes.mat'
    if os.path.isfile(cached_bboxes_file_name):
        return scipy.io.loadmat(cached_bboxes_file_name)['bboxes']

    file = h5py.File(dataset_name + '/' + STRUCTURE_FILE_NAME, 'r')
    digitStruct = file['digitStruct']
    bboxes = digitStruct['bbox']
    parsed_bboxes = []
    for i in range(len(bboxes)):
        bbox = bboxes[i].item()
        heights = get_bboxes_from_h5py_reference(file, file[bbox]['height'])
        widths = get_bboxes_from_h5py_reference(file, file[bbox]['width'])
        tops = get_bboxes_from_h5py_reference(file, file[bbox]['top'])
        lefts = get_bboxes_from_h5py_reference(file, file[bbox]['left'])
        # because one of the numbers is 135458 > 5
        if len(heights) > MAX_LENGTH:
            del heights[MAX_LENGTH:]
            del widths[MAX_LENGTH:]
            del tops[MAX_LENGTH:]
            del lefts[MAX_LENGTH:]
        min_top = np.min(tops)
        min_left = np.min(lefts)
        max_bottom = max([tops[i] + heights[i] for i in range(len(tops))])
        max_right = max([lefts[i] + widths[i] for i in range(len(lefts))])
        parsed_bboxes.append(np.array([min_left, min_top, max_right, max_bottom]))
    parsed_bboxes = np.array(parsed_bboxes)

    scipy.io.savemat(cached_bboxes_file_name, mdict={'bboxes': parsed_bboxes}, oned_as='row')

    return parsed_bboxes

def crop_and_resize_images(dataset_name, parsed_bboxes):
    processed_dataset_directory = 'processed_' + dataset_name
    if os.path.isdir(processed_dataset_directory):
        return

    os.makedirs(processed_dataset_directory)

    for i in range(len(parsed_bboxes)):
        image_file_name = str(i + 1) + '.png'
        full_file_name = dataset_name + '/' + image_file_name
        parsed_bbox = parsed_bboxes[i]
        cropped_image = Image.open(full_file_name).crop((parsed_bbox[0], parsed_bbox[1], parsed_bbox[2], parsed_bbox[3]))
        resized_image = cropped_image.resize([64, 64], Image.ANTIALIAS)
        grayscale_image = rgb2gray(np.array(resized_image))
        img = (((grayscale_image - grayscale_image.min()) / (grayscale_image.max() - grayscale_image.min())) * 255.9).astype(np.uint8)
        Image.fromarray(img).save(processed_dataset_directory + '/' + image_file_name)

def build_dataset(dataset_name, dataset_size):
    dataset = []
    processed_dataset_directory = 'processed_' + dataset_name
    for i in range(dataset_size):
        image_file_name = str(i + 1) + '.png'
        full_file_name = processed_dataset_directory + '/' + image_file_name
        dataset.append(io.imread(full_file_name, as_gray=True) / 255)
    return np.reshape(np.array(dataset), (len(dataset), 64, 64, 1))

def add_conv_layer(inputs, kernel_size):
    inputs = keras.layers.Conv2D(kernel_size, (5, 5), padding='same', kernel_initializer='he_normal', activation="relu")(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.MaxPooling2D()(inputs)
    return keras.layers.Dropout(0.4)(inputs)

def build_output_layer(inputs, labels_count):
    return keras.layers.Dense(labels_count, kernel_initializer='he_normal', activation="softmax")(inputs)

def evaluate(model, test_dataset, test_labels):
    predictions = model.predict(test_dataset)
    right_predictions_count = 0
    for i in range(len(test_dataset)):
        prediction = [np.argmax(predictions[j][i]) for j in range(6)]
        real = [np.asscalar(test_labels[0][i]), np.asscalar(test_labels[1][i]), np.asscalar(test_labels[2][i]), np.asscalar(test_labels[3][i]), np.asscalar(test_labels[4][i]), np.asscalar(test_labels[5][i])]
        if np.array_equal(prediction, real):
            right_predictions_count += 1
        # print(prediction)
        # print(real)
    return right_predictions_count / len(test_dataset)

def show(sample):
    plt.imshow(sample)
    plt.show()

if __name__ == "__main__":

    train_labels = extract_labels(TRAIN_DATASET_NAME)
    print(train_labels.shape)
    test_labels = extract_labels(TEST_DATASET_NAME)
    print(test_labels.shape)

    train_parsed_bboxes = extract_bboxes(TRAIN_DATASET_NAME)
    print(train_parsed_bboxes.shape)
    test_parsed_bboxes = extract_bboxes(TEST_DATASET_NAME)
    print(test_parsed_bboxes.shape)

    crop_and_resize_images(TRAIN_DATASET_NAME, train_parsed_bboxes)
    crop_and_resize_images(TEST_DATASET_NAME, test_parsed_bboxes)

    train_dataset = build_dataset(TRAIN_DATASET_NAME, len(train_parsed_bboxes))
    print(train_dataset.shape)
    test_dataset = build_dataset(TEST_DATASET_NAME, len(test_parsed_bboxes))
    print(test_dataset.shape)

    train_labels_lengths = np.reshape(train_labels[:, 0], (len(train_dataset), 1)) - 1
    train_labels_digits1 = np.reshape(train_labels[:, 1], (len(train_dataset), 1))
    train_labels_digits2 = np.reshape(train_labels[:, 2], (len(train_dataset), 1))
    train_labels_digits3 = np.reshape(train_labels[:, 3], (len(train_dataset), 1))
    train_labels_digits4 = np.reshape(train_labels[:, 4], (len(train_dataset), 1))
    train_labels_digits5 = np.reshape(train_labels[:, 5], (len(train_dataset), 1))

    test_labels_lengths = np.reshape(test_labels[:, 0], (len(test_dataset), 1)) - 1
    test_labels_digits1 = np.reshape(test_labels[:, 1], (len(test_dataset), 1))
    test_labels_digits2 = np.reshape(test_labels[:, 2], (len(test_dataset), 1))
    test_labels_digits3 = np.reshape(test_labels[:, 3], (len(test_dataset), 1))
    test_labels_digits4 = np.reshape(test_labels[:, 4], (len(test_dataset), 1))
    test_labels_digits5 = np.reshape(test_labels[:, 5], (len(test_dataset), 1))

    inputs = keras.layers.Input(shape=(64, 64, 1))
    temp = add_conv_layer(inputs, 48)
    temp = add_conv_layer(inputs, 64)
    temp = add_conv_layer(inputs, 128)
    temp = add_conv_layer(inputs, 160)
    temp = add_conv_layer(inputs, 192)
    temp = add_conv_layer(inputs, 192)
    temp = add_conv_layer(inputs, 192)
    temp = add_conv_layer(inputs, 192)
    temp = keras.layers.Flatten()(temp)
    temp = keras.layers.Dense(3072, kernel_initializer='he_normal', activation="relu")(temp)
    temp = keras.layers.Dropout(0.3)(temp)
    temp = keras.layers.Dense(3072, kernel_initializer='he_normal', activation="relu")(temp)
    temp = keras.layers.Dropout(0.3)(temp)
    outputs = [build_output_layer(temp, MAX_LENGTH), build_output_layer(temp, len(CLASSES)), build_output_layer(temp, len(CLASSES)), build_output_layer(temp, len(CLASSES)), build_output_layer(temp, len(CLASSES)), build_output_layer(temp, len(CLASSES))]

    t = TicToc('learning')
    t.tic()
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(batch_size=256, x=train_dataset, y=[train_labels_lengths, train_labels_digits1, train_labels_digits2, train_labels_digits3, train_labels_digits4, train_labels_digits5], epochs=EPOCHS_NUMBER)
    t.toc()
    print(t.elapsed / 60)
    asd = model.predict(train_dataset)
    print("Accuracy: %s" % evaluate(model, test_dataset, [test_labels_lengths, test_labels_digits1, test_labels_digits2, test_labels_digits3, test_labels_digits4, test_labels_digits5]))
    model.save("model.h5", include_optimizer=False)
