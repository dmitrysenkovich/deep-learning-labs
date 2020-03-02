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

LARGE_DATASET_NAME = 'notMNIST_large/'
SMALL_DATASET_NAME = 'notMNIST_small/'
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

EPOCHS_NUMBER = 200

def build_label_to_sample_map(train_dataset, train_dataset_labels):
    label_to_sample_map = {}
    for clazz in CLASSES:
        label_to_sample_map[clazz] = []
    for i in range(len(train_dataset_labels)):
        sample = train_dataset[i]
        label = train_dataset_labels[i][0]
        label_to_sample_map[label].append(sample)
    return label_to_sample_map

def build_train_set(train_set_size, label_to_sample_map):
    x_train = []
    y_train = []
    sample_size = train_set_size / len(CLASSES)
    train_dataset_indices = {}
    for letter in CLASSES:
        train_dataset_indices[letter] = set()
        while len(train_dataset_indices[letter]) < sample_size:
            train_dataset_indices[letter].add(random.randint(0, len(label_to_sample_map[letter]) - 1))
    for i in range(len(CLASSES)):
        letter = CLASSES[i]
        indices = train_dataset_indices[letter]
        for index in indices:
            image = label_to_sample_map[letter][index]
            x_train.append(image)
            y_train.append(i)
    x_train = np.array(x_train)
    y_train = np_utils.to_categorical(np.array(y_train), len(CLASSES))
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train, train_dataset_indices

def build_train_and_validation_sets(train_set_size, label_to_sample_map):
    x_train, y_train, train_dataset_indices = build_train_set(train_set_size, label_to_sample_map)
    x_val = []
    y_val = []
    validation_dataset_indices = {}
    for letter in CLASSES:
        validation_dataset_indices[letter] = set()
        for index in range(len(label_to_sample_map[letter])):
            if index not in train_dataset_indices[letter]:
                validation_dataset_indices[letter].add(index)
    for i in range(len(CLASSES)):
        letter = CLASSES[i]
        indices = validation_dataset_indices[letter]
        for index in indices:
            image = label_to_sample_map[letter][index]
            x_val.append(image)
            y_val.append(i)
    x_val = np.array(x_val)
    y_val = np_utils.to_categorical(np.array(y_val), len(CLASSES))
    print(x_val.shape)
    print(y_val.shape)
    return x_train, y_train, x_val, y_val

def train_and_compare_with_test(train_set_size, label_to_sample_map, x_test, y_test, model):
    x_train, y_train, large_dataset_indices = build_train_set(train_set_size, label_to_sample_map)
    test_acc = train_and_validate(x_train, y_train, x_test, y_test, model)
    return test_acc

def train_and_compare_with_validation(train_set_size, label_to_sample_map, model):
    x_train, y_train, x_val, y_val = build_train_and_validation_sets(train_set_size, label_to_sample_map)
    return train_and_validate(x_train, y_train, x_val, y_val, model)

def train_and_validate(x_train, y_train, x_val, y_val, model):
    t = TicToc('learning')
    t.tic()
    model.fit(x_train, y_train, epochs=EPOCHS_NUMBER)
    test_loss, test_acc = model.evaluate(x_val, y_val, verbose=2)
    print("Accuracy : ", test_acc)
    t.toc()
    print(t.elapsed)
    return test_acc

def add_conv_layer(inputs, kernel_size):
    inputs = keras.layers.Conv2D(kernel_size, (3, 3), padding='same', kernel_initializer='he_normal', activation="relu")(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.MaxPooling2D()(inputs)
    return keras.layers.Dropout(0.2)(inputs)

def build_output_layer(inputs):
    return keras.layers.Dense(len(CLASSES), kernel_initializer='he_normal', activation="softmax")(inputs)

def show(sample):
    plt.imshow(sample)
    plt.show()

if __name__ == "__main__":

    train_data = loadmat("train_32x32.mat")
    train_dataset = np.transpose(train_data['X'], (3, 0, 1, 2))
    #show(train_dataset[3])
    train_dataset_labels = train_data['y']
    train_dataset_labels = np.where(train_dataset_labels==10, 0, train_dataset_labels)
    label_to_sample_map = build_label_to_sample_map(train_dataset, train_dataset_labels)
    train_dataset = train_dataset / 255
    train_dataset_labels = np_utils.to_categorical(np.array(train_dataset_labels), len(CLASSES))

    test_data = loadmat("test_32x32.mat")
    x_test = np.transpose(test_data['X'], (3, 0, 1, 2)) / 255
    y_test = test_data['y']
    y_test = np.where(y_test==10, 0, y_test)
    y_test = np_utils.to_categorical(np.array(y_test), len(CLASSES))

    inputs = keras.layers.Input(shape=(32, 32, 3))
    temp = add_conv_layer(inputs, 32)
    temp = add_conv_layer(inputs, 48)
    temp = add_conv_layer(inputs, 64)
    temp = add_conv_layer(inputs, 80)
    temp = add_conv_layer(inputs, 128)
    temp = add_conv_layer(inputs, 144)
    temp = add_conv_layer(inputs, 160)
    asd = keras.layers.Flatten()(temp)
    outputs = [build_output_layer(asd)]

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Adam validation accuracy: %s' % train_and_compare_with_validation(40000, label_to_sample_map, model), file=open("results.txt", "a"))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Adam test accuracy: %s' % test_acc, file=open("results.txt", "a"))
