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

LARGE_DATASET_NAME = 'notMNIST_large/'
SMALL_DATASET_NAME = 'notMNIST_small/'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

EPOCHS_NUMBER = 200

def read_data():

    large_dataset_filenames = {}
    for letter in CLASSES:
        for r, d, f in os.walk(LARGE_DATASET_NAME + letter):
            large_dataset_filenames[letter] = f
    print('Large dataset images filenames has been read')
    small_dataset_filenames = {}
    for letter in CLASSES:
        for r, d, f in os.walk(SMALL_DATASET_NAME + letter):
            small_dataset_filenames[letter] = f
    print('Small dataset images filenames has been read')

    t = TicToc('reading datasets')
    t.tic()
    large_dataset_images = {}
    for letter in CLASSES:
        large_dataset_images[letter] = []
        for filename in large_dataset_filenames[letter]:
            large_dataset_images[letter].append(io.imread(LARGE_DATASET_NAME + letter + "/" + filename, as_gray=True) / 255)
    print('Large dataset images has been read')
    small_dataset_images = {}
    for letter in CLASSES:
        small_dataset_images[letter] = []
        for filename in small_dataset_filenames[letter]:
            small_dataset_images[letter].append(io.imread(SMALL_DATASET_NAME + letter + "/" + filename, as_gray=True) / 255)
    print('Small dataset images has been read')
    t.toc()
    print(t.elapsed)

    return large_dataset_images, small_dataset_images

def build_train_set(train_set_size, large_dataset_images):
    x_train = []
    y_train = []
    sample_size = train_set_size / len(CLASSES)
    train_dataset_indices = {}
    for letter in CLASSES:
        train_dataset_indices[letter] = set()
        while len(train_dataset_indices[letter]) < sample_size:
            train_dataset_indices[letter].add(random.randint(0, len(large_dataset_images[letter]) - 1))
    for i in range(len(CLASSES)):
        letter = CLASSES[i]
        indices = train_dataset_indices[letter]
        for index in indices:
            image = large_dataset_images[letter][index]
            x_train.append(image)
            y_train.append(i)
    x_train = np.array(x_train).reshape(len(x_train), 28, 28, 1)
    y_train = np_utils.to_categorical(np.array(y_train), len(CLASSES))
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train, train_dataset_indices

def build_train_and_validation_sets(train_set_size, validation_set_size, large_dataset_images):
    x_train, y_train, train_dataset_indices = build_train_set(train_set_size, large_dataset_images)
    x_val = []
    y_val = []
    sample_size = validation_set_size / len(CLASSES)
    validation_dataset_indices = {}
    for letter in CLASSES:
        validation_dataset_indices[letter] = set()
        while len(validation_dataset_indices[letter]) < sample_size:
            index = random.randint(0, len(large_dataset_images[letter]) - 1)
            if index not in train_dataset_indices[letter]:
                validation_dataset_indices[letter].add(index)
    for i in range(len(CLASSES)):
        letter = CLASSES[i]
        indices = validation_dataset_indices[letter]
        for index in indices:
            image = large_dataset_images[letter][index]
            x_val.append(image)
            y_val.append(i)
    x_val = np.array(x_val).reshape(len(x_val), 28, 28, 1)
    y_val = np_utils.to_categorical(np.array(y_val), len(CLASSES))
    print(x_val.shape)
    print(y_val.shape)
    return x_train, y_train, x_val, y_val

def build_test_set(small_dataset_images):
    x_test = []
    y_test = []
    for i in range(len(CLASSES)):
        letter = CLASSES[i]
        for image in small_dataset_images[letter]:
            x_test.append(image)
            y_test.append(i)
    x_test = np.array(x_test).reshape(len(x_test), 28, 28, 1)
    y_test = np_utils.to_categorical(np.array(y_test), len(CLASSES))
    print(x_test.shape)
    print(y_test.shape)
    return x_test, y_test

def train_and_compare_with_test(train_set_size, large_dataset_images, small_dataset_images, model):
    x_train, y_train, large_dataset_indices = build_train_set(train_set_size, large_dataset_images)
    x_test, y_test = build_test_set(small_dataset_images)

    test_acc = train_and_validate(x_train, y_train, x_test, y_test, model)
    return test_acc

def train_and_compare_with_validation(train_set_size, validation_set_size, large_dataset_images, model):
    x_train, y_train, x_val, y_val = build_train_and_validation_sets(train_set_size, validation_set_size, large_dataset_images)
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

if __name__ == "__main__":

    large_dataset_images, small_dataset_images = read_data()

    # ------------------- without pooling --------------------
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Adam validation accuracy: %s' % train_and_compare_with_validation(200000, 10000, large_dataset_images, model), file=open("results_without_pooling.txt", "a"))
    x_test, y_test = build_test_set(small_dataset_images)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Adam test accuracy: %s' % test_acc, file=open("results_without_pooling.txt", "a"))
    # ------------------- without pooling --------------------




    # ------------------- with pooling --------------------
    # model = keras.Sequential([
    #     keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(128, activation="relu"),
    #     keras.layers.Dense(10, activation='softmax')
    # ])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print('Adam validation accuracy: %s' % train_and_compare_with_validation(200000, 10000, large_dataset_images, model), file=open("results_with_pooling.txt", "a"))
    # x_test, y_test = build_test_set(small_dataset_images)
    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print('Adam test accuracy: %s' % test_acc, file=open("results_with_pooling.txt", "a"))
    # ------------------- with pooling --------------------




    # ------------------- LeNet-5 --------------------
    # model = keras.Sequential([
    #     keras.layers.Conv2D(6, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    #     keras.layers.AveragePooling2D(),
    #     keras.layers.Conv2D(16, (3, 3), activation="relu"),
    #     keras.layers.AveragePooling2D(),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(120, activation="relu"),
    #     keras.layers.Dense(84, activation="relu"),
    #     keras.layers.Dense(10, activation='softmax')
    # ])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # print('Adam validation accuracy: %s' % train_and_compare_with_validation(200000, 10000, large_dataset_images, model), file=open("results_lenet5.txt", "a"))
    # x_test, y_test = build_test_set(small_dataset_images)
    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print('Adam test accuracy: %s' % test_acc, file=open("results_lenet5.txt", "a"))
    # ------------------- LeNet-5 --------------------
