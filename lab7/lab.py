import os
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
import tensorflow.keras

BASE_PATH=""
NEGATIVE_DATASET_DIRECTORY = '/neg/'
POSITIVE_DATASET_DIRECTORY = '/pos/'
TRAIN_DATASET_DIRECTORY = BASE_PATH + 'train'
TEST_DATASET_DIRECTORY = BASE_PATH + 'test'
TRAIN_INDICES_DATASET_FILE_NAME = BASE_PATH + 'train_processed_indices.csv'
TEST_INDICES_DATASET_FILE_NAME = BASE_PATH + 'test_processed_indices.csv'

EPOCHS_NUMBER = 50

TPU_BATCH_SIZE = 128*8

MAX_LENGTH = 128
EMBEDDINE_VECTOR_SIZE=128
LSTM_UNITS_COUNT=128
VOCABULARY_SIZE = 89527


def read_dataset(dataset_directory):
    dataset = []
    labels = []
    positive_reviews_directory = dataset_directory + POSITIVE_DATASET_DIRECTORY
    for review_file_name in os.listdir(positive_reviews_directory):
        full_file_name = positive_reviews_directory + review_file_name
        dataset.append(open(full_file_name, 'r').read())
        labels.append(1)
    negative_reviews_directory = dataset_directory + NEGATIVE_DATASET_DIRECTORY
    for review_file_name in os.listdir(negative_reviews_directory):
        full_file_name = negative_reviews_directory + review_file_name
        dataset.append(open(full_file_name, 'r').read())
        labels.append(0)
    return np.array(dataset), np.array(labels)


def load_vocabulary():
    vocabulary = {}
    i = 0
    with open('imdb.vocab', 'r') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary[line.replace('\n', '')] = i
            i += 1
    return vocabulary


def find_words_from_vocabulary_preserving_order(text, vocabulary):
    found_words_codes = []
    for word in text.split(' '):
        if word in vocabulary:
            found_words_codes.append(vocabulary[word])
    return pd.Series(found_words_codes).drop_duplicates().tolist()


def dataset_to_indices(dataset, vocabulary):
    mapped_dataset = []
    for sample in dataset:
        found_words_codes = find_words_from_vocabulary_preserving_order(sample, vocabulary)
        mapped_dataset.append(found_words_codes)
    return np.array(mapped_dataset)


def save_indices_processed_as_csv(mapped_dataset, labels, output):
    mapped_dataset_in_string = []
    for sample in mapped_dataset:
        mapped_dataset_in_string.append(' '.join(map(str, sample)) if sample else '')
    dataset = np.column_stack([labels, np.array(mapped_dataset_in_string)])
    np.savetxt(output, dataset, delimiter=',', fmt='%s')


def load_indices_processed_as_csv(output):
    csv = pd.read_csv(output)
    labels = csv.iloc[:, 0].to_numpy()
    dataset = []
    for sample in csv.iloc[:, 1:].to_numpy():
        dataset.append(list(map(np.float32, sample[0].split(' '))) if isinstance(sample[0], str) else [])
    return np.array(dataset), labels


if __name__ == "__main__":
    # train_dataset, train_labels = read_dataset(TRAIN_DATASET_DIRECTORY)
    # test_dataset, test_labels = read_dataset(TEST_DATASET_DIRECTORY)
    # print(train_dataset.shape)
    # print(test_dataset.shape)

    # vocabulary = load_vocabulary()

    # mapped_train_dataset = dataset_to_indices(train_dataset, vocabulary)
    # print(mapped_train_dataset.shape)
    # mapped_test_dataset = dataset_to_indices(test_dataset, vocabulary)
    # print(mapped_test_dataset.shape)

    # save_indices_processed_as_csv(mapped_train_dataset, train_labels, TRAIN_INDICES_DATASET_FILE_NAME)
    # save_indices_processed_as_csv(mapped_test_dataset, test_labels, TEST_INDICES_DATASET_FILE_NAME)

    mapped_train_dataset, train_labels = load_indices_processed_as_csv(TRAIN_INDICES_DATASET_FILE_NAME)
    print(np.max([len(v) for v in mapped_train_dataset]))
    print(mapped_train_dataset.shape)
    batch_remainder = mapped_train_dataset.shape[0] % (TPU_BATCH_SIZE)
    mapped_train_dataset=mapped_train_dataset[:-batch_remainder]
    train_labels=train_labels[:-batch_remainder]
    mapped_test_dataset, test_labels = load_indices_processed_as_csv(TEST_INDICES_DATASET_FILE_NAME)
    print(np.max([len(v) for v in mapped_test_dataset]))
    print(mapped_test_dataset.shape)

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(mapped_train_dataset, maxlen=MAX_LENGTH)
    x_test = sequence.pad_sequences(mapped_test_dataset, maxlen=MAX_LENGTH)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        print('Build model...')
        model = Sequential()
        model.add(Embedding(VOCABULARY_SIZE, EMBEDDINE_VECTOR_SIZE))
        model.add(Bidirectional(LSTM(LSTM_UNITS_COUNT, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')
        model.fit(x_train, train_labels, batch_size=TPU_BATCH_SIZE, epochs=EPOCHS_NUMBER)
        score, acc = model.evaluate(x_test, test_labels)
        print('Test accuracy:', acc)
