import os
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Conv1D
import tensorflow.keras
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from ttictoc import TicToc
from sklearn.metrics import mean_squared_error


BASE_PATH=""
DATASET_FILE_NAME = 'Sunspots.csv'


EPOCHS_NUMBER = 200
WINDOW_LENGTH = 34
BATCH_SIZE = 64


def read_spots():
    return pd.read_csv(DATASET_FILE_NAME, usecols=[2]).to_numpy()


def build_dataset(sequence, steps):
    dataset, labels = [], []
    for i in range(len(sequence)):
        end = i + steps
        if end > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end], sequence[end]
        dataset.append(seq_x)
        labels.append(seq_y)
    return np.array(dataset).astype('float32'), np.array(labels).astype('float32')


def map_predictions(predictions):
    mapped_predictions = []
    for prediction in predictions:
        mapped_predictions.append(prediction[0][0])
    return np.array(mapped_predictions).astype('float32')


if __name__ == "__main__":
    spots = read_spots()
    #np.random.shuffle(spots)
    dataset, labels = build_dataset(spots, WINDOW_LENGTH)
    print(dataset.shape)
    print(labels.shape)
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(dataset,
                                                                                          labels,
                                                                                          test_size=1 / 8, shuffle=True)
    print(train_dataset.shape)
    print(train_labels.shape)
    print(validation_dataset.shape)
    print(validation_labels.shape)

    t = TicToc('learning')
    t.tic()
    model = Sequential([
        Conv1D(128, 5, strides=1, padding="same", activation="relu"),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_dataset, train_labels, epochs=EPOCHS_NUMBER, batch_size=BATCH_SIZE)
    t.toc()
    print("Elapsed: %f" % (t.elapsed / 60))
    predictions = model.predict(validation_dataset, verbose=0)
    error = mean_squared_error(validation_labels, map_predictions(predictions))
    print("Test MSE: %s" % error)
    pyplot.plot(validation_labels)
    pyplot.plot(map_predictions(predictions), color='red')
    pyplot.show()
