import os
import os.path

import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from ttictoc import TicToc
import pmdarima as pm

BASE_PATH=""
DATASET_FILE_NAME = BASE_PATH + 'Sunspots.csv'


if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_FILE_NAME, header=0, index_col=0, squeeze=True, usecols=[1, 2]).values
    #np.random.shuffle(dataset)

    rs_fit = pm.auto_arima(dataset,
                           n_jobs=-1,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False, random=True,
                           n_fits=1000)

    print(rs_fit.summary())

    dataset_size = len(dataset)
    print(dataset_size)
    test_split = int(dataset_size / 8)
    train_dataset, validation_dataset = dataset[:-test_split], dataset[-test_split:]
    print(train_dataset.shape)
    print(validation_dataset.shape)
    history = [x for x in train_dataset]
    predictions = []
    tt = TicToc('learning')
    tt.tic()
    for t in range(len(validation_dataset)):
        # fit model
        #model = ARIMA(history, order=(34,1,21))
        model = ARIMA(history, order=(3, 0, 2))
        model_fit = model.fit(disp=False)
        # one step forecast
        yhat = model_fit.forecast()[0]
        # store forecast and ob
        predictions.append(yhat)
        history.append(validation_dataset[t])
        print(t)
    tt.toc()
    print("Elapsed: %f" % (tt.elapsed / 60))
    # evaluate forecasts
    error = mean_squared_error(validation_dataset, predictions)
    print('Test MSE: %s' % error)
    pyplot.plot(validation_dataset)
    pyplot.plot(predictions, color='red')
    pyplot.show()
