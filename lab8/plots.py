import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import datetime
from statsmodels.graphics import tsaplots

BASE_PATH=""
DATASET_FILE_NAME = BASE_PATH + 'Sunspots.csv'



def parser(x):
    return pd.to_datetime(x, format='%Y-%m-%d', errors="ignore")


if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_FILE_NAME, parse_dates=[0], header=0, index_col=0, squeeze=True, usecols=[1, 2], date_parser=parser)
    decomposition = seasonal_decompose(dataset, model='additive')
    #decomposition.plot()

    # data
    # pyplot.plot(list(range(0, len(dataset))), dataset)

    # trend
    # pyplot.title('Trend')
    # pyplot.plot(decomposition.trend)

    # seasonality
    # pyplot.title('Seasonality')
    # pyplot.plot(decomposition.seasonal)

    # autocorrelation
    autocorrelation_plot(dataset)

    # partial autocorrelation
    # tsaplots.plot_pacf(dataset)

    pyplot.show()
