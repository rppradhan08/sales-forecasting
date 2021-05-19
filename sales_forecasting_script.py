import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore")


def model_train(df_main):
    '''Tains Holt Winter's model on entire dataset'''
    model = ExponentialSmoothing(np.asarray(
        df_main['SALES']), seasonal_periods=31, trend='add', seasonal='add')
    return model.fit(optimized=True)


def model_forecast(model, N):
    '''Provides sales forecast on N future samples'''
    future_dates = [pd.Timestamp(END_DATE) + DateOffset(days=i)
                    for i in range(1, N+1)]
    pred = {
        "Date": future_dates,
        "Forecast": model.forecast(N)
    }
    return pred


if __name__ == '__main__':
    # Loading Dataset
    df = pd.read_csv('dataset.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Dealing with Missing values int it provided range
    START_DATE = '2018-09-30'
    END_DATE = '2019-03-25'

    dates = pd.DataFrame(pd.date_range(
        start=START_DATE, end=END_DATE), columns=['DATE'])
    df_main = dates.merge(df, how="left")
    df_main = df_main.set_index('DATE')
    df_main['SALES'].interpolate(method='linear', inplace=True)

    # Training model
    model = model_train(df_main)

    # Forecasting for future N days
    N = int(input("Enter the number of days:"))
    forecast = model_forecast(model, N)
    forecast = pd.DataFrame(forecast)
    print(forecast)
