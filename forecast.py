from sqlalchemy import create_engine
import pandas as pd
import datetime
import config
import pmdarima as pm
import numpy as np
import arch
import statistics
import traceback
pd.set_option('display.max_columns', None)


def initializer(symbol):
    # Get Data
    engine = create_engine(config.psql)
    num_data_points = 255
    one_year_ago = (datetime.datetime.utcnow().date() - datetime.timedelta(days=num_data_points * 1.45)).strftime("%Y-%m-%d")
    query = f"select distinct * from stockdata_hist where symbol = '{symbol}' and tdate > '{one_year_ago}' AND (CAST(tdate AS TIME) = '20:00') limit {num_data_points}"
    df = pd.read_sql_query(query, con=engine).sort_values(by='tdate', ascending=True)

    # Get Forecast Range
    steps = 5
    today = df['tdate'].iloc[-1]
    end_prediction_date = today + datetime.timedelta(days=steps)
    end_friday = end_prediction_date + datetime.timedelta((4-end_prediction_date.weekday()) % 7)
    tomorrow = today+datetime.timedelta(days=1)
    date_range = pd.date_range(tomorrow, end_friday, freq="B")
    period = len(pd.date_range(tomorrow, end_friday, freq="B"))
    return df, tomorrow, date_range, period, engine


def arima(symbol, df, period, date_range):
    df['tdate'] = pd.to_datetime(df['tdate'])
    df.set_index(df['tdate'], inplace=True)
    y = df['tick_close']

    # Model ARIMA parameters
    # model = pm.auto_arima(y, error_action='ignore', trace=True,
    #                       suppress_warnings=True, maxiter=10,
    #                       seasonal=True, m=50)
    # print(type(model))
    # print("get params:")
    # print(model.get_params()['order'])
    # print(type(model.get_params()['order']))
    # print(model.summary())

    m = 7
    order = (1, 1, 1)
    sorder = (0, 0, 1, m)
    model = pm.arima.ARIMA(order, seasonal_order=sorder,
                              start_params=None, method='lbfgs', maxiter=50,
                              suppress_warnings=True, out_of_sample_size=0, scoring='mse',
                              scoring_args=None, trend=None, with_intercept=True)
    model.fit(y)

    # Forecast
    forecasts = model.predict(n_periods=period, return_conf_int=True)  # predict N steps into the future
    flatten = forecasts[1].tolist()

    results_df = pd.DataFrame(flatten, columns=['arima_low', 'arima_high'])
    results_df['arima_forecast'] = forecasts[0]
    results_df['tdate'] = date_range
    results_df['uticker'] = symbol
    results_df['arima_order'] = f"{order} {sorder}"
    results_df['last_price'] = df['tick_close'][-1]
    results_df['last_vwap'] = df['vwap'][-1]
    results_df['arima_diff'] = (results_df['arima_forecast']-results_df['last_price'])/results_df['last_price']
    results_df = results_df[['uticker', 'tdate', 'arima_low', 'arima_forecast', 'arima_high', 'arima_order', 'last_price', 'last_vwap', 'arima_diff']]
    return results_df


def garch_model(df, period, date_range):
    df = df.sort_index(ascending=True)
    df['tdate'] = pd.to_datetime(df['tdate'])
    df.set_index(df['tdate'], inplace=True)
    market = df['tick_close']
    returns = market.pct_change().dropna()
    garch = arch.arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")

    fit_model = garch.fit(update_freq=1)
    forecasts = fit_model.forecast(horizon=period, method='analytic', reindex=False)

    f_mean = forecasts.mean.iloc[0:].iloc[0].reset_index().iloc[:, 1]
    f_vol = np.sqrt(forecasts.variance.iloc[0:]).iloc[0].reset_index().iloc[:, 1]
    f_res = np.sqrt(forecasts.residual_variance.iloc[0:]).iloc[0].reset_index().iloc[:, 1]

    h_vol = statistics.stdev(returns.iloc[::-1])
    h_mean = statistics.mean(returns.iloc[::-1])

    temp_df = pd.DataFrame()
    temp_df['garch_mean'] = f_mean*100
    temp_df['garch_vol'] = f_vol*100

    temp_df['h_vol'] = h_vol*100
    temp_df['h_mean'] = h_mean*100
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df


def main():
    symbol = 'SPY'
    df, tomorrow, date_range, period, engine = initializer(symbol)
    arima_df = arima(symbol, df, period, date_range)
    garch_df = garch_model(df, period, date_range)
    result = pd.concat([arima_df, garch_df], axis=1)
    result['save_date'] = datetime.datetime.utcnow()
    print(result)
    # result.to_sql('arima', engine, if_exists='append', index=False)
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exe:
        print(exe)
        traceback.print_exc()

