import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns
from matplotlib import rcParams
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

def error_metrics(y_pred, y_truth, model_name=None, test=True):
    dict_error = dict()

    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred
    else:
        y_pred = y_pred.to_numpy()

    if isinstance(y_truth, np.ndarray):
        y_truth = y_truth
    else:
        y_truth = y_truth.to_numpy()

    print('\nError metrics for model {}'.format(model_name))

    RMSE = np.sqrt(mean_squared_error(y_truth, y_pred))
    print("RMSE or Root mean squared error: %.2f" % RMSE)

    R2 = r2_score(y_truth, y_pred)
    print('Variance score: %.2f' % R2)

    MAE = mean_absolute_error(y_truth, y_pred)
    print('Mean Absolute Error: %.2f' % MAE)

    MAPE = (np.mean(np.abs((y_truth - y_pred) / y_truth)) * 100)
    print('Mean Absolute Percentage Error: %.2f %%' % MAPE)

    if test:
        train_test = 'test'
    else:
        train_test = 'train'

    name_error = ['model', 'train_test', 'RMSE', 'R2', 'MAE', 'MAPE']
    value_error = [model_name, train_test, RMSE, R2, MAE, MAPE]
    list_error = list(zip(name_error, value_error))

    for error in list_error:
        if error[0] in dict_error:
            dict_error[error[0]].append(error[1])
        else:
            dict_error[error[0]] = [error[1]]

def plot_timeseries(ts, title = 'og', opacity = 1):
    """
    Plot plotly time series of any given timeseries ts
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = ts.index, y = ts.values, name = "observed",
                         line_color = 'lightslategrey', opacity = opacity))

    fig.update_layout(title_text = title,
                  xaxis_rangeslider_visible = True)
    fig.show()


def plot_ts_pred(og_ts, pred_ts, model_name=None, og_ts_opacity=0.5, pred_ts_opacity=0.5):
    """
    Plot plotly time series of the original (og_ts) and predicted (pred_ts) time series values to check how our model performs.
    model_name: name of the model used for predictions
    og_ts_opacity: opacity of the original time series
    pred_ts_opacity: opacity of the predicted time series
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=og_ts.index, y=np.array(og_ts.values), name="Observed",
                             line_color='deepskyblue', opacity=og_ts_opacity))

    try:
        fig.add_trace(go.Scatter(x=pred_ts.index, y=pred_ts, name=model_name,
                                 line_color='lightslategrey', opacity=pred_ts_opacity))
    except:  # if predicted values are a numpy array they won't have an index
        fig.add_trace(go.Scatter(x=og_ts.index, y=pred_ts, name=model_name,
                                 line_color='lightslategrey', opacity=pred_ts_opacity))

    # fig.add_trace(go)
    fig.update_layout(title_text='Observed test set vs predicted energy MWH values using {}'.format(model_name),
                      xaxis_rangeslider_visible=True)
    fig.show()


def train_test(data, test_size=0.15, scale=False, cols_to_transform=None, include_test_scale=False,z='load'):

    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df) * (1 - test_size))

    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])

    X_train = df.drop(z, axis=1).iloc[:test_index]
    y_train = df[z].iloc[:test_index]
    X_test = df.drop(z, axis=1).iloc[test_index:]
    y_test = df[z].iloc[test_index:]

    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])

    return X_train, X_test, y_train, y_test
def add_fourier_terms(df, year_k, week_k, day_k):
    """
    df: dataframe to add the fourier terms to
    year_k: the number of Fourier terms the year period should have. Thus the model will be fit on 2*year_k terms (1 term for
    sine and 1 for cosine)
    week_k: same as year_k but for weekly periods
    day_k:same as year_k but for daily periods
    """

    for k in range(1, year_k + 1):
        # year has a period of 365.25 including the leap year
        df['year_sin' + str(k)] = np.sin(2 * k * np.pi * df.index.dayofyear / 365.25)
        df['year_cos' + str(k)] = np.cos(2 * k * np.pi * df.index.dayofyear / 365.25)

    for k in range(1, week_k + 1):
        # week has a period of 7
        df['week_sin' + str(k)] = np.sin(2 * k * np.pi * df.index.dayofweek / 7)
        df['week_cos' + str(k)] = np.cos(2 * k * np.pi * df.index.dayofweek / 7)

    for k in range(1, day_k + 1):
        # day has period of 24
        df['hour_sin' + str(k)] = np.sin(2 * k * np.pi * df.index.hour / 24)
        df['hour_cos' + str(k)] = np.cos(2 * k * np.pi * df.index.hour / 24)


def trend_model(data, cols_to_transform, l1_space, alpha_space, cols_use=0, scale=True, test_size=0.15,
                include_test_scale=False):
    """
    Tuning, fitting and predicting with an Elastic net regression model.
    data: time series dataframe including X and y variables
    col_use: columns including the y variable to be used from the data
    cols_to_transform: columns to be scaled using StandardScaler if scale = True
    l1_space: potential values to try for the l1_ratio parameter of the elastic net regression
    include_test_scale: If True then the StandardScaler will be fit on the entire dataset instead of just the training set

    A note about l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
    For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    """

    # Creating the train test split
    if cols_use != 0:
        df = data[cols_use]
    else:
        df = data

    X_train, X_test, y_train, y_test = train_test(df, test_size=test_size,
                                                  scale=scale, cols_to_transform=cols_to_transform,
                                                  include_test_scale=include_test_scale)

    # Create the hyperparameter grid
    # l1_space = np.linspace(0, 1, 50)
    param_grid = {'l1_ratio': l1_space, 'alpha': alpha_space}

    # Instantiate the ElasticNet regressor: elastic_net
    elastic_net = ElasticNet()

    # for time-series cross-validation set 5 folds
    tscv = TimeSeriesSplit(n_splits=5)

    # Setup the GridSearchCV object: gm_cv ...trying 5 fold cross validation
    gm_cv = GridSearchCV(elastic_net, param_grid, cv=tscv)

    # Fit it to the training data
    gm_cv.fit(X_train, y_train)

    # Predict on the test set and compute metrics
    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
    print("Tuned ElasticNet R squared: {}".format(r2))
    print("Tuned ElasticNet RMSE: {}".format(np.sqrt(mse)))
    # fitting the elastic net again using the best model from above

    elastic_net_opt = ElasticNet(l1_ratio=gm_cv.best_params_['l1_ratio'])
    elastic_net_opt.fit(X_train, y_train)

    # Plot the coefficients
    _ = plt.figure(figsize=(15, 7))
    _ = plt.plot(range(len(X_train.columns)), elastic_net_opt.coef_)
    _ = plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation=90)
    _ = plt.margins(0.02)
    _ = plt.axhline(0, linewidth=0.5, color='r')
    _ = plt.title('significane of features as per Elastic regularization')
    _ = plt.ylabel('Elastic net coeff')
    _ = plt.xlabel('Features')

    # Plotting y_true vs predicted
    _ = plt.figure(figsize=(5, 5))
    # returns the train and test X and y sets and also the optimal model
    return X_train, X_test, y_train, y_test, elastic_net_opt