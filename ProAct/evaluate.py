
from sklearn.metrics import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def r2_(Y_true, Y_pred, multi_out='uniform_average'):

    # Y_true = Y_true.reshape(-1)
    # Y_pred = Y_pred.reshape(-1)

    r2 = r2_score(Y_true, Y_pred, multioutput=multi_out)

    # r2 = model.score(Y_true, Y_pred)

    # return r2_train, r2_test
    return r2


def mse_(Y_true, Y_pred,multi_out='uniform_average'):

    mse = mean_squared_error(Y_true, Y_pred, multioutput=multi_out)

    return mse


def rmse_(Y_true, Y_pred, multi_out='uniform_average'):

    rmse = mean_squared_error(Y_true, Y_pred, squared=False, multioutput=multi_out)

    return rmse

def mae_(Y_true, Y_pred,multi_out='uniform_average'):

    mae = mean_absolute_error(Y_true, Y_pred, multioutput=multi_out)

    return mae

def rpd_(Y_true, Y_pred):

    rpd = Y_true.std()/np.sqrt(mse_(Y_true, Y_pred))

    return rpd

def msle_(Y_true, Y_pred, multi_out='uniform_average'):

    # scaler = MinMaxScaler()
    # scaler.fit(Y_true, Y_pred)
    # Y_true = scaler.transform(Y_true)
    # Y_pred = scaler.transform(Y_pred)
    #
    # msle = mean_squared_log_error(Y_true, Y_pred, multioutput=multi_out)
    msle = 2
    return msle

def explained_var_(Y_true, Y_pred, multi_out='uniform_average'):

    explained_var = explained_variance_score(Y_true, Y_pred, multioutput=multi_out)

    return explained_var


def get_all_metrics(model, Y_true, Y_pred):

    print('getting here1')

    Y_true = Y_true.reshape(-1)
    Y_pred = Y_pred.reshape(-1)

    r2 = r2_(Y_true, Y_pred)
    mse = mse_(Y_true, Y_pred)
    rmse = rmse_(Y_true, Y_pred)
    mae = mae_(Y_true, Y_pred)
    rpd = rpd_(Y_true, Y_pred)
    msle = msle_(Y_true, Y_pred)
    explained_var = explained_var_(Y_true, Y_pred)

    metrics_df = {'R2':r2, 'MSE':mse, 'RMSE':rmse, 'MAE':mae, 'RPD':rpd,'MSLE':msle, 'Explained Var':explained_var}
    #return dict of metrics - 'R2':89.00
    # return r2, rmse, mse, mae, rpd, explained_var, msle
    return metrics_df

#
# def eval_metrics(Y_true, Y_pred):
#
#     rmse = mean_squared_error(Y_true, Y_pred, squared=False)
#     mse = mean_squared_error(Y_true, Y_pred)
#     mae = mean_absolute_error(Y_true, Y_pred)
#     rpd = Y_true.std()/np.sqrt(mse)        # - ratio of performance to deviation (RPD)
#     explained_var = explained_variance_score(Y_true, Y_pred)
#     msle = mean_squared_log_error(Y_true, Y_pred)
#
#     return rmse, mse, mae, rpd, explained_var, msle
