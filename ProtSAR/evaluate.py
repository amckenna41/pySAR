
#########################################################################
###                         Evaluation                                ###
#########################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

class Evaluate():

    def __init__(self, Y_true, Y_pred):

        #convert input observed and predicted values into numpy arrays and reshape
        # self.Y_true = np.array(Y_true)
        # self.Y_pred = np.array(Y_pred)
        self.Y_true = np.array(Y_true).reshape((-1,1))
        self.Y_pred = np.array(Y_pred).reshape((-1,1))

        #validate that predicted and observed input arrays are of the same length
        #if input predicted and observed arrays are not of the same shape then raise error
        if (self.Y_true.shape) != (self.Y_pred.shape):
            raise ValueError('Observed and predicted values must be of the same length')

        self.r2 = self.r2_()
        self.mse = self.mse_()
        self.rmse = self.rmse_()
        self.mae = self.mae_()
        self.rpd = self.rpd_()
        self.explained_var = self.explained_var_()

    def r2_(self, multioutput='uniform_average'):
        """
        Parameters
        ----------
        Y_true: np.ndarray
            array of observed values
        Y_pred: np.ndarray
            array of predicted values
        multioutput: str (default: 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        r2 : float
            R2 (coefficient of determination) score for observed and predcited
            values.

        """
        r2 = r2_score(self.Y_true, self.Y_pred, multioutput=multioutput)

        return r2

    def mse_(self,multioutput='uniform_average'):
        """

        Calculate MSE (mean square erorr) regression loss score for observed
        and predicted values.

        Parameters
        ----------
        Y_true: np.ndarray
            array of observed values
        Y_pred: np.ndarray
            array of predicted values
        multioutput: str (default: 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        mse : float
            MSE (mean square error) score for observed and predicted values.

        """
        mse = mean_squared_error(self.Y_true, self.Y_pred, multioutput=multioutput)

        return mse

    def rmse_(self, multi_out='uniform_average'):

        """
        Calculate the RMSE (root mean square error) regression loss score for
        inputted observed and predicted values. Uses the same function for
        calculating MSE with the addition squared parameter set to False.

        Parameters
        ----------
        Y_true: np.ndarray
            array of observed values
        Y_pred: np.ndarray
            array of predicted values
        multioutput: str (default: 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        mse : float
            RMSE score for observed and predicted values.
        """
        rmse = mean_squared_error(self.Y_true, self.Y_pred, squared=False, multioutput=multi_out)

        return rmse

    def mae_(self, multi_out='uniform_average'):

        """



        Returns
        -------
        mae : float


        """
        mae = mean_absolute_error(self.Y_true, self.Y_pred, multioutput=multi_out)

        return mae

    def rpd_(self):

        """



        Returns
        -------
        rpd : float


        """
        rpd = self.Y_true.std()/np.sqrt(self.mse_())

        return rpd

    def explained_var_(self, multi_out='uniform_average'):

        """

        Returns
        -------
        explained_var : float


        """
        explained_var = explained_variance_score(self.Y_true, self.Y_pred, multioutput=multi_out)

        return explained_var

    def max_error(self):

        """

        Returns
        -------
        max_error : float


        """
        return metrics.max_error(self.Y_true, self.Y_pred)

    def mean_poisson_deviance(self):

        """
        Calculate Mean Poisson deviance regression loss.


        Returns
        -------
        mean_poisson_deviance : float
            A non-negative floating point value (the best value is 0.0).

        """
        return metrics.mean_poisson_deviance(self.Y_true, self.Y_true)

    def all_metrics(self):

        """
        Calculate all metrics for inputted predicted and observed class labels,
        return a dict with the keys as the metrics names and values as the metric
        values.

        Returns
        -------
        all_metrics_dict : dict
            dictionary of all calculated metrics values.

        """
        #initialise keys and values for metrics dict
        keys = ['R2', 'RMSE', 'MSE', 'MAE', 'RPD','Explained Var']
        vals = [self.r2, self.rmse, self.mse, self.mae, self.rpd, self.explained_var]

        #zip keys and values into a dictionary
        all_metrics_dict = dict(zip(keys,vals))

        return all_metrics_dict

    def __repr__(self):
        return "Instance of Evaluate class(Y_true: {} Y_pred: {})".format(
            self.Y_true, self.Y_pred
        )

    def __str__(self):
        pass
