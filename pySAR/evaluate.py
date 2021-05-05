
################################################################################
#################                    Evaluate                  #################
################################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
    explained_variance_score, max_error, mean_poisson_deviance

class Evaluate():
    """
    An instance of the Evaluate class will calculate various metric values for
    the inputted observed (Y_true) and predicted (Y_pred) arrays, storing the
    results in the class attributed. Evaluate supports metrics: R2, RMSE, MSE,
    MAE, RPD, Explained Variance, Max Error and Mean Poisson Deviance.

    Attributes
    ----------
    Y_true : np.ndarray
        array of observed activity values.
    Y_pred : np.ndarray
        array of predicted activity values

    Methods
    -------
    r2_(multioutput='uniform_average')
    rmse_(multioutput='uniform_average')
    mse_(multioutput='uniform_average')
    mae_(multioutput='uniform_average')
    rpd_()
    explained_var_(multioutput='uniform_average')
    max_error_()
    mean_poisson_deviance_()

    """
    def __init__(self, Y_true, Y_pred):

        #convert input observed and predicted values into numpy arrays and reshape
        self.Y_true = np.array(Y_true).reshape((-1,1))
        self.Y_pred = np.array(Y_pred).reshape((-1,1))

        #validate that predicted and observed input arrays are of the same length,
        #   if input predicted and observed arrays are not same shape then raise error
        if (self.Y_true.shape) != (self.Y_pred.shape):
            raise ValueError('Observed and predicted values must be of the same length, \
                Y_true = {} & Y_pred = {}.'.format(Y_true.shape, Y_pred.shape))

        #calculate all metric values for inputs
        self.r2 = self.r2_()
        self.rmse = self.rmse_()
        self.mse = self.mse_()
        self.mae = self.mae_()
        self.rpd = self.rpd_()
        self.explained_var = self.explained_var_()

    def r2_(self, multioutput='uniform_average'):
        """
        Calculate R^2 (coefficient of determination) regression score function.

        Parameters
        ----------
        multioutput : str (default = 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        r2 : float
            R2 (coefficient of determination) score for observed and predicted values.
        """
        return r2_score(self.Y_true, self.Y_pred, multioutput=multioutput)

    def mse_(self,multioutput='uniform_average'):
        """
        Calculate MSE (mean square error) regression loss score for observed
        and predicted values.

        Parameters
        ----------
        multioutput : str (default = 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        mse : float
            MSE (mean square error) score for observed and predicted values.
        """
        return mean_squared_error(self.Y_true, self.Y_pred, multioutput=multioutput)

    def rmse_(self, multioutput='uniform_average'):
        """
        Calculate the RMSE (root mean square error) regression loss score for
        inputted observed and predicted values. Uses the same function for
        calculating MSE with the squared parameter set to False.

        Parameters
        ----------
        multioutput : str (default = 'uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values -
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.
        Returns
        -------
        mse : float
            RMSE score for observed and predicted values.
        """
        return mean_squared_error(self.Y_true, self.Y_pred, squared=False, multioutput=multioutput)

    def mae_(self, multioutput='uniform_average'):
        """
        Calculate the Mean absolute error regression loss.

        Parameters
        ----------
        multioutput : str
            Defines aggregating of multiple output scores. Array-like value
            defines weights used to average scores.
        Returns
        -------
        mae : float
            If multioutput is ‘raw_values’, then MAE is returned for each output
            separately. If multioutput is ‘uniform_average’ or an ndarray of
            weights, then the weighted average of all output errors is returned.
            The output is a non-negative floating point. The best value is 0.0.
        """
        return mean_absolute_error(self.Y_true, self.Y_pred, multioutput=multioutput)

    def rpd_(self):
        """
        Calculates the ratio of performance to deviation (RPD). RPD is the ratio
        between the standard deviation of a variable and the standard error of
        prediction of that variable by a given model.

        Returns
        -------
        rpd : float
            the RPD score for the model.
        """
        return self.Y_true.std()/np.sqrt(self.mse_())

    def explained_var_(self, multioutput='uniform_average'):
        """
        Calculates the explained variance regression score. Best possible score is 1.0,
        lower values are worse.

        Parameters
        ----------
        multioutput : str (default = 'uniform_average')
            Defines aggregating of multiple output scores. Array-like value
            defines weights used to average scores.
        Returns
        -------
        explained_var : float
            The explained variance or ndarray if ‘multioutput’ is ‘raw_values’.
        """
        return explained_variance_score(self.Y_true, self.Y_pred, multioutput=multioutput)

    def max_error(self):
        """
        Calculates the maximum residual error between observed and predicted
        values.

        Returns
        -------
        max_error : float
            A positive floating point value (the best value is 0.0).
        """
        return max_error(self.Y_true, self.Y_pred)

    def mean_poisson_deviance(self):
        """
        Calculate Mean Poisson deviance regression loss.

        Returns
        -------
        mean_poisson_deviance : float
            A non-negative floating point value (the best value is 0.0).
        """
        return mean_poisson_deviance(self.Y_true, self.Y_true)

    def __repr__(self):
        return "<Evaluate(Y_true: {} Y_pred: {})>".format(
            self.Y_true.shape, self.Y_pred.shape
        )

    def __str__(self):
        return "Instance of Evaluate Class with attribute values: \
                R2: {}, RMSE: {}, MSE: {}, MAE: {}, RPD: {} Explained Variance: {}" \
                .format(self.r2, self.rmse, self.mse, self.mae, self.rpd,self.explained_var
        )
