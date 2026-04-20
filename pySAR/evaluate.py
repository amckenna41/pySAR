################################################################################
#################                    Evaluate                  #################
################################################################################

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
    explained_variance_score, max_error

class Evaluate():
    """
    An instance of the Evaluate class will calculate various metric values for
    the inputted observed (Y_true) and predicted (Y_pred) arrays, storing the
    results in the class attributes. The class supports metrics: R2, RMSE, MSE,
    MAE, RPD, Explained Variance and Max Error.

    Parameters
    ==========
    :Y_true: np.ndarray
        array of observed activity/fitness values.
    :Y_pred: np.ndarray
        array of predicted activity/fitness values.

    Attributes
    ==========
    :r2: float
        R2 (coefficient of determination) score.
    :rmse: float
        Root Mean Square Error score.
    :mse: float
        Mean Square Error score.
    :mae: float
        Mean Absolute Error score.
    :rpd: float
        Ratio of Performance to Deviation. Returns np.inf if MSE is 0.
    :explained_var: float
        Explained Variance score.
    :max_error: float
        Maximum residual error.

    Methods
    =======
    r2_(multioutput='uniform_average'):
        calculate R2 score.
    rmse_(multioutput='uniform_average'):
        calculate RMSE value.
    mse_(multioutput='uniform_average'):
        calculate MSE value.
    mae_(multioutput='uniform_average'):
        calculate MAE value.
    rpd_():
        calculate ratio of performance to deviation.
    explained_var_(multioutput='uniform_average'):
        calculate explained variance.
    max_error_():
        calculate max error.

    Raises
    ======
    :ValueError:
        if Y_true or Y_pred contain NaN values.
    :ValueError:
        if Y_true or Y_pred contain infinite values.
    :ValueError:
        if Y_true and Y_pred are not the same shape after reshaping.
    """
    def __init__(self, Y_true, Y_pred):

        #convert input observed and predicted values into numpy arrays and reshape
        self.Y_true = np.array(Y_true).reshape((-1,1))
        self.Y_pred = np.array(Y_pred).reshape((-1,1))

        #validate inputs contain no NaN or infinite values
        if np.any(np.isnan(self.Y_true)) or np.any(np.isnan(self.Y_pred)):
            raise ValueError('Y_true and Y_pred must not contain NaN values.')
        if np.any(np.isinf(self.Y_true)) or np.any(np.isinf(self.Y_pred)):
            raise ValueError('Y_true and Y_pred must not contain infinite values.')

        #validate that predicted and observed input arrays are of the same length,
        #if not same shape then raise error
        if (self.Y_true.shape != self.Y_pred.shape):
            raise ValueError(f'Observed and predicted values must be of the same shape, '
                f'Y_true = {self.Y_true.shape} & Y_pred = {self.Y_pred.shape}.')

        #calculate all metric values for inputs
        self.r2 = self.r2_()
        self.rmse = self.rmse_()
        self.mse = self.mse_()
        self.mae = self.mae_()
        self.rpd = self.rpd_()
        self.explained_var = self.explained_var_()
        self.max_error = self.max_error_()

    def r2_(self, multioutput='uniform_average'):
        """
        Calculate R^2 (coefficient of determination) regression score function.

        Parameters
        ==========
        :multioutput: str (default='uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values:
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.

        Returns
        =======
        :r2: float
            R2 (coefficient of determination) score for observed and predicted values.
        """
        return r2_score(self.Y_true, self.Y_pred, multioutput=multioutput)

    def mse_(self, multioutput='uniform_average'):
        """
        Calculate MSE (mean square error) regression loss score for observed
        and predicted values.

        Parameters
        ==========
        :multioutput: str (default='uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values:
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.

        Returns
        =======
        :mse: float
            MSE (mean square error) score for observed and predicted values.
        """
        return mean_squared_error(self.Y_true, self.Y_pred, multioutput=multioutput)

    def rmse_(self, multioutput='uniform_average'):
        """
        Calculate the RMSE (root mean square error) regression loss score for
        inputted observed and predicted values. Uses the same function for
        calculating MSE with the squared parameter set to False.

        Parameters
        ==========
        :multioutput: str (default='uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values:
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.

        Returns
        =======
        :rmse: float
            RMSE score for observed and predicted values.
        """
        return np.sqrt(mean_squared_error(self.Y_true, self.Y_pred, multioutput=multioutput))

    def mae_(self, multioutput='uniform_average'):
        """
        Calculate the Mean Absolute Error (MAE) regression loss for
        inputted observed and predicted values.

        Parameters
        ==========
        :multioutput: str (default='uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values:
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.

        Returns
        =======
        :mae: float
            If multioutput is ‘raw_values’, then MAE is returned for each output
            separately. If multioutput is ‘uniform_average’ or an ndarray of
            weights, then the weighted average of all output errors is returned.
            The output is a non-negative floating point. The best value is 0.0.
        """
        return mean_absolute_error(self.Y_true, self.Y_pred, multioutput=multioutput)

    def rpd_(self):
        """
        Calculates the Ratio of Performance to Deviation (RPD). RPD is the ratio
        between the standard deviation of a variable and the standard error of
        prediction of that variable by a given model.

        Parameters
        ==========
        None

        Returns
        =======
        :rpd: float
            the RPD score for the model.
        """
        # reuse already-computed self.mse to avoid a redundant sklearn call
        return self.Y_true.std() / np.sqrt(self.mse) if self.mse > 0 else np.inf

    def explained_var_(self, multioutput='uniform_average'):
        """
        Calculates the Explained Variance regression score. Best possible score is 1.0,
        lower values are worse.

        Parameters
        ==========
        :multioutput: str (default='uniform_average')
            method that defines aggregating of multiple output scores. Default
            is reccomended ('uniform_average'), available values:
            {‘raw_values’, ‘uniform_average’, ‘variance_weighted’}.

        Returns
        =======
        :explained_var: float
            The explained variance or ndarray if ‘multioutput’ is ‘raw_values’.
        """
        return explained_variance_score(self.Y_true, self.Y_pred, multioutput=multioutput)

    def max_error_(self):
        """
        Calculates the maximum residual error between observed and predicted values.

        Parameters
        ==========
        None

        Returns
        =======
        :max_error: float
            A positive floating point value of the maximal residual error 
            (the best value is 0.0).
        """
        return float(max_error(self.Y_true, self.Y_pred))

    def __repr__(self):
        return f"<Evaluate(Y_true: {self.Y_true.shape} Y_pred: {self.Y_pred.shape})>."

    def __str__(self):
        return f"Instance of Evaluate Class with attribute values: \
                R2: {self.r2}, RMSE: {self.rmse}, MSE: {self.mse}, MAE: {self.mae}, RPD: {self.rpd}, Explained Variance: {self.explained_var},\
                    Max Error: {self.max_error}."