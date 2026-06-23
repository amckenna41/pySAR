################################################################################
#################                    Model                     #################
################################################################################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import get_scorer_names
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE, SelectFromModel, SequentialFeatureSelector
from difflib import get_close_matches
from copy import deepcopy
import os
import pickle
import pandas as pd
import numpy as np
# np.seterr is intentionally NOT set globally; divide/invalid warnings are suppressed
# locally via np.errstate() at the call sites where they are expected.

from .evaluate import Evaluate

class Model():
    """
    Class for building, fitting and training a various range of predictive
    regression models and all their related methods and attributes. The 
    model class supports the following regression algorithms: PLS Regression,
    Random Forest, AdaBoost, Bagging, Decision Tree, GradientBoost, Linear
    Regression, Lasso, Ridge, ElasticNet, Support Vector Regression, Stochastic
    Gradient Descent, K Nearest Neighbours (KNN), Extra Trees, Histogram-based
    Gradient Boosting and Gaussian Process Regression.

    Once a model object has been built and fitted to the training data and 
    labels, it can then be used for predicting the sought activity/fitness
    value for unseen test sequences.

    Parameters
    ==========
    :X: np.ndarray
        training data.
    :Y: np.ndarray
        training data labels.
    :algorithm: str
        sklearn regression algorithm to build and fit model with. Value can be 
        an approximate representation of model name, for example: 'plsreg' will 
        initialiase an instance of the PLSRegression model etc. Available 
        algorithms listed above.
    :parameters: dict (default={})
        parameters to use for specific sklearn model when building regression 
        model, by default it is set to {}, meaning all of the models' default 
        parameters are used. Refer to sci-kit learn for full list of available 
        input parameters for each model: https://scikit-learn.org/stable/index.html.
    :test_split: float (default=0.2)
        proportion of the test data to use for building model, default of 0.2 is 
        recommended, meaning 80% of the data used for training and 20% for testing.

    Methods
    =======
    get_model():
        build model using inputted parameters.
    train_test_split(scale=True, test_split=0.2, random_state=None, shuffle=True):
        get train-test split of dataset.
    fit():
        fit model.
    predict():
        predict activity values using trained model and test data.
    save(save_folder):
        save fitted model to save_folder.
    model_fitted():
        return if model has been fitted (true or false)
    hyperparameter_tuning(self, parameters={}, metric='r2', cv=5, n_jobs=None, verbose=2):
        complete hyperparameter tuning of model and its associated parameters.
    feature_selection(method=""):
        undertake feature selection using technique specified by method input
        parameter to find optimal selection of features for maximum predictability
        in model. Supported feature selection methods include SelectKBest, chi2, 
        VarianceThreshold, RFE, SelectFromModel and SequentialFeatureSelector.
    """
    MODEL_CONSTRUCTORS = {
        'plsregression': PLSRegression,
        'randomforestregressor': RandomForestRegressor,
        'adaboostregressor': AdaBoostRegressor,
        'baggingregressor': BaggingRegressor,
        'decisiontreeregressor': DecisionTreeRegressor,
        'linearregression': LinearRegression,
        'lasso': Lasso,
        'ridge': Ridge,
        'sgd': SGDRegressor,
        'stochasticgradientdescent': SGDRegressor,
        'gbr': GradientBoostingRegressor,
        'gradientboostingregressor': GradientBoostingRegressor,
        'svr': SVR,
        'supportvectorregression': SVR,
        'knn': KNeighborsRegressor,
        'kneighborsregressor': KNeighborsRegressor,
        'knearestneighbors': KNeighborsRegressor,
        'elasticnet': ElasticNet,
        'extratreesregressor': ExtraTreesRegressor,
        'extratrees': ExtraTreesRegressor,
        'histgradientboostingregressor': HistGradientBoostingRegressor,
        'histgradientboosting': HistGradientBoostingRegressor,
        'hgbr': HistGradientBoostingRegressor,
        'gaussianprocessregressor': GaussianProcessRegressor,
        'gaussianprocess': GaussianProcessRegressor,
        'gpr': GaussianProcessRegressor,
    }

    def __init__(self, X, Y, algorithm, parameters=None, test_split=0.2):

        self.algorithm = algorithm
        self.test_split = test_split
        self.X = X
        self.Y = Y

        #if no model parameters input, then set to {} meaning default models' parameters are used
        if parameters is None or parameters == [] or parameters == "":
            self.parameters = {}
        else:
            self.parameters = parameters

        #derive valid model names directly from MODEL_CONSTRUCTORS to avoid duplication and sync issues
        self.valid_models = list(self.MODEL_CONSTRUCTORS.keys())

        #raise error if algorithm parameter isnt string type
        if not(isinstance(self.algorithm, str)):
            raise TypeError(f"Algorithm input parameter must be a string, got type {type(self.algorithm)}.")

        #get closest match of valid model from the input algorithm parameter value using difflib
        model_matches = get_close_matches(self.algorithm.lower().strip(),[item.lower().strip() \
            for item in self.valid_models], cutoff=0.5)

        #if algorithm is a valid model then set it to self.algorithm, else raise error
        if (model_matches!=[]):
            self.algorithm = model_matches[0]
        else:
            raise ValueError(f'Input algorithm {self.algorithm} not found in list of available valid models\n{self.valid_models}.')

        #create instance of algorithm object using its sklearn constructor
        self.model = self.get_model()

        #set model_fit to None, specifies if model has been fit or not
        self.model_fit = None

    def get_model(self):
        """
        Create instance of model type specified by input 'algorithm' argument. If
        input 'parameters' = {} then default parameters of sklearn model are used, else set
        the parameters of the model to the values specified in the 'parameters' input.

        Parameters
        ==========
        None

        Returns
        =======
        :model: sklearn.model
            instantiated regression model with default or user-specified parameters.
        """
        constructor = self.MODEL_CONSTRUCTORS.get(self.algorithm.lower().strip())
        if constructor is None:
            raise ValueError('Input Algorithm {} not found in available valid models:\n{}'.
                format(self.algorithm, self.valid_models))

        valid_parameter_names = set(constructor().get_params().keys())
        parameters = {
            key: value for key, value in self.parameters.items()
            if key in valid_parameter_names
        }

        return constructor(**parameters) if parameters else constructor()

    def train_test_split(self, test_split=0.2, scale=True, random_state=None, shuffle=True):
        """
        Split the X and Y input features and labels into random train and test
        subsets. By default a 80:20 split will be used, whereby 80% of the data
        will be used for training and 20% for testing. By default the input will
        be scaled first such that the mean is removed and features scaled to unit
        variance. By default data is shuffled before the split and random state is None.

        Parameters
        ==========
        :scale: bool (default=True)
            if true then scale the features such that they are standardised.
        :test_split: float (default=0.2)
            proportion of the total dataset to use for testing, rest used for training.
        :random_state : float (default=None)
            Controls the shuffling applied to the data before applying the split.
            Popular integer random seeds are 0 and 42, None by default.
        :shuffle: bool (default=True)
            Whether or not to shuffle the data before splitting.

        Returns
        =======
        :self.X_train, self.X_test, self.Y_train, self.Y_test: np.ndarray
            splitted training and test data features and labels.
        """
        #validate that X and Y arrays are of the same size
        if (len(self.X) != len(self.Y)):
            raise ValueError('X and Y input parameters must be of the same length - X: {}, Y: {}.'.
                format(len(self.X), len(self.Y)))

        #reshape input arrays to 2D arrays without mutating the original attributes
        X_values = self.X.values if isinstance(self.X, (pd.DataFrame, pd.Series)) else self.X
        Y_values = self.Y.values if isinstance(self.Y, (pd.DataFrame, pd.Series)) else self.Y

        X_values = np.asarray(X_values)
        Y_values = np.asarray(Y_values)

        if (X_values.ndim != 2):
            X_values = np.reshape(X_values, (-1,1))
        if (Y_values.ndim != 2):
            Y_values = np.reshape(Y_values, (-1,1))

        #if invalid test size input then raise ValueError
        if (test_split <= 0 or test_split >=1):
            raise ValueError(f'test_split must be between 0 and 1 exclusive, got {test_split}.')

        #setting test_split attribute
        self.test_split = test_split     

        #split X and Y into training and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y_values,
            test_size=test_split, random_state=random_state, shuffle=shuffle)

        #scale training data X after splitting to avoid test-set leakage
        if (scale):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler
        else:
            self.scaler = None

        #set X and Y attributes
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = np.reshape(Y_train, (len(Y_train),))
        self.Y_test = np.reshape(Y_test, (len(Y_test),))

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def fit(self):
        """
        Fit model to training data and labels.

        Parameters
        ==========
        None

        Returns
        =======
        :self.model_fit: np.ndarray
            fitted sklearn model of type specified by algorithm attribute.
        """
        self.model_fit = self.model.fit(self.X_train, self.Y_train)
        return self.model_fit

    def predict(self):
        """
        Predict the target values of unseen test data using the 
        trained model.

        Parameters
        ==========
        None

        Returns
        =======
        :self.model_fit.predict(self.X_test): np.ndarray
            array of predicted target values for unseen test data.
        """
        return self.model_fit.predict(self.X_test)

    def save(self, save_folder, model_name="model.pkl"):
        """
        Save fitted model to specified save_folder.

        Parameters
        ==========
        :save_folder: str
            folder to save model to.
        :model_name: str
            filename for model.
        
        Returns
        =======
        None

        Security
        ========
        Models are serialized using pickle. Never load pickle files from untrusted
        sources; deserialization of malicious data can execute arbitrary code.
        """
        #append pickle file extension if not present in filename
        if (os.path.splitext(model_name)[1].lower() != ".pkl"):
            model_name = model_name + ".pkl"
        
        #set save path to folder + filename
        save_path = os.path.join(save_folder, model_name)
        
        #save model in pickle format
        try:
            with open(save_path, 'wb') as file:
                pickle.dump({'model': self.model, 'scaler': getattr(self, 'scaler', None)}, file)
        except pickle.PickleError as e:
            raise RuntimeError(f"Error pickling model with path: {save_path}.") from e

    @classmethod
    def load(cls, path: str, allow_pickle: bool = True) -> 'Model':
        """
        Load a previously saved model and its scaler from a pickle file.

        Parameters
        ==========
        :path: str
            Filepath to the .pkl file created by save().
        :allow_pickle: bool (default=True)
            Must be True to load the file. Set to False to raise a ValueError
            (useful to block accidental loading of untrusted sources).

        Returns
        =======
        :instance: Model
            Reconstructed Model instance with model and scaler restored.

        Security
        ========
        Pickle deserialization can execute arbitrary code. Never load .pkl files
        from untrusted sources. Pass allow_pickle=False to block loading entirely.
        """
        if not allow_pickle:
            raise ValueError(
                "allow_pickle=False: loading pickle files is disabled. "
                "Pass allow_pickle=True only if you trust the source."
            )
        import warnings as _warnings
        _warnings.warn(
            "Model.load() deserializes a pickle file. Never load .pkl files "
            "from untrusted sources as they can execute arbitrary code.",
            UserWarning, stacklevel=2
        )
        if not os.path.isfile(path):
            raise OSError(f'Model file not found at path: {path}.')
        try:
            with open(path, 'rb') as file:
                payload = pickle.load(file)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f'Error unpickling model at path: {path}.') from e

        if not isinstance(payload, dict) or 'model' not in payload:
            raise ValueError(
                f'Unexpected pickle format in {path}. Expected a dict with a "model" key.'
            )

        instance = cls.__new__(cls)
        instance.model = payload['model']
        instance.scaler = payload.get('scaler')
        instance.model_fit = instance.model  # mark as fitted
        instance.algorithm = type(instance.model).__name__.lower()
        instance.parameters = {}
        instance.test_split = 0.2
        instance.valid_models = list(cls.MODEL_CONSTRUCTORS.keys())
        instance.X = None
        instance.Y = None
        instance.X_train = None
        instance.X_test = None
        instance.Y_train = None
        instance.Y_test = None
        return instance

    def hyperparameter_tuning(self, param_grid=None, metric='r2', cv=5, n_jobs=None, verbose=2):
        """
        Hyperparameter tuning of model to find its optimal arrangement of parameters
        using a Grid Search.

        Parameters
        ==========
        :param_grid: dict (default=None)
            dictionary/grid of selected models' parameters and the potential values of each
            that you want to tune.
        :metric: str (default=r2)
            scoring metric used to evaluate the performance of the cross-validated
            model on the test set, R2 by default. List of available scoring metrics
            can be found in documentation:
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :cv: int (default=5)
            Determines the cross-validation splitting strategy, a CV fold of 5 is used by default.
        :n_jobs : int (default=None)
            Number of jobs to run in parallel. None means 1 job.
        :verbose: int (default=2)
            verbosity of output during tuning process. The values and what they mean 
            for this parameter can be found on the documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

        Returns
        =======
        None
        """
        #raise error if train_test_split() hasn't been called yet
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise RuntimeError(
                'train_test_split() must be called before hyperparameter_tuning().'
            )

        #default to empty dict if not provided
        if param_grid is None:
            param_grid = {}

        #input 'param_grid' parameter must be a dict, if not raise error
        if not (isinstance(param_grid, dict)):
            raise TypeError(f'param_grid argument must be of type dict, got type {type(param_grid)}.')

        #input metric must be in available scoring metrics, if not raise error
        valid_scorers = sorted(get_scorer_names())
        if (metric not in valid_scorers):
            raise ValueError(
                f"Invalid scoring metric {metric} not in list of available Sklearn Scoring Metrics:\n{valid_scorers}."
            )

        #cv must be of type int and be between 5 and 10, if not then default of 5 is used
        if not isinstance(cv, int) or cv < 5 or cv > 10:
            import warnings
            warnings.warn(f'Invalid cv value {cv!r}; must be an int between 5 and 10. Defaulting to 5.', UserWarning, stacklevel=2)
            cv = 5

        #copy to avoid mutating caller's dict; filter out parameter names invalid for this model
        param_grid = {p: v for p, v in param_grid.items() if p in self.model.get_params()}

        #create deep copy of model
        model_copy = deepcopy(self.model)

        #grid search of hyperparameter space for model
        grid_search = GridSearchCV(estimator=model_copy, param_grid=param_grid, \
            cv=cv, scoring=metric, n_jobs=n_jobs, verbose=verbose, error_score=0)

        #fit X and Y to best model found in grid search
        grid_result = grid_search.fit(self.X_train, self.Y_train)

        #predict values of unseen test data using best found model
        best_model_pred = grid_result.predict(self.X_test)
        
        #create instance of Evaluate class and calculate metrics from best model
        evaluation = Evaluate(self.Y_test,best_model_pred)
        
        #print out results of grid search
        print('\n#############################################################')
        print('################### Hyperparameter Results ###################')
        print('#############################################################\n')

        print('######################### Parameters ########################\n')
        print(f'# Best Params: {grid_result.best_params_}')
        print(f'# Model Type: {repr(self)}')
        print(f'# Scoring Metric: {metric}')
        print(f'# Number of CV folds: {cv}')
        print(f'# Test Split: {self.test_split}\n')

        print('######################### Metrics ###########################\n')
        print(f'# Best Score (R2): {grid_result.best_score_}')
        print(f'# RMSE: {evaluation.rmse} ')
        print(f'# MSE: {evaluation.mse} ')
        print(f'# MAE: {evaluation.mae}')
        print(f'# RPD: {evaluation.rpd}')
        print(f'# Explained Variance: {evaluation.explained_var}\n')
        print('##############################################################')
        
        self.grid_result = grid_result
        return self.grid_result

    def model_fitted(self):
        """
        Return if model has been fitted, true or false.

        Parameters
        ==========
        None
        
        Returns
        =======
        :True/False: bool
            true if model (self.model) has been fitted, false if not.
        """
        return (self.model_fit is not None)

    def feature_selection(self, method="", k=None):
        """
        Feature selection/dimensionality reduction on dataset and models.
        Return the best applicable features found using the technique selected
        from method input parameter.

        Parameters
        ==========
        :method: str (default="")
            feature selection method to use. One of: selectkbest, chi2,
            variancethreshold, rfe, selectfrommodel, sequentialfeatureselector.
        :k: int or None (default=None)
            number of features to select for SelectKBest/chi2 methods.
            Defaults to 1 for selectkbest and 2 for chi2 when None.

        Returns
        =======
        :X_new: np.ndarray
            best found features using training data.

        References
        ==========
        [1] https://scikit-learn.org/stable/modules/feature_selection.html
        """
        #list of available sklearn feature selection techniques
        valid_feature_selection = ["selectkbest", "chi2", "variancethreshold", "rfe",
            "selectfrommodel", "sequentialfeatureselector"]

        #get closest valid feature selection method
        feature_matches = get_close_matches(method.lower().strip(), [item.lower().strip() \
            for item in valid_feature_selection], cutoff=0.6)
        selected_method = feature_matches[0] if feature_matches else "selectkbest"

        #apply feature selection method according to input parameter
        if selected_method == 'selectkbest':
            k_val = k if k is not None else 1
            X_new = SelectKBest(f_regression, k=k_val).fit_transform(self.X, self.Y)
        elif selected_method == "variancethreshold":
            X_new = VarianceThreshold(1).fit_transform(self.X, self.Y)
        elif selected_method == "chi2":
            # chi2 is a classification scorer and requires non-negative features; f_regression
            # is used here because this class is exclusively for regression tasks.
            # Defaults to k=2 features (wider feature set than selectkbest's default of 1).
            k_val = k if k is not None else 2
            X_new = SelectKBest(f_regression, k=k_val).fit_transform(self.X, self.Y)
        elif selected_method == "rfe":
            selector = RFE(self.model, n_features_to_select=5, step=1)
            X_new = selector.fit_transform(self.X, self.Y)
        elif selected_method == "sequentialfeatureselector":
            selector = SequentialFeatureSelector(self.model, n_features_to_select=3)
            X_new = selector.fit_transform(self.X, self.Y)
        elif selected_method == "selectfrommodel":
            selector = SelectFromModel(estimator=deepcopy(self.model))
            X_new = selector.fit_transform(self.X, self.Y)
        else:
            k_val = k if k is not None else 1
            X_new = SelectKBest(f_regression, k=k_val).fit_transform(self.X, self.Y)

        return X_new

    def cv_score(self, cv: int = 5, metric: str = 'r2', n_jobs: int = None) -> np.ndarray:
        """
        Evaluate the model using k-fold cross-validation on the full (X, Y) data.

        Unlike :meth:`train_test_split` + :meth:`fit`, this method does not
        permanently alter the model's fitted state; a deep copy of the model is
        used internally so the original :attr:`model_fit` is preserved.

        Parameters
        ==========
        :cv: int (default=5)
            Number of cross-validation folds.
        :metric: str (default='r2')
            Sklearn scoring string.  See
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :n_jobs: int (default=None)
            Number of parallel jobs.  ``None`` means 1; ``-1`` means all CPUs.

        Returns
        =======
        :scores: np.ndarray
            Array of *cv* scores, one per fold.

        Raises
        ======
        :RuntimeError
            If :meth:`train_test_split` has not been called yet (X/Y unavailable).
        :ValueError
            If *metric* is not a valid sklearn scoring string.
        """
        if self.X is None or self.Y is None:
            raise RuntimeError('X and Y must be set before calling cv_score(). '
                               'Call train_test_split() first.')

        valid_scorers = sorted(get_scorer_names())
        if metric not in valid_scorers:
            raise ValueError(
                f"Invalid scoring metric '{metric}'. "
                f"See sklearn.metrics.get_scorer_names() for valid options."
            )

        if not isinstance(cv, int) or cv < 2:
            import warnings as _warnings
            _warnings.warn(f'Invalid cv value {cv!r}; must be an int >= 2. Defaulting to 5.',
                           UserWarning, stacklevel=2)
            cv = 5

        X_values = self.X.values if isinstance(self.X, (pd.DataFrame, pd.Series)) else self.X
        Y_values = self.Y.values if isinstance(self.Y, (pd.DataFrame, pd.Series)) else self.Y
        X_values = np.asarray(X_values)
        Y_values = np.asarray(Y_values).ravel()

        scores = cross_val_score(
            deepcopy(self.model),
            X_values,
            Y_values,
            cv=cv,
            scoring=metric,
            n_jobs=n_jobs,
        )
        return scores

######################          Getters & Setters          ######################

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, val):
        self._X = val

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, val):
        self._Y = val

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def test_split(self):
        return self._test_split

    @test_split.setter
    def test_split(self, val):
        if not isinstance(val, (int, float)):
            raise TypeError(f"test_split must be a numeric value, got {type(val).__name__}.")
        if not 0 < val < 1:
            raise ValueError(f"test_split must be between 0 and 1 exclusive, got {val}.")
        self._test_split = val

    @property
    def valid_models(self):
        return self._valid_models

    @valid_models.setter
    def valid_models(self,val):
        self._valid_models = val

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self,val):
        self._parameters = val

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self,val):
        self._algorithm = val

    @property
    def model_fit(self):
        return self._model_fit

    @model_fit.setter
    def model_fit(self,val):
        self._model_fit = val

    def __str__(self):
        return (
            f"Model of type {type(self.model).__name__} using parameters {self.parameters}, "
            f"model has been fitted = {self.model_fitted()}."
        )

    def __repr__(self):
        """ Object representation of class instance. """
        return type(self.model).__name__

    def __eq__(self, other):
        """ Checking if 2 sklearn models are the same. """
        return self.model == other.model

    def __sizeof__(self):
        """ Get size of sklearn model. """
        return self.model.__sizeof__()