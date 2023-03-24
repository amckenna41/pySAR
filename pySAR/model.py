################################################################################
#################                    Model                     #################
################################################################################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import SCORERS
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, SelectFromModel, SequentialFeatureSelector
from difflib import get_close_matches
from copy import deepcopy
import numpy as np
import os
import pickle
import pandas as pd

from .evaluate import Evaluate

class Model():
    """
    Class for building, fitting and training a various range of predictive
    regression models and all their related methods and attributes. The 
    model class supports the following regression algoriths: PLS Regression,
    Random Forest, AdaBoost, Bagging, Decision Tree, GradientBoost, Linear
    Regression, Lasso, Ridge, Support Vector Regression, Stochastic Gradient
    Descent and K Nearest Neighbours (KNN).

    Attributes
    ----------
    :X : np.ndarray
        training data.
    :Y : np.ndarray
        training data labels.
    :algorithm : str
        sklearn regression algorithm to build and fit model with. Value can be 
        an approximate representation of model name, for example: 'plsreg' will 
        initialiase an instance of the PLSRegression model etc. Available 
        algorithms listed above.
    :parameters : dict (default={})
        parameters to use for specific sklearn model when building regression 
        model, by default it is set to {}, meaning all of the models' default 
        parameters are used. Refer to sci-kit learn for full list of available 
        input parameters for each model: https://scikit-learn.org/stable/index.html.
    :test_split : float (default=0.2)
        proportion of the test data to use for building model, default of 0.2 is 
        reccomended.

    Methods
    -------
    get_model():
        build model using inputtted parameters.
    train_test_split(scale = True, test_split = 0.2, random_state=None, shuffle=True):
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
        in model.
    """
    def __init__(self, X, Y, algorithm, parameters={}, test_split=0.2):

        self.algorithm = algorithm
        self.test_split = test_split
        self.X = X
        self.Y = Y

        #if no model parameters input, then set to {} meaning default models' parameters are used
        if (parameters == [] or parameters == ""):
            self.parameters = {}
        else:
            self.parameters = parameters

        #list of valid models available to use for this class
        self.valid_models = ['plsregression', 'randomforestregressor', 'adaboostregressor',\
                            'baggingregressor', 'decisiontreeregressor', 'gbr', 'gradientboostingregressor', 
                            'linearregression', 'lasso', 'ridge', 'svr', 'supportvectorregression', 'sgd',
                            'stochasticgradientdescent', 'kneighborsregressor', 'knearestneighbors', 'knn']

        #raise error if algorithm parameter isnt string type
        if not(isinstance(self.algorithm, str)):
            raise TypeError("Algorithm input parameter must be a string.")

        #get closest match of valid model from the input algorithm parameter value using difflib
        model_matches = get_close_matches(self.algorithm.lower().strip(),[item.lower().strip() \
            for item in self.valid_models], cutoff=0.5)

        #if algorithm is a valid model then set it to self.algorithm, else raise error
        if (model_matches!=[]):
            self.algorithm = model_matches[0]
        else:
            raise ValueError('Input algorithm {} not found in list of available valid models {}.'.format(
                    self.algorithm, self.valid_models))

        #create instance of algorithm object using its sklearn constructor
        self.model = self.get_model()

        #set model_fit to None, specifies if model has been fit or not
        self.model_fit = None

    def get_model(self):
        """
        Create instance of model type specifed by input 'algorithm' argument. If
        input 'parameters' = {} then default parameters of sklearn model are used, else set
        the parameters of the model to the values specified in the 'parameters' input.

        Parameters
        ----------
        None

        Returns
        -------
        :model : sklearn.model
            instantiated regression model with default or user-specified parameters.
        """
        #use if/elif statements to get matching model specified by user in algorithm attribute
        if (self.algorithm.lower().strip() == 'plsregression'):

            #get parameters of sklearn model and check that user inputted
            #parameters are available in the model, only use those that are valid.
            model_params = set(dir(PLSRegression()))
            parameters = [i for i in model_params if i in self.parameters]
            
            #use default model parameters if user input parameters is empty {}, else
            #use user-specified parameters
            if (parameters != {} or parameters != []):
                model = PLSRegression(**self.parameters)
            else:
                model = PLSRegression()

        elif (self.algorithm.lower().strip() == 'randomforestregressor'):

            model_params = set(dir(RandomForestRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = RandomForestRegressor(**self.parameters)
            else:
                model = RandomForestRegressor()

        elif (self.algorithm.lower().strip() == 'adaboostregressor'):

            model_params = set(dir(AdaBoostRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = AdaBoostRegressor(**self.parameters)
            else:
                model = AdaBoostRegressor()

        elif (self.algorithm.lower().strip() == 'baggingregressor'):

            model_params = set(dir(BaggingRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = BaggingRegressor(**self.parameters)
            else:
                model = BaggingRegressor()

        elif (self.algorithm.lower().strip() == 'decisiontreeregressor'):

            model_params = set(dir(DecisionTreeRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = DecisionTreeRegressor(**self.parameters)
            else:
                model = DecisionTreeRegressor()

        elif (self.algorithm.lower().strip() == 'linearregression'):

            model_params = set(dir(LinearRegression()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = LinearRegression(**self.parameters)
            else:
                model = LinearRegression()

        elif (self.algorithm.lower().strip() == 'lasso'):

            model_params = set(dir(Lasso()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = Lasso(**self.parameters)
            else:
                model = Lasso()

        elif (self.algorithm.lower().strip() == 'ridge'):

            model_params = set(dir(Ridge()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = Ridge(**self.parameters)
            else:
                model = Ridge()

        elif (self.algorithm.lower().strip() == 'sgd' or \
            self.algorithm.lower().strip() == 'stochasticgradientdescent'):

            model_params = set(dir(SGDRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = SGDRegressor(**self.parameters)
            else:
                model = SGDRegressor()

        elif (self.algorithm.lower().strip() == 'gbr' or \
            self.algorithm.lower().strip() == 'gradientboost' or \
            self.algorithm.lower().strip() == 'gradientboostingregressor'):

            model_params = set(dir(GradientBoostingRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = GradientBoostingRegressor(**self.parameters)
            else:
                model = GradientBoostingRegressor()

        elif (self.algorithm.lower().strip() == 'svr' or \
            self.algorithm.lower().strip() == 'supportvectorregression'):

            model_params = set(dir(SVR()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = SVR(**self.parameters)
            else:
                model = SVR()

        elif (self.algorithm.lower().strip() == 'knn' or \
           self.algorithm.lower().strip() == 'kneighborsregressor' or \
           self.algorithm.lower().strip() == 'knearestneighbors'):

            model_params = set(dir(KNeighborsRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if (parameters != {} or parameters != []):
                model = KNeighborsRegressor(**self.parameters)
            else:
                model = KNeighborsRegressor()
        #no matching valid algorithm/model found
        else:
            raise ValueError('Input Algorithm ({}) not found in available valid models: {}'.
                format(self.algorithm, self.valid_models))

        return model

    def train_test_split(self, test_split=0.2, scale=True, random_state=None, shuffle=True):
        """
        Split the X and Y input features and labels into random train and test
        subsets. By default a 80:20 split will be used, whereby 80% of the data
        will be used for training and 20% for testing. By default the input will
        be scaled first such that the mean is removed and features scaled to unit
        variance. By default data is shuffled before the split and random state is None.

        Parameters
        ----------
        :scale : bool (default=True)
            if true then scale the features such that they are standardised.
        :test_split : float (default=0.2)
            proportion of the total dataset to use for testing.
        :random_state : float (default=None)
            Controls the shuffling applied to the data before applying the split.
            Popular integer random seeds are 0 and 42, None by default.
        :shuffle : bool (default=True)
            Whether or not to shuffle the data before splitting.

        Returns
        -------
        :self.X_train, self.X_test, self.Y_train, self.Y_test : np.ndarray
            splitted training and test data features and labels.
        """
        #validate that X and Y arrays are of the same size
        if (len(self.X) != len(self.Y)):
            raise ValueError('X and Y input parameters must be of the same length - X: {}, Y: {}.'.
                format(len(self.X), len(self.Y)))

        #reshape input arrays to 2D arrays
        if (self.X.ndim != 2):
            self.X = np.reshape(self.X, (-1,1))
        if (self.Y.ndim != 2):
            if (isinstance(self.Y, pd.Series)):
                self.Y = np.reshape(self.Y.values, (-1,1))
            else:
                self.Y = np.reshape(self.Y, (-1,1))

        #if invalid test size input then set to default 0.2
        if (test_split <= 0 or test_split >=1):
            test_split = 0.2

        #setting test_split attribute
        self.test_split = test_split     

        #scale training data X, if scale=True
        if (scale):
            self.X = StandardScaler().fit_transform(self.X)

        #split X and Y into training and test data
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y,
            test_size=test_split, random_state=random_state, shuffle=shuffle)

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
        ----------
        None

        Returns
        -------
        :self.model_fit : np.ndarray
            fitted sklearn model of type specified by algorithm attribute.
        """
        self.model_fit = self.model.fit(self.X_train, self.Y_train)
        return self.model_fit

    def predict(self):
        """
        Predict the target values of unseen test data using the model.

        Parameters
        ----------
        None

        Returns
        -------
        :self.model_fit.predict(self.X_test) : np.ndarray
            array of predicted target values for unseen test data.
        """
        return self.model_fit.predict(self.X_test)

    def save(self, save_folder, model_name="model.pkl"):
        """
        Save fitted model to specified save_folder.

        Parameters
        ----------
        :save_folder : str
            folder to save model to.
        
        Returns
        -------
        None
        """
        #append pickle file extension if not present in filename
        if (os.path.splitext(model_name)[1] == "" and 
            os.path.splitext(model_name)[1] != "pkl"):
            model_name = model_name + ".pkl"
        
        #set save path to folder + filename
        save_path = os.path.join(save_folder, model_name)
        
        #save model in pickle format
        try:
            with open(save_path, 'wb') as file:
                pickle.dump(self.model, file)
        except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
            print("Error pickling model with path: {}.".format(save_path))

    def hyperparameter_tuning(self, param_grid={}, metric='r2', cv=5, n_jobs=None, verbose=2):
        """
        Hyperparamter tuning of model to find its optimal arrangment of parameters
        using a Grid Search.

        Parameters
        ----------
        :param_grid : dict (default={})
            dictionary/grid of selected models' parameter and the potential values of each
            that you want to tune.
        :metric : str (default=r2)
            scoring metric used to evaluate the performance of the cross-validated
            model on the test set, R2 by default. List of available scoring metrics
            can be found in documentation:
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :cv : int (default=5)
            Determines the cross-validation splitting strategy.
        :n_jobs : int (default=None)
            Number of jobs to run in parallel. None means 1 job.
        :verbose: int (default=2)
            verbosity of output during tuning process. The values and what they mean 
            for this parameter can be found on the documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

        Returns
        -------
        None
        """
        #input 'param_grid' parameter must be a dict, if not raise error
        if not (isinstance(param_grid, dict)):
            raise TypeError('param_grid argument must be of type dict, got type {}.'.format(type(parameters)))

        #input metric must be in available scoring metrics, if not raise error
        if (metric not in sorted(SCORERS.keys())):
            raise UndefinedMetricWarning('Invalid scoring metric, {} not in available Sklearn Scoring Metrics: {}.\n'\
                .format(metric, SCORERS.keys()))

        #cv must be of type int and be between 5 and 10, if not then default of 5 is used
        if not ((isinstance(cv, int)) or (cv<5 or cv>10)):
            cv = 5

        #iterate through all parameter names to check if they are correct for model,
        #if parameter not found in model params then delete from dictionary
        for p in list(param_grid.keys()):
            if (p not in (list(self.model.get_params().keys()))):
                del param_grid[p]

        #create deep copy of model
        model_copy = deepcopy(self.model)

        #grid search of hyperparameter space for model
        grid_search = GridSearchCV(estimator=model_copy, param_grid=param_grid, \
            cv=cv, scoring=metric, n_jobs=n_jobs, verbose=verbose, error_score=0)

        #fit X and Y to best model found in grid search
        grid_result = grid_search.fit(self.X_train, self.Y_train)

        #get best grid search metics values
        mean_test = grid_result.cv_results_['mean_test_score']
        std_test = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        #predict values of unseen test data using best found model
        best_model_pred = grid_result.predict(self.X_test)
        
        #create instance of Evaluate class and calculate metrics from best model
        eval = Evaluate(self.Y_test,best_model_pred)
        
        #print out results of grid search
        print('\n############################################################')
        print('################### Hyperparamter Results ###################')
        print('#############################################################\n')

        print('######################### Parameters ########################\n')
        print('# Best Params -> {}'.format(grid_result.best_params_))
        print('# Model Type -> {}'.format(repr(self)))
        print('# Scoring Metric -> {}'.format(metric))
        print('# Number of CV folds -> {}'.format(cv))
        print('# Test Split -> {}'.format(self.test_split))

        print('######################### Metrics ###########################\n')
        print('# Best Score (R2) -> {}'.format(grid_result.best_score_))
        print('# RMSE -> {} '.format(eval.rmse))
        print('# MSE -> {} '.format(eval.mse))
        print('# MAE -> {}'.format(eval.mae))
        print('# RPD -> {}'.format(eval.rpd))
        print('# Explained Variance -> {}\n'.format(eval.explained_var))
        print('##############################################################')
        
        self.grid_result = grid_result

    def model_fitted(self):
        """
        Return if model has been fitted, true or false.

        Parameters
        ----------
        None
        
        Returns
        -------
        :True/False : bool
            true if model (self.model) has been fitted, false if not.
        """
        return (self.model_fit != None)

    def feature_selection(self, method=""):
        """
        Feature selection/dimensionality reduction on dataset and models.
        Return the best applicable features found using the technique selected
        from method input parameter.

        Parameters
        ----------
        :method : str (default="")
            feature selection method to use.

        Returns
        -------
        :X_new : np.ndarray
            best found features from training data.
        
        References
        ----------
        [1] https://scikit-learn.org/stable/modules/feature_selection.html
        """
        #list of available sklearn feature selection techniques
        valid_feature_selection = ["selectkbest", "chi2", "variancethreshold", "rfe", 
            "selectfrommodel", "sequentialfeatureselector"]

        #get closest valid feature selection method
        feature_matches = get_close_matches(method.lower().strip(), [item.lower().strip() \
            for item in valid_feature_selection], cutoff=0.5)

        #apply feature selection method according to input parameter
        if (feature_matches == 'selectkbest'):
            X_new = SelectKBest(chi2, k=1).fit_transform(self.X, self.Y)
        elif (feature_matches == "variancethreshold"):
            X_new = VarianceThreshold(1).fit_transform(self.X, self.Y)
        elif (feature_matches == "chi2"):
            X_new = chi2().fit_transform(self.X, self.Y)
        elif (feature_matches == "rfe"):
            pass
        elif (feature_matches == "sequentialfeatureselector"):
            pass
        elif (feature_matches == "selectfrommodel"):
            pass
        else:
            X_new = SelectKBest(chi2, k=2).fit_transform(self.X, self.Y)

        return X_new
        
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

    @property
    def valid_models(self):
        return self._valid_models

    @valid_models.setter
    def valid_models(self,val):
        self._valid_models = val

    def __str__(self):
        return "Model of type {} using parameters {}, model has been fitted = {}.".format(
            type(self.model).__name__, self.parameters, self.model_fitted())

    def __repr__(self):
        """ Object representation of class instance """
        return type(self.model).__name__

    def __eq__(self, other):
        """ Checking if 2 sklearn models are the same """
        return self.model == other.model

    def __sizeof__(self):
        """ Get size of sklearn model """
        return self.model.__sizeof__()