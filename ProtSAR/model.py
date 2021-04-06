

#########################################################################
###                         Model                                     ###
#########################################################################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, ParameterGrid, ParameterSampler, HalvingRandomSearchCV, train_test_split
from sklearn.metrics import SCORERS
from difflib import get_close_matches
import pandas as pd
import numpy as np
import inspect

from evaluate import Evaluate



class Model():
    """

    Class for building, fitting and training predictive models and to store all
    their related function and attributes.

    Parameters
    ----------
    algorithm : str
        sklearn regression algorithm to build and fit model with
    parameters : dict (default = {})
        parameters to use for building regression model, by default the models'
        default parameters are used.
    test_split : int
        size of the test data.

    Returns
    -------

    """
    def __init__(self, algorithm,parameters={}, test_split = 0.2):

        self.algorithm = algorithm
        self.parameters = parameters
        self.test_split = test_split
        self.standardScaler = StandardScaler()
        self.minMaxScaler = MinMaxScaler()

        #list of valid models available to use for this class
        self.valid_models = ['PlsRegression','RandomForestRegressor','AdaBoostRegressor',\
                            'BaggingRegressor','DecisionTreeRegressor','LinearRegression',\
                            'Lasso','SVR','KNeighborsRegressor', 'KNN']

        #get closest match of valid model from the input algorithm parameter value
        # modelMatches = get_close_matches(self.algorithm,self.valid_models, cutoff=0.4)
        modelMatches = get_close_matches(self.algorithm.lower(),[item.lower() for item in self.valid_models], cutoff=0.4)

        #if algorithm is a valid model then set it to self.algorithm else raise error
        if modelMatches!=[]:
            self.algorithm = modelMatches[0]
        else:
            raise ValueError('Input algorithm ('+ self.algorithm + ') not in available models \n\n'+ ' '.join(self.valid_models))

        #create instance of algorithm object
        self.model = self.get_model()

    def get_model(self):
        """
        Create instance of model type specifed by input 'algorithm' argument. If
        parameters input parameter = {} then default parameters are used else set
        the parameters of the model to the values specified in the parameters input
        parameter.

        Returns
        -------
        model : sklearn.model
            instantiated regression model with default or user-specified parameters.

        """
        if self.algorithm.lower() == 'plsregression':

            #get parameters of sklearn model and check that user inputted
            #  parameters are available in the model, only use those that are valid.
            model_params = set(dir(PLSRegression()))
            parameters = [i for i in model_params if i in self.parameters]

            #use default model parameters if user input parameters is empty, else
            #   use user-specified parameters.
            if parameters != {}:
                model = PLSRegression(**self.parameters)
            else:
                model = PLSRegression()

        elif self.algorithm.lower() == 'randomforestregressor':

            model_params = set(dir(RandomForestRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = RandomForestRegressor(**self.parameters)
            else:
                model = RandomForestRegressor()

        elif self.algorithm.lower() == 'adaboostregressor':

            model_params = set(dir(AdaBoostRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = AdaBoostRegressor(**self.parameters)
            else:
                model = AdaBoostRegressor()

        elif self.algorithm.lower() == 'baggingregressor':

            model_params = set(dir(BaggingRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = BaggingRegressor(**self.parameters)
            else:
                model = BaggingRegressor()

        elif self.algorithm.lower() == 'decisiontreeregressor':

            model_params = set(dir(DecisionTreeRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = DecisionTreeRegressor(**self.parameters)
            else:
                model = DecisionTreeRegressor()

        elif self.algorithm.lower() == 'linearregression':

            model_params = set(dir(LinearRegression()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = LinearRegression(**self.parameters)
            else:
                model = LinearRegression()

        elif self.algorithm.lower() == 'lasso':

            model_params = set(dir(Lasso()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = Lasso(**self.parameters)
            else:
                model = Lasso()

        elif self.algorithm.lower() == 'svr':

            model_params = set(dir(SVR()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = SVR(**self.parameters)
            else:
                model = SVR()

        elif self.algorithm.lower() == 'knn' or \
           self.algorithm.lower() == 'kneighborsregressor':
           
            model_params = set(dir(KNeighborsRegressor()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != {}:
                model = KNeighborsRegressor(**self.parameters)
            else:
                model = KNeighborsRegressor()
        else:
            raise ValueError('Input Algorithm ({}) is not valid'.format(self.algorithm))
            return None

        return model

    def train_test_split(self, X, Y, scale = True, test_size = 0.2, random_state=None):
        """
        Split the X and Y input features and labels into random train and test
        subsets. By default a 80:20 split will be used, whereby 80% of the data
        will be used for training and 20% for testing. By default the input will
        be scaled first such that the mean is removed and features scaled to unit
        variance. By default data is shuffled before the split and random is None.

        Parameters
        ----------
        X: np.ndarray
            array of feaure data.
        Y: np.ndarray
            array of observed label values.

        Returns
        -------
        self.X_train, self.X_test, self.Y_train, self.Y_test : np.ndarray
            split training and test data features and labels.

        """
        if (len(X)!=len(Y)):
            raise ValueError('X and Y input parameters must be of the same length')

        if (test_size <= 0 or test_size >=1):
            test_size = 0.2
        #
        # if isinstance(X, list) and isinstance(Y,list):
        #     assert len(x)==len(y), 'X and Y must be the same length'
        #
        # elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        #     assert X.shape[0] == Y.shape, 'X and Y must be the same length'

        #scale X
        if scale:
            X = self.standardScaler.fit_transform(X)

        #split X and Y into training and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = np.reshape(Y_train, (len(Y_train),))
        self.Y_test = np.reshape(Y_test, (len(Y_test),))

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def fit(self):
        """
        Fit model to training data and target values X and Y.

        Returns
        -------
        self.model_fit: np.ndarray
            fitted model of type self.algorithm

        """
        self.model_fit = self.model.fit(self.X_train, self.Y_train)
        return self.model_fit

    def predict(self):
        """
        Predict the target values of unseen data using the model.

        Returns
        -------
        self.model_fit.predict(self.X_test): np.ndarray
            array of predicted target values for unseen data.

        """
        return self.model_fit.predict(self.X_test)

    def save(self, save_folder):
        """
        Save fitted model to specified save_folder.

        Parameters
        ----------
        save_folder: str
            folder to save model to

        Returns
        -------

        """
        save_path = os.path.join(save_folder, 'model.pkl')

        try:
            with open(save_path, 'wb') as file:
                pickle.dump(self.model, file)
        except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
            print("Error pickling model with path: {} ".format(save_path))

    def hyperparameter_tuning(self,parameters, metric='r2', cv=5):
        """
        Hyperparamter tuning of model to find its optimal arrangment of
        parameters.

        Parameters
        ----------
        parameters: dict
            dictionary of parameter names and their values.
        metric: str (default: r2)
            scoring metric used to evaluate the performance of the cross-validated
            model on the test set.
        cv: int (default: 5)
            Determines the cross-validation splitting strategy.

        Returns
        -------
        metrics_df: pd.DataFrame
            dataframe of best results and the associated parameters from the
            hyperparameter tuning process.

        """
        #assert input parameters are of the correct type and accepted values
        assert isinstance(parameters, dict), 'Parameters argument must be of type dict'
        assert metric in sorted(SCORERS.keys()), 'Scoring must be in available Sklearn Scoring Metrics:\n{}'.format(sklearn.metrics.SCORERS.keys())
        assert cv >=5 and cv <=10, 'The number of cross-validation folds must be between 5 and 10'
        for p in list(parameters.keys()):
            assert p in (list(self.model.get_params().keys())), '{} not in available model parameters:\n{}'.format(p, list(self.model.get_params().keys()))

        #grid search of hyperparameter space for model
        model_copy = self.copy()
        grid_search = GridSearchCV(estimator=model_copy, param_grid=parameters,\
            n_jobs=-1, cv=cv, scoring=metric,error_score=0)

        #fit X and Y to best model found in grid search
        grid_result = grid_search.fit(self.X_train, self.Y_train)

        #get best grid search metics values
        mean_test = grid_result.cv_results_['mean_test_score']
        std_test = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        self.best_score = grid_result.best_score_
        self.best_params = grid_result.best_params_

        #predict values of unseen X data
        best_model_pred = grid_result.predict(self.X_test)

        #create instance of Evaluate class and calculate metrics from best model
        eval = Evaluate(self.Y_test,best_model_pred)
        all_metrics = eval.all_metrics()

        print("Best result of %f using parameters: %s" % (grid_result.best_score_, grid_result.best_params_))
        print('R2: ',eval.r2)
        print('RMSE: ',eval.rmse)
        print('MSE: ',eval.mse)
        print('MAE: ',eval.mae)

        # def halving_grid_search(self):
        #
        # # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html
        #     pass
        #
        # def randomised_grid_search(self):
        #     pass

    def copy(self):
        """
        Make a copy of the sklearn model stored in self.model instance variable.

        Returns
        -------
        model_copy : sklearn.model
            deep copy of model.

        """
        model_copy = self.model

        return model_copy

    def modelFitted(self):
        """
        Return if model has been fitted, true or false.

        Returns
        -------
        True/False : bool
            true if model (self.model) has been fitted, false if not.

        """
        return (hasattr(self, 'model_fit'))

    def get_tunable_params(self):
        pass

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
    def valid_models(self):
        return self._valid_models

    @valid_models.setter
    def valid_models(self,val):
        self._valid_models = val

    def __str__(self):

        return "Model of type {} using {} parameters, model fit = {}".format(
            type(self.model).__name__, self.parameters, self.modelFitted())

    def __repr__(self):

        return type(self.model).__name__
        # return "Model of type {} using {} parameters, model fit = {}".format(
        #     type(self.model).__name__, self.parameters, self.modelFitted())

    def __sizeof__(self):

        return self.model.__sizeof__()



# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report model performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#





#Add below methods to classes??
# ['__abstractmethods__',
#  '__class__',
#  '__delattr__',
#  '__dict__',
#  '__dir__',
#  '__doc__',
#  '__eq__',
#  '__format__',
#  '__ge__',
#  '__getattribute__',
#  '__getstate__',
#  '__gt__',
#  '__hash__',
#  '__init__',
#  '__init_subclass__',
#  '__le__',
#  '__lt__',
#  '__module__',
#  '__ne__',
#  '__new__',
#  '__reduce__',
#  '__reduce_ex__',
#  '__repr__',
#  '__setattr__',
#  '__setstate__',
#  '__sizeof__',
#  '__str__',
#  '__subclasshook__',
#  '__weakref__',
