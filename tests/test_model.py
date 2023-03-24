################################################################################
#################             Model Module Tests               #################
################################################################################

import unittest
import sklearn
import numpy as np
import shutil
#### Suppress Sklearn warnings ####
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.exceptions import UndefinedMetricWarning

from pySAR.model import *

class ModelTests(unittest.TestCase):
    """
    Test suite for testing model module and functionality 
    in pySAR package. 

    Test Cases
    ----------
    test_model:
        testing correct overall Model class and module functionality.
    test_model_input_closeness:
        testing correct input closeness functionality.
    test_train_test_split:
        testing correct train-test split functionality.
    test_predict:
        testing correct predict functionality.
    test_save:
        testing correct model saving functionality.
    test_parameters:
        testing correct parameters functionality.
    test_hyperparamter_tuning:
        testing correct hyperparameter tuning functionality.
    test_feature_selection:
        testing correct feature selection functionality.
    """
    def setUp(self):
        """ Create dummy data. """
        self.dummy_X = np.random.ranf(size=100)
        self.dummy_X_2D = np.random.ranf((100, 50)) #50 sequences 
        self.dummy_X_2 = np.random.ranf(size=50)
        self.dummy_Y = np.random.randint(10, size=100)
        self.dummy_Y_2 = np.random.randint(20, size=50)
        self.dummy_Y_2D = np.random.ranf((50,1)) #50 sequences 

        self.test_folder = os.path.join('tests', 'test_model_output') #test model folder
        os.mkdir(self.test_folder)

    def test_model(self):
        """ Test Case to check each model type & its associated parameters & attributes. """
        test_models = ['PLSRegression', 'RandomForestRegressor', 'AdaBoostRegressor',\
                            'BaggingRegressor', 'DecisionTreeRegressor', 'LinearRegression',\
                            'Lasso', 'SVR', 'KNeighborsRegressor', 'GradientBoostingRegressor', 'Ridge']

        #iterate through all available algorithms and test them
        for test_mod in range(0, len(test_models)):
   
            model = Model(self.dummy_X, self.dummy_Y, test_models[test_mod])
#1.)
            #checking model object is of the correct sklearn model datatype
            self.assertEqual(type(model.model).__name__, test_models[test_mod],
                'Model type is not correct, expected {}, got {}.'.format(
                    test_models[test_mod], type(model.model).__name__))
#2.)        #assert that model has not been fitted
            self.assertFalse(model.model_fitted(), 'Model should not be fitted on initialisation')
#3.)        #verify that parameters input param = {} meaning the default params for the model are used
            self.assertEqual(model.parameters, {},
                'Default Parameters attribute should be an empty dict, but got {}.'.format(model.parameters))
#4.)        #verify test split attribute is 0.2, its default value
            self.assertEqual(model.test_split, 0.2,
                'Default test split attribute should be 0.2, but got {}.'.format(model.test_split))
#5.)        #verify that input model type is a valid model for the class
            self.assertTrue(model.algorithm in [item.lower() for item in model.valid_models],
                'Input algorithm {} not in available algorithms:\n {}'.format(model.algorithm, model.valid_models))
#6.)        #verify repr representation of model object is correct
            self.assertEqual(repr(model), test_models[test_mod],
                'Repr function expected to be {}, but got {}.'.format(test_models[test_mod], repr(model)))
#7.)        #verify algorithm is a regression
            self.assertTrue(sklearn.base.is_regressor(model.model),
                'Model type should be a sklearn regressor.')
#8.)           
            if (self.dummy_X.ndim == 1):
                model = Model(self.dummy_X, self.dummy_Y, 'plsreg', parameters={'n_components': 1})
            model.train_test_split()
            model.fit()
            self.assertTrue(model.model_fitted(), 'Model has not been fitted')

    def test_model_input_closeness(self):
        """ Test case for testing the algorithm closeness function used to get the
            closest available algorithm to the algorithm input into the class. """
#1.)
        model = Model(self.dummy_X, self.dummy_Y, 'plsreg')
        self.assertEqual(model.algorithm, "plsregression", 
            "Expected algorithm to be plsregression, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "PLSRegression",
            "Expected representation of model object to be PLSRegression, got {}".format(repr(model)))
#2.)
        model = Model(self.dummy_X, self.dummy_Y, 'randomfor')
        self.assertEqual(model.algorithm, "randomforestregressor",
            "Expected algorithm to be randomforestregressor, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "RandomForestRegressor",
            "Expected representation of model object to be RandomForestRegressor, got {}".format(repr(model)))
#3.)
        model = Model(self.dummy_X, self.dummy_Y, 'adaboo')
        self.assertEqual(model.algorithm, "adaboostregressor",
            "Expected algorithm to be adaboostregressor, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "AdaBoostRegressor",
            "Expected representation of model object to be AdaBoostRegressor, got {}".format(repr(model)))
#4.)
        model = Model(self.dummy_X, self.dummy_Y, 'bagging')
        self.assertEqual(model.algorithm, "baggingregressor",
            "Expected algorithm to be baggingregressor, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "BaggingRegressor")
#5.)
        model = Model(self.dummy_X, self.dummy_Y, 'decisiontree')
        self.assertEqual(model.algorithm, "decisiontreeregressor",
            "Expected algorithm to be decisiontreeregressor, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "DecisionTreeRegressor",
            "Expected representation of model object to be DecisionTreeRegressor, got {}".format(repr(model)))
#6.)
        model = Model(self.dummy_X, self.dummy_Y, 'linear')
        self.assertEqual(model.algorithm, "linearregression",
            "Expected algorithm to be linearregression, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "LinearRegression",
            "Expected representation of model object to be LinearRegression, got {}".format(repr(model)))
#7.)
        model = Model(self.dummy_X, self.dummy_Y, 'lass')
        self.assertEqual(model.algorithm, "lasso",
            "Expected algorithm to be lasso, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "Lasso",
            "Expected representation of model object to be Lasso, got {}".format(repr(model)))
#8.)
        model = Model(self.dummy_X, self.dummy_Y, 'kneighbors')
        self.assertEqual(model.algorithm, "knearestneighbors",
            "Expected algorithm to be knearestneighbors, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "KNeighborsRegressor",
            "Expected representation of model object to be KNeighborsRegressor, got {}".format(repr(model)))
#9.)
        model = Model(self.dummy_X, self.dummy_Y, 'sv')
        self.assertEqual(model.algorithm, "svr",
            "Expected algorithm to be svr, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "SVR",
            "Expected representation of model object to be SVR, got {}".format(repr(model)))
#10.)
        model = Model(self.dummy_X, self.dummy_Y, 'rid')
        self.assertEqual(model.algorithm, "ridge",
            "Expected algorithm to be ridge, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "Ridge",
            "Expected representation of model object to be Ridge, got {}".format(repr(model)))
#11.)
        model = Model(self.dummy_X, self.dummy_Y, 'gbr')
        self.assertEqual(model.algorithm, "gbr",
            "Expected algorithm to be gbr, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "GradientBoostingRegressor",
            "Expected representation of model object to be GradientBoostingRegressor, got {}".format(repr(model)))
#12.)
        model = Model(self.dummy_X, self.dummy_Y, 'sg')
        self.assertEqual(model.algorithm, "sgd",
            "Expected algorithm to be sgd, got {}.".format(model.algorithm))
        self.assertEqual(repr(model), "SGDRegressor",
            "Expected representation of model object to be SGDRegressor, got {}".format(repr(model)))
#13.)
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            bad_model = Model(self.dummy_X, self.dummy_Y, 'abcdefg')
            bad_model = Model(self.dummy_X, self.dummy_Y, 'notamodel')
            bad_model = Model(self.dummy_X, self.dummy_Y, '123')
            bad_model = Model(self.dummy_X, self.dummy_Y, 'blahblahblah')
#14.)
        with self.assertRaises(TypeError, msg="Type Error raised, input must be of type string."):
            bad_model = Model(self.dummy_X, self.dummy_Y, 12345)
            bad_model = Model(self.dummy_X, self.dummy_Y, 5.60)
            bad_model = Model(self.dummy_X, self.dummy_Y, False)

    def test_train_test_split(self):
        """ Testing splitting up dataset into training and test data. """
#1.)
        model = Model(self.dummy_X_2D, self.dummy_Y, 'plsreg')
        X_train, X_test, Y_train, Y_test = model.train_test_split()

        self.assertTrue(len(X_train) == 80, "Expected 80 rows in training data, got {}.".format(len(X_train)))
        self.assertTrue(len(Y_train) == 80, "Expected 80 rows in training data labels, got {}.".format(len(Y_train)))
        self.assertTrue(len(X_test) == 20, "Expected 20 rows in test data, got {}.".format(len(X_test)))
        self.assertTrue(len(Y_test) == 20, "Expected 20 rows in test data labels, got {}.".format(len(Y_test)))

        self.assertIsInstance(X_train, np.ndarray, "X_train training data expected to be a numpy array.")
        self.assertIsInstance(Y_train, np.ndarray, "Y_train training data labels expected to be a numpy array.")
        self.assertIsInstance(X_test, np.ndarray, "X_test test data expected to be a numpy array.")
        self.assertIsInstance(Y_test, np.ndarray, "Y_test test data labels expected to be a numpy array.")
#2.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'adaboost')
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.5)

        self.assertTrue(len(X_train) == 25, "Expected 25 rows in training data, got {}.".format(len(X_train)))
        self.assertTrue(len(Y_train) == 25, "Expected 25 rows in training data labels, got {}.".format(len(Y_train)))
        self.assertTrue(len(X_test) == 25, "Expected 25 rows in test data, got {}.".format(len(X_test)))
        self.assertTrue(len(Y_test) == 25, "Expected 25 rows in test data labels, got {}.".format(len(Y_test)))

        self.assertIsInstance(X_train, np.ndarray, "X_train training data expected to be a numpy array.")
        self.assertIsInstance(Y_train, np.ndarray, "Y_train training data labels expected to be a numpy array.")
        self.assertIsInstance(X_test, np.ndarray, "X_test test data expected to be a numpy array.")
        self.assertIsInstance(Y_test, np.ndarray, "Y_test test data labels expected to be a numpy array.")
#3.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'bagging')
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=1234) #if test_split <0 or >1 then use 0.2 default

        self.assertTrue(len(X_train) == 40, "Expected 40 rows in training data, got {}.".format(len(X_train)))
        self.assertTrue(len(Y_train) == 40, "Expected 40 rows in training data labels, got {}.".format(len(Y_train)))
        self.assertTrue(len(X_test) == 10, "Expected 10 rows in test data, got {}.".format(len(X_test)))
        self.assertTrue(len(Y_test) == 10, "Expected 10 rows in test data labels, got {}.".format(len(Y_test)))

        self.assertIsInstance(X_train, np.ndarray, "X_train training data expected to be a numpy array.")
        self.assertIsInstance(Y_train, np.ndarray, "Y_train training data labels expected to be a numpy array.")
        self.assertIsInstance(X_test, np.ndarray, "X_test test data expected to be a numpy array.")
        self.assertIsInstance(Y_test, np.ndarray, "Y_test test data labels expected to be a numpy array.")
#4.)
        model = Model(self.dummy_X_2, self.dummy_Y, 'plsreg')
        with self.assertRaises(ValueError, msg='Value Error raised, invalid test_split type input.'):
            X_train, X_test, Y_train, Y_test = model.train_test_split(test_split="ABCD")

    def test_predict(self):
        """ Testing the prediction of values for unseen sequences using the trained model. """
#1.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'knn')
        X_train, X_test, Y_train, Y_test = model.train_test_split()
        model.fit()
        Y_pred = model.predict()

        self.assertIsInstance(Y_pred, np.ndarray, "Predicted output labels expected to be numpy array.")
        self.assertEqual(len(Y_pred), len(Y_test), "Predicted output labels and test data should be same length.")
#2.)
        model = Model(self.dummy_X, self.dummy_Y, 'plsreg', parameters={"n_components": 1})
        X_train, X_test, Y_train, Y_test = model.train_test_split()
        model.fit()
        Y_pred = model.predict()

        self.assertIsInstance(Y_pred, np.ndarray, "Predicted output labels expected to be numpy array.")
        self.assertEqual(len(Y_pred), len(Y_test), "Predicted output labels and test data should be same length.")

    def test_save(self):
        """ Testing save function that saves pickle of model to specified folder. """
#1.)
        model = Model(self.dummy_X, self.dummy_Y, 'gbr')
        X_train, X_test, Y_train, Y_test = model.train_test_split()
        model.fit()
        model.save(self.test_folder, model_name='test_model.pkl')
        
        self.assertTrue(os.path.isfile(os.path.join(self.test_folder, 'test_model.pkl')), 
            "Expected model pickle to be saved to test folder.")
#2.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'lasso')
        X_train, X_test, Y_train, Y_test = model.train_test_split()
        model.fit()
        model.save(self.test_folder, 'test_model2.pkl')

        self.assertTrue(os.path.isfile(os.path.join(self.test_folder, 'test_model2.pkl')), 
            "Expected model pickle to be saved to test folder.")

    def test_parameters(self):
        """ Testing parameters of Model class for specified algorithm match that of the sklearn 
            models' parameters. """
#1.)
        #create instance of PLS model using Model class & creating instance
        #using SKlearn libary, comparing if the parameters of both instances are equal
        pls_parameters = {"n_components": 20, "scale": False, "max_iter": 200}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="PlsRegression", parameters=pls_parameters)
        pls_model = PLSRegression(n_components=20, scale="svd", max_iter=200)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(pls_model.get_params()), 
                "Parameter {} should be in list of parameters:\n{}".format(k, list(pls_model.get_params())))
#2.)
        rf_parameters = {"n_estimators": 200, "max_depth": 50, "min_samples_split": 10}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="RandomForest", parameters=rf_parameters)
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=50, min_samples_split=10)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(rf_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(rf_model.get_params())))
#3.)
        knn_parameters = {"n_neighbors": 10, "weights": "distance", "algorithm": "ball_tree"}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="KNN", parameters=knn_parameters)
        knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm="kd_tree")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(knn_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(knn_model.get_params())))
#4.)
        svr_parameters = {"kernel": "poly", "degree": 5, "coef0": 1}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="SVR",parameters=svr_parameters)
        svr_model = SVR(kernel='poly', degree=5, coef0=1)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(svr_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(svr_model.get_params())))
#5.)
        ada_parameters = {"n_estimators": 150, "learning_rate": 1.2, "loss": "square"}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="AdaBoost", parameters=ada_parameters)
        ada_model = AdaBoostRegressor(n_estimators=150, learning_rate=1.2, loss="square")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(ada_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(ada_model.get_params())))
#6.)
        bagging_parameters = {"n_estimators": 50, "max_samples": 1.5, "max_features": 2}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="Bagging", parameters=bagging_parameters)
        bagging_model = BaggingRegressor(n_estimators=50, max_samples=1.5, max_features="square")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(bagging_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(bagging_model.get_params())))
#7.)
        lasso_parameters = {"alpha": 1.5, "max_iter": 500, "tol": 0.004}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="lasso", parameters=lasso_parameters)
        lasso_model = Lasso(alpha=1.5, max_iter=500, tol=0.004)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(lasso_model.get_params()),
                "Parameter {} should be in list of parameters:\n{}".format(k, list(lasso_model.get_params())))

    def test_hyperparamter_tuning(self):
        """ Testing hyperparamter tuning function. """
#1.)
        model = Model(self.dummy_X, self.dummy_Y, algorithm="adaboost")
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.2)
        model.fit()
        param_grid = {'n_estimators': [50,100,150], 'learning_rate': [0.5,0.75,1], 'loss': ['linear','exponential']}
        model.hyperparameter_tuning(metric="neg_root_mean_squared_error", param_grid=param_grid, verbose=0, cv=10)
        
        self.assertEqual(str(type(model.grid_result)), "<class 'sklearn.model_selection._search.GridSearchCV'>", 
            "Expected grid result to be of type sklearn.model_selection._search, got {}.".format(type(model.grid_result)))
        self.assertEqual(model.grid_result.cv, 10, 
            "Expected there to be 10 cross-validation folds, got {}.".format(model.grid_result.cv))
        self.assertEqual(model.grid_result.error_score, 0, 
            "Expected the error score to be 0, got {}.".format(model.grid_result.error_score))
        self.assertEqual(model.grid_result.scoring, 'neg_root_mean_squared_error', 
            "Expected the scoring metric to be neg_root_mean_squared_error, got {}.".format(model.grid_result.scoring))
        self.assertEqual(model.grid_result.verbose, 0, 
            "Expected the verbosity to be 0, got {}.".format(model.grid_result.verbose))
        self.assertEqual(model.grid_result.param_grid, param_grid, 
            "Expected the parameter grid to be an empty dict, got {}.".format(model.grid_result.param_grid))
        self.assertEqual(str(type(model.grid_result.estimator)), "<class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>", 
            "Expected the estimator to be an AdaBoostRegressor, got {}.".format(str(type(model.grid_result.estimator))))
#2.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="randomforest")
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.2)
        model.fit()
        param_grid = {'max_depth': [2,3,4], 'n_estimators': [100,200,250], 'criterion': ['squared_error', 'absolute_error']}
        model.hyperparameter_tuning(param_grid=param_grid, verbose=0, cv=5)
        
        self.assertEqual(str(type(model.grid_result)), "<class 'sklearn.model_selection._search.GridSearchCV'>", 
            "Expected grid result to be of type sklearn.model_selection._search, got {}.".format(type(model.grid_result)))
        self.assertEqual(model.grid_result.cv, 5, 
            "Expected there to be 5 cross-validation folds, got {}.".format(model.grid_result.cv))
        self.assertEqual(model.grid_result.error_score, 0, 
            "Expected the error score to be 0, got {}.".format(model.grid_result.error_score))
        self.assertEqual(model.grid_result.scoring, 'r2', 
            "Expected the scoring metric to be r2, got {}.".format(model.grid_result.scoring))
        self.assertEqual(model.grid_result.verbose, 0, 
            "Expected the verbosity to be 0, got {}.".format(model.grid_result.verbose))
        self.assertEqual(model.grid_result.param_grid, param_grid, 
            "Expected the parameter grid to be an empty dict, got {}.".format(model.grid_result.param_grid))
        self.assertEqual(str(type(model.grid_result.estimator)), "<class 'sklearn.ensemble._forest.RandomForestRegressor'>", 
            "Expected the estimator to be an RandomForestRegressor, got {}.".format(str(type(model.grid_result.estimator))))
#3.)
        with self.assertRaises(UndefinedMetricWarning):
            model.hyperparameter_tuning(metric="invalid_metric", verbose=0, cv=10)
#4.)
        with self.assertRaises(UndefinedMetricWarning):
            model.hyperparameter_tuning(metric="R2", verbose=0, cv=5)
#5.)
        with self.assertRaises(TypeError):
            model.hyperparameter_tuning(parameters='wrongType')

    def test_feature_selection(self):
        """ Testing Feature Selection function. """ 
        pass
    
    def tearDown(self):
        """ Delete any temp data used for tests. """
        del self.dummy_X
        del self.dummy_X_2
        del self.dummy_X_2D
        del self.dummy_Y
        del self.dummy_Y_2
        del self.dummy_Y_2D
        shutil.rmtree(self.test_folder, ignore_errors=False, onerror=None)

if __name__ == '__main__':
    #run all model tests
    unittest.main(verbosity=2)
