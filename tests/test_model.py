################################################################################
#################             Model Module Tests               #################
################################################################################

import unittest
import sklearn
import numpy as np

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
    test_parameters:
        testing correct parameters functionality.
    test_copy:
        testing correct copy functionality.
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

    def test_model(self):
        """ Test Case to check each model type & its associated parameters & attributes. """
        test_models = ['PLSRegression','RandomForestRegressor','AdaBoostRegressor',\
                            'BaggingRegressor','DecisionTreeRegressor','LinearRegression',\
                            'Lasso','SVR','KNeighborsRegressor','GradientBoostingRegressor','Ridge']

        #iterate through all available algorithms and test them
        for test_mod in range(0, len(test_models)):
   
            model = Model(test_models[test_mod])
#1.)
            #checking model object is of the correct sklearn model datatype
            self.assertEqual(type(model.model).__name__, test_models[test_mod],
                'Model type is not correct, wanted {}, got {} '.format(
                    test_models[test_mod], type(model.model).__name__
                ))
#2.)        #assert that model has not been fitted
            self.assertFalse(model.model_fitted(), 'Model should not be fitted \
                on initialisation')
#3.)        #verify that parameters input param = {} meaning the default params for the model are used
            self.assertEqual(model.parameters,{},
                'Default Parameters attribute should be an empty dict, but got {}'.format(model.parameters))
#4.)        #verify test split attribute is = 0.2, its default value
            self.assertEqual(model.test_split, 0.2,
                'Default test split attribute should be 0.2, but got {}'.format(model.test_split))
#5.)        #verify that input model type is a valid model for the class
            self.assertTrue(model.algorithm in [item.lower() for item in model.valid_models],
                'Input algorithm {} not in available algorithms: {}'.format(model.algorithm, model.valid_models))
#6.)        #verify repr representation of model object is correct
            self.assertEqual(repr(model), test_models[test_mod],
                'Repr function should return {}, but got {}'.format(test_models[test_mod], repr(model)))
#7.)        #verify algorithm is a regression
            self.assertTrue(sklearn.base.is_regressor(model.model),
                'Model type should be a sklearn regressor.')
#8.)        
            #if (self.dummy_X.shape[1] == 1):
            #model = Model('plsreg', parameters={'n_components':1})
            #else:
            #fit model and assert it has been fitted
            model.train_test_split(self.dummy_X_2D, self.dummy_Y)
            model.fit()
            self.assertTrue(model.model_fitted(), 'Model has not been fitted')

    def test_model_input_closeness(self):
        """ Test case for testing the algorithm closeness function used to get the
            closest available algorithm to the algorithm input into the class. """
#1.)
        model = Model('plsreg')
        self.assertEqual(model.algorithm, "plsregression")
        self.assertEqual(repr(model), "PLSRegression")
#2.)
        model = Model('randomfor')
        self.assertEqual(model.algorithm, "randomforestregressor")
        self.assertEqual(repr(model), "RandomForestRegressor")
#3.)
        model = Model('adaboo')
        self.assertEqual(model.algorithm, "adaboostregressor")
        self.assertEqual(repr(model), "AdaBoostRegressor")
#4.)
        model = Model('bagging')
        self.assertEqual(model.algorithm, "baggingregressor")
        self.assertEqual(repr(model), "BaggingRegressor")
#5.)
        model = Model('decisiontree')
        self.assertEqual(model.algorithm, "decisiontreeregressor")
        self.assertEqual(repr(model), "DecisionTreeRegressor")
#6.)
        model = Model('linear')
        self.assertEqual(model.algorithm, "linearregression")
        self.assertEqual(repr(model), "LinearRegression")
#7.)
        model = Model('lass')
        self.assertEqual(model.algorithm, "lasso")
        self.assertEqual(repr(model), "Lasso")
#8.)
        model = Model('kneighbors')
        self.assertEqual(model.algorithm, "kneighborsregressor")
        self.assertEqual(repr(model), "KNeighborsRegressor")
#9.)
        model = Model('sv')
        self.assertEqual(model.algorithm, "svr")
        self.assertEqual(repr(model), "SVR")
#10.)
        model = Model('rid')
        self.assertEqual(model.algorithm, "ridge")
        self.assertEqual(repr(model), "Ridge")
#11.)
        model = Model('gbr')
        self.assertEqual(model.algorithm, "gbr")
        self.assertEqual(repr(model), "GradientBoostingRegressor")
#12.)
        model = Model('sg')
        self.assertEqual(model.algorithm, "sgd")
        self.assertEqual(repr(model), "SGDRegressor")
#13.)
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            bad_model = Model('abcdefg')
            bad_model = Model('notamodel')
            bad_model = Model('123')
            bad_model = Model('blahblahblah')

    def test_train_test_split(self):
        """ Testing splitting up dataset into training and test data. """
#1.)
        model = Model('plsreg')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2D, self.dummy_Y)

        self.assertTrue(len(X_train) == 80)
        self.assertTrue(len(Y_train) == 80)
        self.assertTrue(len(X_test) == 20)
        self.assertTrue(len(Y_test) == 20)
#2.)
        model = Model('plsreg')
        with self.assertRaises(ValueError, msg='Value Error raised, invalid X and Y data given.'):
            X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2, self.dummy_Y)
#3.)
        model = Model('adaboostreg')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2D, self.dummy_Y, test_size=0.5)

        self.assertTrue(len(X_train) == 50)
        self.assertTrue(len(Y_train) == 50)
        self.assertTrue(len(X_test) == 50)
        self.assertTrue(len(Y_test) == 50)
#4.)
        model = Model('bagging')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2, self.dummy_Y_2, test_size=0.1)

        self.assertTrue(len(X_train) == 45)
        self.assertTrue(len(Y_train) == 45)
        self.assertTrue(len(X_test) == 5)
        self.assertTrue(len(Y_test) == 5)

    def test_predict(self):
        """ Testing the prediction of values for unseen sequences using the trained model. """
#1.)
        model = Model('knn')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2, self.dummy_Y_2)
        model.fit()
#2.)
        Y_pred = model.predict()
        self.assertIsInstance(Y_pred, np.ndarray)
        self.assertEqual(len(Y_pred), len(Y_test))

    def test_parameters(self):
        """ Testing parameters of Model class. """
#1.)
        #create instance of PLS model using Model class & creating instance
        #   using SKlearn libary, comparing if the parameters of both instances are equal
        pls_parameters = {"n_components": 20, "scale": False, "max_iter": 200}
        model = Model(algorithm="PlsRegression", parameters=pls_parameters)
        pls_model = PLSRegression(n_components=20, scale="svd", max_iter=200)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(pls_model.get_params()))
#2.)
        rf_parameters = {"n_estimators": 200, "max_depth": 50,"min_samples_split": 10}
        model = Model(algorithm="RandomForest", parameters=rf_parameters)
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=50, min_samples_split=10)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(rf_model.get_params()))
#3.)
        knn_parameters = {"n_neighbors": 10, "weights": "distance", "algorithm": "ball_tree"}
        model = Model(algorithm="KNN", parameters=knn_parameters)
        knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm="kd_tree")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(knn_model.get_params()))
#4.)
        svr_parameters = {"kernel": "poly", "degree": 5, "coef0": 1}
        model = Model(algorithm="SVR",parameters=svr_parameters)
        svr_model = SVR(kernel='poly', degree=5, coef0=1)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(svr_model.get_params()))
#5.)
        ada_parameters = {"n_estimators": 150, "learning_rate": 1.2, "loss": "square"}
        model = Model(algorithm="AdaBoost", parameters=ada_parameters)
        ada_model = AdaBoostRegressor(n_estimators=150, learning_rate=1.2, loss="square")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(ada_model.get_params()))
#6.)
        bagging_parameters = {"n_estimators": 50, "max_samples": 1.5, "max_features": 2}
        model = Model(algorithm="Bagging", parameters=bagging_parameters)
        bagging_model = BaggingRegressor(n_estimators=50, max_samples=1.5, max_features="square")

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(bagging_model.get_params()))
#7.)
        lasso_parameters = {"alpha": 1.5, "max_iter": 500, "tol": 0.004}
        model = Model(algorithm="lasso", parameters=lasso_parameters)
        lasso_model = Lasso(alpha=1.5, max_iter=500, tol=0.004)

        for k, v in model.model.get_params().items():
            self.assertIn(k, list(lasso_model.get_params()))

    def test_hyperparamter_tuning(self):
        """ Testing hyperparamter tuning function. """
#1.)
        model = Model(algorithm="PLSReg")
        test_split = 0.2

        with self.assertRaises(TypeError, msg='Type Error raised, invalid parameter data type given.'):
            model.hyperparameter_tuning(parameters='wrongType')

        with self.assertRaises(UndefinedMetricWarning, msg='Undefined Metric Error raised, invalid hyperparamter tuning metric given.'):
            model.hyperparameter_tuning(metric='blahblah')
#2.)
        if (self.dummy_X_2D.shape[1] == 1):
            model = Model('plsreg', parameters={'n_components':1})

        #get training and test dataset split
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2D, self.dummy_Y, test_size=test_split)

        # print("model")
        # print(model.model.n_components)
        # print("self.dummy_X_2D")
        # print(self.dummy_X_2D.shape)

        # print("self.dummy_Y")
        # print(self.dummy_Y.shape)
        #fit predictive model   
        model.fit()
        
        #undertake hyperparameter tuning
        model.hyperparameter_tuning()
    
    def test_feature_selection(self):
        """ Testing Feature Selection function. """ 
    
    def tearDown(self):
        """ Delete any temp data used for tests. """
        del self.dummy_X
        del self.dummy_X_2
        del self.dummy_X_2D
        del self.dummy_Y
        del self.dummy_Y_2
        del self.dummy_Y_2D

if __name__ == '__main__':
    #run all model tests
    unittest.main(verbosity=2)
