################################################################################
#################             Model Module Tests               #################
################################################################################

import os
import sys
import unittest
import sklearn
import numpy as np

#### Suppress Sklearn warnings ####
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
###################################

from sklearn.cross_decomposition import PLSRegression

from pySAR.model import *


class ModelTests(unittest.TestCase):

    def setUp(self):
        """ Create dummy data. """

        self.dummy_X = np.random.ranf(size=100)
        self.dummy_X_2 = np.random.ranf(size=50)
        self.dummy_Y = np.random.randint(10,size=100)
        self.dummy_Y_2 = np.random.randint(20,size=50)

    def test_model(self):
        """ Test Case to check each model type & its associated parameters & attributes. """

        test_models = ['PLSRegression','RandomForestRegressor','AdaBoostRegressor',\
                            'BaggingRegressor','DecisionTreeRegressor','LinearRegression',\
                            'Lasso','SVR','KNeighborsRegressor']

        #iterate through all available algorithms and test them
        for test_mod in range(0,len(test_models)):

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
            self.assertTrue(model.algorithm in [item.lower() \
                for item in model.valid_models],
                'Input algorithm {} not in available algorithms: {}'.format(model.algorithm, model.valid_models))
#6.)        #verify repr represenation of model object is correct
            self.assertEqual(repr(model), test_models[test_mod],
                'Repr function should return {}, but got {}'.format(test_models[test_mod], repr(model)))
#7.)        #verify algorithm is a regression
            self.assertTrue(sklearn.base.is_regressor(model.model),
                'Model type should be a sklearn regressor.')

#8.)        #fit model and assert it has been fitted
            model.train_test_split(self.dummy_X, self.dummy_Y)
            model.fit()
            self.assertTrue(model.model_fitted(), 'Model has not been fitted')

    def test_model_input_closeness(self):
        """ Test case for testing the algorithm closeness function used to get the
            closest available algorithm to the algorithm input into the class. """

#1.)
        model = Model('plsreg')
        self.assertEqual(model.algorithm, "plsregression")
        self.assertEqual(repr(model), "PLSRegression")

        model = Model('randomfor')
        self.assertEqual(model.algorithm, "randomforestregressor")
        self.assertEqual(repr(model), "RandomForestRegressor")

        model = Model('adaboo')
        self.assertEqual(model.algorithm, "adaboostregressor")
        self.assertEqual(repr(model), "AdaBoostRegressor")

        model = Model('bagg')
        self.assertEqual(model.algorithm, "baggingregressor")
        self.assertEqual(repr(model), "BaggingRegressor")

        model = Model('decisiontree')
        self.assertEqual(model.algorithm, "decisiontreeregressor")
        self.assertEqual(repr(model), "DecisionTreeRegressor")

        model = Model('linear')
        self.assertEqual(model.algorithm, "linearregression")
        self.assertEqual(repr(model), "LinearRegression")

        model = Model('lass')
        self.assertEqual(model.algorithm, "lasso")
        self.assertEqual(repr(model), "Lasso")

        model = Model('kneighbors')
        self.assertEqual(model.algorithm, "kneighborsregressor")
        self.assertEqual(repr(model), "KNeighborsRegressor")

        model = Model('sv')
        self.assertEqual(model.algorithm, "svr")
        self.assertEqual(repr(model), "SVR")
#2.)
        with self.assertRaises(ValueError):
            bad_model = Model('abcdefg')

        with self.assertRaises(ValueError):
            bad_model = Model('rand')

        with self.assertRaises(ValueError):
            bad_model = Model('123')

        with self.assertRaises(ValueError):
            bad_model = Model('blahblahblah')

    def test_train_test_split(self):
        """ Testing splitting up dataset into training and test data. """
#1.)
        model = Model('plsreg')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X, self.dummy_Y)

        self.assertTrue(len(X_train) == 80)
        self.assertTrue(len(Y_train) == 80)
        self.assertTrue(len(X_test) == 20)
        self.assertTrue(len(Y_test) == 20)
#2.)
        model = Model('plsreg')
        with self.assertRaises(ValueError):
            X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X_2, self.dummy_Y)
#3.)
        model = Model('adaboostreg')
        X_train, X_test, Y_train, Y_test = model.train_test_split(self.dummy_X, self.dummy_Y, test_size=0.5)

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

        Y_pred = model.predict()
        self.assertIsInstance(Y_pred, np.ndarray)
        self.assertEqual(len(Y_pred), len(Y_test))
# ***
    def test_parameters(self):

        pls_parameters = {"n_components":20,"scale":False, "max_iter":200}

        model = Model(algorithm="PlsRegression",parameters={})

        pls_model = PLSRegression(n_components=20, scale="svd", max_iter=200)

        # self.assertEqual(model.model.parameters.items(), pls_parameters.items())

        for k, v in pls_parameters.items():

            self.assertIn(k, list(pls_model.get_params()))


        rf_parameters = {}

        # model = Model('PlsRegression',parameters = pls_parameters)
        #
        # pls_model = PLSRegression(n_components=20, algorithm="svd", max_iter=200)
        #
        # self.assertEqual(model.parameters.items, pls_parameters.items)
        #
        # for k, v in pls_parameters:
        #
        #     self.assertIn(k, list(pls_model.get_params()))
        #


        # bagging_parameters = {}
        #
        # model = Model('PlsRegression',parameters = pls_parameters)
        #
        # pls_model = PLSRegression(n_components=20, algorithm="svd", max_iter=200)
        #
        # self.assertEqual(model.parameters.items, pls_parameters.items)
        #
        # for k, v in pls_parameters:
        #
        #     self.assertIn(k, list(pls_model.get_params()))
        #
        # pass

    def test_hyperparamter_tuning(self):
        pass

    def test_copy(self):
        pass

    def tearDown(self):

        del self.dummy_X
        del self.dummy_Y

        pass

if __name__ == '__main__':
    #run all model tests
    unittest.main(verbosity=2)
