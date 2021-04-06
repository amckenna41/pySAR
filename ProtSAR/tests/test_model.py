################################################################################
#################             Model Module Tests               #################
################################################################################

import os
import sys
from model import *
from globals import *
import unittest
import sklearn
import numpy as np

class ModelTests(unittest.TestCase):

    def setUp(self):

        #creatre model save dir
        self.dummy_X = np.random.ranf(size=100)
        self.dummy_Y = np.random.randint(10,size=100)


    def test_model_types(self):

        """
        Test Case to check each model type and its associated parameters and
        attributes.

        """
        test_models = ['PLSRegression','RandomForestRegressor']

        for test_mod in range(0,len(test_models)):

            model = Model(test_models[test_mod])

            #checking model object is of the correct sklearn model datatype
            self.assertEqual(type(model.model).__name__, test_models[test_mod])
            # self.assertIsInstance(model.model,sklearn.cross_decomposition.PLSRegression)

            #assert that model has not been fitted
            self.assertFalse(model.modelFitted())
            #fit model and assert it has been fitted
            model.fit(self.dummy_X, self.dummy_Y)
            self.assertTrue(model.modelFitted())

            #verify that parameters input param = {} meaning the defauly params for the model are used
            self.assertEqual(model.parameters,{})
            #assert model/algorithm name is ==
            self.assertEqual(model.algorithm.lower(), test_models[test_mod].lower())
            #verify test split attribute is = 0.2, its default value
            self.assertEqual(model.test_split, 0.2)
            #verify that input model type is a valid model for the class
            self.assertTrue(model.algorithm in model.valid_models)

        #############################################

        # model = Model('RandomForestRegressor')
        # self.assertEqual(type(self.model).__name__, 'RandomForestRegressor')
        # self.assertIsInstance(model.model,sklearn.ensemble.RandomForestRegressor)
        #
        # #assert that model has not been fitted
        # self.assertFalse(model.modelFitted())
        # #fit model and assert it has been fitted
        # model.fit(self.dummy_X, self.dummy_Y)
        # self.assertTrue(model.modelFitted())
        #
        # #verify that parameters input param = {} meaning the defauly params for the model are used
        # self.assertEqual(model.parameters,{})
        # #assert model/algorithm name is == 'plsregression'
        # self.assertEqual(model.algorithm.lower(), 'plsregression')
        # #verify test split attribute is = 0.2, its default value
        # self.assertEqual(model.test_split, 0.2)
        # #verify that input model type is a valid model for the class
        # self.assertTrue(model.algorithm in model.valid_models)
        #
        #
        #
        #
        # model = Modoel('AdaBoostRegressor')
        # self.assertEqual(type(self.model).__name__, 'AdaBoostRegressor')
        #
        # model = Model('BaggingRegressor')
        # self.assertEqual(type(self.model).__name__, 'BaggingRegressor')
        #
        # model = Modoel('LinearRegression')
        # self.assertEqual(type(self.model).__name__, 'LinearRegression')
        #
        # model = Model('DecisionTreeRegressor')
        # self.assertEqual(type(self.model).__name__, 'DecisionTreeRegressor')
        #
        # model = Modoel('Lasso')
        # self.assertEqual(type(self.model).__name__, 'Lasso')
        #
        # model = Model('SVR')
        # self.assertEqual(type(self.model).__name__, 'SVR')
        #
        # model = Model('KNN')
        # self.assertEqual(type(self.model).__name__, 'KNN')

    def test_model_input_closeness(self):

        model = Model('plsreg')
        self.assertEqual(model.algorithm, "PLSRegression")

        self.assertEqual(repr(model), "PLSRegression")
        #test if inputting similar like algorithm names still gives similar output
        # e.g model = Model('plsreg') etc

        pass
    def test_model_instances():



        pass

    def test_X_Y(self):
        #assert X and Y have the same length
        pass

    def test_predict(self):
        pass
    # sklearn.utils.validation.check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=<built-in function all>)

    # def test_testsplit():
    #
    #     model = Model('LinearRegression')
    #     self.ass
    #     pass
    # @unittest.skip("Don't want to overload the FTP server each time tests are run")
    # def test_download(self):

    def test_parameters():
        pass

    def test_sizeof():
        pass

    def tearDown(self):

        #remove model save dir
        pass
if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests.test_aaindex -v (-v to give more verbose output)
# python -m unittest test_module1 test_module2
# python -m unittest test_module.TestClass
# python -m unittest test_module.TestClass.test_method
# python -m unittest tests/test_something.py
# python -m unittest -v test_module
# python -m unittest discover -s project_directory -p "*_test.py"
    # @unittest.expectedFailure
# assertIs(a, b)
# assertIsNot(a, b)

# assertIsNone(x)
# assertIsNotNone(x)

# assertIn(a, b)

# assertNotIn(a, b)
# assertIsInstance(a, b)
#
# assertNotIsInstance(a, b)

# assertGreaterer()
# assertLess()


#with self.assertRaises(ValueError):
#   calc.divide(10,0)
