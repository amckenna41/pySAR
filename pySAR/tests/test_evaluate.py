################################################################################
#################             Evaluate Module Tests            #################
################################################################################

import numpy as np
import unittest
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

from evaluate import Evaluate

class EvaluateTests(unittest.TestCase):

    def setUp(self):
        # self.a = np.random.ranf(10, size=100)
        self.a = np.random.ranf(size=100).reshape((-1,1))
        self.a_ = np.random.ranf(size=100).reshape((-1,1))
        self.b = np.random.ranf(size =80).reshape((-1,1))
        # self.b = np.array.randint
        self.c = np.array([1,2,3,4,5,6,7,8,9,10]).reshape((-1,1))
        self.d = np.array([10,9,8,7,6,5,4,3,2,1]).reshape((-1,1))
        self.e = np.random.randint(20, size=10).reshape((-1,1))
        self.f = np.random.randint(20, size=10).reshape((-1,1))

    def test_evaluate(self):

        #assert that value error is raised when input parameters to evaluate
        #   class are mot of the same shape
        with self.assertRaises(ValueError):
            fail_eval = Evaluate(self.a,self.b)

        eval = Evaluate(self.a,self.a_)
        metrics = ['r2', 'mse', 'rmse', 'mae', 'rpd', 'explained_var']

        #assert metric names exist as instance variables in Evaluate class
        for met in range(0,len(metrics)):
            self.assertTrue(metrics[met] in list(eval.__dict__.keys()))

        #testing Y_true variable in the Evaluate object has been set according to Y_true input parameter
        np_equality = np.testing.assert_array_equal(self.a,eval.Y_true)
        self.assertIsNone(np_equality)

        #testing Y_pred variable in the Evaluate object has been set according to Y_pred input parameter
        np_equality = np.testing.assert_array_equal(self.a_,eval.Y_pred)
        self.assertIsNone(np_equality)

        #numpy.testing function will raise an assertion error if the arrays are not equal
        with self.assertRaises(AssertionError):
             np.testing.assert_array_equal(self.a_,eval.Y_true )

        self.assertTrue(eval.Y_true.shape == (100,1))
        self.assertTrue(eval.Y_pred.shape == (100,1))

        #############################################
        eval = Evaluate(self.c,self.d)
        metrics = ['r2', 'mse', 'rmse', 'mae', 'rpd', 'explained_var']

        #assert metric names exist as instance variables in Evaluate class
        for met in range(0,len(metrics)):
            self.assertTrue(metrics[met] in list(eval.__dict__.keys()))


        np_equality = np.testing.assert_array_equal(self.c,eval.Y_true)
        self.assertIsNone(np_equality)

        np_equality = np.testing.assert_array_equal(self.d,eval.Y_pred)
        self.assertIsNone(np_equality)

        #numpy.testing function will raise an assertion error if the arrays are not equal
        with self.assertRaises(AssertionError):
             np.testing.assert_array_equal(self.d,eval.Y_true )

        self.assertTrue(eval.Y_true.shape == (10,1))
        self.assertTrue(eval.Y_pred.shape == (10,1))

    def test_r2(self):

        print('Testing R2 Evaluate function...')

        eval = Evaluate(self.a, self.a)
        #assert that R2 = 1 if Y_true and Y_pred arrays are equal
        self.assertEqual(eval.r2, 1)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2,float)

        #############################################
        eval = Evaluate(self.c, self.d)
        #assert r2 value is around the pre-calculated value for c=Y_true, d=Y_pred
        self.assertAlmostEqual(eval.r2, -3.0)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2,float)

        #############################################
        eval = Evaluate(self.e, self.f)
        actual_value = r2_score(e,f)
        self.assertAlmostEqual(eval.r2,actual_value)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2,float)

    def test_mse(self):

        print('Testing MSE Evaluate function...')

        eval = Evaluate(self.a, self.a)

        self.assertEqual(eval.mse, 1)
        self.assertIsInstance(eval.mse,float)

        #############################################
        eval = Evaluate(self.c, self.d)

        self.assertIsInstance(eval.mse,float)

    #
    # def test_rmse(self):
    #
    #     print('Testing RMSE Evaluate function...')
    #
    #     pass
    #
    # def test_mae(self):
    #
    #     print('Testing MAE Evaluate function...')
    #
    #     pass
    #
    # def test_rpd(self):
    #
    #     print('Testing RPD Evaluate function...')
    #
    #     pass
    #
    # def test_explainedVar(self):
    #
    #     print('Testing Explained Variance Evaluate function...')
    #
    #     pass
    #
    # def test_maxError(self):
    #
    #     print('Testing Max Error Evaluate function...')
    #
    #     pass
    #
    # def test_meanPoissonDeviance(self):
    #
    #     print('Testing Mean PoissonDeviance Evaluate function...')
    #
    #     pass


#test 2 inputs to funcrtion are of the same size.
# self.assertAlmostEqual(kappa, 0.0)
