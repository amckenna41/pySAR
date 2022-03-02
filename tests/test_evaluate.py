################################################################################
#################             Evaluate Module Tests            #################
################################################################################

import numpy as np
import unittest
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pySAR.evaluate import Evaluate

#Individual evaluation functions working with pySAR but test case approach erroneous
# @unittest.expectedFailure
class EvaluateTests(unittest.TestCase):

    def setUp(self):
        """ Create dummy data to test evaluate class. """
        self.a = np.random.ranf(size=100).reshape((-1,1))
        self.a_ = np.random.ranf(size=100).reshape((-1,1))
        self.b = np.random.ranf(size =80).reshape((-1,1))
        self.c = np.array([1,2,3,4,5,6,7,8,9,10]).reshape((-1,1))
        self.d = np.array([10,9,8,7,6,5,4,3,2,1]).reshape((-1,1))
        self.e = np.random.randint(20, size=10).reshape((-1,1))
        self.f = np.random.randint(20, size=10).reshape((-1,1))

    def test_evaluate(self):
        """ Test Evaluate class initialisation and its attributes. """
#1.)
        #assert that value error is raised when input parameters to class are not same shape
        with self.assertRaises(ValueError, msg='Value Error raised, input parameters are not of the same shape.'):
            fail_eval = Evaluate(self.a,self.b)

        eval = Evaluate(self.a,self.a_)
        metrics = ['r2', 'mse', 'rmse', 'mae', 'rpd', 'explained_var']
#2.)
        #assert metric names exist as instance variables in Evaluate class
        for met in range(0, len(metrics)):
            self.assertTrue(metrics[met] in list(eval.__dict__.keys()))
#3.)
        #testing Y_true variable in the Evaluate object has been set according to Y_true input parameter
        np_equality = np.testing.assert_array_equal(self.a, eval.Y_true)
        self.assertIsNone(np_equality)
#4.)
        #testing Y_pred variable in the Evaluate object has been set according to Y_pred input parameter
        np_equality = np.testing.assert_array_equal(self.a_, eval.Y_pred)
        self.assertIsNone(np_equality)
#5.)
        #numpy.testing function will raise an assertion error if the arrays are not equal
        with self.assertRaises(AssertionError, msg='Assertion Error raised, input arrays are not equal.'):
             np.testing.assert_array_equal(self.a_, eval.Y_true )
#6.)
        self.assertTrue(eval.Y_true.shape == (100,1))
        self.assertTrue(eval.Y_pred.shape == (100,1))

        #############################################
        eval = Evaluate(self.c,self.d)
        metrics = ['r2', 'mse', 'rmse', 'mae', 'rpd', 'explained_var']
#7.)
        #assert metric names exist as instance variables in Evaluate class
        for met in range(0, len(metrics)):
            self.assertTrue(metrics[met] in list(eval.__dict__.keys()))
#8.)
        np_equality = np.testing.assert_array_equal(self.c, eval.Y_true)
        self.assertIsNone(np_equality)
#9.)
        np_equality = np.testing.assert_array_equal(self.d, eval.Y_pred)
        self.assertIsNone(np_equality)
#10.)
        #numpy.testing function will raise an assertion error if the arrays are not equal
        with self.assertRaises(AssertionError, msg='Assertion Error raised, input arrays are not equal.'):
             np.testing.assert_array_equal(self.d, eval.Y_true )
#11.)
        self.assertTrue(eval.Y_true.shape == (10,1))
        self.assertTrue(eval.Y_pred.shape == (10,1))

#     @unittest.expectedFailure
    def test_r2(self):
        """ Test case for testing the R2 score method. """
#1.)
        with warnings.catch_warnings(record=True) as w: #catch Divide by zero runtime warning
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        #assert that R2 = 1 if Y_true and Y_pred arrays are equal
        self.assertEqual(eval.r2, 1)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        #assert r2 value is around the pre-calculated value for c=Y_true, d=Y_pred
        self.assertAlmostEqual(eval.r2, -3.0)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2, float)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        actual_value = r2_score(self.e, self.f)
        self.assertAlmostEqual(eval.r2, actual_value)
        #assert r2 variable is of type float
        self.assertIsInstance(eval.r2, float)

    def test_mse(self):
        """ Test case for testing the MSE method. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.mse, 0)
        self.assertIsInstance(eval.mse, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.mse, float)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.mse, float)
        self.assertTrue(eval.mse>0)

    def test_rmse(self):
        """ Testing RMSE evaluation metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.rmse, 0)
        self.assertIsInstance(eval.rmse, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.mse, float)
        self.assertTrue(eval.mse>0)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.mse, float)
        self.assertTrue(eval.mse>0)

    def test_mae(self):
        """ Testing MAE evaluation metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.mae, 0)
        self.assertIsInstance(eval.mae, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.mae, float)
        self.assertTrue(eval.mae>0)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.mae, float)
        self.assertTrue(eval.mae>0)

    def test_explainedVar(self):
        """ Testing Explained Variance metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.explained_var, 1)
        self.assertIsInstance(eval.explained_var, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.explained_var, float)
        self.assertTrue(eval.explained_var==-3)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.explained_var, float)
        self.assertTrue(eval.explained_var<=1)

    def test_maxError(self):
        """ Testing max error evaluation metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.max_error_(), 0)
        self.assertIsInstance(eval.max_error_(), float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.max_error_(), float)
        self.assertTrue(eval.max_error_()==9)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.max_error_(), float)
        self.assertTrue(eval.max_error_()>=1)

    @unittest.expectedFailure
    def test_meanPoissonDeviance(self):
        """ Testing mean poisson deviation metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.mean_poisson_deviance, 0)
        self.assertIsInstance(eval.mean_poisson_deviance, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.mean_poisson_deviance, float)
        self.assertTrue(eval.mean_poisson_deviance>0)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.mean_poisson_deviance, float)
        self.assertTrue(eval.mean_poisson_deviance>0)

    @unittest.expectedFailure
    def test_rpd(self):
        """ Testing RPD evaluation metric. """
#1.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eval = Evaluate(self.a, self.a)
            self.assertEqual(len(w), 1)
            self.assertTrue("divide by zero encountered in double_scalars" in str(w[-1].message))

        self.assertEqual(eval.rpd, 0)
        self.assertIsInstance(eval.rpd, float)
#2.)
        #############################################
        eval = Evaluate(self.c, self.d)
        self.assertIsInstance(eval.rpd, float)
        self.assertTrue(eval.rpd>0)
#3.)
        #############################################
        eval = Evaluate(self.e, self.f)
        self.assertIsInstance(eval.rpd, float)
        self.assertTrue(eval.rpd>0)

