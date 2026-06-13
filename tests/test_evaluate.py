################################################################################
#################            Evaluate Module Tests            #################
################################################################################

import unittest
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, max_error

import pySAR.evaluate as eval_

unittest.TestLoader.sortTestMethodsUsing = None


class EvaluateTests(unittest.TestCase):
    """
    Test suite for testing evaluate module and functionality in pySAR package.

    Test Cases
    ==========
    test_evaluate_init:
        testing Evaluate class initialisation and metric attribute assignment.
    test_metric_methods:
        testing metric methods return expected values for known inputs.
    test_rpd_zero_mse:
        testing rpd_ returns np.inf when mse is 0.
    test_nan_inf_validation:
        testing ValueError is raised when inputs contain NaN or infinite values.
    test_shape_validation:
        testing ValueError is raised when observed and predicted arrays have different lengths.
    test_multioutput_parameter:
        testing multioutput parameter handling in metric methods.
    test_repr:
        testing __repr__ output format.
    test_str:
        testing __str__ output format and content.
    """

    def setUp(self):
        """Create deterministic sample arrays for metric validation."""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 4.7])
        self.eval_obj = eval_.Evaluate(self.y_true, self.y_pred)

    def test_evaluate_init(self):
        """Testing Evaluate class initialisation and metric attribute assignment."""
#1.)
        self.assertIsInstance(self.eval_obj, eval_.Evaluate,
            f"Expected Evaluate instance, got {type(self.eval_obj)}.")
#2.)
        self.assertEqual(self.eval_obj.Y_true.shape, (5, 1),
            f"Expected Y_true shape (5, 1), got {self.eval_obj.Y_true.shape}.")
        self.assertEqual(self.eval_obj.Y_pred.shape, (5, 1),
            f"Expected Y_pred shape (5, 1), got {self.eval_obj.Y_pred.shape}.")
#3.)
        for metric_name in ["r2", "rmse", "mse", "mae", "rpd", "explained_var", "max_error"]:
            self.assertTrue(hasattr(self.eval_obj, metric_name),
                f"Expected Evaluate instance to have attribute '{metric_name}'.")

    def test_metric_methods(self):
        """Testing metric methods return expected values for known inputs."""
        y_t = self.y_true.reshape(-1, 1)
        y_p = self.y_pred.reshape(-1, 1)
#1.)
        self.assertAlmostEqual(self.eval_obj.r2_(), r2_score(y_t, y_p), places=12,
            msg="r2_ value does not match sklearn r2_score output.")
#2.)
        self.assertAlmostEqual(self.eval_obj.mse_(), mean_squared_error(y_t, y_p), places=12,
            msg="mse_ value does not match sklearn mean_squared_error output.")
#3.)
        self.assertAlmostEqual(self.eval_obj.rmse_(), np.sqrt(mean_squared_error(y_t, y_p)), places=12,
            msg="rmse_ value does not match sqrt(mse) output.")
#4.)
        self.assertAlmostEqual(self.eval_obj.mae_(), mean_absolute_error(y_t, y_p), places=12,
            msg="mae_ value does not match sklearn mean_absolute_error output.")
#5.)
        self.assertAlmostEqual(self.eval_obj.explained_var_(), explained_variance_score(y_t, y_p), places=12,
            msg="explained_var_ value does not match sklearn explained_variance_score output.")
#6.)
        self.assertAlmostEqual(self.eval_obj.max_error_(), float(max_error(y_t, y_p)), places=12,
            msg="max_error_ value does not match sklearn max_error output.")

    def test_rpd_zero_mse(self):
        """Testing rpd_ returns nan when mse is 0 (division by zero)."""
#1.)
        perfect_true = np.array([0.0, 1.0, 2.0, 3.0])
        perfect_pred = np.array([0.0, 1.0, 2.0, 3.0])
        eval_perfect = eval_.Evaluate(perfect_true, perfect_pred)
        self.assertTrue(np.isnan(eval_perfect.rpd),
            f"Expected rpd to be nan for perfect predictions (zero MSE), got {eval_perfect.rpd}.")

    def test_nan_inf_validation(self):
        """Testing ValueError is raised when inputs contain NaN or infinite values."""
#1.)
        with self.assertRaises(ValueError):
            eval_.Evaluate(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
#2.)
        with self.assertRaises(ValueError):
            eval_.Evaluate(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))
#3.)
        with self.assertRaises(ValueError):
            eval_.Evaluate(np.array([1.0, np.inf, 3.0]), np.array([1.0, 2.0, 3.0]))
#4.)
        with self.assertRaises(ValueError):
            eval_.Evaluate(np.array([1.0, 2.0, 3.0]), np.array([1.0, -np.inf, 3.0]))

    def test_shape_validation(self):
        """Testing ValueError is raised when observed and predicted arrays have different lengths."""
#1.)
        with self.assertRaises(ValueError):
            eval_.Evaluate(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]))

    def test_multioutput_parameter(self):
        """Testing multioutput parameter handling in metric methods."""
#1.)
        r2_raw = self.eval_obj.r2_(multioutput='raw_values')
        mse_raw = self.eval_obj.mse_(multioutput='raw_values')
        mae_raw = self.eval_obj.mae_(multioutput='raw_values')
        rmse_raw = self.eval_obj.rmse_(multioutput='raw_values')
        exp_var_raw = self.eval_obj.explained_var_(multioutput='raw_values')

        for arr, name in [(r2_raw, 'r2_'), (mse_raw, 'mse_'), (mae_raw, 'mae_'), (rmse_raw, 'rmse_'), (exp_var_raw, 'explained_var_')]:
            self.assertIsInstance(arr, np.ndarray,
                f"Expected numpy.ndarray for {name} with multioutput='raw_values', got {type(arr)}.")
            self.assertEqual(arr.shape, (1,),
                f"Expected shape (1,) for {name} with single-target data, got {arr.shape}.")

    def test_repr(self):
        """Testing __repr__ output format."""
#1.)
        repr_out = repr(self.eval_obj)
        self.assertIsInstance(repr_out, str,
            f"Expected __repr__ output to be str, got {type(repr_out)}.")
        self.assertIn("Evaluate", repr_out,
            f"Expected __repr__ to contain 'Evaluate', got {repr_out}.")
        self.assertIn("Y_true: (5, 1)", repr_out,
            f"Expected __repr__ to contain Y_true shape, got {repr_out}.")
        self.assertIn("Y_pred: (5, 1)", repr_out,
            f"Expected __repr__ to contain Y_pred shape, got {repr_out}.")

    def test_str(self):
        """Testing __str__ output format and content."""
#1.)
        str_out = str(self.eval_obj)
        self.assertIsInstance(str_out, str,
            f"Expected __str__ output to be str, got {type(str_out)}.")
#2.)
        for token in ["R2", "RMSE", "MSE", "MAE", "RPD", "Explained Variance", "Max Error"]:
            self.assertIn(token, str_out,
                f"Expected __str__ output to contain token '{token}', got:\n{str_out}")


if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)
