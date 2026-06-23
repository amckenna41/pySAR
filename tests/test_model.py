################################################################################
#################             Model Module Tests               #################
################################################################################

import os
import tempfile
import unittest
import sklearn
import numpy as np
import shutil
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from pySAR.model import Model

# @unittest.skip("")
class ModelTests(unittest.TestCase):
    """
    Test suite for testing model module and functionality 
    in pySAR package. 

    Test Cases
    ==========
    test_model:
        testing correct overall Model class and module functionality.
    test_model_input_closeness:
        testing correct input closeness functionality.
    test_train_test_split:
        testing correct train-test split functionality.
    test_predict:
        testing correct predict functionality.
    test_save:
        testing correct model saving functionality, including dict format and scaler persistence.
    test_load:
        testing Model.load() classmethod round-trip and error handling.
    test_cv_score:
        testing Model.cv_score() cross-validation method.
    test_parameters:
        testing correct parameters functionality.
    test_hyperparamter_tuning:
        testing correct hyperparameter tuning functionality.
    test_feature_selection:
        testing correct feature selection functionality.
    """
    def setUp(self):
        """ Create dummy data. """
        self.rng = np.random.default_rng(42)
        self.dummy_X = self.rng.random(100)
        self.dummy_X_2D = self.rng.random((100, 50)) #50 sequences 
        self.dummy_X_2 = self.rng.random(50)
        self.dummy_Y = self.rng.integers(0, 10, size=100)
        self.dummy_Y_2 = self.rng.integers(0, 20, size=50)
        self.dummy_Y_2D = self.rng.random((50, 1)) #50 sequences 

        #test model folder
        self.test_folder = tempfile.mkdtemp(prefix='test_model_output_')

    def test_model(self):
        """ Test Case to check each model type & its associated parameters & attributes. """
        test_models = ['PLSRegression', 'RandomForestRegressor', 'AdaBoostRegressor',\
                            'BaggingRegressor', 'DecisionTreeRegressor', 'LinearRegression',\
                            'Lasso', 'SVR', 'KNeighborsRegressor', 'GradientBoostingRegressor', 'Ridge',\
                            'ElasticNet', 'ExtraTreesRegressor', 'HistGradientBoostingRegressor',\
                            'GaussianProcessRegressor']

        #iterate through all available algorithms/models and test each
        for test_mod in range(0, len(test_models)):
   
            model = Model(self.dummy_X, self.dummy_Y, test_models[test_mod])
#1.)
            #checking model object is of the correct sklearn model datatype
            self.assertEqual(type(model.model).__name__, test_models[test_mod],
                f'Model type is not correct, expected {test_models[test_mod]}, got {type(model.model).__name__}.')
#2.)        #assert that model has not been fitted
            self.assertFalse(model.model_fitted(), 'Model should not be fitted on initialisation.')
#3.)        #verify that parameters input param = {} meaning the default params for the model are used
            self.assertEqual(model.parameters, {},
                f'Default Parameters attribute should be an empty dict, but got {model.parameters}.')
#4.)        #verify test split attribute is 0.2, its default value
            self.assertEqual(model.test_split, 0.2,
                f'Default test split attribute should be 0.2, but got {model.test_split}.')
#5.)        #verify that input model type is a valid model for the class
            self.assertTrue(model.algorithm in [item.lower() for item in model.valid_models],
                f'Input algorithm {model.algorithm} not in available algorithms:\n{model.valid_models}.')
#6.)        #verify repr representation of model object is correct
            self.assertEqual(repr(model), test_models[test_mod],
                f'Repr function expected to be {test_models[test_mod]}, but got {repr(model)}.')
#7.)        #verify algorithm is a regression
            self.assertTrue(sklearn.base.is_regressor(model.model), 'Model type should be a sklearn regressor.')
#8.)           
            if (self.dummy_X.ndim == 1):
                model = Model(self.dummy_X, self.dummy_Y, 'plsreg', parameters={'n_components': 1})
            model.train_test_split()
            model.fit()
            self.assertTrue(model.model_fitted(), 'Model should be fitted')

    def test_model_input_closeness(self):
        """ Test case for testing the algorithm closeness function used to get the
            closest available algorithm to the algorithm input into the class. """
        aliases = [
            ('plsreg', 'plsregression', 'PLSRegression'),
            ('randomfor', 'randomforestregressor', 'RandomForestRegressor'),
            ('adaboost', 'adaboostregressor', 'AdaBoostRegressor'),
            ('bagging', 'baggingregressor', 'BaggingRegressor'),
            ('decisiontree', 'decisiontreeregressor', 'DecisionTreeRegressor'),
            ('linear', 'linearregression', 'LinearRegression'),
            ('lass', 'lasso', 'Lasso'),
            ('kneighbors', 'knearestneighbors', 'KNeighborsRegressor'),
            ('sv', 'svr', 'SVR'),
            ('rid', 'ridge', 'Ridge'),
            ('gbr', 'gbr', 'GradientBoostingRegressor'),
            ('sg', 'sgd', 'SGDRegressor'),
            ('elasticnet', 'elasticnet', 'ElasticNet'),
            ('extratrees', 'extratrees', 'ExtraTreesRegressor'),
            ('hgbr', 'hgbr', 'HistGradientBoostingRegressor'),
            ('gpr', 'gpr', 'GaussianProcessRegressor'),
        ]

        for input_name, expected_algorithm, expected_repr in aliases:
            with self.subTest(alias=input_name):
                model = Model(self.dummy_X, self.dummy_Y, input_name)
                self.assertEqual(
                    model.algorithm,
                    expected_algorithm,
                    f"Expected algorithm to be {expected_algorithm}, got {model.algorithm}.")
                self.assertEqual(
                    repr(model),
                    expected_repr,
                    f"Expected representation of model object to be {expected_repr}, got {repr(model)}.")
#13.)
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            Model(self.dummy_X, self.dummy_Y, 'abcdefg')
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            Model(self.dummy_X, self.dummy_Y, 'notamodel')
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            Model(self.dummy_X, self.dummy_Y, '123')
        with self.assertRaises(ValueError, msg='Value Error raised, invalid model/algorithm name given.'):
            Model(self.dummy_X, self.dummy_Y, 'blahblahblah')
#14.)
        with self.assertRaises(TypeError, msg="Type Error raised, input must be of type string."):
            Model(self.dummy_X, self.dummy_Y, 12345)
        with self.assertRaises(TypeError, msg="Type Error raised, input must be of type string."):
            Model(self.dummy_X, self.dummy_Y, 5.60)
        with self.assertRaises(TypeError, msg="Type Error raised, input must be of type string."):
            Model(self.dummy_X, self.dummy_Y, False)

    def test_train_test_split(self):
        """ Testing splitting up dataset into training and test data. """
#1.)
        model = Model(self.dummy_X_2D, self.dummy_Y, 'plsreg')
        original_x = self.dummy_X_2D.copy()
        original_y = self.dummy_Y.copy()
        X_train, X_test, Y_train, Y_test = model.train_test_split()

        self.assertEqual(len(X_train), 80, f"Expected 80 rows in training data, got {len(X_train)}.")
        self.assertEqual(len(Y_train), 80, f"Expected 80 rows in training data labels, got {len(Y_train)}.")
        self.assertEqual(len(X_test), 20, f"Expected 20 rows in test data, got {len(X_test)}.")
        self.assertEqual(len(Y_test), 20, f"Expected 20 rows in test data labels, got {len(Y_test)}.")

        self.assertIsInstance(X_train, np.ndarray, "X_train training data expected to be a numpy array.")
        self.assertIsInstance(Y_train, np.ndarray, "Y_train training data labels expected to be a numpy array.")
        self.assertIsInstance(X_test, np.ndarray, "X_test test data expected to be a numpy array.")
        self.assertIsInstance(Y_test, np.ndarray, "Y_test test data labels expected to be a numpy array.")
        self.assertTrue(np.array_equal(model.X, original_x), "Model.X should not be mutated by train_test_split.")
        self.assertTrue(np.array_equal(model.Y, original_y), "Model.Y should not be mutated by train_test_split.")
#2.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'adaboost')
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.5)

        self.assertEqual(len(X_train), 25, f"Expected 25 rows in training data, got {len(X_train)}.")
        self.assertEqual(len(Y_train), 25, f"Expected 25 rows in training data labels, got {len(Y_train)}.")
        self.assertEqual(len(X_test), 25, f"Expected 25 rows in test data, got {len(X_test)}.")
        self.assertEqual(len(Y_test), 25, f"Expected 25 rows in test data labels, got {len(Y_test)}.")

        self.assertIsInstance(X_train, np.ndarray, "X_train training data expected to be a numpy array.")
        self.assertIsInstance(Y_train, np.ndarray, "Y_train training data labels expected to be a numpy array.")
        self.assertIsInstance(X_test, np.ndarray, "X_test test data expected to be a numpy array.")
        self.assertIsInstance(Y_test, np.ndarray, "Y_test test data labels expected to be a numpy array.")
        unscaled_model = Model(self.dummy_X_2, self.dummy_Y_2, 'adaboost')
        X_train_unscaled, X_test_unscaled, _, _ = unscaled_model.train_test_split(test_split=0.5, scale=False, random_state=0)
        scaled_model = Model(self.dummy_X_2, self.dummy_Y_2, 'adaboost')
        X_train_scaled, X_test_scaled, _, _ = scaled_model.train_test_split(test_split=0.5, scale=True, random_state=0)
        self.assertFalse(np.allclose(X_train_unscaled, X_train_scaled),
            "Scaled and unscaled training data should differ when scale=True.")
        self.assertFalse(np.allclose(X_test_unscaled, X_test_scaled),
            "Scaled and unscaled test data should differ when scale=True.")
#3.)
        #test_split outside (0, 1) should raise ValueError, not silently reset
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'bagging')
        with self.assertRaises(ValueError,
                msg='ValueError expected when test_split is outside (0, 1).'):
            model.train_test_split(test_split=1234)
#4.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'plsreg')
        with self.assertRaises(TypeError, msg='Type Error raised, invalid test_split type input.'):
            model.train_test_split(test_split="ABCD")
#5.)
        #invalid test_split passed via __init__ should raise ValueError through the setter
        with self.assertRaises(ValueError,
                msg='ValueError expected when test_split is outside (0, 1) via __init__.'):
            Model(self.dummy_X_2, self.dummy_Y_2, 'plsreg', test_split=5.0)
#6.)
        #non-numeric test_split via __init__ should raise TypeError through the setter
        with self.assertRaises(TypeError, msg='TypeError expected for non-numeric test_split via __init__.'):
            Model(self.dummy_X_2, self.dummy_Y_2, 'plsreg', test_split="invalid")

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
#3.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'ridge')
        X_train, X_test, Y_train, Y_test = model.train_test_split()
        model.fit()
        model.save(self.test_folder, 'test_model3')

        self.assertTrue(os.path.isfile(os.path.join(self.test_folder, 'test_model3.pkl')), 
            "Expected .pkl extension to be appended when missing.")
#4.)    saved pickle must contain a dict with 'model' and 'scaler' keys
        import pickle
        pkl_path = os.path.join(self.test_folder, 'test_model3.pkl')
        with open(pkl_path, 'rb') as fh:
            payload = pickle.load(fh)
        self.assertIsInstance(payload, dict,
            "Saved pickle should be a dict.")
        self.assertIn('model', payload,
            "Saved pickle dict must contain a 'model' key.")
        self.assertIn('scaler', payload,
            "Saved pickle dict must contain a 'scaler' key.")
#5.)    scaler is None when scale=False
        unscaled = Model(self.dummy_X_2, self.dummy_Y_2, 'lasso')
        unscaled.train_test_split(scale=False)
        unscaled.fit()
        unscaled.save(self.test_folder, 'test_model_unscaled.pkl')
        with open(os.path.join(self.test_folder, 'test_model_unscaled.pkl'), 'rb') as fh:
            up = pickle.load(fh)
        self.assertIsNone(up['scaler'],
            "Scaler should be None when scale=False was used during train_test_split.")
#6.)    scaler is a fitted StandardScaler when scale=True
        scaled = Model(self.dummy_X_2, self.dummy_Y_2, 'lasso')
        scaled.train_test_split(scale=True)
        scaled.fit()
        scaled.save(self.test_folder, 'test_model_scaled.pkl')
        from sklearn.preprocessing import StandardScaler
        with open(os.path.join(self.test_folder, 'test_model_scaled.pkl'), 'rb') as fh:
            sp = pickle.load(fh)
        self.assertIsInstance(sp['scaler'], StandardScaler,
            "Scaler should be a fitted StandardScaler when scale=True was used.")

    def test_load(self):
        """ Testing Model.load() classmethod that reconstructs a Model from a saved pickle. """
#1.)    round-trip: save then load and verify model predictions match
        model = Model(self.dummy_X_2D, self.dummy_Y, 'randomforest',
                      parameters={'n_estimators': 10, 'random_state': 0})
        model.train_test_split(test_split=0.2, scale=True, random_state=0)
        model.fit()
        original_preds = model.predict()
        pkl_path = os.path.join(self.test_folder, 'test_load_model.pkl')
        model.save(self.test_folder, 'test_load_model.pkl')

        loaded = Model.load(pkl_path)
        self.assertIsInstance(loaded, Model,
            f"Model.load() should return a Model instance, got {type(loaded)}.")
        self.assertTrue(loaded.model_fitted(),
            "Loaded model should report as fitted.")
        self.assertEqual(type(loaded.model).__name__, 'RandomForestRegressor',
            f"Loaded model type should be RandomForestRegressor, got {type(loaded.model).__name__}.")
#2.)    loaded model's scaler attribute is restored
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(loaded.scaler, StandardScaler,
            "Scaler on loaded model should be a StandardScaler.")
#3.)    predictions from the loaded model match those of the original
        loaded_preds = loaded.model_fit.predict(model.X_test)
        np.testing.assert_array_almost_equal(
            original_preds, loaded_preds,
            err_msg="Loaded model should produce identical predictions to the original."
        )
#3.)    loaded model without scaler (scale=False)
        no_scaler = Model(self.dummy_X_2, self.dummy_Y_2, 'lasso')
        no_scaler.train_test_split(scale=False)
        no_scaler.fit()
        ns_path = os.path.join(self.test_folder, 'no_scaler_model.pkl')
        no_scaler.save(self.test_folder, 'no_scaler_model.pkl')
        loaded_ns = Model.load(ns_path)
        self.assertIsNone(loaded_ns.scaler,
            "Scaler should be None on a model saved without scaling.")
#4.)    OSError when path does not exist
        with self.assertRaises(OSError,
                msg='OSError expected when loading from a non-existent path.'):
            Model.load(os.path.join(self.test_folder, 'does_not_exist.pkl'))
#5.)    allow_pickle=False raises ValueError
        with self.assertRaises(ValueError,
                msg='ValueError expected when allow_pickle=False.'):
            Model.load(pkl_path, allow_pickle=False)
#6.)    loading emits a UserWarning about pickle security
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Model.load(pkl_path)
        warning_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(any("pickle" in m.lower() or "untrusted" in m.lower() for m in warning_msgs),
            "Model.load() should emit a UserWarning mentioning pickle safety.")

    def test_cv_score(self):
        """ Testing Model.cv_score() cross-validation method. """
#1.)    basic invocation returns an ndarray with cv entries
        model = Model(self.dummy_X_2D, self.dummy_Y, 'randomforest',
                      parameters={'n_estimators': 10, 'random_state': 0})
        model.train_test_split(test_split=0.2, random_state=0)
        scores = model.cv_score(cv=5, metric='r2')

        self.assertIsInstance(scores, np.ndarray,
            f"cv_score() should return np.ndarray, got {type(scores)}.")
        self.assertEqual(len(scores), 5,
            f"Expected 5 cv scores, got {len(scores)}.")
#2.)    metric parameter is forwarded (neg_mean_squared_error gives negative values)
        scores_mse = model.cv_score(cv=3, metric='neg_mean_squared_error')
        self.assertEqual(len(scores_mse), 3,
            "Expected 3 scores for cv=3.")
        self.assertTrue(np.all(scores_mse <= 0),
            "neg_mean_squared_error scores should be <= 0.")
#3.)    invalid metric raises ValueError
        with self.assertRaises(ValueError,
                msg='ValueError expected for invalid scoring metric.'):
            model.cv_score(metric='not_a_real_metric')
#4.)    cv_score does not permanently alter model_fit
        was_fitted = model.model_fitted()
        model.cv_score(cv=3)
        self.assertEqual(model.model_fitted(), was_fitted,
            "cv_score() should not change the fitted state of the model.")


    def test_parameters(self):
        """ Testing parameters of Model class for specified algorithm match that of the sklearn 
            models' parameters. """
#1.)
        pls_parameters = {"n_components": 20, "scale": False, "max_iter": 200}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="PlsRegression", parameters=pls_parameters)
        self.assertEqual(model.model.get_params()["n_components"], 20)
        self.assertEqual(model.model.get_params()["scale"], False)
        self.assertEqual(model.model.get_params()["max_iter"], 200)
#2.)
        rf_parameters = {"n_estimators": 200, "max_depth": 50, "min_samples_split": 10}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="RandomForest", parameters=rf_parameters)
        self.assertEqual(model.model.get_params()["n_estimators"], 200)
        self.assertEqual(model.model.get_params()["max_depth"], 50)
        self.assertEqual(model.model.get_params()["min_samples_split"], 10)
#3.)
        knn_parameters = {"n_neighbors": 10, "weights": "distance", "algorithm": "ball_tree"}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="KNN", parameters=knn_parameters)
        self.assertEqual(model.model.get_params()["n_neighbors"], 10)
        self.assertEqual(model.model.get_params()["weights"], "distance")
        self.assertEqual(model.model.get_params()["algorithm"], "ball_tree")
#4.)
        svr_parameters = {"kernel": "poly", "degree": 5, "coef0": 1}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="SVR",parameters=svr_parameters)
        self.assertEqual(model.model.get_params()["kernel"], "poly")
        self.assertEqual(model.model.get_params()["degree"], 5)
        self.assertEqual(model.model.get_params()["coef0"], 1)
#5.)
        ada_parameters = {"n_estimators": 150, "learning_rate": 1.2, "loss": "square"}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="AdaBoost", parameters=ada_parameters)
        self.assertEqual(model.model.get_params()["n_estimators"], 150)
        self.assertEqual(model.model.get_params()["learning_rate"], 1.2)
        self.assertEqual(model.model.get_params()["loss"], "square")
#6.)
        bagging_parameters = {"n_estimators": 50, "max_samples": 1.5, "max_features": 2}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="Bagging", parameters=bagging_parameters)
        self.assertEqual(model.model.get_params()["n_estimators"], 50)
        self.assertEqual(model.model.get_params()["max_samples"], 1.5)
        self.assertEqual(model.model.get_params()["max_features"], 2)
#7.)
        lasso_parameters = {"alpha": 1.5, "max_iter": 500, "tol": 0.004}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="lasso", parameters=lasso_parameters)
        self.assertEqual(model.model.get_params()["alpha"], 1.5)
        self.assertEqual(model.model.get_params()["max_iter"], 500)
        self.assertEqual(model.model.get_params()["tol"], 0.004)
#8.)
        filtered_parameters = {"n_components": 2, "invalid_parameter": 999}
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="plsreg", parameters=filtered_parameters)
        self.assertEqual(model.model.get_params()["n_components"], 2)
        self.assertNotIn("invalid_parameter", model.model.get_params(),
            "Invalid parameters should be filtered out before model construction.")

    def test_hyperparameter_tuning(self):
        """ Testing hyperparamter tuning functionality. """
#1.)
        model = Model(self.dummy_X, self.dummy_Y, algorithm="adaboostregressor")
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.2)
        model.fit()
        param_grid = {'n_estimators': [50,100,150], 'learning_rate': [0.5,0.75,1], 'loss': ['linear','exponential']}
        model.hyperparameter_tuning(metric="neg_root_mean_squared_error", param_grid=param_grid, verbose=0, cv=10)
        
        self.assertIsInstance(model.grid_result, GridSearchCV,
            f"Expected grid result to be GridSearchCV, got {type(model.grid_result)}.")
        self.assertEqual(model.grid_result.cv, 10, 
            f"Expected there to be 10 cross-validation folds, got {model.grid_result.cv}.")
        self.assertEqual(model.grid_result.error_score, 0, 
            f"Expected the error score to be 0, got {model.grid_result.error_score}.")
        self.assertEqual(model.grid_result.scoring, 'neg_root_mean_squared_error', 
            f"Expected the scoring metric to be neg_root_mean_squared_error, got {model.grid_result.scoring}.")
        self.assertEqual(model.grid_result.verbose, 0, 
            f"Expected the verbosity to be 0, got {model.grid_result.verbose}.")
        self.assertEqual(model.grid_result.param_grid, param_grid, 
            f"Expected the parameter grid to be an empty dict, got {model.grid_result.param_grid}.")
        self.assertIsInstance(model.grid_result.estimator, AdaBoostRegressor,
            f"Expected the estimator to be an AdaBoostRegressor, got {type(model.grid_result.estimator)}.")
#2.)
        model = Model(self.dummy_X_2, self.dummy_Y_2, algorithm="randomforest")
        X_train, X_test, Y_train, Y_test = model.train_test_split(test_split=0.2)
        model.fit()
        param_grid = {'max_depth': [2,3,4], 'n_estimators': [100,200,250], 'criterion': ['squared_error', 'absolute_error']}
        model.hyperparameter_tuning(param_grid=param_grid, verbose=0, cv=5)
        
        self.assertIsInstance(model.grid_result, GridSearchCV,
            f"Expected grid result to be GridSearchCV, got {type(model.grid_result)}.")
        self.assertEqual(model.grid_result.cv, 5, 
            f"Expected there to be 5 cross-validation folds, got {model.grid_result.cv}.")
        self.assertEqual(model.grid_result.error_score, 0, 
            f"Expected the error score to be 0, got {model.grid_result.error_score}.")
        self.assertEqual(model.grid_result.scoring, 'r2', 
            f"Expected the scoring metric to be r2, got {model.grid_result.scoring}.")
        self.assertEqual(model.grid_result.verbose, 0, 
            f"Expected the verbosity to be 0, got {model.grid_result.verbose}.")
        self.assertEqual(model.grid_result.param_grid, param_grid, 
            f"Expected the parameter grid to be an empty dict, got {model.grid_result.param_grid}.")
        self.assertIsInstance(model.grid_result.estimator, RandomForestRegressor,
            f"Expected the estimator to be a RandomForestRegressor, got {type(model.grid_result.estimator)}.")
#3.)
        with self.assertRaises(ValueError):
            model.hyperparameter_tuning(metric="invalid_metric", verbose=0, cv=10)
        with self.assertRaises(ValueError):
            model.hyperparameter_tuning(metric="R2", verbose=0, cv=5)
#4.)
        with self.assertRaises(TypeError):
            model.hyperparameter_tuning(param_grid='wrongType')
        with self.assertRaises(TypeError):
            model.hyperparameter_tuning(param_grid=123)

    def test_feature_selection(self):
        """ Testing Feature Selection functionality. """ 
        feature_X = self.rng.random((30, 6)) * 10
        feature_Y = self.rng.integers(1, 5, size=30)

#1.)
        model = Model(feature_X, feature_Y, 'randomforest')
        selected = model.feature_selection("selectkbest")
        self.assertIsInstance(selected, np.ndarray, "Feature selection output should be numpy array.")
        self.assertEqual(selected.shape, (30, 1), "SelectKBest should return 1 selected feature.")
#2.)
        variance_selected = model.feature_selection("variancethreshold")
        self.assertIsInstance(variance_selected, np.ndarray, "VarianceThreshold output should be numpy array.")
        self.assertEqual(variance_selected.shape[0], 30, "Feature selection should preserve row count.")
#3.)
        chi2_selected = model.feature_selection("chi2")
        self.assertEqual(chi2_selected.shape, (30, 2), "chi2 branch should return 2 selected features.")
#4.)
        rfe_selected = model.feature_selection("rfe")
        self.assertEqual(rfe_selected.shape, (30, 5), "RFE should return 5 selected features.")
#5.)
        select_from_model_selected = model.feature_selection("selectfrommodel")
        self.assertEqual(select_from_model_selected.shape[0], 30, "SelectFromModel should preserve row count.")
#6.)
        fallback_selected = model.feature_selection("unknown_method_name")
        self.assertEqual(fallback_selected.shape, (30, 1), "Unknown methods should fall back to SelectKBest with 1 feature.")
#7.)    configurable k parameter for selectkbest
        selected_k3 = model.feature_selection("selectkbest", k=3)
        self.assertEqual(selected_k3.shape, (30, 3),
            "feature_selection('selectkbest', k=3) should return 3 features.")
#8.)    configurable k parameter for chi2
        chi2_selected_k1 = model.feature_selection("chi2", k=1)
        self.assertEqual(chi2_selected_k1.shape, (30, 1),
            "feature_selection('chi2', k=1) should return 1 feature.")

    def test_invalid_model_usage(self):
        """ Testing model error behavior when called out of sequence. """
        model = Model(self.dummy_X_2, self.dummy_Y_2, 'knn')

        with self.assertRaises(AttributeError, msg='Predict before fit should raise AttributeError.'):
            model.predict()

        with self.assertRaises(AttributeError, msg='Fit before train_test_split should raise AttributeError.'):
            model.fit()

    def test_hyperparameter_tuning_before_train_test_split_raises(self):
        """ hyperparameter_tuning() called before train_test_split() should raise RuntimeError. """
        model = Model(self.dummy_X_2D, self.dummy_Y, 'randomforest')
        param_grid = {'n_estimators': [50, 100]}
        with self.assertRaises(RuntimeError,
                msg='RuntimeError expected when hyperparameter_tuning called before train_test_split.'):
            model.hyperparameter_tuning(param_grid=param_grid, verbose=0)
    
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
