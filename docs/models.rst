Models
======

The ``Model`` class (``pySAR/model.py``) provides a unified interface for building, fitting,
evaluating, and saving scikit-learn regression models. All 16 supported algorithms share the same 
API: construct with feature data and labels, split the dataset, fit, predict, and optionally 
run hyperparameter tuning or feature selection.

.. code-block:: python

    from pySAR.model import Model
    import numpy as np

    X = np.random.rand(100, 50)   # feature matrix
    Y = np.random.rand(100)       # activity labels

    model = Model(X, Y, algorithm="randomforest", parameters={"n_estimators": 200})
    model.train_test_split(test_split=0.2)
    model.fit()
    predictions = model.predict()

----

Supported Algorithms
--------------------

Algorithm names are matched with fuzzy matching (``difflib``), so approximate strings
(e.g. ``"randomforest"``, ``"plsreg"``, ``"knn"``) are accepted.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Algorithm
     - Accepted Aliases
     - scikit-learn Class
   * - PLS Regression
     - ``plsregression``
     - ``PLSRegression``
   * - Random Forest
     - ``randomforestregressor``
     - ``RandomForestRegressor``
   * - AdaBoost
     - ``adaboostregressor``
     - ``AdaBoostRegressor``
   * - Bagging
     - ``baggingregressor``
     - ``BaggingRegressor``
   * - Decision Tree
     - ``decisiontreeregressor``
     - ``DecisionTreeRegressor``
   * - Gradient Boosting
     - ``gbr``, ``gradientboost``, ``gradientboostingregressor``
     - ``GradientBoostingRegressor``
   * - Histogram Gradient Boosting
     - ``histgradientboosting``, ``histgradientboostingregressor``, ``hgbr``
     - ``HistGradientBoostingRegressor``
   * - Linear Regression
     - ``linearregression``
     - ``LinearRegression``
   * - Lasso
     - ``lasso``
     - ``Lasso``
   * - Ridge
     - ``ridge``
     - ``Ridge``
   * - ElasticNet
     - ``elasticnet``
     - ``ElasticNet``
   * - Support Vector Regression
     - ``svr``, ``supportvectorregression``
     - ``SVR``
   * - Stochastic Gradient Descent
     - ``sgd``, ``stochasticgradientdescent``
     - ``SGDRegressor``
   * - K-Nearest Neighbours
     - ``knn``, ``kneighborsregressor``, ``knearestneighbors``
     - ``KNeighborsRegressor``
   * - Extra Trees
     - ``extratrees``, ``extratreesregressor``
     - ``ExtraTreesRegressor``
   * - Gaussian Process
     - ``gaussianprocess``, ``gaussianprocessregressor``, ``gpr``
     - ``GaussianProcessRegressor``

----

Parameters
----------

``Model.__init__(X, Y, algorithm, parameters={}, test_split=0.2)``

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``X``
     - —
     - Feature matrix (``np.ndarray``).
   * - ``Y``
     - —
     - Activity/fitness labels (``np.ndarray``).
   * - ``algorithm``
     - —
     - Name (or approximate name) of the sklearn regression algorithm to use.
   * - ``parameters``
     - ``{}``
     - Dict of algorithm-specific parameters passed directly to the sklearn constructor. An empty dict uses all sklearn defaults.
   * - ``test_split``
     - ``0.2``
     - Fraction of data reserved for testing (default 80/20 train/test split).

The ``parameters`` dict maps directly to the chosen sklearn model's constructor keyword
arguments. For example, to set the number of estimators for a Random Forest:

.. code-block:: python

    model = Model(X, Y, algorithm="randomforest", parameters={"n_estimators": 500, "max_depth": 10})

Full parameter lists for each algorithm are available in the
`scikit-learn documentation <https://scikit-learn.org/stable/index.html>`_.

----

train_test_split
----------------

Splits ``X`` and ``Y`` into training and test sets and optionally applies standard scaling.

.. code-block:: python

    model.train_test_split(test_split=0.2, scale=True, random_state=42, shuffle=True)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``test_split``
     - ``0.2``
     - Proportion of observations to reserve for the test set.
   * - ``scale``
     - ``True``
     - Whether to apply ``StandardScaler``; the scaler is fit on the training set only and then used to transform both sets, preventing data leakage.
   * - ``random_state``
     - ``None``
     - Seed for reproducible splits.
   * - ``shuffle``
     - ``True``
     - Whether to shuffle observations before splitting.

----

fit and predict
---------------

After splitting, call ``fit()`` to train the model on ``X_train`` / ``Y_train``,
then ``predict()`` to generate predictions on ``X_test``:

.. code-block:: python

    model.fit()
    predictions = model.predict()

``predict()`` returns an ``np.ndarray`` of predicted activity values for the test set.

----

Hyperparameter Tuning
---------------------

``hyperparameter_tuning()`` uses scikit-learn's ``GridSearchCV`` to exhaustively search a
user-supplied parameter grid and report the best configuration.

.. code-block:: python

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10],
    }
    model.hyperparameter_tuning(param_grid=param_grid, metric="r2", cv=5)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``param_grid``
     - ``{}``
     - Dict mapping parameter names to lists of candidate values. Keys that are not valid for the current model are silently ignored.
   * - ``metric``
     - ``"r2"``
     - scikit-learn scoring metric used to rank candidates. Any metric returned by ``sklearn.metrics.get_scorer_names()`` is accepted.
   * - ``cv``
     - ``5``
     - Number of cross-validation folds (clamped to the range 5–10).
   * - ``n_jobs``
     - ``None``
     - Number of parallel jobs. ``None`` means 1; ``-1`` uses all available cores.
   * - ``verbose``
     - ``2``
     - Verbosity level passed to ``GridSearchCV``.

After the search, the best parameters, R² score, RMSE, MSE, MAE, RPD, and explained
variance are printed. The ``GridSearchCV`` result object is stored in ``model.grid_result``.

----

Feature Selection
-----------------

``feature_selection()`` applies a dimensionality-reduction technique to ``X`` and ``Y``
and returns the reduced feature matrix.

.. code-block:: python

    X_reduced = model.feature_selection(method="rfe")

    # Select the top-3 features with SelectKBest
    X_top3 = model.feature_selection(method="selectkbest", k=3)

    # Select 1 feature using chi2
    X_chi2 = model.feature_selection(method="chi2", k=1)

The ``k`` parameter controls how many features are retained for the ``selectkbest`` and
``chi2`` methods (defaults: ``1`` and ``2`` respectively). It is ignored for other
methods.

The following methods are supported. Names are matched approximately, so ``"kbest"``
will resolve to ``selectkbest``, for example.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``selectkbest`` / ``chi2``
     - Selects features by the highest scores under the ``f_regression`` scoring function (``SelectKBest``). Use the ``k`` parameter to control the number of features selected.
   * - ``variancethreshold``
     - Removes features whose variance does not exceed a threshold (``VarianceThreshold``).
   * - ``rfe``
     - Recursive Feature Elimination — iteratively removes the least important features as ranked by the model (``RFE``).
   * - ``selectfrommodel``
     - Selects features whose importance weights exceed a threshold derived from a fitted copy of the model (``SelectFromModel``).
   * - ``sequentialfeatureselector``
     - Greedy forward/backward selection, adding or removing one feature at a time (``SequentialFeatureSelector``).

----

Saving a Model
--------------

Trained models (and their associated scaler) are serialised to disk using ``pickle``:

.. code-block:: python

    model.save(save_folder="results/", model_name="my_model")
    # saves to results/my_model.pkl

The ``.pkl`` extension is appended automatically if omitted. The saved file contains a
``dict`` with two keys — ``'model'`` (the fitted sklearn estimator) and ``'scaler'``
(the fitted ``StandardScaler``, or ``None`` if scaling was disabled). Both are required
for reproducible predictions on new data.

To check whether a model has already been fitted before saving:

.. code-block:: python

    if model.model_fitted():
        model.save("results/")

.. warning::
   Only load ``.pkl`` files from trusted sources. Deserialising a malicious pickle file
   can execute arbitrary code.

----

Loading a Saved Model
---------------------

Use the ``Model.load()`` class method to restore a previously saved model:

.. code-block:: python

    from pySAR.model import Model

    loaded = Model.load("results/my_model.pkl")
    loaded.model_fitted()   # True
    loaded.scaler           # StandardScaler or None

``load()`` reconstructs a ``Model`` instance with both the fitted estimator and the
original scaler restored, so the loaded model can immediately be used to generate
predictions for new feature matrices.

.. code-block:: python

    # scale new data the same way the training data was scaled
    X_new_scaled = loaded.scaler.transform(X_new)
    preds = loaded.model.predict(X_new_scaled)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``path``
     - —
     - Absolute or relative path to a ``.pkl`` file created by ``save()``.
   * - ``allow_pickle``
     - ``True``
     - Set to ``False`` to raise ``ValueError`` immediately, disabling deserialization.
       Use this as a safety gate in environments where pickle files should never be loaded.

Raises ``OSError`` if the file does not exist, ``ValueError`` if ``allow_pickle=False``
or if the pickle does not have the expected ``{'model': ..., 'scaler': ...}`` structure.
A ``UserWarning`` is always emitted reminding callers not to load pickles from untrusted
sources.

----

Cross-Validation
----------------

``cv_score()`` evaluates the model using k-fold cross-validation on the full (X, Y)
data **without** permanently altering the fitted state — a deep copy of the model is
used internally so ``model_fit`` is preserved.

.. code-block:: python

    model.train_test_split(test_split=0.2)
    scores = model.cv_score(cv=5, metric="r2")
    print(scores)        # array of 5 per-fold R² scores
    print(scores.mean()) # mean cross-validated R²

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``cv``
     - ``5``
     - Number of cross-validation folds (must be ≥ 2; values < 2 trigger a warning and default to 5).
   * - ``metric``
     - ``'r2'``
     - Sklearn scoring string. Any value returned by ``sklearn.metrics.get_scorer_names()`` is accepted.
   * - ``n_jobs``
     - ``None``
     - Number of parallel jobs. ``None`` means 1; ``-1`` uses all available cores.

Returns a ``np.ndarray`` of *cv* scores, one per fold. Raises ``RuntimeError`` if
``train_test_split()`` has not been called yet, or ``ValueError`` for an unrecognised
scoring metric.

----

Config File
-----------

The algorithm and its parameters can be set in the ``[model]`` section of a pySAR
config JSON file:

.. code-block:: json

    {
        "model": {
            "algorithm": "randomforest",
            "parameters": {
                "n_estimators": 500,
                "max_depth": 10
            },
            "test_split": 0.2
        }
    }

This is equivalent to:

.. code-block:: python

    model = Model(X, Y, algorithm="randomforest",
                  parameters={"n_estimators": 500, "max_depth": 10},
                  test_split=0.2)
