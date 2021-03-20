
from sklearn.model_selection import GridSearchCV
import pandas as pd

class HyperparameterTuning():

    def __init__(self, model, parameters, X, Y, metric='r2', cv=5):

        self.model = model
        self.parameters = parameters
        self.X =
        self.Y = Y
        self.metric = metric
        self.cv = cv

        assert metric in sorted(sklearn.metrics.SCORERS.keys())
        assert cv >=5 && cv <=10

    def hyperparameter_tuning(self, verbose):

        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=metric,error_score=0)
        grid_result = grid_search.fit(X_train, Y_train)

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        self.best_score = grid_result.best_score_
        self.best_params = grid_result.best_params_

        if verbose:
            print("Best results for index %s: %f using %s" % (index, grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':
    pass

# def hyperparameter_tuning(model, grid, X_train, Y_train, index, metric='r2', cv=10, verbose = False):     #metric='accuracy' is more for classification? changing to r2_score
#
#     assert metric in sorted(sklearn.metrics.SCORERS.keys())
#
#     grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=metric,error_score=0)
#     grid_result = grid_search.fit(X_train, Y_train)
#
#     # summarize results
#     if verbose:
#        print("Best results for index %s: %f using %s" % (index, grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#
#     if verbose:
#       for mean, stdev, param in zip(means, stds, params):
#           print("%f (%f) with: %r" % (mean, stdev, param))
#
#     # classifier_scores["PLS"] = grid_result.best_score_
#     # classifier_params["PLS"] = grid_result.best_params_
#
#     return grid_result.best_score_, grid_result.best_params_
