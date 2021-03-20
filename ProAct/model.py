
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import pandas as pd
import numpy as np
import inspect

standardScaler = StandardScaler()
minMaxScaler = MinMaxScaler()



class Model():

    def __init__(self, algorithm,parameters={}):

        self.algorithm = algorithm
        self.parameters = parameters

        self.validModels = self.valid_models()

        modelMatches = get_close_matches(self.algorithm,self.validModels)

        if modelMatches!=[]:
            self.algorithm = modelMatches[0]
        else:
            raise ValueError('Input algorithm ('+ self.algorithm + ') not in available models /n '+self.validModels)

        self.model = self.get_model()

    # from difflib import get_close_matches - validation for when using puts in model close to existing model:
    #e.g plsregresion instead of plsregression
    #matches = get_close_matches(model_, all_models))
    #chosen_model = matches[0]

    def get_model(self):

        if self.algorithm.lower() == 'plsregression':

            model_params = set(dir(PLSRegression()))
            parameters = [i for i in model_params if i in self.parameters]

            if parameters != []:
                model = PLSRegression(**self.parameters)
            else:
                model = PLSRegression()

        # elif self.algorithm.lower() == 'randomforestregressor':
        #     model = RandomForestRegressor(**self.parameters)
        # elif self.algorithm.lower() == 'adaboostregressor':
        #     model = AdaBoostRegressor(**self.parameters)
        # elif self.algorithm.lower() == 'baggingregressor':
        #     model = BaggingRegressor(**self.parameters)
        # elif self.algorithm.lower() == 'decisiontreeregressor':
        #     model = DecisionTreeRegressor(**self.parameters)
        # elif self.algorithm.lower() == 'linearregression':
        #     model = LinearRegression(**self.parameters)
        # elif self.algorithm.lower() == 'lasso':
        #     model = Lasso(**self.parameters)
        # elif self.algorithm.lower() == 'svr':
        #     model = SVR(**self.parameters)
        # elif self.algorithm.lower() == 'knn':
        #     model = KNeighborsRegressor(**self.parameters)

        return model


    def build(self,X, Y,scale=True, test_size =0.2):



    # model.build(encoded_aa_power, proAct.get_activity())


        if scale:
            X = standardScaler.fit_transform(X)
            #****
        print('here 2')

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.model.fit(X_train, Y_train)

        self.model_fit = self.model

        # return self.model_fit

    def valid_models(self):

        validModels = ['PlsRegression','RandomForestRegressor','AdaBoostRegressor','BaggingRegressor',
                        'DecisionTreeRegressor','LinearRegression','Lasso','SVR','KNeighborsRegressor']

        return validModels


# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report model performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#





#Add below methods to classes??
# ['__abstractmethods__',
#  '__class__',
#  '__delattr__',
#  '__dict__',
#  '__dir__',
#  '__doc__',
#  '__eq__',
#  '__format__',
#  '__ge__',
#  '__getattribute__',
#  '__getstate__',
#  '__gt__',
#  '__hash__',
#  '__init__',
#  '__init_subclass__',
#  '__le__',
#  '__lt__',
#  '__module__',
#  '__ne__',
#  '__new__',
#  '__reduce__',
#  '__reduce_ex__',
#  '__repr__',
#  '__setattr__',
#  '__setstate__',
#  '__sizeof__',
#  '__str__',
#  '__subclasshook__',
#  '__weakref__',
