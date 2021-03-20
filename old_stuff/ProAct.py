from aaindex import *
from proteinSpectra import *
from evaluate import *
from utils import *
from data import *
from descriptors import *
from model import *
from plots import *

import pandas as pd
import argparse

#1.) Dataset path and name passed as parameter to class - __init__ func
#2.) Import dataset and seqeunces into dataframe
#3.) Pre-processing all sequences
#4.) Import AAIndex Class
#5.) Exit class initialisation process
#6.) Class func to calcualte AAIndex def calculate_aai(index):
#7.) Class func to calculate descriptors def calcualte_descriptor(desc)
#8.) If AAI and Descriptors applied, concat function to concatenate features
#9.) Class func to build predictve model from accumulated features
#10.) Class func to plot and output results
class ProAct():

    def __init__(self, dataset, algorithm, aa_indices="", descriptors=""):

        self.dataset = dataset      #dataset path
        self.aa_indices = list(aa_indices)
        self.descriptors = list(descriptors)
        self.algorithm = algorithm

        # dataset_path = os.path.join('data',dataset +'.txt')
        self.data = pd.read_csv(self.dataset, sep=",", header=0)

    #     if type(aa_indices) is list:
    #         for x in aa_indices list:
    #             get aa index encoding
    #             encoded_vals = aa_index.get_feature_from_code(aa_indices[x])['values']
    #             proteinDSP = ProteinDSP()
    #             proteinDSP.get_power()
    #             append power spec to DF for each Index
    # else:
    #         encoded_vals = aa_index.get_feature_from_code(aa_indices[x])['values']
    #         proteinDSP = ProteinDSP()
    #         proteinDSP.get_power()
    #         append power spec to DF for each Index


    # def get_dataset(self):
    #
    #     # # assert dataset in ['t50', 'enantioselectivity', 'absorption','localization'], 'Dataset type must be ....'
    #     #
    #     # dataset_path = os.path.join('data',dataset +'.txt')
    #     # data = pd.read_csv(dataset_path, sep=",", header=0)
    #
    #     return self.data

    def get_aaindex_features(self, spectra='power', window_type="", filter_type=""):

        aa_index_features = []
        if (self.aa_indices != []):
            aa_index = AAIndex()
            for index in range(0,len(self.aa_indices)):
                assert index in aa_index.get_feature_codes()

                aa_index_features.append(aa_index.get_feature_from_code(index)['values'])

        data_with_spectra = self.get_dataset(self.dataset).copy()

        proteinDSP = ProteinDSP(spectra, window_type, filter_type)
        aa_index_encoding = []
        for feature in aa_index_features:
            aa_index_encoding.append(proteinDSP.get_power())

        return aa_index_encoding

    # def get_descriptor_features(self):
    #
    #     if self.descriptors != []:
    #         desc_features = []
    #         for d in range(0, len(self.descriptors)):
    #             desc_features.append(desc.get_descriptor(d))
    #
    #     return desc_features

    def calculate_descriptors():
        pass

def main(args):

    dataset = args.dataset
    aa_indices_filepath = args.aa_indices_filepath
    aa_indices = args.aa_indices
    descriptors = args.descriptors
    algorithm = args.algorithm

    proAct = ProAct(dataset,aa_indices, descriptors, algorithm)

    data = proAct.get_dataset(dataset)

    if (aa_indices != []):
        aa_index_features = proAct.get_aaindex_features()
        data['AAIndex_Features'] = aa_index_features
        X_aaindex = get_AAIndex_X(aa_index_features)
        X_aaindex = pd.DataFrame(X_aaindex)

    if (descriptors != []):
        desc = Descriptor()
        data['Descriptor_features'] = descriptor_features
        X_desc = proAct.get_descriptor_features(desc)

    #append features list to data DF

    if (aa_indices != [] and descriptors != []):
        X = [X_aaindex, X_desc]
    elif (aa_indices != []):
        X = X_aaindex
    elif (descriptors != []):
        X = X_desc

    Y = get_Y(dataset)

    #train test split
    model = proAct.get_model()
    model = proAct.build_model(X, Y)

    Y_pred = model.predict(X_test)

    r2 = model.score(X_train, Y_train)
    # rmse =
    # mse =
    # rpd = ....


    # rmse, mse, mae, rpd, explained_var, msle = eval_metrics(Y_test, Y_pred)
    #model.predict

    #evaluate model

    #put results into Df or csv

    #plot results



#REMOVE NAN from Descripti

check for NAN's in DF
t50_descriptors.isnull().values.any()

count number of NAN's
t50_descriptors.isnull().sum().sum()

drop nan
t50_descriptors.dropna(inplace=True)

Drop all columns that have NA's
    >>> df.dropna(axis=1, how='all')

Probs better option - replacing NA's with 0
t50_descriptors.fillna(0, inplace=True) or df.replace(np.nan, 0)

ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

#count infinite values:
np.isfinite(t50_descriptors).sum().sum()

count = np.isinf(t50_descriptors).values.sum()
count

if __name__ == "__main__":

    # Program Arguments
    # **************
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dataset', '--dataset', type=str, default="t50", required = False,
                        help='')
    parser.add_argument('-aa_indices_filepath', '--aa_indices_filepath', type=str, default=(os.path.join('data','aa_index')), required = False,
                        help='')
    parser.add_argument('-aa_indices', '--aa_indices', type=str, default="VELV850101", required = False,
                        help='')
    parser.add_argument('-descriptors', '--descriptors', type=str, default="aa_comp", required = False,
                        help='')
    parser.add_argument('-algorithm', '--algorithm', type=str, default="PLSRegression", required = False,
                        help='')

    args = parser.parse_args()

    main(args)
