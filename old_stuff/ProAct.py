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


    pass


#
# def get_aaindex(indices):
#
#     loop through indices and get all index values
#     self.aaindex[]
#
#     using_aaindex_feautures = True
#
#     if using_desc_features ==1:
#         concat with aa_index_features
#     return self.concatenated_aaindex_features
#
# def get_descriptors(desc):
#
#     loop through all dwscriptors:
#     if self.descriptors doesnt exist then run import_descriptors
#
#     using_desc_feautures = True
#
#     if aa_index_features ==1:
#         concat with using_desc_feautures
#     return self.concartenatred_descriptor_feautes

#main for testing
# def main2(args):
#
#     input_path = str(args.input_data)
#
#     assert os.path.isfile(input_path), 'Input data file not correct filepath'
#     _, file_extension = os.path.splitext(input_path)
#     print(file_extension)
#
#     if (file_extension == '.yml') or (file_extension == '.yaml'):
#         print('hello')
#         with open(input_path) as f:
#             # The FullLoader parameter handles the conversion from YAML
#             # scalar values to Python the dictionary format
#             data = yaml.load(f, Loader=yaml.FullLoader)
#
#     elif (file_extension == '.json'):
#         with open(input_path) as f:
#             data = json.load(f)
#         dataset = data['dataset']
#         activity = data['activity']
#
#         print(dataset)
#         print(activity)
#
#         #parse json
#     else:
#         raise ValueError('Input data file must be of type yml or json')


# def encode_aaindices(self, combo2 = False, verbose=True):
#
#     aaindex = AAIndex()
#     model = self.model.copy()
#     features = aaindex.get_feature_codes(combo2)
#
#     r2_ = []
#     mse_ = []
#     index_ = []
#     rmse_ = []
#     rpd_ = []
#     mae_ = []
#     explained_var_ = []
#     index_count = 1
#
#     aai_df = pd.DataFrame(columns=['Index','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])
#
#     for feature in features:
#
#         if verbose:
#           print('\nIndex {} using {} spectrum ###### {}/{}'.format(feature , self.spectrum,index_count, len(features)))
#           index_count+=1
#
#         if combo2:
#
#             encoded_seqs = proAct.aaindex_encoding(aaindex,feature[0])
#             proDSP = ProDSP(encoded_seqs, spectrum=self.spectrum, window=self.window, filter=self.filter)
#             proDSP.encode_seqs()
#             aa_1 = pd.DataFrame(proDSP.spectrum_encoding)
#
#             encoded_seqs = proAct.aaindex_encoding(aaindex,feature[1])
#             proDSP = ProDSP(encoded_seqs, spectrum=self.spectrum, window=self.window, filter=self.filter)
#             proDSP.encode_seqs()
#             aa_2 = pd.DataFrame(proDSP.spectrum_encoding)
#
#             X_ = [aa_1, aa_2]
#             X = pd.concat(X_, axis = 1)
#
#             X = scaler.fit_transform(X)
#
#         else:
#
#             encoded_seqs = proAct.aaindex_encoding(aaindex,feature)
#             proDSP = ProDSP(encoded_seqs, spectrum=self.spectrum, window=self.window, filter=self.filter)
#             proDSP.encode_seqs()
#             X = pd.DataFrame(proDSP.spectrum_encoding)
#
#
#         X_train, X_test, Y_train, Y_test = model.train_test_split(X, self.get_activity(), test_size = 0.2)
#
#         model_copy = self.model.copy()
#         model_copy.fit(X_train, Y_train)
#         Y_pred = model_copy.predict(X_test)
#
#         eval = Evaluate(model_copy, Y_test, Y_pred)
#
#         index_.append(feature)
#         r2_.append(eval.r2)
#         rmse_.append(eval.rmse)
#         mse_.append(eval.mse)
#         rpd_.append(eval.rpd)
#         mae_.append(eval.mae)
#         explained_var_.append(eval.explained_var)
#
#     aa_df['Index'] = index_
#     aa_df['R2'] = r2_
#     aa_df['RMSE'] = rmse_
#     aa_df['MSE'] = mse_
#     aa_df['RPD'] = rpd_
#     aa_df['MAE'] = mae_
#     aa_df['Explained Var'] = explained_var_
#
#     aa_df.sort_values(by=['R2'], ascending=False, inplace=True)
#
#     return aa_df
