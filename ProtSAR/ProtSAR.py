
#########################################################################
###                         ProtSAR                                   ###
#########################################################################

import pandas as pd
import numpy as np
import datetime, time
import argparse
import itertools
import pickle
import yaml
import io
import os
from difflib import get_close_matches
import json

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import  AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
import utils as utils
from plots import plot_reg
import descriptors as descriptors

####Add StanardScaler after every AAIndex encoding and before model building####

class ProtSAR():

    def __init__(self,data_json=None,dataset="",seq_col="sequence", activity="",\
        aa_indices="", window="hamming", filter="",spectrum="power", descriptors="", \
        algorithm="",parameters={}, test_split=0.2):

        self.data_json = data_json
        self.dataset = dataset
        self.seq_col = seq_col
        self.activity = activity
        self.window = window
        self.filter = filter
        self.spectrum = spectrum
        self.aa_indices = aa_indices
        self.descriptors = descriptors
        self.algorithm = algorithm
        self.parameters = parameters
        self.test_split = test_split

        if self.data_json!="" and self.data_json!=None:

            self.dataset, self.seq_col, self.activity, self.aa_indices, self.window, \
            self.filter, self.spectrum, self.descriptors, self.algorithm, \
            self.parameters, self.test_split = utils.parse_json(data_json)

        self.data = self.read_data(data_json)

        self.num_seqs = len(self.data[self.seq_col])
        self.seq_len = len(self.data[self.seq_col][0])

        self.preprocessing()

        self.model = Model(algorithm=self.algorithm, parameters=self.parameters, test_split=self.test_split)

        utils.create_output_dir()

        all_attrs = vars(self)
        # del all_attrs['data']

        # with open("sample.json", "w") as outfile:
        #     json.dump(all_attrs, outfile)
        # print(', '.join("%s: %s" % item for item in all_attrs.items()))
        # a_file = open("data.pkl", "wb")
        # pickle.dump(all_attrs, a_file)
        # a_file.close()

        #output parameters to dict/json/yml

    def read_data(self, data_json):

        dataset_file = os.path.join(DATA_DIR,self.dataset)

        if (os.path.isfile(dataset_file)):
            try:
                data = pd.read_csv(dataset_file, sep=",", header=0)
                return data
            except IOError as e:
                print('Error opening dataset file: ',dataset)
        else:
            raise IOError('Dataset file not found in directory: {}'.format(dataset_file))

    def preprocessing(self):

        seqColMatches = get_close_matches(self.seq_col,self.data.columns)

        if seqColMatches!=[]:
            self.seq_col = seqColMatches[0]
        else:
            raise ValueError('Sequence Column not present in dataset columns - {} /n '.format(self.data.columns))

        self.data[self.seq_col] = utils.remove_gaps(self.data[self.seq_col])

        invalid_seqs = utils.valid_sequence(self.get_seqs())
        if invalid_seqs!=None:
            raise ValueError('Invalid Amino Acids found in protein sequence dataset: {}'.format(invalid_seqs))

        activityMatches = get_close_matches(self.activity,self.data.columns)

        if activityMatches!=[]:
            self.activity = activityMatches[0]
        else:
            raise ValueError('Activity Column not present in dataset columns - {} /n '.format(list(self.data.columns)))

        self.data[self.activity].replace([np.inf,-np.inf], np.nan)
        self.data[self.activity].fillna(0,inplace=True)

    # def aaindex_encoding(self, aaindex):
    def aaindex_encoding(self, indices):

        aaindex = AAIndex()

        encoded_indices = []

        if not isinstance(indices, list):

            if indices == None or indices == "":
                raise ValueError('No AA Indices have been input, aa_indices attribute is empty')

            # encoded_aai = aaindex.get_feature_from_code(self.aa_indices)['values']

            encoded_aai = aaindex.get_feature_from_code(indices)['values']
            # encoded_vals = list((aa_index.get_feature_from_code(indices)['values']).values())

            temp_seq_vals = []
            temp_all_seqs = []

            for protein in range(0, len(self.data[self.seq_col])):
                for aa in self.data[self.seq_col][protein]:
                    temp_seq_vals.append(encoded_aai[aa])

                temp_all_seqs.append(temp_seq_vals)
                temp_seq_vals = []

            temp_all_seqs = utils.zero_padding(temp_all_seqs)

            temp_all_seqs = np.array(temp_all_seqs)
            # encoded_aai_reshaped = np.reshape(temp_all_seqs, (self.get_num_seqs(), self.get_seq_len()))

            return temp_all_seqs

        else:

            print('here1')
            encoded_aai_reshaped = np.zeros((self.num_seqs, self.seq_len))

            #if multiple indices used then calcualte FFT encoding for each and then concatenate after each calculation
            for ind in range(0,len(indices)):
                encoded_aai = aaindex.get_feature_from_code(indices[ind])['values']

                temp_seq_vals = []
                temp_all_seqs = []
                #reshaping issue here caused by zero padding - all seqs were not of the same length
                for protein in range(0, len(self.data[self.seq_col])):
                    for aa in self.data[self.seq_col][protein]:
                        temp_seq_vals.append(encoded_aai[aa])

                    temp_all_seqs.append(temp_seq_vals)
                    temp_seq_vals = []

                temp_all_seqs = utils.zero_padding(temp_all_seqs)

                temp_all_seqs =np.array(temp_all_seqs)

                # encoded_aai_ = np.reshape(temp_all_seqs, (self.get_num_seqs(), self.get_seq_len()))

                #in first iteration through indices set encoded_ai_ to zeros initialised np array, else concatenate to array in previous iteration
                if ind == 0:
                    encoded_aai_reshaped = temp_all_seqs
                else:
                    encoded_aai_reshaped = np.concatenate((encoded_aai_reshaped,temp_all_seqs), axis=1)

            self.encoded_aai_reshaped = encoded_aai_reshaped

            return self.encoded_aai_reshaped


    def encode_aaindices(self, verbose=True):

        aaindex = AAIndex()
        # model = self.model.copy()

        aai_df = pd.DataFrame(columns=['Index','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        print('encoding using',self.aa_indices)
        encoded_seqs = self.aaindex_encoding(self.aa_indices)
        proDSP = ProDSP(encoded_seqs, spectrum=self.spectrum, window=self.window, filter=self.filter)
        proDSP.encode_seqs()
        print('spectral encoding', proDSP.spectrum_encoding.shape)
        X = pd.DataFrame(proDSP.spectrum_encoding)

        # if len(self.aa_indices)>1:
        #     scale=True
        # else:
        #     scale=False

        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X, self.get_activity(), test_size = self.test_split)

        self.model.fit()
        Y_pred = self.model.predict()

        eval = Evaluate(Y_test, Y_pred)

        print(self.aa_indices)

        # aai_df['Index'] = ' '.join(self.aa_indices)
        # aai_df['Index'] = self.aa_indices
        # aai_df['R2'] = eval.r2
        # aai_df['RMSE'] = eval.rmse
        # aai_df['MSE'] = eval.mse
        # aai_df['RPD'] = eval.rpd
        # aai_df['MAE'] = eval.mae
        # aai_df['Explained Var'] = eval.explained_var

        aai_dict = eval.all_metrics()

        print('aai_dict',aai_dict)
        self.output_results(aai_dict)

        plot_reg(Y_test, Y_pred, eval.r2)

        utils.save_results(aai_dict, 'aai_encoding')

        return aai_dict

    def desc_encoding(self):

        #split this into 2 funcs: 1 does the desc encoding and then 1 does the model building and results
        if self.descriptors == None or self.descriptors == "":
            raise ValueError('No descriptors have been input, descriptors attribute is empty')

        encoded_desc = []
        encoded_desc_vals = []

        desc = descriptors.Descriptors(self.data[self.seq_col])
        # if not isinstance(self.descriptors, list):

        # encoded_desc = desc.get_descriptor_encoding(self.descriptors)
        for d in range(0,len(self.descriptors)):
            encoded_desc = desc.get_descriptor_encoding(self.descriptors[d])
            encoded_desc_vals.append(list(encoded_desc))
            encoded_desc = []

        X = pd.DataFrame(np.concatenate(encoded_desc_vals))

        return X

    def encoded_desc(self):

        desc_df = pd.DataFrame(columns=['Descriptor','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        X = self.desc_encoding()

        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X, self.get_activity(), test_size = self.test_split)

        self.model.fit()
        Y_pred = self.model.predict()

        eval = Evaluate(Y_test, Y_pred)

        # aai_df['Index'] = ' '.join(self.aa_indices)
        # desc_df['Descriptor'] = self.aa_indices
        # desc_df['R2'] = eval.r2
        # desc_df['RMSE'] = eval.rmse
        # desc_df['MSE'] = eval.mse
        # desc_df['RPD'] = eval.rpd
        # desc_df['MAE'] = eval.mae
        # desc_df['Explained Var'] = eval.explained_var

        desc_dict = eval.all_metrics()

        self.output_results(desc_dict)

        plot_reg(Y_test, Y_pred, eval.r2)

        utils.save_results(desc_dict, 'desc_encoding')

        return desc_dict

        # encoded_desc_vals = (list(itertools.chain.from_iterable(encoded_desc_vals)))
        # encoded_desc_vals = np.reshape(encoded_desc_vals, (self.get_num_seqs(), len(encoded_desc[0])))

    def encode_aai_descriptor(self):

        """
        Encode using both AAI1 indices and the descriptors. Function will parse
        the classes input parameters to get the required AA indices and descriptors
        to use and calculate their respective values. The two outputs from the
        individual encoding strategies will be concatenated together and used in
        the building of the predictve models. The resulting model assets and its
        results will be exported to the directory pointed to by the global variable
        OUTPUT_DIR.

        Parameters
        ----------

        Returns
        -------
        mse : float

        """
        if (self.descriptors == None or self.descriptors == "") \
            or (self.aa_indices == None or self.aa_indices == ""):
            raise ValueError('AAI and descriptor values both need to be present')

        aaindex = AAIndex()
        aai_desc_df = pd.DataFrame(columns=['Index_Descriptor','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        aai_encoding = self.aaindex_encoding(aaindex)
        aai_encoding = pd.DataFrame(aai_encoding)
        descriptor_encoding=self.desc_encoding()

        aai_desc = [aai_encoding, descriptor_encoding]
        X = pd.concat(aai_desc, axis = 1)

        print('aai_encoding',aai_encoding.shape)
        print('descriptor_encoding',descriptor_encoding.shape)

        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X, self.get_activity(), test_size = self.test_split)

        self.model.fit()
        Y_pred = self.model.predict()

        eval = Evaluate(Y_test, Y_pred)

        index_desc = (str(self.aa_indices) + str(self.descriptors))
        aai_desc_df['Index_Descriptor'] = index_desc
        aai_desc_df['R2'] = eval.r2
        aai_desc_df['RMSE'] = eval.rmse
        aai_desc_df['MSE'] = eval.mse
        aai_desc_df['RPD'] = eval.rpd
        aai_desc_df['MAE'] = eval.mae
        aai_desc_df['Explained Var'] = eval.explained_var

        aai_desc_dict = eval.all_metrics()

        plot_reg(Y_test, Y_pred, eval.r2)

        self.output_results(aai_desc_dict)

        utils.save_results(aai_desc_dict, 'aai_desc_encoding')


    def output_results(self, results):

        print('\n####################################')
        print('############# Results ##############')
        print('####################################\n')

        print('############ Parameters ############\n')
        print('Dataset -> {}\nActivity -> {}\nSpectrum -> {}\nWindow -> {}\nFilter -> {}\nAAIndices -> {}\nDescriptors -> {}\nAlgorithm -> {}\nParameters -> {}\nTest Split -> {}\n'
                        .format(self.dataset,self.activity, self.spectrum, self.window, self.filter,
                                self.aa_indices, self.descriptors, repr(self.model), self.parameters, self.test_split))

        print('############# Metrics #############\n')
        print('# R2: {}'.format(results['R2']))
        print('# RMSE: {} '.format(results['RMSE']))
        print('# MSE: {} '.format(results['MSE']))
        print('# MAE: {}'.format(results['MAE']))
        print('# RPD {}'.format(results['RPD']))
        print('# Variance {}\n'.format(results['Explained Var']))
        print('###################################')

    def get_seqs(self):

        return self.data[self.seq_col]

    def get_names(self):

        return self.data['name']

    def get_activity(self):

        return self.data[self.activity].values.reshape((-1,1))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indenst=4)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, val):
        self._dataset = val

    @property
    def seq_col(self):
        return self._seq_col

    @seq_col.setter
    def seq_col(self, val):
        self._seq_col = val

    @property
    def activity(self):
        return self._activity

    @activity.setter
    def activity(self, val):
        self._activity = val

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        self._window = val

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, val):
        self._filter = val

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, val):
        self._spectrum = val

    @property
    def aa_indices(self):
        return self._aa_indices

    @aa_indices.setter
    def aa_indices(self, val):
        self._aa_indices = val

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, val):
        self._descriptors = val

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, val):
        self._algorithm = val

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        self._parameters = val

    @property
    def test_split(self):
        return self._test_split

    @test_split.setter
    def test_split(self, val):
        self._test_split = val

    @property
    def num_seqs(self):
        return self._num_seqs

    @num_seqs.setter
    def num_seqs(self, val):
        self._num_seqs = val

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, val):
        self._seq_len = val

    def __len__(self):

        return len(self.data['name'])

    def __str__(self):
        pass

    def __repr__(self):
        pass
