
#in this class the user will test and evaluate the different encoding combinations
#results will be output and from there the user can enact the ProAct Class, building their chosen model

import pandas as pd
import numpy as np
import utils as utils
import datetime, time
import argparse
import itertools
import pickle
import yaml
import io
from tqdm import tqdm
from os import path, makedirs, remove
from difflib import get_close_matches
import json

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import  AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
from ProtSAR import ProtSAR
import utils as utils
from descriptors import Descriptors
from plots import plot_reg

class Encoding(ProtSAR):

    # """
    # Digital Signal Processing on protein sequences. Transform protein sequences into their spectral form
    # via an Fast Fourier Transform (FFT). The output of a Fourier Transform is a complex number C made up of
    # an imaginary I and real R component. **what does an FFT do??
    #
    # Parameters
    # ----------
    # encoded_sequences : numpy array
    #     protein sequences encoded via a specific AAI index feature value.
    #     encoded_sequences has to be at least 2 dimensions, containing at least more
    #     than 1 protein seqence. These encoded sequences are used to generate the various
    #     protein spectra.
    # spectrum : str
    #     protein spectra to generate from the protein sequences:
    # window : str
    #     window function to apply to the output of the FFT.
    # filter: str
    #     filter function to apply to the output of the FFT.
    #
    # Returns
    # -------
    #
    # """

    # def __init__(self, input_data, model, aaindex = None, desc = None):
    def __init__(self, data_json=None,dataset="",seq_col="sequence", activity="", window="hamming", filter="",
                    spectrum="",algorithm="", parameters={}, test_split=0.2):

        super().__init__(data_json=data_json,dataset=dataset,seq_col=seq_col,
                activity=activity, window=window, filter=filter,spectrum=spectrum,
                algorithm=algorithm, parameters=parameters, test_split=test_split)

        self.data = self.read_data(data_json)

        print(self.model)
        print('algorithm',self.algorithm)
        print('dataset',self.dataset)
        print('model',self.model)
        utils.create_output_dir()
    # aa_df = encoding.aai_encoding(aaindex, combo2 = False, cutoff=1, verbose=True)

    def aai_encoding(self, combo2 = False, cutoff=1, verbose=False):

        """
        Encoding all protein sequences using each of the indices in the AAI1.
        Each index encoding will be used as the feature data to build the
        predictive regression models. The metrics evaluated from the model
        for each index will be collated into a dataframe and returned.

        Returns
        -------
        aaindex_metrics_ : pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using all indices in the AAI1 for the AAI encoding strategy.

        """
        # assert(type(aaindex) ==)
        aaindex = AAIndex()
        #initialise dataframe to store all output results of AAI encoding
        aaindex_metrics_df = pd.DataFrame(columns=['Index','Category','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        #lists to store calculated metrics for each AAI1 index
        aa_list = []
        index_ = []
        category_ = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        index_count = 1

        #if encoding using combinations of 2 AAI indices then get all index combinations
        #   else get the normal list of all indices
        if combo2:
            all_features = aaindex.get_feature_codes(combo=2)
        else:
            all_features = aaindex.get_feature_codes()

        print('\n\n#######################################################################################\n')
        print('Encoding using {} AAI combinations with the parameters:\n\nSpectrum: {}\nWindow Function: {} \
            \nFilter:{}\nAlgorithm: {}\nParameters: {}\nTest Split: {}\n'.format(len(all_features), self.spectrum,\
            self.window, self.filter, repr(self.model), self.parameters, self.test_split))
        print('#######################################################################################\n')

        #cutoff index used if you only want to use a proprtion of all indices to build models with
        #   cutoff index multiplied with the total number of features and the value used as the
        #       index for the for loop.
        cutoff_index = int(len(all_features) * cutoff)
        # cutoff_index = 50

        '''
        1.) Get AAI index - encode it into a protein spectra according to the spectrum
        input parameter.
        2.) Build model using encoded AAI spectral features.
        3.) Predict and evaluate the model using the test data.
        4.) Append index and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all indices.
        6.) Output results into a final dataframe, save it and return.
        '''
        # for index in all_features[:cutoff_index]:
        for index in tqdm(all_features[:cutoff_index],unit=" indices",position=0,desc="AAIndex"):

            #if verbose, print out the current index and the current number of indices
            #   iterated through so far, increment counter to keep track of this
            if verbose:
                print('\nIndex {} ###### {}/{}'.format(index , index_count, len(all_features)))
            index_count+=1

            #if getting all
            if combo2:

                for i in index:
                    encoded_seqs = self.aaindex_encoding(i) #can call this as func inherits from ProAct

                    proDSP = ProDSP(encoded_seqs)
                    proDSP.encode_seqs()
                    aa_list.append(proDSP.spectrum_encoding)

                    category_.append(aaindex.get_category(i))

                    #need to concatenate the 2 x 261 x 466 AAI arrays into one 261 x 932
                aa_list_concat = np.concatenate((aa_list[0],aa_list[1]),axis =1)

                #transform encoded features into a dataframe
                X = pd.DataFrame(aa_list_concat)

            else:

                encoded_seqs = self.aaindex_encoding(index)
                proDSP = ProDSP(encoded_seqs)
                proDSP.encode_seqs()
                #append category of current AAI index
                category_.append(aaindex.get_category(index))

                #transform encoded features into a dataframe
                X = pd.DataFrame(proDSP.spectrum_encoding)
                # ***errror here?? proDSP.spectrum_encoding gives array of 0's

            #get observed class labels from the dataset
            Y  = self.get_activity()

            #split feature data and class labels into train and test data
            X_train, X_test, Y_train, Y_test  = self.model.train_test_split(X, Y)

            #fit model to training data
            model_fit = self.model.fit()
            #predict class labels for the test data
            Y_pred = self.model.predict()

            #initilaise instance of Evaluate class
            eval = Evaluate(Y_test,Y_pred)

            #append metrics from model to their respective lists
            index_.append(index)
            r2_.append(eval.r2)
            rmse_.append(eval.rmse)
            mse_.append(eval.mse)
            rpd_.append(eval.rpd)
            mae_.append(eval.mae)
            explained_var_.append(eval.explained_var)

        #if using combinations of 2 AAI indices then transform the category_
        #   list such that the 2 categories for the 2 indices are accounted for
        #       in one entry in the list & resulting dataframe.
        if combo2:
            category_= [ ','.join(x) for x in zip(category_[0::2], category_[1::2]) ]

        #set dataframe column values to the values of the accumulated lists.
        aaindex_metrics_= aaindex_metrics_df.copy()
        aaindex_metrics_['Index'] = index_
        #SPLIT INTO INDEX 1 , INDEX 2 ? Would allow for indices to be more easily searched
        aaindex_metrics_['Category'] = category_
        aaindex_metrics_['R2'] = r2_
        aaindex_metrics_['RMSE'] = rmse_
        aaindex_metrics_['MSE'] = mse_
        aaindex_metrics_['RPD'] = rpd_
        aaindex_metrics_['MAE'] = mae_
        aaindex_metrics_['Explained Var'] = explained_var_

        aaindex_metrics_ = aaindex_metrics_.sort_values(by=['R2'],ascending=False)

        #set results dataframe according to the number of combinations of AAI used.
        if combo2:
            save_path = 'aaindex_combo2_results'
        else:
            save_path = 'aaindex_results'

        #save results dataframe, saved to OUTPUT_DIR by default.
        utils.save_results(aaindex_metrics_,save_path)

        return aaindex_metrics_

    def descriptor_encoding(self, desc_combo = 1, verbose=True):

        """
        Encoding all protein sequences using each of the available physicochemical
        descriptors. Each descriptor encoding will be used as the feature data to
        build the predictive regression models. The metrics evaluated from the model
        for each descriptor will be collated into a dataframe and returned.

        Returns
        -------
        desc_metrics_ : pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using all descriptors for the descriptors encoding strategy.

        """

        #initialise Descriptor object with protein sequences, set all_desc to get all descriptors at once
        desc = Descriptors(self.data[self.seq_col], all_desc = True)

        #print(dir(desc))
        desc_metrics_df = pd.DataFrame(columns=['Descriptor','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        desc_list = []
        descriptor = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        msle_ = []
        desc_count = 1

        all_descriptors = desc.all_descriptors_list(desc_combo)
        print(all_descriptors)
        featureIndex = 1

        '''
        1.) Get current descriptor value from all_descriptors list.
        2.) Build model using descriptor features from current descriptor.
        3.) Predict and evaluate the model using the test data.
        4.) Append descriptor and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all descriptors.
        6.) Output results into a final dataframe, save it and return.
        '''
        for descr in all_descriptors:

            desc_ = pd.DataFrame()
            desc_list = []

            print('Descriptor: {} ###### {}/{}'.format(descr , desc_count, len(all_descriptors)))
            desc_count+=1
            if desc_combo == 2 or desc_combo == 3:

                for de in descr:
                    desc_list.append(getattr(desc, de))
                desc_ = pd.concat(desc_list,axis=1)
                # desc_ = pd.DataFrame(desc_list)
            else:
                desc_ = getattr(desc, descr)

            X = desc_

            if X.shape[0] != self.num_seqs:
              raise ValueError('Feature data doesnt have the correct number of sequences: {},\
                      proabable error in getting descriptor values'.format(self.num_seqs))

            Y  = self.get_activity()

            #if using the PlsRegression algorithm and there is only 1 feature (1-dimension)
            #in the feature data X then create a new PLSReg model with the n_components
            #parameter set to 1 instead of the default 2 - this stops the error:
            #ValueError - Invalid Number of Components: 2
            if X.shape[1] == 1 and repr(self.model) == "PLSRegression":
              tmp_model = Model('plsreg',parameters={'n_components':1})
              X_train, X_test, Y_train, Y_test  = tmp_model.train_test_split(X, Y)
              model_fit = tmp_model.fit()
              Y_pred = tmp_model.predict()
            else:
              X_train, X_test, Y_train, Y_test  = self.model.train_test_split(X, Y)
              model_fit = self.model.fit()
              Y_pred = self.model.predict()

            eval = Evaluate(Y_test,Y_pred)

            descriptor.append(descr)
            r2_.append(eval.r2)
            rmse_.append(eval.rmse)
            mse_.append(eval.mse)
            rpd_.append(eval.rpd)
            mae_.append(eval.mae)
            explained_var_.append(eval.explained_var)

        desc_metrics_= desc_metrics_df.copy()
        desc_metrics_['Descriptor'] = descriptor
        desc_metrics_['R2'] = r2_
        desc_metrics_['RMSE'] = rmse_
        desc_metrics_['MSE'] = mse_
        desc_metrics_['RPD'] = rpd_
        desc_metrics_['MAE'] = mae_
        desc_metrics_['Explained Var'] = explained_var_

        desc_metrics_ = desc_metrics_.sort_values(by=['R2'],ascending=False)

        if desc_combo == 2:
            save_path = 'desc_combo2_results'
        elif desc_combo == 3:
            save_path = 'desc_combo3_results'
        else:
            save_path = 'desc_results'

        utils.save_results(desc_metrics_,save_path)

        return desc_metrics_

    def aai_descriptor_encoding(self, desc_combo=1, verbose=True):

        """
        Encoding all protein sequences using each of the indices in the AAI1 as well
        as the descriptors. The sequences can be encoded using 1 AAI + 1 Descriptor,
        2 Descriptors or 3 Descriptors, dictated by the desc_combo input parameter:
        set this to 1,2 or 3 for what encoding combination to use, default is 1.
        Each encoding will be used as the feature data to build the predictive
        regression models. The metrics evaluated from the model for each AAI +
        Descriptor encoding combination strategy will be collated into a dataframe
        and returned.

        Returns
        -------
        aai_desc_metrics_df_ : pd.DataFrame


        """
        aaindex = AAIndex()
        desc = Descriptors(self.data[self.seq_col], all_desc = True)
        # aaindex_metrics_df = pd.DataFrame(columns=['Index_Descriptor','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])
        aaindex_metrics_df = pd.DataFrame(columns=['Index','Descriptor','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        index_ = []
        descriptor_ = []
        desc_list = []
        index_descriptor = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        index_count = 1
        desc_count = 1

        #get list of all descriptors
        all_descriptors = desc.all_descriptors_list(desc_combo)

        '''
        1.) Get AAI index - encode it into a protein spectra according to the spectrum
        input parameter.
        2.) Get all 15 descriptor values and concatenate to AAI encoding features.
        3.) Build model using concatenated AAI and Descriptor features as the training
        data.
        4.) Predict and evaluate the model using the test data.
        5.) Append index, descriptor and calculated metrics to lists.
        6.) Repeat steps 1 - 5 for all indices in the AAI.
        7.) Output results into a final dataframe, save it and return.
        '''
        for feature in (aaindex.get_feature_codes()):

            if verbose:
                print('\nIndex {} ###### {}/{}'.format(feature , index_count, len(aaindex.get_feature_codes())))
            index_count+=1

            encoded_seqs = self.aaindex_encoding(feature) #can call this as func inherits from ProAct
            proDSP = ProDSP(encoded_seqs)
            proDSP.encode_seqs()

            X_aai = pd.DataFrame(proDSP.spectrum_encoding)

            for descr in all_descriptors:

                desc_ = pd.DataFrame()
                desc_list = []

                print('Descriptor: {} ###### {}/{}'.format(descr , desc_count, len(all_descriptors)))
                desc_count+=1
                if desc_combo == 2 or desc_combo == 3:

                    for de in descr:
                        desc_list.append(getattr(desc, de))
                                # desc_ = pd.concat(desc_list,axis=1)  check if this is needed here

                    desc_ = desc_list

                    if desc_combo == 2:
                        desc_list_concat = np.concatenate((desc_[0],desc_[1]),axis =1)
                    elif desc_combo == 3:
                        desc_list_concat = np.concatenate((desc_[0],desc_[1],desc_[2]),axis =1)

                    desc_ = desc_list_concat
                    print('desclist',desc_list_concat)
                    print((desc_list_concat.shape))

                else:
                    desc_ = getattr(desc, descr)

                X = pd.DataFrame(desc_)

                Y  = self.get_activity()

                if X.shape[1] == 1 and repr(self.model) == "PLSRegression":
                  tmp_model = Model('plsreg',parameters={'n_components':1})
                  X_train, X_test, Y_train, Y_test  = tmp_model.train_test_split(X, Y)
                  model_fit = tmp_model.fit()
                  Y_pred = tmp_model.predict()
                else:
                  X_train, X_test, Y_train, Y_test  = self.model.train_test_split(X, Y)
                  model_fit = self.model.fit()
                  Y_pred = self.model.predict()

                eval = Evaluate(Y_test,Y_pred)
                index_.append(feature)
                descriptor_.append(descr)
                r2_.append(eval.r2)
                rmse_.append(eval.rmse)
                mse_.append(eval.mse)
                rpd_.append(eval.rpd)
                mae_.append(eval.mae)
                explained_var_.append(eval.explained_var)

            desc_count = 1

        aai_desc_metrics_df_= aaindex_metrics_df.copy()
        aai_desc_metrics_df_['Index'] = index_
        aai_desc_metrics_df_['Descriptor'] = descriptor_
        # aai_desc_metrics_df_['Index_Descriptor'] = (list(map(list, zip(aaindex_metrics_df_['Index'], aaindex_metrics_df_['Descriptor']))))
        # aai_desc_metrics_df_['Index_Descriptor'] = (list(map(list, zip(index_, descriptor_))))
        # aai_desc_metrics_df_.drop(['Index','Descriptor'],axis=1, inplace=True)
        aai_desc_metrics_df_['R2'] = r2_
        aai_desc_metrics_df_['RMSE'] = rmse_
        aai_desc_metrics_df_['MSE'] = mse_
        aai_desc_metrics_df_['RPD'] = rpd_
        aai_desc_metrics_df_['MAE'] = mae_
        aai_desc_metrics_df_['Explained Var'] = explained_var_

        #sort results by R2
        aai_desc_metrics_df_ = aai_desc_metrics_df_.sort_values(by=['R2'],ascending=False)

        #set save path according to the descriptor combinations type
        if desc_combo == 2:
            save_path = 'aaindex_descCombo2_results'
        elif desc_combo == 3:
            save_path = 'aaindex_descCombo3_results'
        else:
            save_path = 'aaindex_desc_results'

        #save results dataframe to specified save_path
        utils.save_results(aai_desc_metrics_df_,save_path)

        return aai_desc_metrics_df_


    def __str__(self):
        pass

    def __len__(self):
        pass
