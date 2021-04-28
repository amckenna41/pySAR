
################################################################################
#################                    Encoding                  #################
################################################################################

import pandas as pd
import numpy as np
import utils as utils
import datetime, time
import argparse
import itertools
import pickle
import yaml
import io
import sys
# from tqdm import tqdm
from tqdm.auto import tqdm
from os import path, makedirs, remove
from difflib import get_close_matches
import json

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import  AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
from pySAR import PySAR
import utils as utils
from descriptors import Descriptors
from plots import plot_reg

#update comments/docstring of aai_encoding func
#update comments/docstring of aai_desc_encoding func
#functuonality where user can input what descriptors / aa_indices they want to encode with??
#update comnments for input params of all 3 encodings - now accepted user defined indices & desc!

class Encoding(PySAR):
    """
    The use-case of this class is when you have a dataset of protein sequences with
    a sought-after protein activity value and you want to measure this activity
    value for new and unseen sequences that have not had their activity value
    experimentally measured. The encoding class allows for evaluation of a variety
    of potential techniques at which to numerically encode the protein sequences,
    allowing for the builiding of predictive regression models that can ultimately
    predict the activity value of an unseen protein sequence. The strategies each
    generate a huge number of potential models that you can then assess for performance
    and predictability, selecting the best-performing model out of all those evaluated.

    The encoding class inherits from the main PySAR module and allows for a
    dataset of protein sequences to be encoded through 3 main strategies: AAI Indices,
    protein Descriptors and AAI Indices + protein Descriptors. The encoding class
    and its methods differ from the PySAR class by allowing for the encoding of
    all available features. To date, there are 566 indices supported in the AAI and
    pySAR supports 15 different descriptors. The features can be encoeed using
    different combinations, for example, 1 2 or 3 descriptors can be used for the
    descriptor and AAI + Descriptor encoding strategies. In total, this class
    supports over 410,000 potential ways at which to encode the protein sequences
    in the building of the predictive models.

    Parameters
    ----------
    Refer to pySAR module doctring for description of parameters

    Methods
    -------
    aai_encoding(use_dsp=True, verbose=True, cutoff_index=1):
        encoding protein sequences using indices from the AAI
    descriptor_encoding(desc_combo=1, verbose=True):
        encoding protein sequences using protein descriptors from descrtipors module
    aai_descriptor_encoding(desc_combo=1, verbose=True, use_dsp=True):
        encoding protein sequences using indices from the AAI in concatenation
        with the protein descriptors from the descriptors module
    """

    def __init__(self, dataset="",seq_col="sequence", activity="", \
            algorithm="", parameters={}, test_split=0.2,descriptors_csv="descriptors.csv"):

        super().__init__(dataset=dataset,seq_col=seq_col, \
                activity=activity, algorithm=algorithm, parameters=parameters,
                test_split=test_split, descriptors_csv=descriptors_csv)

        # self.data = self.read_data(data_json)

        #create output directory to store all of the program's outputs
        utils.create_output_dir()

    def aai_encoding(self, use_dsp=True, aai_list=None, spectrum='power',window='hamming',
        filter_="", verbose=True, cutoff_index=1):
        """
        Encoding all protein sequences using each of the available indices in the
        AAI. The protein spectra of the AAI indices will be generated if use_dsp is true,
        dictated by the instance attributes: spectrum, window and filter. If not true then
        the encoced sequences from the AAI will be used. Each encoding will be
        used as the feature data to build the predictive regression models. To date,
        there are 566 indices in the AAI, therefore 566 total models can be built using
        this encoding strategy. The metrics evaluated from the model for each AAI
        encoding combination will be collated into a dataframe, saved and returned.

        Parameters
        ----------
        use_dsp : bool (default = True)
            if true then pass AAI indices encoding through the DSP pipeline to
            create protein spectra, dictated by instance attributes: spectrum,
            window and filter. If false then AAI indices encoding used as feature data.
        verbose : bool (default = True)
            if true, the progress of the encoding will be output, else output will
            only occur when the encoding has finished.
        cutoff_index : float (default = 1)
            set the proportion of AAindex indices to use for the encoding process.
            Default=1 means all AAI indices will be used, 0.5 means 50% of indices used etc.
            This parameter primarily just used for speeding up testing the functions etc.

        Returns
        -------
        aaindex_metrics_df : pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using all indices in the AAI for the AAI encoding strategy.
        """

        #initialise dataframe to store all output results of AAI encoding
        aaindex_metrics_df = pd.DataFrame(columns=['Index','Category','R2', 'RMSE',
            'MSE', 'RPD', 'MAE', 'Explained Var'])

        #lists to store results for each predictive model
        aa_list = []
        index_ = []
        category_ = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        index_count = 1        #counters to keep track of current index

        if aai_list == None or aai_list == [] or aai_list == "":
            all_indices = self.aaindex.get_record_codes()
        else:
            all_indices = aai_list

        #get list of all indices in the AAI
        # all_indices = self.aaindex.get_record_codes()

        print('\n\n#######################################################################################\n')
        print('Encoding using {} AAI combinations with the parameters:\n\nSpectrum: {}\nWindow Function: {} \
            \nFilter:{}\nAlgorithm: {}\nParameters: {}\nTest Split: {}\n'.format(len(all_indices), spectrum,\
            window, filter_, repr(self.model), self.parameters, self.test_split))
        print('#######################################################################################\n')

        #cutoff index used if you only want to use a proprtion of all indices to build models with
        #   cutoff index multiplied with the total number of features and the value used as the
        #       index for the for loop, default=1 meaning all indices used
        cutoff_index = int(len(all_indices) * cutoff_index)
        start = time.time() #start counter

        '''
        1.) Get AAI index encoding of protein sequences, if using DSP (use_dsp = True),
        create instance of proDSP class and generate protein spectra from the AAI
        indices, according to instance parameters: spectrum, window and filter.
        2.) Build model using encoded AAI indices or protein spectra as features.
        3.) Predict and evaluate the model using the test data.
        4.) Append index and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all indices.
        6.) Output results into a final dataframe, save it and return.
        '''

        #using tqdm package to create a progress bar showing encoding progress
        #file=sys.stdout to stop error where iterations were printing out of order
        for index in tqdm(all_indices[:cutoff_index],unit=" indices",position=0,
            desc="AAIndex",file=sys.stdout):

            #get AAI indices encoding for sequences according to index var
            encoded_seqs = self.get_aai_enoding(index)

            #generate protein spectra from proDSP class if use_dsp is true
            if use_dsp:
                proDSP = ProDSP(encoded_seqs, spectrum=spectrum,
                    window=window, filter_=filter_)
                proDSP.encode_seqs()
                X = pd.DataFrame(proDSP.spectrum_encoding)
            else:
                X = pd.DataFrame(encoded_seqs)

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

            #append values/results from current AAI encoding iteration to lists
            index_.append(index)
            category_.append(self.aaindex.get_category(index))
            r2_.append(eval.r2)
            rmse_.append(eval.rmse)
            mse_.append(eval.mse)
            rpd_.append(eval.rpd)
            mae_.append(eval.mae)
            explained_var_.append(eval.explained_var)

        end = time.time()
        elapsed = end - start
        print('\n\n##############################################################')
        print('Elapsed Time for AAI + Descriptor Encoding: {0:.3f} seconds'.format(elapsed))

        #set columns in the output dataframe to each of the values/metrics lists
        aaindex_metrics_= aaindex_metrics_df.copy()
        aaindex_metrics_['Index'] = index_
        aaindex_metrics_['Category'] = category_
        aaindex_metrics_['R2'] = r2_
        aaindex_metrics_['RMSE'] = rmse_
        aaindex_metrics_['MSE'] = mse_
        aaindex_metrics_['RPD'] = rpd_
        aaindex_metrics_['MAE'] = mae_
        aaindex_metrics_['Explained Var'] = explained_var_

        #sort results by R2
        aaindex_metrics_ = aaindex_metrics_.sort_values(by=['R2'],ascending=False)

        #save results dataframe, saved to OUTPUT_DIR by default.
        utils.save_results(aaindex_metrics_,'aaindex_results')

        return aaindex_metrics_

    def descriptor_encoding(self, desc_list=None, desc_combo=1, verbose=True):
        """
        Encoding all protein sequences using each of the available physicochemical
        and structural descriptors. The sequences can be encoded using combinations
        of 1, 2 or 3 of these descriptors, dictated by the desc_combo input parameter:
        set this to 1,2 or 3 for what encoding combination to use, default is 1. Each
        descriptor encoding will be used as the feature data to build the predictive
        regression models. With 15 descriptors supported by pySAR this means there
        can be 15, 105 and 455 total predictive models built for 1, 2 or 3 descriptors,
        respecitvely. The metrics evaluated from the model for each Descriptor encoding
        combination will be collated into a dataframe and saved and returned.

        Parameters
        ----------
        desc_combo : int (default = 1)
            combination of descriptors to use.
        verbose : bool (default = True)
            if true, progress of the encoding will be output else no output will
            be output until encoding has finished, true is recccomended.

        Returns
        -------
        desc_metrics_df_ : pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using all descriptors for the descriptors encoding strategy.
        """
        #initialise Descriptor object with protein sequences, set all_desc to calculate all descriptors
        desc = Descriptors(self.data[self.seq_col], all_desc = True)

        #create dataframe to store output results from models
        desc_metrics_df = pd.DataFrame(columns=['Descriptor','Group','R2', 'RMSE',
            'MSE', 'RPD', 'MAE', 'Explained Var'])

        #lists to store results for each predictive model
        desc_list = []
        descriptor = []
        descriptor_group_ = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        msle_ = []
        desc_count = 1  #counters to keep track of current index & descriptor

        if desc_list == None or desc_list == [] or desc_list == "":
            all_descriptors = desc.all_descriptors_list(desc_combo)
        else:
            all_descriptors = desc_list

        #get list of all descriptors
        # all_descriptors = desc.all_descriptors_list(desc_combo)

        print('\n\n##############################################################\n')
        print('Encoding using {} descriptor combinations with the parameters:\n \
            \nAlgorithm: {}\nParameters: {}\nTest Split: {}\n'.format(len(all_descriptors),\
            repr(self.model), self.parameters, self.test_split))
        print('##################################################################\n')

        start = time.time() #start counter

        '''
        1.) Get current descriptor value or combination of descriptors from all_descriptors list.
        2.) Build model using descriptor features from current descriptor(s).
        3.) Predict and evaluate the model using the test data.
        4.) Append descriptor(s) and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all descriptors.
        6.) Output results into a final dataframe, save it and return.
        '''
        for descr in tqdm(all_descriptors,unit=" descriptor",position=0,desc="Descriptors",file=sys.stdout):

            desc_ = pd.DataFrame()           #reset descriptor DF and list
            desc_list = []
            desc_count+=1          #counter to keep track of current descriptor

            #if using 2 or 3 descriptors, append each descriptor & its category to list
            if desc_combo == 2 or desc_combo == 3:
                for de in descr:

                    desc_list.append(getattr(desc, de))
                    descriptor_group_.append(desc.descriptor_groups[de])

                desc_ = pd.concat(desc_list,axis=1) #concatenate descriptors
                # desc_ = desc_list
                # desc_ = pd.DataFrame(desc_list)
            else:
                desc_ = getattr(desc, descr)
                descriptor_group_.append(desc.descriptor_groups[descr])

            X = desc_

            #get protein activity values
            Y  = self.get_activity()

            '''
            if using the PlsRegression algorithm and there is only 1 feature (1-dimension)
            in the feature data X then create a new PLSReg model with the n_components
            parameter set to 1 instead of the default 2 - this stops the error:
            ValueError - Invalid Number of Components: 2

            Also, get train/test split, fit model and predict activity of test data.
            '''
            if X.shape[1] == 1 and repr(self.model) == "PLSRegression":
              tmp_model = Model('plsreg',parameters={'n_components':1})
              X_train, X_test, Y_train, Y_test  = tmp_model.train_test_split(X, Y)
              model_fit = tmp_model.fit()
              Y_pred = tmp_model.predict()
            else:
              X_train, X_test, Y_train, Y_test  = self.model.train_test_split(X, Y)
              model_fit = self.model.fit()
              Y_pred = self.model.predict()

            #create instance of Evaluate class
            eval = Evaluate(Y_test,Y_pred)

            #append values/results from current descriptor encoding iteration to lists
            descriptor.append(descr)
            r2_.append(eval.r2)
            rmse_.append(eval.rmse)
            mse_.append(eval.mse)
            rpd_.append(eval.rpd)
            mae_.append(eval.mae)
            explained_var_.append(eval.explained_var)

        end = time.time()
        elapsed = end - start
        print('\n\n##############################################################')
        print('Elapsed Time for Descriptor Encoding: {0:.3f} seconds'.format(elapsed))

        #if using combinations of 2 or 3 descriptors, group every 2 or 3 descriptor
        #   groups into one element in the descriptor group list
        if desc_combo == 2:
          descriptor_group_= [ ','.join(x) for x in zip(descriptor_group_[0::2],
            descriptor_group_[1::2]) ]
        elif desc_combo == 3:
          descriptor_group_= [ ','.join(x) for x in zip(descriptor_group_[0::3],
            descriptor_group_[1::3],  descriptor_group_[2::3]) ]

        #set columns in the output dataframe to each of the values/metrics lists
        desc_metrics_df_= desc_metrics_df.copy()
        desc_metrics_df_['Descriptor'] = descriptor
        desc_metrics_df_['Group'] = descriptor_group_
        desc_metrics_df_['R2'] = r2_
        desc_metrics_df_['RMSE'] = rmse_
        desc_metrics_df_['MSE'] = mse_
        desc_metrics_df_['RPD'] = rpd_
        desc_metrics_df_['MAE'] = mae_
        desc_metrics_df_['Explained Var'] = explained_var_

        #sort results by R2
        desc_metrics_df_ = desc_metrics_df_.sort_values(by=['R2'],ascending=False)

        #set save path according to the descriptor combinations type
        if desc_combo == 2:
            save_path = 'desc_combo2_results'
        elif desc_combo == 3:
            save_path = 'desc_combo3_results'
        else:
            save_path = 'desc_results'

        #save results dataframe to specified save_path
        utils.save_results(desc_metrics_df_,save_path)

        return desc_metrics_df_

    def aai_descriptor_encoding(self, aai_list=None, desc_list=None, desc_combo=1,
        use_dsp = True, spectrum='power', window='hamming', filter_="",verbose=True, cutoff_index=1):
        """
        Encoding all protein sequences using each of the indices in the AAI as well
        as the descriptors. The sequences can be encoded using 1 AAI + 1 Descriptor,
        2 Descriptors or 3 Descriptors, dictated by the desc_combo input parameter:
        set this to 1,2 or 3 for what encoding combination to use, default is 1.
        The protein spectra of the AAI indices will be generated if use_dsp is true,
        dictated by the class attributes: spectrum, window and filter.
        Each encoding will be used as the feature data to build the predictive
        regression models. To date, there are 566 indices and pySAR supports 15
        descriptors so the encoding process will generate 8490, ~59000 and ~257000
        models, when using 1, 2 or 3 descriptors + AAI indices, respectively. The
        metrics evaluated from the model for each AAI + Descriptor encoding combination
        will be collated into a dataframe and saved and returned.

        Parameters
        ----------
        desc_combo : int (default = 1)
            combination of descriptors to use with the AAI indices.
        verbose : bool (default = True)
            if true, progress of the encoding will be output else no output will
            be output until encoding has finished, true is recccomended.
        use_dsp : bool (default = True)
            if true then pass AAI indices encoding through the DSP pipeline to
            create protein spectra, dictated by instance attributes: spectrum,
            window and filter. If false then AAI indices encoding used as feature data.
        Returns
        -------
        aai_desc_metrics_df_ : pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using AAI indices + descriptors encoding strategy.
        """
        #create instance of Descriptors class, all_desc = True so all descriptor
        #   values will be calculated and or imported from descriptors csv
        desc = Descriptors(self.data[self.seq_col], all_desc = True)

        #create dataframe to store output results from models
        aai_desc_metrics_df = pd.DataFrame(columns=['Index','Category', 'Descriptor'\
            'Descriptor Group','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'])

        #lists to store results for each predictive model
        index_ = []
        index_category_ = []
        descriptor_ = []
        descriptor_group_ = []
        desc_list = []
        r2_ = []
        mse_ = []
        rmse_ = []
        rpd_ = []
        mae_ = []
        explained_var_ = []
        index_count = 1     #counters to keep track of current index & descriptor
        desc_count = 1

        if aai_list == None or aai_list == [] or aai_list == "":
            all_indices = self.aaindex.get_record_codes()
        else:
            all_indices = aai_list

        if desc_list == None or desc_list == [] or desc_list == "":
            all_descriptors = desc.all_descriptors_list(desc_combo)
        else:
            all_descriptors = desc_list

        #get list of all descriptors
        # all_descriptors = desc.all_descriptors_list(desc_combo)

        print('\n\n##############################################################\n')
        print('Encoding using {} AAI and {} descriptor combinations with the parameters:\n \
            Window: {}\nFilter: {}\nSpectrum: {}\nAlgorithm: {}\nParameters: {}\n \
            Test Split: {}\n'.format(len(self.aaindex.get_record_codes()), len(all_descriptors),
                window, filter_, spectrum, repr(self.model), self.parameters, self.test_split))
        print('##################################################################\n')

        start = time.time() #start counter

        '''
        1.) Get AAI index encoding of protein sequences, if using DSP, create instance
        of proDSP class and generate protein spectra from the AAI indices, according to
        instance parameters: spectrum, window and filter.
        2.) Get all 15 descriptor values and concatenate to AAI encoding features.
        3.) Build model using concatenated AAI and Descriptor features as the training data.
        4.) Predict and evaluate the model using the test data.
        5.) Append index, descriptor and calculated metrics to lists.
        6.) Repeat steps 1 - 5 for all indices in the AAI.
        7.) Output results into a final dataframe, save it and return.
        '''
        for index in tqdm(all_indices,unit=" indices",desc="AAIndex"):

            #get AAI indices encoding for sequences according to index var
            encoded_seqs = self.get_aai_enoding(index)

            #generate protein spectra from proDSP class if use_dsp is true
            if use_dsp:
                proDSP = ProDSP(encoded_seqs, spectrum=spectrum,
                    window=window, filter_=filter_)
                proDSP.encode_seqs()
                X_aai = pd.DataFrame(proDSP.spectrum_encoding)
            else:
                X_aai = pd.DataFrame(encoded_seqs)

            for descr in tqdm(all_descriptors, leave=False,unit=" descriptor",desc="Descriptors"):

                #reset descriptor DF and list
                desc_ = pd.DataFrame()
                desc_list = []
                desc_count+=1   #counter to keep track of current descriptor

                #if using 2 or 3 descriptors, append each descriptor & its category to list
                if desc_combo == 2 or desc_combo == 3:
                    for de in descr:

                        desc_list.append(getattr(desc, de)) #get descriptor attribute
                        descriptor_group_.append(desc.descriptor_groups[de])

                                # desc_ = pd.concat(desc_list,axis=1)  check if this is needed here
                    desc_ = desc_list

                    #concatenate each descriptor dataframe into one
                    if desc_combo == 2:
                        desc_list_concat = np.concatenate((desc_[0],desc_[1]),axis = 1)
                    elif desc_combo == 3:
                        desc_list_concat = np.concatenate((desc_[0],desc_[1],desc_[2]),axis = 1)

                    desc_ = desc_list_concat

                #if only using 1 descriptor
                else:
                    desc_ = getattr(desc, descr) #get descriptor attribute
                    descriptor_group_.append(desc.descriptor_groups[descr])

                X = pd.DataFrame(desc_)

                #get protein activity values
                Y  = self.get_activity()

                '''
                if using the PlsRegression algorithm and there is only 1 feature (1-dimension)
                in the feature data X then create a new PLSReg model with the n_components
                parameter set to 1 instead of the default 2 - this stops the error:
                ValueError - Invalid Number of Components: 2.

                Also, get train/test split, fit model and predict activity of test data.
                '''
                if X.shape[1] == 1 and repr(self.model) == "PLSRegression":
                  tmp_model = Model('plsreg',parameters={'n_components':1})
                  X_train, X_test, Y_train, Y_test  = tmp_model.train_test_split(X, Y)
                  model_fit = tmp_model.fit()
                  Y_pred = tmp_model.predict()

                else:
                  X_train, X_test, Y_train, Y_test  = self.model.train_test_split(X, Y)
                  model_fit = self.model.fit()
                  Y_pred = self.model.predict()

                #create instance of Evaluate class
                eval = Evaluate(Y_test,Y_pred)

                #append values/results from current encoding iteration to lists
                index_.append(index)
                index_category_.append(self.aaindex.get_category(index))
                descriptor_.append(descr)
                # descriptor_group_.append(desc.descriptor_groups[descr])
                r2_.append(eval.r2)
                rmse_.append(eval.rmse)
                mse_.append(eval.mse)
                rpd_.append(eval.rpd)
                mae_.append(eval.mae)
                explained_var_.append(eval.explained_var)

            desc_count = 1  #reset descriptor counter

        end = time.time()
        elapsed = end - start
        print('\n\n##############################################################')
        print('Elapsed Time for AAI + Descriptor Encoding: {0:.3f} seconds'.format(elapsed))

        #if using combinations of 2 or 3 descriptors, group every 2 or 3 descriptor
        #   groups into one element in the descriptor group list
        if desc_combo == 2:
          descriptor_group_= [ ','.join(x) for x in zip(descriptor_group_[0::2],
            descriptor_group_[1::2]) ]
        elif desc_combo == 3:
          descriptor_group_= [ ','.join(x) for x in zip(descriptor_group_[0::3],
            descriptor_group_[1::3],  descriptor_group_[2::3]) ]

        #set columns in the output dataframe to each of the values/metrics lists
        aai_desc_metrics_df_= aai_desc_metrics_df.copy()
        aai_desc_metrics_df_['Index'] = index_
        aai_desc_metrics_df_['Category'] = index_category_
        aai_desc_metrics_df_['Descriptor'] = descriptor_
        aai_desc_metrics_df_['Descriptor Group'] = descriptor_group_
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
        return "Instance of Encoding Class with attribute values: \
                \nDataset: {}\n, Activity: {}\n,Algorithm: {}\n Parameters: {}\n\
                Test Split: {}\n".format(
                    self.dataset, self.activity,self.algorithm,\
                    self.parameters, self.test_split
        )

    def __repr__(self):
        return "<Encoding: {} >".format(self)
