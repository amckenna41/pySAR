################################################################################
#################                     pySAR                    #################
################################################################################

import pandas as pd
import numpy as np
import datetime, time
import argparse
import itertools
import pickle
import io
import os
from difflib import get_close_matches
import json

from .globals_ import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from .aaindex import AAIndex
from .model import Model
from .proDSP import ProDSP
from .evaluate import Evaluate
from .utils import *
from .plots import plot_reg
from .descriptors import Descriptors

class PySAR():
    """
    The PySAR class is the main class for the pySAR software. The class allows for
    the encoding of protein sequences via a plethora of techniques, mainly via
    AAI Indices and or strucutrual and physiochemical protein descriptors that
    are then used as features in the building of a predictive model created to
    map the protein sequences to a sought-after activity value (activity attribute),
    this is known as a sequence activity relationship (SAR). Three main encoding
    strategies are possible in the class, namely using AAI Indices or protein
    descriptors as well as AAI Indices + Descriptors. The class accepts strings
    or lists of AAI Indices or descriptors and then passes these through a pipeline
    to get the required encoding of the respective sequences. The calculated encodings
    of the sequences are used as features in the building of the predictive model
    that will predict the acitivty values for new unseen protein sequences. After
    the encoding process, various metrics will be captured and stored in a local
    output folder according to the OUTPUT_FOLDER global var as well as a regression
    plot showing how well the model fits to the test data.

    Attributes
    ----------
    dataset : str (default = "")
        full path to dataset or name of dataset if it is stored in DATA_DIR.
    seq_col : str (default = "sequence")
        name of column in dataset that stores the protein sequences. By default
        the class will look for a column called 'sequence'.
    activity : str (default = "")
        name of activity column in dataset.
    algorithm : str (default = "")
        name of regression model to use for building the predictive models, class
        will accept full name or approximate name of model e.g "PLSReg", "plsregg" and
        "PLSRegression" will all build a PlsRegression model.
    parameters : dict (default = {})
        dictionary of parameters to use for the predictive model. By default the
        default parameters of the model will be used.
    test_split : float (default = 0.2)
        specifies the proportion of the dataset to use for testing. By default a
        80:20 split will be used, meaning 80% of the data will be used for training
        and 20% for testing.
    descriptors_csv : str (default = "descriptors.csv")
        csv file storing the pre-calculated descriptor values for the sequences
        in the dataset. By default the class will look for a file named
        "descriptors.csv" in the DATA_DIR and will use its contents as the
        descriptor features, instead of having to recalculate all descriptors for the dataset.

    Methods
    -------
    read_data():

    preprocessing()

    get_aai_encoding()

    aai_encoding()

    desc_encoding()

    aai_desc_encoding()

    output_results()
    """
    def __init__(self,dataset="",seq_col="sequence", activity="",algorithm="",\
        parameters={},test_split=0.2, descriptors_csv="descriptors.csv"):

        self.dataset = dataset
        self.seq_col = seq_col
        self.activity = activity
        self.algorithm = algorithm
        self.parameters = parameters
        self.test_split = test_split
        self.descriptors_csv = descriptors_csv
        self.aai_indices = None
        self.descriptors = None

        #create instance of AAIndex class
        self.aaindex = AAIndex()

        #import and read dataset
        self.data = self.read_data()

        #pre-process dataset and protein sequences
        self.preprocessing()                            #Reorder these??

        #get number of rows and cols of dataset
        self.num_seqs = len(self.data[self.seq_col])
        self.seq_len = len(max(self.data[self.seq_col],key=len))

        #create instance of model class of type specified by algorithm
        self.model = Model(algorithm=self.algorithm, parameters=self.parameters)

        #updating algoritm attribute
        self.algoritm = repr(self.model)

        #create output folder to store all model assets and results
        create_output_dir()

    def read_data(self):
        """
        Read in dataset according to name from 'dataset' attribute. By default
        the dataset should be stored in DATA_DIR.

        """
        dataset_file = os.path.join(DATA_DIR,self.dataset)

        #read in dataset csv if found in path, if not raise error
        if not (os.path.isfile(dataset_file)):
            if not (os.path.isfile(self.dataset)):
                raise OSError('Dataset filepath is not correct: {}'.format(self.dataset))
            dataset_file = self.dataset

        try:
            data = pd.read_csv(dataset_file, sep=",", header=0)
            return data
        except:
            raise IOError('Error opening dataset file: {}'.format(dataset_file))

    def preprocessing(self):
        """
        Pre-process protein sequences in dataset. Validate column names, check
        for invalid amino acids in sequences and remove any NAN or +/- infinity values.

        """
        #get closest match for sequence column name in dataset
        seq_col_matches = get_close_matches(self.seq_col,self.data.columns, cutoff=0.4)

        #set sequence col to the first match found, else raise error
        if seq_col_matches!=[]:
            self.seq_col = seq_col_matches[0]
        else:
            raise ValueError('Sequence Column ({}) not present in dataset columns \
                - {} /n '.format(self.seq_col, self.data.columns))

        #remove any gaps found in sequences in dataset
        self.data[self.seq_col] = remove_gaps(self.data[self.seq_col])

        #verify no invalid amino acids found in sequences, if so then raise error
        invalid_seqs = valid_sequence(self.get_seqs())
        if invalid_seqs!=None:
            raise ValueError('Invalid Amino Acids found in protein sequence \
                dataset: {}'.format(invalid_seqs))

        #get closest match for activity column name in dataset
        activity_matches = get_close_matches(self.activity,self.data.columns,cutoff=0.4)

        #set activity col to the first match found, else raise error
        if activity_matches!=[]:
            self.activity = activity_matches[0]
        else:
            raise ValueError('Activity Column ({}) not present in dataset columns  \
                - {} /n '.format(self.activity,list(self.data.columns)))

        #remove any +/- infinity values or any Null/NAN's from activity values
        self.data[self.activity].replace([np.inf,-np.inf], np.nan)
        self.data[self.activity].fillna(0,inplace=True)

    def get_aai_enoding(self, indices=None):
        """
        Get AAI index encoding values for index specified by indices for each amino
        acid in each of the protein sequences in dataset. If multiple indices
        specified then encode protein sequences with each index and concatenate.

        Parameters
        ----------
        indices : str/list (default = None)
            string or list of AAI indices.

        Returns
        -------
        encoded_seqs : np.ndarray
            array of the encoded protein sequences in dataset.
        """
        #validate AAI indices are present in the input parameter, if not raise error
        if indices == None or indices == "":
            raise ValueError('AAI indices input parameter cannot be None or empty.')

        encoded_indices = []

        #if single index input passed into parameter
        if not isinstance(indices, list):

            #get dict of amino acid encoding values for index specified by indices
            encoded_aai = self.aaindex.get_record_from_code(indices)['values']

            #initialise temp arrays to store encoded sequences
            temp_seq_vals = []
            temp_all_seqs = []

            #iterate through each protein sequence and amino acid, getting the AAI index encoding value
            for protein in range(0, len(self.data[self.seq_col])):
                for aa in self.data[self.seq_col][protein]:
                    temp_seq_vals.append(encoded_aai[aa])

                #append encoding and reset temp array
                temp_all_seqs.append(temp_seq_vals)
                temp_seq_vals = []

            #zero-pad encoding list so that sequences are all the same length
            temp_all_seqs = zero_padding(temp_all_seqs)

            #convert list of lists into array
            temp_all_seqs = np.array(temp_all_seqs, dtype="float32")

            encoded_seqs = temp_all_seqs

            return encoded_seqs

        #if multiple indices (list) passed into input parameter
        else:

            #create zeros numpy array to store encoded sequence output
            encoded_aai_ = np.zeros((self.num_seqs, self.seq_len))

            #if multiple indices used then calcualte AAI index encoding for each and
            #   then concatenate after each calculation
            for ind in range(0,len(indices)):
                encoded_aai = self.aaindex.get_record_from_code(indices[ind])['values']

                temp_seq_vals = []
                temp_all_seqs = []

                for protein in range(0, len(self.data[self.seq_col])):
                    for aa in self.data[self.seq_col][protein]:
                        temp_seq_vals.append(encoded_aai[aa])

                    temp_all_seqs.append(temp_seq_vals)
                    temp_seq_vals = []

                temp_all_seqs = zero_padding(temp_all_seqs)

                temp_all_seqs =np.array(temp_all_seqs, dtype="float32")

                #in first iteration through indices set encoded_aai_ to zeros
                #   initialised np array, else concatenate to array in previous iteration
                if ind == 0:
                    encoded_aai_ = temp_all_seqs
                else:
                    encoded_aai_ = np.concatenate((encoded_aai_,temp_all_seqs), axis=1)

            encoded_seqs = encoded_aai_

            return encoded_seqs

    def encode_aai(self,spectrum=None, indices=None, window=None, filter_=None,use_dsp=True):
        """
        Encode using AAI indices. If multiple indices input then calculate each
        and concatenate them. Build predictive model from AAI feature data. The
        resulting model assets and its results will be exported to the directory
        pointed to by the global variable OUTPUT_DIR. If use_dsp is true then
        pass AAI Indices through a DSP transformation specified by DSP parameters:
        spectrum, window & filter.

        Parameters
        ----------
        spectrum : str
            type of protein spectrum to use for DSP transformation.
        indices : str/list (default = None)
            string or list of indices from the AAI.
        window : str
            name of window function to apply to Fourier Transform.
        filter_ : str
            name of filter function to apply to ouput of Fourier Transform.
        use_dsp : bool (default = True)
            whether to transform AAI index encodings via DSP techniques and a Fourier
            transform. If false then no protein spectra will be generated.

        Returns
        -------
        aai_df : pd.Series
            pandas series storing metrics and results of encoding.
        """
        #validate AAI indices are present in the input parameter
        if indices == None or indices == "":
            raise ValueError('AAI indices input parameter cannot be None or empty.')

        self.aai_indices = indices

        #if input spectrum is none or empty, raise error.
        if use_dsp:
            if spectrum == None or spectrum == "":
                raise ValueError('Spectrum cannot be None or empty, got {}'.format(spectrum))

        #pandas series to store all output results
        aai_df = pd.Series(index=['Index','Category','R2', 'RMSE', 'MSE', 'RPD',
            'MAE', 'Explained Var'],dtype='object')

        #get AAI index encodings specified by indices input parameter
        encoded_seqs = self.get_aai_enoding(indices)

        #if use_dsp true then get protein spectra from AAI indices using ProDSP class
        #   else use the AAI indices encoding's themselves as the feature data (X)
        if use_dsp:
            proDSP = ProDSP(encoded_seqs, spectrum=spectrum, window=window, filter_=filter_)
            proDSP.encode_seqs()
            X = pd.DataFrame(proDSP.spectrum_encoding)
        else:
            X = pd.DataFrame(encoded_seqs)

        #get training and test dataset split
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size = self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        #get categories for all indices in self.aai_indices
        if isinstance(self.aai_indices, list):
            index_cat = ""
            for i in range(0,len(self.aai_indices)):
                index_cat += index_cat + ', '+ \
                    self.aaindex.get_category_from_record(self.aai_indices[i])
        else:
            index_cat = self.aaindex.get_category_from_record(self.aai_indices)

        #set results Series variables
        aai_df['Index'] = indices
        aai_df['Category'] = str(index_cat)
        aai_df['R2'] = eval.r2
        aai_df['RMSE'] = eval.rmse
        aai_df['MSE'] = eval.mse
        aai_df['RPD'] = eval.rpd
        aai_df['MAE'] = eval.mae
        aai_df['Explained Var'] = eval.explained_var

        #print out results from encoding
        self.output_results(aai_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2)

        #save results of encoding to output folder specified by OUTPUT_FOLDER
        save_results(aai_df, 'aai_encoding')

        #reset aai_indices instance variable
        # self.aai_indices = ""

        return aai_df

    def get_descriptor_encoding(self, descriptors=None):
        """
        Calculate inputted descriptor(s) requried for the encoding process.
        Get closest match to user inputted descriptor using difflib library.
        If a single descriptor is input then calculate it and return, if list of
        descriptors input then calculate descriptor values for each and concatenate.

        Parameters
        ----------
        descriptors : str/list (default = None)
            string or list of protein descriptors

        Returns
        -------
        encoded_desc : pd.DataFrame
            pandas dataFrame of calculated descriptor values according to user
            inputted descriptor.
        """
        #raise error if no descriptors specified in input
        if descriptors == None or descriptors == "":
            raise ValueError('No descriptors have been input.')

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.data[self.seq_col])

        #get closest valid available descriptors from input descriptor parameter,
        #   if a list of descriptors passed in as the input parameter then get
        #       all valid descriptors in list
        if isinstance(descriptors, list):
            for de in range(0,len(descriptors)):
                desc_matches = get_close_matches(descriptors[de],
                    descr.valid_descriptors(),cutoff=0.4)
                descriptors[de] = desc_matches[0]
                if descriptors[de] == []:
                    raise ValueError('No approximate descriptor found from one entered: {}'.format(de))
        else:
            desc_matches = get_close_matches(descriptors,descr.valid_descriptors(),cutoff=0.4)
            descriptors = desc_matches[0]
            if descriptors == []:
                raise ValueError('No approximate descriptor found from one entered: {}'.format(descriptors))

        #initialise temp lists and DF to store encoded descriptor values
        encoded_desc_temp = []
        encoded_desc_vals = []
        encoded_desc_temp = pd.DataFrame()

        #if single descriptor passed in, not a list, get descriptor values for sequences
        if not isinstance(descriptors, list):
            encoded_desc_temp = descr.get_descriptor_encoding(descriptors)

            #raise value error if descriptor is empty
            if (encoded_desc_temp.empty):
                raise ValueError('Descriptors {} cannot be empty or None'.format(descriptors))

            encoded_desc = encoded_desc_temp

            return encoded_desc

        #if list of descriptors passed as input, iterate and get each descriptors' values
        else:
            for d in range(0,len(descriptors)):
              encoded_desc_temp = descr.get_descriptor_encoding(descriptors[d])

              #raise value error if descriptor is empty
              if (encoded_desc_temp.empty):
                  raise ValueError('Descriptors {} cannot be empty or None'.format(descriptors[d]))
                    # self.descriptors[d]))

              encoded_desc_vals.append(encoded_desc_temp)
              encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe

            #concatenate dataframes of descriptors
            encoded_desc = pd.concat(encoded_desc_vals,axis=1)

        return encoded_desc

    def encode_desc(self, descriptor=None):
        """
        Encode protein sequences using protein physiochemical and structural
        descriptors and build predictive model from the descriptor feature data.
        If multiple descriptors input then calculate each and concatenate them.
        The resulting model assets and its results will be exported to the
        directory pointed to by the global variable OUTPUT_DIR.

        Parameters
        ----------
        descriptor : str/list (default = None)
            string or list of protein descriptors

        Returns
        -------
        desc_df : pd.Series
            pandas series storing metrics and results of encoding.
        """
        #raise error if no descriptor specified in input
        if descriptor == None or descriptor == "":
            raise ValueError('No descriptors have been input, descriptors attribute is empty.')

        self.descriptors = descriptor

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.data[self.seq_col])

        #pandas series to store all output results
        desc_df = pd.Series(index=['Descriptor','Group','R2', 'RMSE', 'MSE',
            'RPD', 'MAE', 'Explained Var'],dtype='object')

        X = self.get_descriptor_encoding(descriptors = self.descriptors)

        #get training and test dataset split
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size = self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        desc_group = ""

        #get groups for all descriptors in self.desciptors
        if isinstance(self.descriptors, list):
            for i in range(0,len(self.descriptors)):
                desc_group += desc_group + ', '+ \
                    descr.descriptor_groups['_'+self.descriptors[i]]
        else:
            desc_cat = descr.descriptor_groups['_'+self.descriptors]

        #set results Series variables
        desc_df['Descriptor'] = descriptor
        desc_df['Group'] = desc_cat
        desc_df['R2'] = eval.r2
        desc_df['RMSE'] = eval.rmse
        desc_df['MSE'] = eval.mse
        desc_df['RPD'] = eval.rpd
        desc_df['MAE'] = eval.mae
        desc_df['Explained Var'] = eval.explained_var

        #print out results from encoding
        self.output_results(desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2)

        #save results of encoding to output folder
        save_results(desc_df, 'desc_encoding')

        #reset descriptors instance variable
        # self.descriptors = ""

        return desc_df

    def encode_aai_desc(self, indices=None,descriptors=None,spectrum='power',
        window="", filter_=None,use_dsp=True):
        """
        Encode using both AAI indices and the descriptors. The two outputs from
        the individual encoding strategies will be concatenated together and
        used in the building of a predictive model. The resulting model assets
        and its results will be exported to the directory pointed to by the global
        variable OUTPUT_DIR. If use_dsp is true then pass AAI Indices through a
        DSP transformation specified by parameters: spectrum, window & filter.

        Parameters
        ----------
        spectrum : str (default = "power")
            type of protein spectrum to use for DSP transformation.
        indices : str/list (default = None)
            string or list of indices from the AAI.
        descriptors : str/list (default = None)
            string or list of protein descriptors.
        window : str
            name of window function to apply to output of Fourier Transform.
        filter_ : str
            name of filter function to apply to ouput of Fourier Transform.
        use_dsp : bool (default = True)
            whether to transform AAI index encodings via DSP techniques and a Fourier
            transform. If false then no protein spectra will be generated and the
            normal encoded AAI indices will be used as the model feature data.

        Returns
        -------
        aai_desc_df : pd.Series
            pandas series storing metrics and results of encoding.
        """

        #validate AAI indices and Descriptors are present in the input parameters
        #   or instance variables, return error if either is None
        if (descriptors == None or descriptors == "") or \
            (indices == None or indices == ""):
                raise ValueError('AAI Indices and Descriptor input parameters must \
                    not be empty or None')

        self.aai_indices = indices           #set instance attributes
        self.descriptors = descriptors

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.data[self.seq_col])

        #create output results Series
        aai_desc_df = pd.Series(index=['Index','Category','Descriptor',
            'Group','R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Var'],dtype='object')

        #get AAI index encoding features
        aai_encoding = self.get_aai_enoding(indices)
        aai_encoding = pd.DataFrame(aai_encoding)

        #if AAI indices encoding is empty, raise error
        if (aai_encoding.empty):
            raise ValueError('AAI Indices {} cannot be empty or None'.format(indices))

        #get descriptor encoding features
        # descriptor_encoding = self.desc_encoding(descriptors)
        descriptor_encoding = self.get_descriptor_encoding(descriptors)

        #if descriptors encoding is empty, raise error
        if (descriptor_encoding.empty):
            raise ValueError('Descriptors {} cannot be empty or None'.format(descriptors))

        #concatenate AAI index and Descriptor features
        aai_desc = [aai_encoding, descriptor_encoding]
        X = pd.concat(aai_desc, axis = 1)

        #get training and test dataset split
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size = self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        index_cat = ""

        #get categories for all indices in self.aai_indices
        if isinstance(self.aai_indices, list):
            for i in range(0,len(self.aai_indices)):
                index_cat += index_cat + ', '+ \
                    self.aaindex.get_category_from_record(self.aai_indices[i])
        else:
            index_cat = self.aaindex.get_category_from_record(self.aai_indices)

        desc_group = ""

        #get groups for all descriptors in self.desciptors
        if isinstance(self.descriptors, list):
            for i in range(0,len(self.descriptors)):
                desc_group += desc_group + ', '+ \
                    descr.descriptor_groups[self.descriptors[i]]
        else:
            desc_cat = descr.descriptor_groups[self.descriptors]

        #set output dataframe columns
        aai_desc_df['Index'] = str(self.aai_indices)
        aai_desc_df['Category'] = str(index_cat)
        aai_desc_df['Descriptor'] = str(self.descriptors)
        aai_desc_df['Group'] = str(desc_cat)
        aai_desc_df['R2'] = eval.r2
        aai_desc_df['RMSE'] = eval.rmse
        aai_desc_df['MSE'] = eval.mse
        aai_desc_df['RPD'] = eval.rpd
        aai_desc_df['MAE'] = eval.mae
        aai_desc_df['Explained Var'] = eval.explained_var

        #print out results from encoding
        self.output_results(aai_desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2)

        #save results of encoding to output folder
        save_results(aai_desc_df, 'aai_desc_encoding')

        return aai_desc_df

    def output_results(self, results):
        """
        Print out the predictive model parameters/attributes and its results.

        Parameters
        ----------
        results : dict/pd.Series
            dictionary or Series of metrics and their associated values.
        """
        print('\n#############################################################')
        print('######################## Results ############################\n')
        print('#############################################################\n')
        print('####################### Parameters ##########################\n')
        print('# Dataset -> {}\n# Dataset Size -> {}\n# Sequence Length -> {} \
            \n# Activity -> {}\n# AAI Indices -> {}\n# Descriptors -> {}\n# Algorithm -> {}\
                \n# Model Parameters -> {}\n# Test Split -> {}\n'.format(
                self.dataset,self.num_seqs, self.seq_len, self.activity, self.aai_indices,
                self.descriptors, repr(self.model), self.model.model.get_params(),
                self.test_split
                ))
        print('######################## Metrics ############################\n')
        print('# R2: {}'.format(results['R2']))
        print('# RMSE: {} '.format(results['RMSE']))
        print('# MSE: {} '.format(results['MSE']))
        print('# MAE: {}'.format(results['MAE']))
        print('# RPD {}'.format(results['RPD']))
        print('# Variance {}\n'.format(results['Explained Var']))
        print('#############################################################\n')

    def get_seqs(self):
        """
        Return all protein sequences within the dataset.

        Returns
        -------
        self.data[self.seq_col] : pd.Series
            pandas series of all protein sequences in the dataset.
        """
        return self.data[self.seq_col]

    def get_activity(self):
        """
        Return all activity/fitness values for the sequences within the dataset.

        Returns
        -------
        self.data[self.activity] : np.ndarray
            array of all activity/fitness values for the protein sequences
        """
        return self.data[self.activity].values.reshape((-1,1))

######################          Getters & Setters          ######################

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
    def aai_indices(self):
        return self._aai_indices

    @aai_indices.setter
    def aai_indices(self, val):
        self._aai_indices = val

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

################################################################################

    def __str__(self):
        return "Instance of PySAR class, using parameters: {}".format(self.__dict__)

    def __repr__(self):
        return "<PySAR: {} >".format(self)
