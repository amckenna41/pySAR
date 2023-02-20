################################################################################
#################                     pySAR                    #################
################################################################################

import pandas as pd
import numpy as np
import os
import sys
from difflib import get_close_matches
import json
from json import JSONDecodeError

from .globals_ import DATA_DIR
from aaindex import aaindex1
from .model import Model
from .pyDSP import PyDSP
from .evaluate import Evaluate
from .utils import *
from .plots import plot_reg
from .descriptors import Descriptors

class PySAR():
    """
    The PySAR class is the main class for the pySAR software. The class allows for
    the encoding of protein sequences via a plethora of techniques, mainly via
    AAI Indices and or strucutrual and physiochemical protein descriptors that
    are then used as features in the building of a predictive regression model created to
    map the protein sequences to a sought-after activity value (activity attribute),
    this is known as a sequence activity relationship (SAR). Three main encoding
    strategies are possible in the class, namely using AAI Indices or protein
    descriptors as well as AAI Indices + Descriptors. Additionally, the protein sequences
    can be encoded using Digital Signal Processing (DSP) techniques, mainly through the use
    of informational protein spectra, this is achieved via the pyDSP class in the software.
    The class accepts strings or lists of AAI Indices or descriptors and then passes 
    these through a pipeline to get the required encoding of the respective sequences. 
    The calculated encodings of the sequences are used as features in the building 
    of the predictive models that will predict the acitivty values for new unseen 
    protein sequences. After the encoding process, various metrics will be captured 
    and stored in a local output folder according to the OUTPUT_FOLDER global var 
    as well as a regression plot showing how well the model, and the selected protein 
    feature attributes, fit to the test data of unseen protein sequences.

    Parameters
    ----------    
    :config_file (str): 
        path to configuration file.
        
    Methods
    -------
    read_data():
        read dataset of protein sequences.
    preprocessing():
        pre-process / clean protein sequence dataset.
    get_aai_encoding(indices):
        get AAI encoding for user inputted index/indices.
    encode_aai(indices=None):
        get encoded protein sequences according to user inputted index/indices, applying 
        DSP if applicable. 
    get_descriptor_encoding(descriptors):
        calculate user inputted descriptor/descriptors according to user input.
    encode_desc(descriptor=None):
        get encoded protein sequences according to user inputted descriptor/descriptors.
    encode_aai_desc(indices=None, descriptors=None)
        get encoded protein sequences according to user inputted AA index/indices and 
        descriptor/descriptors.
    output_results(results):
        print out the predictive model parameters/attributes and its results.
    get_seqs():
        return protein sequences
    get_activity():
        return array of protein sequences' activity values.
    """
    def __init__(self, config_file=""):

        self.config_file = config_file
        self.params = {}

        config_filepath = ""
    
        #open json config file       
        if not (isinstance(config_file, str) or config_file is None):
            raise TypeError('JSON config file must be a filepath of type string, not of type {}.'.format(type(config_file)))
        if (os.path.splitext(config_file)[1] == ""):
            config_file = config_file + '.json' #append extension if only filename input
        if (os.path.isfile(self.config_file)):
            config_filepath = self.config_file
        elif (os.path.isfile(os.path.join('config', self.config_file))):
            config_filepath = os.path.join('config', self.config_file)
        else:
            raise OSError('JSON config file not found at path: {}.'.format(config_filepath))
        try:
            with open(config_filepath) as f:
                self.params = json.load(f)
        except JSONDecodeError as e:
            print('Error parsing config JSON file: {}.'.format(config_filepath))
            sys.exit()

        #create instance of Map class so parameters can be accessed via dot notation
        self.params = Map(self.params)
        self.dataset = self.params.dataset[0]["dataset"]
        self.sequence_col = self.params.dataset[0]["sequence_col"]
        self.activity = self.params.dataset[0]["activity"]

        #model parameters
        self.algorithm = self.params.model[0]["algorithm"]
        self.parameters = self.params.model[0]["parameters"]
        self.test_split = self.params.model[0]["test_split"]

        #aai parameters
        self.aai_indices = None

        #descriptors parameters
        # self.descriptors = self.params["descriptors"]
        self.descriptors = None

        #pyDSP parameters
        self.use_dsp = self.params.pyDSP[0]["use_dsp"]
        if (self.use_dsp):
            self.spectrum = self.params.pyDSP[0]["spectrum"]
            self.window = self.params.pyDSP[0]["window"]
            self.filter = self.params.pyDSP[0]["filter"]

        #import and read dataset
        self.data = self.read_data()

        #array of protein sequences
        self.sequences = self.data[self.sequence_col]

        #pre-process dataset and protein sequences
        self.preprocessing()

        #get number of rows and cols of dataset
        self.num_seqs = len(self.data[self.sequence_col])
        self.seq_len = len(max(self.data[self.sequence_col], key=len))

        #create instance of Descriptors class
        self.descriptor = Descriptors(self.config_file, protein_seqs=self.data[self.sequence_col])

        #create instance of model class of type specified by algorithm parameter
        self.model = Model(self.algorithm, parameters=self.parameters, test_split=self.test_split)

        #updating algorithm attribute
        self.algorithm = repr(self.model)

        #create output folder to store all model assets and results
        create_output_dir()

    def read_data(self):
        """
        Read in dataset according to name from 'dataset' attribute. By default
        the dataset should be stored in DATA_DIR.
        
        Parameters
        ----------
        None

        Returns
        -------
        :data (pd.DataFrame): 
            dataframe of imported dataset.      
        """
        filepath = ""

        #read in dataset csv if found in path, if not raise error
        if (os.path.isfile(os.path.join(DATA_DIR, self.dataset))):
            filepath = os.path.join(DATA_DIR, self.dataset)
        elif (os.path.isfile(self.dataset)):
            filepath = self.dataset
        else:
            raise OSError('Dataset filepath is not correct: {}.'.format(filepath))

        #read in dataset csv
        try:
            data = pd.read_csv(filepath, sep=",", header=0)
            return data
        except:
            raise IOError('Error opening dataset file: {}.'.format(filepath))

    def preprocessing(self):
        """
        Pre-process protein sequences in dataset. Validate column names, check
        for invalid amino acids in sequences, remove any gaps in sequence and 
        remove any NAN or +/- infinity values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #get closest match for sequence column name in dataset
        sequence_col_matches = get_close_matches(self.sequence_col, self.data.columns, cutoff=0.4)

        #set sequence col to the first match found, else raise error
        if (sequence_col_matches != []):
            self.sequence_col = sequence_col_matches[0]
        else:
            raise ValueError('Sequence Column ({}) not present in dataset columns - {} /n.'.
                format(self.sequence_col, self.data.columns))

        #remove any gaps found in sequences in dataset
        self.data[self.sequence_col] = remove_gaps(self.data[self.sequence_col])

        #verify no invalid amino acids found in sequences, if so then raise error
        invalid_seqs = valid_sequence(self.get_seqs())
        if (invalid_seqs != None):
            raise ValueError('Invalid Amino Acids found in protein sequence dataset: {}.'.
                format(invalid_seqs))

        #get closest match for activity column name in dataset
        activity_matches = get_close_matches(self.activity, self.data.columns, cutoff=0.4)

        #set activity col to the first match found, else raise error
        if (activity_matches != []):
            self.activity = activity_matches[0]
        else:
            raise ValueError('Activity Column ({}) not present in dataset columns  - {} /n.'.
                format(self.activity,list(self.data.columns)))

        #remove any +/- infinity values or any Null/NAN's from activity values
        self.data[self.activity].replace([np.inf, -np.inf], np.nan)
        self.data[self.activity].fillna(0, inplace=True)

    def get_aai_encoding(self, indices=None):
        """ 
        Get AAI index encoding values for index specified by indices for each amino
        acid in each of the protein sequences in dataset. If multiple indices
        specified then encode protein sequences with each index and concatenate.

        Parameters
        ----------
        :indices : str/list (default=None):
            string or list of AAI indices.
        
        Returns
        -------
        :encoded_seqs : np.ndarray:
            array of the encoded protein sequences in dataset via user input index/indices.
        """
        #validate AAI indices are present in the input parameter, if not raise error
        if (indices == None or indices == ""):
            raise ValueError('AAI indices input parameter cannot be None or empty.')

        #check input indices is of correct type (str/list), if not raise type error
        if (not isinstance(indices, str) and (not isinstance(indices, list))):
            raise TypeError("Input indices parameter must be a string or list, got {}.".format(type(indices)))

        #cast index string to list, split multiple indices using comma
        if (isinstance(indices, str)):
            if (',' in indices):
                indices = indices.split(',')  #split on ',' just in case multiple indices passed in as str
            else:
                indices = [indices]

        #create zeros numpy array to store encoded sequence output
        encoded_aai_ = np.zeros((self.num_seqs, self.seq_len*len(indices)))

        #if multiple indices used then calcualte AAI index encoding for each and
        #then concatenate after each calculation
        for ind in range(0, len(indices)):

            #get values of aaindex record
            encoded_aai = aaindex1[indices[ind]].values

            #initialise temp arrays to store encoded sequences
            temp_seq_vals = []
            temp_all_seqs = []

            #iterate through each protein sequence and amino acid, getting the AAI index encoding value
            for protein in range(0, len(self.data[self.sequence_col])):
                for aa in self.data[self.sequence_col][protein]:
                    temp_seq_vals.append(encoded_aai[aa])

                #append encoding and reset temp array
                temp_all_seqs.append(temp_seq_vals)
                temp_seq_vals = []

            #zero-pad encoding list so that sequences are all the same length
            temp_all_seqs = zero_padding(temp_all_seqs)

            #convert list of lists into array
            temp_all_seqs = np.array(temp_all_seqs, dtype="float32")

            #in first iteration through indices set encoded_aai_ to zero-initialised 
            # numpy array, else concatenate to array in previous iteration
            if (ind == 0):
                encoded_aai_ = temp_all_seqs
            else:
                encoded_aai_ = np.concatenate((encoded_aai_, temp_all_seqs), axis=1)
            
        return encoded_aai_

    def encode_aai(self, indices=None, show_plot=False, print_results=True):
        """
        Encode using AAI indices. If multiple indices input then calculate each
        and concatenate them. Build predictive model from AAI feature data. The
        resulting model assets and its results will be exported to the directory
        pointed to by the global variable OUTPUT_DIR. If use_dsp config parameter 
        is true then pass AAI Indices through a DSP transformation specified by 
        the config's DSP parameters (spectrum, window & filter) via the PyDSP 
        module and class.

        Parameters
        ----------
        :indices : str/list (default=None)
            string or list of indices from the AAI.
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding process
            (mainly used for testing).
            
        Returns
        -------
        :aai_df : pd.Series
            pandas series storing metrics and results of encoding.
        """
        #validate AAI indices are present in the input parameter
        if (indices == None or indices == "" or indices == []):
            raise ValueError('AAI indices input parameter cannot be None or empty.')

        #check input indices is of correct type (str/list), if not raise type error
        if ((not isinstance(indices, str)) and (not isinstance(indices, list))):
            raise TypeError("Input indices parameter must be a string or list, got {}.".format(type(indices)))

        self.aai_indices = indices

        #pandas series to store all output results
        # aai_df = pd.Series(index=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'RPD',
        #     'MAE', 'Explained Variance'], dtype='object') 
        # aai_df = pd.DataFrame(columns=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Variance'])

        #get AAI index encodings specified by indices input parameter
        encoded_seqs = self.get_aai_encoding(indices)

        #if use_dsp true then get protein spectra from AAI indices using PyDSP class,
        #else use the AAI indices encoding's themselves as the feature/training data (X)
        if (self.use_dsp):
            #if input spectrum is none or empty, raise error.
            if (self.spectrum == None or self.spectrum == ""):
                raise ValueError('Spectrum cannot be None or empty, got {}.'.format(self.spectrum))
            pyDSP = PyDSP(self.config_file, protein_seqs=encoded_seqs)
            pyDSP.encode_seqs()
            X = pd.DataFrame(pyDSP.spectrum_encoding)
        else:
            X = pd.DataFrame(encoded_seqs)  #no DSP applied to encoded sequences

        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        col_counter = 1
        for col in X.columns:
            X.rename(columns={col: "aai_" + str(col_counter)}, inplace=True)
            col_counter+=1

        #get training and test dataset split from model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        index_cat = []

        #get categories for all indices in self.aai_indices
        if (isinstance(self.aai_indices, list)):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]

        #create comma seperated list of categories
        index_cat = ', '.join(index_cat)

        #create output dataframe, set first row to attribute/metric values
        aai_df = pd.DataFrame(columns=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Variance'])
        aai_df.loc[0] = [indices, str(index_cat).strip(), eval.r2, eval.rmse, eval.mse, eval.rpd, eval.mae, eval.explained_var]

        #convert index and category from Object -> String datatypes
        aai_df['Index'] = aai_df['Index'].astype(pd.StringDtype())
        aai_df['Category'] = aai_df['Category'].astype(pd.StringDtype())
        
        #print out results from encoding
        if (print_results):
            self.output_results(aai_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, show_plot)

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
        :descriptors : str/list (default=None)
            string or list of protein descriptors

        Returns
        -------
        :encoded_desc : pd.DataFrame
            pandas dataFrame of calculated descriptor values according to user
            inputted descriptor(s).
        """
        #raise error if no descriptors specified in input
        if (descriptors == None or descriptors == "" or descriptors == []): 
            raise ValueError('Descriptors input parameter cannot be None or empty.')
        
        #check input descriptor is of correct type (str) or list, if not raise type error
        if (not isinstance(descriptors, str) and (not isinstance(descriptors, list))):
            raise TypeError("Input descriptor parameter must be a str or list, got {}.".format(type(descriptors)))

        #cast descriptors parameter to a list by creating comma seperated list
        if (isinstance(descriptors, str)):
            descriptors = descriptors.split(',')

        #remove any leading or trailing whitespace from descriptors
        descriptors = [de.strip() for de in descriptors]

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.data[self.sequence_col])
        
        temp_descriptors = []
        #get closest valid available descriptors from input descriptor parameter,
        #   if a list of descriptors passed in as the input parameter then get
        #       all valid descriptors in list
        for de in range(0, len(descriptors)):
            desc_matches = get_close_matches(descriptors[de],
                descr.valid_descriptors(), cutoff=0.4)
            descriptors[de] = desc_matches[0]
            if (descriptors[de] == []):
                raise ValueError('No approximate descriptor found from one entered: {}.'.format(de))
            temp_descriptors.append(desc_matches[0])
        
        self.descriptors = temp_descriptors

        #initialise temp lists and DF to store encoded descriptor values
        encoded_desc_temp = []
        encoded_desc_vals = []
        encoded_desc_temp = pd.DataFrame()

        #iterate and get each descriptors' values
        for d in range(0, len(descriptors)):
            encoded_desc_temp = descr.get_descriptor_encoding(descriptors[d])

            #raise value error if descriptor is empty
            if (encoded_desc_temp.empty):
                raise ValueError('Descriptors {} cannot be empty or None.'.format(descriptors[d]))

            encoded_desc_vals.append(encoded_desc_temp) #append to array of all descriptor values
            encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe for next iteration

        #concatenate dataframes of descriptors
        encoded_desc = pd.concat(encoded_desc_vals, axis=1)

        return encoded_desc

    def encode_desc(self, descriptor=None, show_plot=False, print_results=True):
        """
        Encode protein sequences using protein physiochemical and or structural
        descriptors and build predictive model from the descriptor feature/training 
        data. If multiple descriptors input then calculate each and concatenate 
        them. The resulting model assets and its results will be exported to the
        directory pointed to by the global variable OUTPUT_DIR.

        Parameters
        ----------
        :descriptor : str/list (default=None)
            string or list of protein descriptors
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding 
            process (mainly used for testing). 

        Returns
        -------
        :desc_df : pd.Series
            pandas series storing metrics and results of encoding.
        """
        #raise error if no descriptor specified in input
        if (descriptor == None or descriptor == ""):
            raise ValueError('No descriptors have been input, descriptors attribute is empty.')

        #check input descriptor is of correct type (str), if not raise type error
        if not (isinstance(descriptor, str)):
            raise TypeError("Input descriptor parameter must be a string, got {}.".format(type(descriptor)))

        self.descriptors = descriptor

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.data[self.sequence_col])

        #get closest matching descriptor from descriptor input parameter
        desc_matches = get_close_matches(self.descriptors, descr.valid_descriptors(), cutoff=0.4)
        if (desc_matches != []):
            self.descriptors = desc_matches[0]  #set desc to closest descriptor match found
        else:
            raise ValueError('Could not find a match for the input descriptor in available valid descriptors:\n {}.'.
                format(self.descriptors, descr.valid_descriptors()))

        #pandas series to store all output results
        desc_df = pd.Series(index=['Descriptor', 'Group', 'R2', 'RMSE', 'MSE',
            'RPD', 'MAE', 'Explained Variance'], dtype='object')

        # if isinstance(descriptors, list):
        #     for de in range(0,len(descriptors)):
        #         desc_matches = get_close_matches(descriptors[de],
        #             descr.valid_descriptors(),cutoff=0.4)
        #         descriptors[de] = desc_matches[0]
        #         if descriptors[de] == []:
        #             raise ValueError('No approximate descriptor found from one entered: {}'.format(de))
        # else:
        #     desc_matches = get_close_matches(descriptors,descr.valid_descriptors(),cutoff=0.4)
        #     descriptors = desc_matches[0]
        #     if descriptors == []:
        #         raise ValueError('No approximate descriptor found from one entered: {}'.format(descriptors))

        #set training data (X) to descriptor-encoded protein sequences
        X = self.get_descriptor_encoding(descriptors=self.descriptors)

        #get training and test dataset split using Model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        desc_group = ""

        #get groups for all descriptors in self.desciptors
        if (isinstance(self.descriptors, list)):
            for i in range(0, len(self.descriptors)):
                desc_group += desc_group + ', ' + \
                    descr.descriptor_groups[self.descriptors[i]]
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #set results Series variables
        desc_df['Descriptor'] = descriptor
        desc_df['Group'] = desc_group
        desc_df['R2'] = eval.r2
        desc_df['RMSE'] = eval.rmse
        desc_df['MSE'] = eval.mse
        desc_df['RPD'] = eval.rpd
        desc_df['MAE'] = eval.mae
        desc_df['Explained Variance'] = eval.explained_var

        #ensure aai indices attribute doesn't show up in output results
        if (self.aai_indices != None):
            self.aai_indices = None

        #print out results from encoding
        if (print_results):
            self.output_results(desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, show_plot)

        #save results of encoding to output folder
        save_results(desc_df, 'desc_encoding')

        #reset descriptors instance variable
        self.descriptors = None 

        return desc_df

    def encode_aai_desc(self, indices=None, descriptors=None, show_plot=False, print_results=True):
        """
        Encode using both AAI indices and the physiochemical/structural descriptors. 
        The two outputs from the individual encoding strategies will be concatenated 
        together and used in the building of a predictive regression model. The 
        resulting model assets and its results will be exported to the directory 
        pointed to by the global variable OUTPUT_DIR. If the config parameter 
        use_dsp is true then pass AAI Indices through a DSP transformation 
        specified by DSP parameters (spectrum, window & filter) via the PyDSP 
        class/module.

        Parameters
        ----------
        :indices : str/list (default=None)
            string or list of indices from the AAI.
        :descriptors : str/list (default=None)
            string or list of protein descriptors.
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding process
            (mainly used for testing).
            
        Returns
        -------
        :aai_desc_df : pd.Series
            pandas series storing metrics and results of encoding.
        """
        #validate AAI indices and Descriptors are present in the input parameters
        #   or instance variables, return error if either is None
        if (descriptors == None or descriptors == "") or \
            (indices == None or indices == ""):
                raise ValueError('AAI Indices and Descriptor input parameters must not be empty or None.')

        #check input descriptor & indices are of correct type (str/list), if not raise type error
        if (not isinstance(indices, str) and (not isinstance(indices, list)) or \
                (not isinstance(descriptors, str) and (not isinstance(indices, list)))):
            raise TypeError("Input indices and descriptors parameter must be of type string or list.")

        #set instance attributes
        self.aai_indices = indices           
        self.descriptors = descriptors

        #create output results Series
        # aai_desc_df = pd.Series(index=['Index', 'Category', 'Descriptor', 'Group', 'R2', 
        #     'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Variance'], dtype='object')

        aai_desc_df = pd.DataFrame(columns=['Index', 'Category', 'Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Variance'])

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.data[self.sequence_col])

        #get AAI index encoding features, convert into dataframe
        aai_encoding = self.get_aai_encoding(indices)
        
        #set dataframe columns to format of aai_1, aai_2 etc
        aai_encoding = pd.DataFrame(aai_encoding)
        columns = ["aai_" + str(x) for x in range(1,len(aai_encoding.columns)+1)]
        aai_encoding.columns = columns

        #if AAI indices encoding is empty, raise error
        if (aai_encoding.empty):
            raise ValueError('AAI Indices {} cannot be empty or None.'.format(indices))

        #get descriptor encoding features
        descriptor_encoding = self.get_descriptor_encoding(descriptors)

        #if descriptors encoding is empty, raise error
        if (descriptor_encoding.empty):
            raise ValueError('Descriptors {} cannot be empty or None.'.format(descriptors))

        # aai_encoding.reset_index(inplace=True, drop=True)

        # aai_encoding = aai_encoding.reset_index()
        descriptor_encoding.reset_index(inplace=True, drop=True)

        #concatenate AAI index and Descriptor features to get training data (X)
        X = pd.concat([aai_encoding, descriptor_encoding], axis=1)

        #get training and test dataset split using Model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(X,
            self.get_activity(), test_size=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        index_cat = []

        #get categories for all indices in self.aai_indices
        if (isinstance(self.aai_indices, list)):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]
        
        #seperate index categories into comma seperated string
        index_cat = ', '.join(index_cat)

        desc_group = ""

        #get groups for all descriptors in self.descriptors
        if (isinstance(self.descriptors, list)):
            for i in range(0, len(self.descriptors)):
                desc_group += desc_group + ', ' + \
                    descr.descriptor_groups[self.descriptors[i]]
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        # aai_desc_df['Index'] = str(indices)
        # aai_desc_df['Category'] = str(index_cat).strip()
        # aai_desc_df['Descriptor'] = str(self.descriptors)
        # aai_desc_df['Group'] = str(desc_cat)
        # aai_desc_df['R2'] = eval.r2
        # aai_desc_df['RMSE'] = eval.rmse
        # aai_desc_df['MSE'] = eval.mse
        # aai_desc_df['RPD'] = eval.rpd
        # aai_desc_df['MAE'] = eval.mae
        # aai_desc_df['Explained Variance'] = eval.explained_var

        #set output dataframe columns
        aai_desc_df.loc[0] = [indices, str(index_cat).strip(), str(self.descriptors), str(desc_group), eval.r2, 
            eval.rmse, eval.mse, eval.rpd, eval.mae, eval.explained_var]

        #print out results from encoding
        if (print_results):
            self.output_results(aai_desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, show_plot)

        #save results of encoding to output folder
        save_results(aai_desc_df, 'aai_desc_encoding')

        return aai_desc_df

    def output_results(self, results):
        """
        Print out the predictive model parameters/attributes and its results.

        Parameters
        ----------
        :results : dict/pd.Series
            dictionary or Series of metrics and their associated values.

        Returns
        -------
        None
        """
        print('#############################################################')
        print('####################### Parameters ##########################\n')
        print('# Dataset -> {}\n# Dataset Size -> {}\n# Sequence Length -> {} \
            \n# Activity -> {}'.format(self.dataset, self.num_seqs, self.seq_len, self.activity))
        if (self.aai_indices is not None):
            print('# AAI Indices -> {}'.format(self.aai_indices))
            if (self.use_dsp):
                print('  # Using DSP -> {}\n  # Spectrum -> {}\n  # Window Function -> {} \
                \n  # Filter Function -> {}\n'.format(
                    self.use_dsp, self.spectrum, self.window, self.filter))  
        print('# Descriptors -> {}\n# Algorithm -> {}\
                \n# Model Parameters -> {}\n# Test Split -> {}'.format(
                self.descriptors, repr(self.model), self.model.model.get_params(), self.test_split
                ))
        print('\n#############################################################')
        print('######################## Results ############################')
        print('# R2: {}'.format(results['R2'].values[0]))
        print('# RMSE: {} '.format(results['RMSE'].values[0]))
        print('# MSE: {} '.format(results['MSE'].values[0]))
        print('# MAE: {}'.format(results['MAE'].values[0]))
        print('# RPD {}'.format(results['RPD'].values[0]))
        print('# Explained Variance {}\n'.format(results['Explained Variance'].values[0]))
        print('#############################################################\n')

    def get_seqs(self):
        """
        Return all protein sequences within the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        :self.data[self.sequence_col] : pd.Series
            pandas series of all protein sequences in the dataset.
        """
        return self.data[self.sequence_col]

    def get_activity(self):
        """
        Return all activity/fitness values for the sequences within the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        :self.data[self.activity] : np.ndarray
            array of all activity/fitness values for the protein sequences
        """
        # return self.data[self.activity].values.reshape((-1,1))
        return self.data[self.activity]

######################          Getters & Setters          ######################

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, val):
        self._dataset = val

    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, val):
        self._sequences = val

    @property
    def sequence_col(self):
        return self._sequence_col

    @sequence_col.setter
    def sequence_col(self, val):
        self._sequence_col = val

    @property
    def activity(self):
        return self._activity

    @activity.setter
    def activity(self, val):
        self._activity = val

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

    # def __str__(self):
    #     return "Instance of PySAR class, using parameters: {}.".format(self.__dict__)

    # def __repr__(self):
    #     return "<PySAR: {}>".format(self)