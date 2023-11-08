################################################################################
#################                     pySAR                    #################
################################################################################

import pandas as pd
import numpy as np
import os
from difflib import get_close_matches
import json
from json import JSONDecodeError
import textwrap

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
    the encoding of protein sequences via a plethora of techniques, mainly via AAI 
    Indices and or strucutrual, biochemical and physiochemical protein descriptors that are 
    then used as features in the building of predictive regression ML models created to map the 
    protein sequences to a sought-after activity/fitness value (activity attribute), this is 
    known as a Sequence Activity Relationship (SAR) or Sequence Function Relationship (SFR). 
    Creating this mapping from sequence to activity/fitness then allows for the future prediction
    of the sought activity/fitness value for unseen protein sequences.

    Three main encoding strategies are possible in the class and in the software, 
    namely using AAI Indices or protein descriptors as well as AAI Indices + Descriptors. 
    Additionally, the protein sequences can be encoded using Digital Signal Processing (DSP) 
    techniques, mainly through the use of informational protein spectra, this is achieved 
    via the pyDSP class in the software. This class accepts strings or lists of AAI Indices 
    or descriptors and then passes these through a pipeline to get the required numerical 
    encoding of the respective sequences. The calculated encodings of the sequences are 
    used as features in the building of the predictive ML models that will then predict the 
    acitivty values for new unseen protein sequences. After the encoding process, 
    various metrics will be captured and stored in a local output folder according to the 
    OUTPUT_FOLDER global var as well as a regression plot showing how well the model, 
    and the selected protein feature attributes, fit to the test data of unseen protein 
    sequences.

    The class has one main input parameter (config_file), that is the filename or filepath
    to the configuration file that contains all the required parameters for the encoding
    strategy/process. The class also accepts a variable number of keyword arguments
    (**kwargs) that will override the config file parameter values of the same name if 
    they are passed in.

    Parameters
    ==========   
    :config_file : str 
        path to configuration file.
    **kwargs: dict
        keyword arguments and values passed into constructor. The keywords should be 
        the same name and form of those in the configuration file. The keyword values
        input take precedence over those in the config files.
        
    Methods
    =======
    read_data():
        read dataset of protein sequences.
    preprocessing():
        pre-process / clean protein sequence dataset.
    get_aai_encoding(indices):
        get AAI encoding for user inputted index/indices.
    encode_aai(aai_indices=None, show_plot=False, print_results=True, output_folder=""):
        full pipeline for encoding protein sequences according to user specified 
        index/indices from the respective records in the AAI database using the 
        get_aai_encoding() function, and outputting the results with all the predictability 
        metrics. Also applying a DSP pipeline if applicable. 
    get_descriptor_encoding(descriptors=None):
        calculate user inputted descriptor/descriptors using the input protein sequences
        and protpy package.
    encode_descriptor(descriptors=None, show_plot=False, print_results=True, output_folder=""):
        full pipeline for encoding protein sequences according to user inputted descriptor/descriptors,
        calculated using the get_descriptor_encoding() function and the protpy package and outputting
        the results with all the predictability metrics. 
    encode_aai_descriptor(aai_indices=None, descriptors=None, show_plot=False, print_results=True, output_folder=""):
        full pipeline for encoding protein sequences according to user specified index/indices 
        in concatenation with descriptor/descriptors using the get_aai_encoding() and 
        get_descriptor_encoding() functions. Output the results with all the predictability
        metrics. 
    output_results(results):
        print out the predictive model parameters/attributes and its results.
    """
    def __init__(self, config_file="", **kwargs):

        self.config_file = config_file
        self.config_parameters = {}

        config_filepath = ""
    
        #append extension if only filename input
        if (os.path.splitext(config_file)[1] == ''):
            config_file = config_file + '.json' 
        
        #open json config file and read in parameters
        if not (isinstance(config_file, str) or config_file is None):
            raise TypeError('JSON config file must be a filepath of type string, got type {}.'.format(type(config_file)))
        if (os.path.isfile(self.config_file)):
            config_filepath = self.config_file
        elif (os.path.isfile(os.path.join('config', self.config_file))):
            config_filepath = os.path.join('config', self.config_file)
        else:
            raise OSError('JSON config file {} not found at path: {}.'.format(self.config_file, config_filepath))
        try:
            with open(config_filepath) as f:
                self.config_parameters = json.load(f)
        except:
            raise JSONDecodeError('Error parsing config JSON file: {}.'.format(config_filepath))

        #create instance of Map class so parameters can be accessed via dot notation
        self.config_parameters = Map(self.config_parameters)

        #dataset parameters
        self.dataset = kwargs.get('dataset') if 'dataset' in kwargs else self.config_parameters.dataset["dataset"]
        self.sequence_col = kwargs.get('sequence_col') if 'sequence_col' in kwargs else self.config_parameters.dataset["sequence_col"] 
        self.activity_col = kwargs.get('activity_col') if 'activity_col' in kwargs else self.config_parameters.dataset["activity"]

        #model parameters
        self.model_parameters = kwargs.get('model_parameters') if 'model_parameters' in kwargs else self.config_parameters.model["parameters"]
        self.algorithm = kwargs.get('algorithm') if 'algorithm' in kwargs else self.config_parameters.model["algorithm"]
        self.test_split = kwargs.get('test_split') if 'test_split' in kwargs else self.config_parameters.model["test_split"]

        #aai parameters
        self.aai_indices = None

        #descriptors parameters
        self.descriptors = None

        #pyDSP parameters - use_dsp, spectrum, window function, window filter
        self.use_dsp = kwargs.get('use_dsp') if 'use_dsp' in kwargs else self.config_parameters.pyDSP["use_dsp"]
        self.dsp_parameters = kwargs.get('dsp_parameters') if 'dsp_parameters' in kwargs else self.config_parameters.pyDSP
        self.filter_parameters = kwargs.get('filter_parameters') if 'filter_parameters' in kwargs else self.dsp_parameters["filter"]
        self.spectrum = kwargs.get('spectrum') if 'spectrum' in kwargs else self.config_parameters.pyDSP["spectrum"]
        self.window_type = kwargs.get('window_type') if 'window_type' in kwargs else self.config_parameters.pyDSP["window"]["type"] 
        self.filter_type = kwargs.get('filter_type') if 'filter_type' in kwargs else self.config_parameters.pyDSP["filter"]["type"]

        #set use_dsp variable to true if any of the DSP parameters passed in as kwargs
        if (('spectrum' or 'window_type' or 'filter_type') in kwargs):
            self.use_dsp = True

        #import and read dataset
        self.data = self.read_data()

        #array of protein sequences
        self.sequences = self.data[self.sequence_col]

        #array of activity values
        self.activity = self.data[self.activity_col]

        #pre-process dataset and protein sequences
        self.preprocessing()

        #get number of rows and cols of dataset
        self.num_seqs = len(self.sequences)
        self.sequence_length = len(max(self.sequences, key=len))

        #feature space dimensions used in building the model
        self.feature_space = ()

        #create instance of Descriptors class
        self.descriptor = Descriptors(self.config_file, protein_seqs=self.sequences)

    def read_data(self):
        """
        Read in dataset according to file name from 'dataset' attribute.
        
        Parameters
        ==========
        None

        Returns
        =======
        :data: pd.DataFrame 
            dataframe of imported dataset.      
        """
        #read in dataset csv if found in path, if not raise error
        if not (os.path.isfile(self.dataset)):
            raise OSError('Dataset filepath is not correct: {}.'.format(self.dataset))

        #read in dataset csv
        try:
            data = pd.read_csv(self.dataset, sep=",", header=0)
            return data
        except:
            raise IOError('Error opening dataset file: {}.'.format(self.dataset))

    def preprocessing(self):
        """
        Pre-process protein sequences in dataset. Validate column names, check
        for invalid amino acids in sequences, remove any gaps in sequence and 
        remove any NAN or +/- infinity values.

        Parameters
        ==========
        None

        Returns
        =======
        None
        """
        #get closest match for sequence column name in dataset
        sequence_col_matches = get_close_matches(self.sequence_col, self.data.columns, cutoff=0.6)

        #set sequence col to the first match found, else raise error
        if (sequence_col_matches != []):
            self.sequence_col = sequence_col_matches[0]
        else:
            raise ValueError('Sequence column ({}) not present in dataset columns:\n{}.'.
                format(self.sequence_col, self.data.columns))

        #remove any gaps found in sequences in dataset
        self.sequences = remove_gaps(self.sequences)

        #verify no invalid amino acids found in sequences, if so then raise error
        invalid_seqs = valid_sequence(self.sequences)
        if (invalid_seqs != None):
            raise ValueError('Invalid amino acids found in protein sequence dataset: {}.'.format(invalid_seqs))

        #get closest match for activity column name in dataset
        activity_matches = get_close_matches(self.activity_col, self.data.columns, cutoff=0.6)

        #set activity col to the first match found, else raise error
        if (activity_matches != []):
            self.activity_col = activity_matches[0]
        else:
            raise ValueError('Activity column ({}) not present in dataset columns:\n{}.'.
                format(self.activity_col,list(self.data.columns)))

        #remove any +/- infinity values or any Null/NAN's from activity values
        self.data[self.activity_col].replace([np.inf, -np.inf], np.nan)
        self.data[self.activity_col].fillna(0, inplace=True)

    def get_aai_encoding(self, aai_indices=None):
        """ 
        Get AAI index encoding values for input index/indices and their respective
        record values from the AAI database. Encode each amino acid in the protein
        sequences in the dataset to the respective values specified in the AAI 
        The index/indices should be in the form of the properties accession number 
        which is the 10 length alphanumeric code that represents each property within 
        the AAI database. If multiple indices/accession numbers input then encode 
        protein sequences with each index and concatenate.

        Parameters
        ==========
        :aai_indices: str/list (default=None)
            string or list of AAI indices/accession numbers.
        
        Returns
        =======
        :encoded_seqs: np.ndarray
            array of the encoded protein sequences in dataset via user input index/indices.
        """
        #validate AAI indices are present in the input parameter, if not raise error
        if (aai_indices == None or aai_indices == ""):
            raise ValueError('AAI indices input parameter cannot be None or empty: {}.'.format(aai_indices))

        #check input indices is of correct type (str/list), if not raise type error
        if (not isinstance(aai_indices, str) and (not isinstance(aai_indices, list))):
            raise TypeError("Input indices parameter must be a string or list, got {}.".format(type(aai_indices)))

        #cast index string to list, split multiple indices using comma
        if (isinstance(aai_indices, str)):
            if (',' in aai_indices):
                aai_indices = aai_indices.split(',')  #split on ',' just in case multiple indices passed in as str
            else:
                aai_indices = [aai_indices]

        #create zeros numpy array to store encoded sequence output
        encoded_aai_ = np.zeros((self.num_seqs, self.sequence_length*len(aai_indices)))

        #if multiple indices used then calculate AAI index encoding for each and concatenate after each calculation
        for index in range(0, len(aai_indices)):

            #get values from aaindex record using its accession number and the aaindex package
            encoded_aai = aaindex1[aai_indices[index]].values

            #initialise temp arrays to store encoded sequences
            temp_seq_vals = []
            temp_all_seqs = []

            #iterate through each protein sequence and amino acid, getting the AAI index encoding value
            for protein in range(0, len(self.sequences)):
                for aa in self.sequences[protein]:
                    temp_seq_vals.append(encoded_aai[aa])

                #append encoding and reset temp array
                temp_all_seqs.append(temp_seq_vals)
                temp_seq_vals = []

            #zero-pad encoding list so that sequences are all the same length
            temp_all_seqs = zero_padding(temp_all_seqs)

            #convert list of lists into array
            temp_all_seqs = np.array(temp_all_seqs, dtype="float32")

            #in first iteration through aai_indices (index=0) set encoded_aai_ to zero-initialised 
            #numpy array, else concatenate to the array in previous iteration
            if (index == 0):
                encoded_aai_ = temp_all_seqs
            else:
                encoded_aai_ = np.concatenate((encoded_aai_, temp_all_seqs), axis=1)
            
        return encoded_aai_

    def encode_aai(self, aai_indices=None, show_plot=False, print_results=True, output_folder=""):
        """
        Full pipeline for encoding proteins sequences in dataset using the input AAI indices 
        from the AAI database. If multiple indices/accession numbers input then calculate each 
        and concatenate them. Build predictive regression ML model from encoded AAI feature data 
        for predicting the activity/fitness values of unseen sequences. 
        
        The resulting model assets and its results will be exported to the directory pointed to 
        by the global var OUTPUT_DIR. If use_dsp config parameter is true then pass AAI 
        Indices through a DSP transformation pipeline specified by the config's DSP parameters 
        (spectrum, window & filter) via the PyDSP module and class.

        Parameters
        ==========
        :aai_indices: str/list (default=None)
            string or list of indices/accession numbers from the AAI.
        :show_plot: bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results: bool (default=True)
            if true, output verbose output of results and parameters from encoding process.
        :output_folder: str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        =======
        :aai_df: pd.Dataframe
            pandas Dataframe storing metrics and results of encoding.
        """
        #validate AAI indices are present in the input parameter
        if (aai_indices == None or aai_indices == "" or aai_indices == []):
            raise ValueError('AAI indices input parameter cannot be None or empty: {}.'.format(aai_indices))

        #check input indices is of correct type (str/list), if not raise type error
        if ((not isinstance(aai_indices, str)) and (not isinstance(aai_indices, list))):
            raise TypeError("Input indices parameter must be a string or list, got {}.".format(type(aai_indices)))

        self.aai_indices = aai_indices

        #if list of one element with multiple indices, split them into list of individual elements
        if isinstance(self.aai_indices, list) and len(self.aai_indices) == 1:
            self.aai_indices = self.aai_indices[0].replace(' ', '').split(',')

        #convert string indices into comma seperated list, remove whitespace
        if isinstance(self.aai_indices, str):
            self.aai_indices = self.aai_indices.replace(' ', '').split(',')

        #sort list of indices into alphabetical order
        self.aai_indices.sort()

        #dataframe to store encoding of inputted aai indices
        aai_encoding_df = pd.DataFrame()

        #iterate over each index, calculate its encoding, apply DSP functionality if applicable, concat into one dataframe
        for index in self.aai_indices:

            #get AAI index encodings specified by indices input parameter
            encoded_seqs = self.get_aai_encoding(index)

            #if use_dsp true then get protein spectra from encoded sequences via the AAI indices using PyDSP class,
            #else use the AAI indices encoding's themselves as the feature/training data (X)
            if (self.use_dsp):
                #if input spectrum is none or empty, raise error.
                if (self.spectrum == None or self.spectrum == ""):
                    raise ValueError('Spectrum cannot be None or empty: {}.'.format(self.spectrum))
                pyDSP = PyDSP(self.config_file, protein_seqs=encoded_seqs)
                pyDSP.encode_sequences()
                X = pd.DataFrame(pyDSP.spectrum_encoding) #set training data to FFT spectrum encoding
            else:
                X = pd.DataFrame(encoded_seqs)  #no DSP applied to encoded sequences

            #concat encoding of current aai index with other encodings for training data
            aai_encoding_df = pd.concat([aai_encoding_df, X], axis=1)

        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(aai_encoding_df.columns) + 1)]
        aai_encoding_df.columns = columns

        #set class variable to the training data feature space
        self.feature_space = aai_encoding_df.shape

        #create instance of model class of type specified by algorithm parameter using X and Y data
        self.model = Model(aai_encoding_df, self.activity, self.algorithm, parameters=self.model_parameters)
        
        #updating algorithm attribute
        self.algorithm = repr(self.model)

        #get training and test dataset split from model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(test_split=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        #get categories for all indices in self.aai_indices
        index_cat = []
        if (isinstance(self.aai_indices, list)):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]

        #create comma seperated list of categories
        index_cat = ', '.join(index_cat)

        #create output dataframe, set first row to attribute/metric values
        aai_df = pd.DataFrame(columns=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])
        aai_df.loc[0] = [', '.join(self.aai_indices), str(index_cat).strip(), eval.r2, eval.rmse, eval.mse, eval.mae, eval.rpd, eval.explained_var]

        #convert index and category from default Object type -> String datatypes
        aai_df['Index'] = aai_df['Index'].astype(pd.StringDtype())
        aai_df['Category'] = aai_df['Category'].astype(pd.StringDtype())
        
        #print out results from encoding
        if (print_results):
            self.output_results(aai_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, output_folder, show_plot)

        #save results of encoding to output folder specified by input param
        save_results(aai_df, 'aai_results', output_folder=output_folder)
        
        #reset aai_indices instance variable
        #self.aai_indices = ""

        return aai_df

    def get_descriptor_encoding(self, descriptors=None):
        """
        Calculate inputted descriptor(s), using the Descriptors class and custom-built 
        protpy package, requried for the encoding process. Get closest match to user 
        inputted string or list of descriptors using difflib library. If a single 
        descriptor is input then calculate it and return, if list of descriptors input 
        then calculate each descriptor's value and concatenate.

        Parameters
        ==========
        :descriptors: str/list (default=None)
            string or list of protein descriptor names.

        Returns
        =======
        :encoded_desc: pd.DataFrame
            pandas dataframe of calculated descriptor values according to user
            inputted descriptor(s).
        """
        #raise error if no descriptors specified in input
        if (descriptors == None or descriptors == "" or descriptors == []): 
            raise ValueError('Descriptors input parameter cannot be None or empty: {}.'.format(descriptors))
        
        #check input descriptor is of correct type str or list, if not raise type error
        if (not isinstance(descriptors, str) and (not isinstance(descriptors, list))):
            raise TypeError("Input descriptor parameter must be a str or list, got {}.".format(type(descriptors)))

        #cast descriptors parameter to a list if it is a str by creating comma seperated list
        if (isinstance(descriptors, str)):
            descriptors = descriptors.split(',')

        #remove any leading or trailing whitespace from descriptors
        descriptors = [de.strip() for de in descriptors]

        #create instance of Descriptors class using data in instance variable and config file
        descr = Descriptors(self.config_file, protein_seqs=self.sequences)
        
        #store list of correct descriptor names from ones user input using the difflib library
        temp_descriptors = []

        #get closest valid available descriptor name from input descriptor parameter,
        #if a list of descriptors passed in as the input parameter then get
        #all valid descriptors in list
        for de in range(0, len(descriptors)):
            desc_matches = get_close_matches(descriptors[de],
                descr.valid_descriptors, cutoff=0.6)
            descriptors[de] = desc_matches[0]
            if (descriptors[de] == []):
                raise ValueError('No approximate descriptor found from one input: {}.'.format(de))
            temp_descriptors.append(desc_matches[0])
        
        #initialise temp lists and DF to store encoded descriptor values
        encoded_desc_temp = []
        encoded_desc_vals = []
        encoded_desc_temp = pd.DataFrame()

        #iterate and get each descriptors' values using Descriptor class and protpy package
        for d in range(0, len(descriptors)):
            encoded_desc_temp = descr.get_descriptor_encoding(descriptors[d])
            #raise value error if descriptor is empty/None
            if (encoded_desc_temp.empty):
                raise ValueError('Descriptor cannot be empty or None: {}.'.format(descriptors[d]))
            encoded_desc_vals.append(encoded_desc_temp) #append to array of all descriptor values
            encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe for next iteration

        #concatenate dataframes of descriptors
        encoded_desc = pd.concat(encoded_desc_vals, axis=1)

        return encoded_desc

    def encode_descriptor(self, descriptors=None, show_plot=False, print_results=True, output_folder=""):
        """
        
        Full pipeline for encoding the protein sequences in the dataset using protein 
        physiochemical, biochemical and or structural descriptors, using the Descriptors 
        class and custom-built protpy package, and build predictive ML regression model 
        from the descriptor feature/training data. This model is then used to calculate
        the activity/fitness value of unseen test sequences. If multiple descriptors input 
        then calculate each and concatenate them. The resulting model assets and its metric's 
        results will be exported to the directory pointed to by the global variable OUTPUT_DIR.

        Parameters
        ==========
        :descriptors: str/list (default=None)
            string or list of protein descriptor names. 
        :show_plot: bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results: bool (default=True)
            if true, output verbose output of results and parameters from encoding process.
        :output_folder: str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        =======
        :desc_df: pd.DataFrame
            pandas dataframe storing metrics and results of encoding.
        """
        #raise error if no descriptor specified in input
        if (descriptors == None or descriptors == ""):
            raise ValueError('Descriptors input parameter cannot be None or empty: {}.'.format(descriptors))

        #check input descriptor is of correct type (str or list), if not raise type error
        if (not (isinstance(descriptors, str))) and (not (isinstance(descriptors, list))):
            raise TypeError("Input descriptor parameter must be a string or list, got {}.".format(type(descriptors)))

        #set class attribute        
        self.descriptors = descriptors

        #if multiple descriptors input as str, split into comma seperated list
        if isinstance(self.descriptors, str):
            self.descriptors = self.descriptors.replace(' ', '').split(',')

        #if list of multiple descriptors input in one string, seperate into commas seperated list of individual elements
        if isinstance(self.descriptors, list) and len(self.descriptors) == 1:
            self.descriptors = self.descriptors[0].replace(' ', '').split(',')

        #sort list of descriptors into alphabetical order
        self.descriptors.sort()

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.sequences)

        #pandas dataframe to store all output results
        desc_df = pd.DataFrame(columns=['Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])
        
        #object to store sequence encodings for each input descriptor
        descriptor_encoding_df = pd.DataFrame()

        #iterate over each input descriptor, calculate its encoding from its respective function, concatenate with main encoding object
        for desc in range(0, len(self.descriptors)):

            #get closest matching descriptor from descriptor input parameter using difflib library
            desc_matches = get_close_matches(self.descriptors[desc], descr.valid_descriptors, cutoff=0.6)
            if (desc_matches != []):
                self.descriptors[desc] = desc_matches[0]
            else:
                raise ValueError('Could not find a match for the input descriptor ({}) in list of valid descriptors:\n{}.'.
                    format(self.descriptors[desc], descr.valid_descriptors))

            #concatenate encoding of current descriptor to main encodng object
            descriptor_encoding_df = pd.concat([descriptor_encoding_df, self.get_descriptor_encoding(descriptors=self.descriptors[desc])], axis=1)
        
        #set class variable to the training data feature space
        self.feature_space = descriptor_encoding_df.shape

        #create instance of model class of type specified by algorithm parameter using X and Y data
        self.model = Model(descriptor_encoding_df, self.activity, self.algorithm, parameters=self.model_parameters)

        #updating algorithm attribute
        self.algorithm = repr(self.model)

        #get training and test dataset split using Model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(test_split=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        #get groups for all descriptors in self.desciptors, put multiple descriptor groups into comma seperated list
        if (isinstance(self.descriptors, list)):
            desc_group = []
            for desc_ in self.descriptors:
                desc_group.append(descr.descriptor_groups[desc_])
            desc_group = ', '.join(desc_group)
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #add metric values to output dataframe
        desc_df.loc[0] = [', '.join(self.descriptors), desc_group, eval.r2, eval.rmse, eval.mse, eval.mae, eval.rpd, eval.explained_var]

        #convert Descriptor and Group from default Object type -> String datatypes
        desc_df['Descriptor'] = desc_df['Descriptor'].astype(pd.StringDtype())
        desc_df['Group'] = desc_df['Group'].astype(pd.StringDtype())

        #ensure aai indices attribute doesn't show up in output results
        if (self.aai_indices != None):
            self.aai_indices = None

        #print out results from encoding
        if (print_results):
            self.output_results(desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, output_folder, show_plot)

        #save results of encoding to output folder
        save_results(desc_df, 'desc_results', output_folder=output_folder)

        #reset descriptors instance variable
        self.descriptors = None 

        return desc_df

    def encode_aai_descriptor(self, aai_indices=None, descriptors=None, show_plot=False, print_results=True, output_folder=""):
        """
        Encode using both AAI indices and the physiochemical/structural descriptors from
        the get_aai_encoding() and get_descriptor_encoding() functions. The two outputs 
        from the individual encoding strategies, previously described above, will be 
        concatenated together and used in the building of a predictive regression ML 
        model. The resulting model assets and its results will be exported to the 
        directory pointed to by the global variable OUTPUT_DIR. If the config parameter 
        use_dsp is true then pass AAI Indices through a DSP transformation pipeline 
        specified by the DSP parameters (spectrum, window & filter) via the PyDSP 
        class/module.

        Parameters
        ==========
        :aai_indices: str/list (default=None)
            string or list of indices/accession numbers from the AAI database.
        :descriptors: str/list (default=None)
            string or list of protein descriptors names.
        :show_plot: bool (default=False)
            display regression plot of best predictive model. If false then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results: bool (default=True)
            if true, output verbose output of results and parameters from encoding process.
        :output_folder: str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        =======
        :aai_desc_df : pd.Dataframe
            pandas dataframe storing metrics and results of encoding.
        """
        #validate AAI indices and Descriptors are present in the input parameters, return error if either is None
        if (descriptors == None or descriptors == "") or (aai_indices == None or aai_indices == ""):
                raise ValueError('AAI Indices and Descriptor input parameters must not be empty or None.')

        #check input descriptor & indices are of correct type (str/list), if not raise type error
        if (not isinstance(aai_indices, str) and (not isinstance(aai_indices, list)) or \
                (not isinstance(descriptors, str) and (not isinstance(descriptors, list)))):
            raise TypeError("Input AAI indices and descriptors parameter must be of type string or list.")

        #set instance attributes
        self.aai_indices = aai_indices           
        self.descriptors = descriptors

        #if list of multiple descriptors input in one string, seperate into commas seperated list of individual elements
        if isinstance(self.descriptors, list) and len(self.descriptors) == 1:
            self.descriptors = self.descriptors[0].replace(' ', '').split(',')

        #convert descriptors into comma seperated list if str input, remove whitespace
        if isinstance(self.descriptors, str):
            self.descriptors = self.descriptors.replace(' ', '').split(',')

        #if list of one element with multiple indices, split them into list of individual elements
        if isinstance(self.aai_indices, list) and len(self.aai_indices) == 1:
            self.aai_indices = self.aai_indices[0].replace(' ', '').split(',')

        #convert string indices into comma seperated list, remove whitespace
        if isinstance(self.aai_indices, str):
            self.aai_indices = self.aai_indices.replace(' ', '').split(',')
        
        #sort list of indices into alphabetical order
        self.aai_indices.sort()

        #sort list of descriptors into alphabetical order
        self.descriptors.sort()

        #create output results Dataframe
        aai_desc_df = pd.DataFrame(columns=['Index', 'Category', 'Descriptor', 'Group', 'R2', 'RMSE', \
            'MSE', 'MAE', 'RPD', 'Explained Variance'])

        #dataframe to store the encodings for the AAI indices
        aai_encoding_df = pd.DataFrame()

        #iterate over each index code, calculate its encoding features using respective function, convert to dataframe and concat to main dataframe
        for index in self.aai_indices:
            aai_encoding_df = pd.concat([aai_encoding_df, pd.DataFrame(self.get_aai_encoding(index))], axis=1)
        
        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(aai_encoding_df.columns) + 1)]
        aai_encoding_df.columns = columns

        #if AAI indices encoding is empty, raise error
        if (aai_encoding_df.empty):
            raise ValueError('AAI Indices encoding cannot be empty or None: {}.'.format(aai_indices))

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.sequences)

        #dataframe to store the encodings for the descriptors
        descriptor_encoding_df = pd.DataFrame()

        #iterate over each input descriptor, calculate its encoding from its respective function, concatenate with main encoding object
        for desc in range(0, len(self.descriptors)):

            #get closest matching descriptor from descriptor input parameter using difflib library
            desc_matches = get_close_matches(self.descriptors[desc], descr.valid_descriptors, cutoff=0.6)
            if (desc_matches != []):
                self.descriptors[desc] = desc_matches[0]
            else:
                raise ValueError('Could not find a match for the input descriptor ({}) in list of valid descriptors:\n{}.'.
                    format(self.descriptors[desc], descr.valid_descriptors))

            #concatenate encoding of current descriptor to main encodng object
            descriptor_encoding_df = pd.concat([descriptor_encoding_df, self.get_descriptor_encoding(descriptors=self.descriptors[desc])], axis=1)
        
        #reset index for aai indices and in descriptors output dataframe
        aai_encoding_df.reset_index(inplace=True, drop=True)
        descriptor_encoding_df.reset_index(inplace=True, drop=True)

        #concatenate AAI index and Descriptor features to get training data (X)
        X = pd.concat([aai_encoding_df, descriptor_encoding_df], axis=1)

        #set class variable to the training data feature space
        self.feature_space = X.shape

        #create instance of model class of type specified by algorithm parameter using X and Y data
        self.model = Model(X, self.activity, self.algorithm, parameters=self.model_parameters)
        
        #updating algorithm attribute
        self.algorithm = repr(self.model)

        #get training and test dataset split using Model class
        X_train, X_test, Y_train, Y_test = self.model.train_test_split(test_split=self.test_split)

        #fit predictive model
        self.model.fit()

        #predict activity values for test data
        Y_pred = self.model.predict()

        #create instance of Evaluate class which will get all the evaluation metrics
        eval = Evaluate(Y_test, Y_pred)

        #get categories for all indices in self.aai_indices
        index_cat = []
        if (isinstance(self.aai_indices, list)):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]
        
        #seperate index categories into comma seperated string
        index_cat = ', '.join(index_cat)

        #get groups for all descriptors in self.desciptors, can be string or list of descriptors
        if (isinstance(self.descriptors, list)):
            desc_group = []
            for desc_ in self.descriptors:
                desc_group.append(descr.descriptor_groups[desc_])
            desc_group = ', '.join(desc_group)
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #set output dataframe columns
        aai_desc_df.loc[0] = [', '.join(self.aai_indices), str(index_cat).strip(), ', '.join(self.descriptors), str(desc_group), eval.r2, 
            eval.rmse, eval.mse, eval.mae, eval.rpd, eval.explained_var]

        #convert Index, Category, Descriptor and Group from default Object type -> String datatypes
        # aai_desc_df['Index'] = aai_desc_df['Index'].astype(pd.StringDtype())
        aai_desc_df['Index'] = aai_desc_df['Index'].astype("string")
        aai_desc_df['Category'] = aai_desc_df['Category'].astype(pd.StringDtype())
        aai_desc_df['Descriptor'] = aai_desc_df['Descriptor'].astype(pd.StringDtype())
        aai_desc_df['Group'] = aai_desc_df['Group'].astype(pd.StringDtype())

        #print out results from encoding
        if (print_results):
            self.output_results(aai_desc_df)

        #plot regression plot for predictive model
        plot_reg(Y_test, Y_pred, eval.r2, output_folder, show_plot)

        #save results of encoding to output folder
        save_results(aai_desc_df, 'aai_desc_results', output_folder=output_folder)

        return aai_desc_df

    def output_results(self, results):
        """
        Print out the predictive model parameters/attributes and its results.

        Parameters
        ==========
        :results: dict/pd.Series
            dictionary or Series of metrics and their associated values.

        Returns
        =======
        None
        """
        #create text wrapper for aai indices, descriptors and model parameters text
        line_length = 90

        print('##########################################################################################')
        print('###################################### Parameters ########################################\n')
        if (not (self.aai_indices is None)) and (len(self.aai_indices) <= 10):
            print(textwrap.fill('# AAI Indices: {}'.format(', '.join(self.aai_indices)), line_length))        
            if (self.use_dsp):
                print('# DSP Parameters:\n  # Spectrum: {}\n  # Window Function: {} \
                \n  # Filter Function: {}'.format(self.spectrum, self.window_type, self.filter_type))  
        if (self.descriptors is not None):
            print(textwrap.fill('# Descriptors: {}'.format(', '.join(self.descriptors)), line_length))
        print('# Configuration File: {}\n# Dataset: {}\n# Number of Sequences/Sequence Length: {} x {} \
            \n# Target Activity: {}'.format(os.path.basename(self.config_file), self.dataset, self.num_seqs, self.sequence_length, self.activity_col))
        print("# Algorithm: {}".format(repr(self.model)))
        if (self.model_parameters == "" or self.model_parameters is None or self.model_parameters == {}):
            # print('# Model Parameters: {}'.format("\n\t".join(tw.wrap(', '.join(temp_model_parameters.model.get_params())))))        
            print(textwrap.fill('# Model Parameters: {}'.format(self.model.model.get_params()), line_length))
        else:
            print(textwrap.fill('# Model Parameters: {}'.format(self.model_parameters), line_length))
        print('# Test Split: {}\n# Feature Space: {}'.format(self.test_split, self.feature_space))

        print('\n##########################################################################################')
        print('######################################## Results #########################################\n')
        print('# R2: {}'.format(results['R2'].values[0]))
        print('# RMSE: {} '.format(results['RMSE'].values[0]))
        print('# MSE: {} '.format(results['MSE'].values[0]))
        print('# MAE: {}'.format(results['MAE'].values[0]))
        print('# RPD {}'.format(results['RPD'].values[0]))
        print('# Explained Variance {}\n'.format(results['Explained Variance'].values[0]))
        print('##########################################################################################\n')

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
    def activity_col(self):
        return self._activity_col

    @activity_col.setter
    def activity_col(self, val):
        self._activity_col = val

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
    def model_parameters(self):
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, val):
        self._model_parameters = val

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
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, val):
        self._sequence_length = val

    # def __str__(self):
    #     return "Instance of PySAR class, using parameters: {}.".format(self.__dict__)

    # def __repr__(self):
    #     return "<PySAR: {}>".format(self)