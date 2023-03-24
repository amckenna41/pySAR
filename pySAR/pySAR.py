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
    AAI Indices and or strucutrual, biochemical and physiochemical protein descriptors that
    are then used as features in the building of predictive regression models created to
    map the protein sequences to a sought-after activity/fitness value (activity attribute),
    this is known as a Sequence Activity Relationship (SAR) or Sequence Function Relationship
    (SFR). Three main encoding strategies are possible in the class and in the software, 
    namely using AAI Indices or protein descriptors as well as AAI Indices + Descriptors. 
    Additionally, the protein sequences can be encoded using Digital Signal Processing (DSP) 
    techniques, mainly through the use of informational protein spectra, this is achieved 
    via the pyDSP class in the software. This class accepts strings or lists of AAI Indices 
    or descriptors and then passes these through a pipeline to get the required numerical 
    encoding of the respective sequences. The calculated encodings of the sequences are 
    used as features in the building of the predictive models that will then predict the 
    acitivty values for new unseen protein sequences. After the encoding process, 
    various metrics will be captured and stored in a local output folder according to the 
    OUTPUT_FOLDER global var as well as a regression plot showing how well the model, 
    and the selected protein feature attributes, fit to the test data of unseen protein 
    sequences.

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
    encode_aai(indices=None, show_plot=False, print_results=True, output_folder=""):
        get encoded protein sequences according to user inputted index/indices, applying 
        DSP if applicable. 
    get_descriptor_encoding(descriptors):
        calculate user inputted descriptor/descriptors according to user input.
    encode_descriptor(descriptor=None, show_plot=False, print_results=True, output_folder=""):
        get encoded protein sequences according to user inputted descriptor/descriptors.
    encode_aai_descriptor(indices=None, descriptors=None, show_plot=False, print_results=True, output_folder=""):
        get encoded protein sequences according to user inputted AA index/indices and 
        descriptor/descriptors.
    output_results(results):
        print out the predictive model parameters/attributes and its results.
    """
    def __init__(self, config_file=""):

        self.config_file = config_file
        self.parameters = {}

        config_filepath = ""
    
        #open json config file and read in parameters
        if not (isinstance(config_file, str) or config_file is None):
            raise TypeError('JSON config file must be a filepath of type string, not of type {}.'.format(type(config_file)))
        if (os.path.splitext(config_file)[1] == ""):
            config_file = config_file + '.json' #append extension if only filename input
        if (os.path.isfile(self.config_file)):
            config_filepath = self.config_file
        elif (os.path.isfile(os.path.join('config', self.config_file))):
            config_filepath = os.path.join('config', self.config_file)
        else:
            raise OSError('JSON config file not found at path: {}.'.format(self.config_file))
        try:
            with open(config_filepath) as f:
                self.parameters = json.load(f)
        except:
            raise JSONDecodeError('Error parsing config JSON file: {}.'.format(config_filepath))

        #create instance of Map class so parameters can be accessed via dot notation
        self.parameters = Map(self.parameters)
        self.dataset = self.parameters.dataset["dataset"] 
        if (os.path.splitext(self.dataset)[1] == ''):
            self.dataset = self.dataset + ".txt" #append extension if just filename input
        self.sequence_col = self.parameters.dataset["sequence_col"]
        self.activity_col = self.parameters.dataset["activity"]

        #model parameters
        self.model_parameters = self.parameters.model["parameters"]
        self.algorithm = self.parameters.model["algorithm"]
        self.test_split = self.parameters.model["test_split"]

        #aai parameters
        self.aai_indices = None

        #descriptors parameters
        self.descriptors = None

        #pyDSP parameters
        self.use_dsp = self.parameters.pyDSP["use_dsp"]
        if (self.use_dsp):
            self.spectrum = self.parameters.pyDSP["spectrum"]
            self.window_type = self.parameters.pyDSP["window"]["type"]
            self.filter_type = self.parameters.pyDSP["filter"]["type"]

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
        self.seq_len = len(max(self.sequences, key=len))

        #create instance of Descriptors class
        self.descriptor = Descriptors(self.config_file, protein_seqs=self.sequences)

    def read_data(self):
        """
        Read in dataset according to file name from 'dataset' attribute. By default
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
            raise ValueError('Sequence Column ({}) not present in dataset columns:\n{}'.
                format(self.sequence_col, self.data.columns))

        #remove any gaps found in sequences in dataset
        self.sequences = remove_gaps(self.sequences)

        #verify no invalid amino acids found in sequences, if so then raise error
        invalid_seqs = valid_sequence(self.sequences)
        if (invalid_seqs != None):
            raise ValueError('Invalid Amino Acids found in protein sequence dataset: {}.'.
                format(invalid_seqs))

        #get closest match for activity column name in dataset
        activity_matches = get_close_matches(self.activity_col, self.data.columns, cutoff=0.4)

        #set activity col to the first match found, else raise error
        if (activity_matches != []):
            self.activity_col = activity_matches[0]
        else:
            raise ValueError('Activity Column ({}) not present in dataset columns:\n{}'.
                format(self.activity_col,list(self.data.columns)))

        #remove any +/- infinity values or any Null/NAN's from activity values
        self.data[self.activity_col].replace([np.inf, -np.inf], np.nan)
        self.data[self.activity_col].fillna(0, inplace=True)

    def get_aai_encoding(self, indices=None):
        """ 
        Get AAI index encoding values for index specified by indices for each amino
        acid in each of the protein sequences in dataset. The index/indices should be
        in the form of the properties accession number which is the 10 length 
        alphanumeric code that represents each property within the AAI database. If 
        multiple indices/accession numbers input then encode protein sequences with 
        each index and concatenate.

        Parameters
        ----------
        :indices : str/list (default=None):
            string or list of AAI indices/accession numbers.
        
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

        #if multiple indices used then calculate AAI index encoding for each and
        #then concatenate after each calculation
        for index in range(0, len(indices)):

            #get values from aaindex record using its accession number
            encoded_aai = aaindex1[indices[index]].values

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

            #in first iteration through indices (index=0) set encoded_aai_ to zero-initialised 
            #numpy array, else concatenate to the array in previous iteration
            if (index == 0):
                encoded_aai_ = temp_all_seqs
            else:
                encoded_aai_ = np.concatenate((encoded_aai_, temp_all_seqs), axis=1)
            
        return encoded_aai_

    def encode_aai(self, indices=None, show_plot=False, print_results=True, output_folder=""):
        """
        Encode using AAI indices from the AAI database. If multiple 
        indices/accession numbers input then calculate each and concatenate them. 
        Build predictive model from AAI feature data. The resulting model assets 
        and its results will be exported to the directory pointed to by the global 
        variable OUTPUT_DIR. If use_dsp config parameter is true then pass AAI 
        Indices through a DSP transformation specified by the config's DSP parameters 
        (spectrum, window & filter) via the PyDSP module and class.

        Parameters
        ----------
        :indices : str/list (default=None)
            string or list of indices/accession numbers from the AAI.
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding process.
        :output_folder : str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        -------
        :aai_df : pd.Dataframe
            pandas Dataframe storing metrics and results of encoding.
        """
        #validate AAI indices are present in the input parameter
        if (indices == None or indices == "" or indices == []):
            raise ValueError('AAI indices input parameter cannot be None or empty.')

        #check input indices is of correct type (str/list), if not raise type error
        if ((not isinstance(indices, str)) and (not isinstance(indices, list))):
            raise TypeError("Input indices parameter must be a string or list, got {}.".format(type(indices)))

        self.aai_indices = indices

        #get AAI index encodings specified by indices input parameter
        encoded_seqs = self.get_aai_encoding(indices)

        #if use_dsp true then get protein spectra from encoded sequences via the AAI indices using PyDSP class,
        #else use the AAI indices encoding's themselves as the feature/training data (X)
        if (self.use_dsp):
            #if input spectrum is none or empty, raise error.
            if (self.spectrum == None or self.spectrum == ""):
                raise ValueError('Spectrum cannot be None or empty, got {}.'.format(self.spectrum))
            pyDSP = PyDSP(self.config_file, protein_seqs=encoded_seqs)
            pyDSP.encode_seqs()
            X = pd.DataFrame(pyDSP.spectrum_encoding) #set training data to FFT spectrum encoding
        else:
            X = pd.DataFrame(encoded_seqs)  #no DSP applied to encoded sequences

        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(X.columns) + 1)]
        X.columns = columns

        #create instance of model class of type specified by algorithm parameter using X and Y data
        self.model = Model(X, self.activity, self.algorithm, parameters=self.model_parameters)
        
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

        index_cat = []

        #get categories for all indices in self.aai_indices
        if (isinstance(self.aai_indices, list)):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]

        #create comma seperated list of categories
        index_cat = ', '.join(index_cat)

        #join list of indices into single string
        if (isinstance(indices, list)):
            indices = ', '.join(indices)

        #create output dataframe, set first row to attribute/metric values
        aai_df = pd.DataFrame(columns=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])
        aai_df.loc[0] = [indices, str(index_cat).strip(), eval.r2, eval.rmse, eval.mse, eval.mae, eval.rpd, eval.explained_var]

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
        Calculate inputted descriptor(s), using the Descriptors class and 
        custom-built protpy package, requried for the encoding process. 
        Get closest match to user inputted string or list of descriptors 
        using difflib library. If a single descriptor is input then 
        calculate it and return, if list of descriptors input then 
        calculate each descriptor's value and concatenate.

        Parameters
        ----------
        :descriptors : str/list (default=None)
            string or list of protein descriptor names.

        Returns
        -------
        :encoded_desc : pd.DataFrame
            pandas dataFrame of calculated descriptor values according to user
            inputted descriptor(s).
        """
        #raise error if no descriptors specified in input
        if (descriptors == None or descriptors == "" or descriptors == []): 
            raise ValueError('Descriptors input parameter cannot be None or empty.')
        
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
                descr.valid_descriptors, cutoff=0.4)
            descriptors[de] = desc_matches[0]
            if (descriptors[de] == []):
                raise ValueError('No approximate descriptor found from one entered: {}.'.format(de))
            temp_descriptors.append(desc_matches[0])
        
        self.descriptors = temp_descriptors

        #initialise temp lists and DF to store encoded descriptor values
        encoded_desc_temp = []
        encoded_desc_vals = []
        encoded_desc_temp = pd.DataFrame()

        #iterate and get each descriptors' values using Descriptor class and protpy package
        for d in range(0, len(descriptors)):
            encoded_desc_temp = descr.get_descriptor_encoding(descriptors[d])
            #raise value error if descriptor is empty
            if (encoded_desc_temp.empty):
                raise ValueError('Descriptor ({}) cannot be empty or None.'.format(descriptors[d]))
            encoded_desc_vals.append(encoded_desc_temp) #append to array of all descriptor values
            encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe for next iteration

        #concatenate dataframes of descriptors
        encoded_desc = pd.concat(encoded_desc_vals, axis=1)

        return encoded_desc

    def encode_descriptor(self, descriptor=None, show_plot=False, print_results=True, output_folder=""):
        """
        Encode protein sequences using protein physiochemical, biochemical and or 
        structural descriptors, using the Descriptors class and custom-built protpy
        package, and build predictive model from the descriptor feature/training data. 
        If multiple descriptors input then calculate each and concatenate them. 
        The resulting model assets and its results will be exported to the directory 
        pointed to by the global variable OUTPUT_DIR.

        Parameters
        ----------
        :descriptor : str/list (default=None)
            string or list of protein descriptor names. 
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding 
            process.
        :output_folder : str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        -------
        :desc_df : pd.DataFrame
            pandas dataframe storing metrics and results of encoding.
        """
        #raise error if no descriptor specified in input
        if (descriptor == None or descriptor == ""):
            raise ValueError('Descriptors input parameter cannot be None or empty.')

        #check input descriptor is of correct type (str), if not raise type error
        if not (isinstance(descriptor, str)):
            raise TypeError("Input descriptor parameter must be a string, got {}.".format(type(descriptor)))

        self.descriptors = descriptor

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.sequences)

        #get closest matching descriptor from descriptor input parameter using difflib library
        desc_matches = get_close_matches(self.descriptors, descr.valid_descriptors, cutoff=0.6)
        if (desc_matches != []):
            self.descriptors = desc_matches[0]  #set desc to closest descriptor match found
        else:
            raise ValueError('Could not find a match for the input descriptor ({}) in available valid descriptors:\n {}.'.
                format(self.descriptors, descr.valid_descriptors))

        #pandas dataframe to store all output results
        desc_df = pd.DataFrame(columns=['Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])

        #set training data (X) to descriptor-encoded protein sequences
        X = self.get_descriptor_encoding(descriptors=self.descriptors)

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

        desc_group = ""

        #get groups for all descriptors in self.desciptors
        if (isinstance(self.descriptors, list)):
            for i in range(0, len(self.descriptors)):
                desc_group += desc_group + ', ' + \
                    descr.descriptor_groups[self.descriptors[i]]
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #add metric values to dataframe
        desc_df.loc[0] = [descriptor, desc_group, eval.r2, eval.rmse, eval.mse, eval.mae, eval.rpd, eval.explained_var]

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

    def encode_aai_descriptor(self, indices=None, descriptors=None, show_plot=False, print_results=True, output_folder=""):
        """
        Encode using both AAI indices and the physiochemical/structural descriptors. 
        The two outputs from the individual encoding strategies, previously described
        above, will be concatenated together and used in the building of a 
        predictive regression model. The resulting model assets and its results will 
        be exported to the directory pointed to by the global variable OUTPUT_DIR. 
        If the config parameter use_dsp is true then pass AAI Indices through a DSP 
        transformation specified by the DSP parameters (spectrum, window & filter) via 
        the PyDSP class/module.

        Parameters
        ----------
        :indices : str/list (default=None)
            string or list of indices/accession numbers from the AAI database.
        :descriptors : str/list (default=None)
            string or list of protein descriptors names.
        :show_plot : bool (default=False)
            display regression plot of best predictive model. If False then the plot
            will just be saved to the output folder, else it'll be displayed & also saved.
        :print_results : bool (default=True)
            if true, output verbose output of results and parameters from encoding process.
        :output_folder : str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        -------
        :aai_desc_df : pd.Dataframe
            pandas dataframe storing metrics and results of encoding.
        """
        #validate AAI indices and Descriptors are present in the input parameters
        #or instance variables, return error if either is None
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

        #create output results Dataframe
        aai_desc_df = pd.DataFrame(columns=['Index', 'Category', 'Descriptor', 'Group', 'R2', 'RMSE', \
            'MSE', 'MAE', 'RPD', 'Explained Variance'])

        #create instance of Descriptors class using data in instance variable
        descr = Descriptors(self.config_file, protein_seqs=self.sequences)

        #get AAI index encoding features using respective function, convert to dataframe
        aai_encoding = pd.DataFrame(self.get_aai_encoding(indices))
        
        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(aai_encoding.columns) + 1)]
        aai_encoding.columns = columns

        #if AAI indices encoding is empty, raise error
        if (aai_encoding.empty):
            raise ValueError('AAI Indices encoding ({}) cannot be empty or None.'.format(indices))

        #get descriptor encoding features using respective function
        descriptor_encoding = self.get_descriptor_encoding(descriptors)

        #if descriptors encoding is empty, raise error
        if (descriptor_encoding.empty):
            raise ValueError('Descriptor encoding ({}) cannot be empty or None.'.format(descriptors))

        #aai_encoding = aai_encoding.reset_index()
        #reset index in descriptors output dataframe
        descriptor_encoding.reset_index(inplace=True, drop=True)

        #concatenate AAI index and Descriptor features to get training data (X)
        X = pd.concat([aai_encoding, descriptor_encoding], axis=1)

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

        #set output dataframe columns
        aai_desc_df.loc[0] = [indices, str(index_cat).strip(), str(self.descriptors), str(desc_group), eval.r2, 
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
            \n# Activity -> {}'.format(self.dataset, self.num_seqs, self.seq_len, self.activity_col))
        if (self.aai_indices is not None and len(self.aai_indices) <= 10):
            print('# AAI Indices -> {}'.format(self.aai_indices))
            if (self.use_dsp):
                print('# Using DSP -> {}\n  # Spectrum -> {}\n  # Window Function -> {} \
                \n  # Filter Function -> {}\n'.format(
                    self.use_dsp, self.spectrum, self.window_type, self.filter_type))  
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
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, val):
        self._seq_len = val

    # def __str__(self):
    #     return "Instance of PySAR class, using parameters: {}.".format(self.__dict__)

    # def __repr__(self):
    #     return "<PySAR: {}>".format(self)