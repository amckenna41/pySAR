################################################################################
#################                     pySAR                    #################
################################################################################

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import logging
from pathlib import Path
from difflib import get_close_matches
from typing import Optional
import json
import textwrap

from aaindex import aaindex1
from .model import Model
from .pyDSP import PyDSP
from .evaluate import Evaluate
from .utils import Map, valid_sequence, remove_gaps, zero_padding, save_results
from .plots import plot_reg
from .descriptors import Descriptors
from .globals_ import get_current_datetime as _get_current_datetime

class PySAR():
    """
    The PySAR class is the main class for the pySAR software. The class allows for
    the encoding of protein sequences via a plethora of techniques, mainly via AAI 
    Indices and or structural, biochemical and physicochemical protein descriptors that are 
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
    activity values for new unseen protein sequences. After the encoding process, 
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
    def __init__(self, config_file="", logger: Optional[logging.Logger] = None, **kwargs):

        self.kwargs = kwargs  # keyword arguments override config parameters
        self.config_parameters = {}
        self.logger = logger
        self.aai_indices = None
        self.descriptors = None
        self.feature_space = ()

        self._load_config(config_file)
        self._extract_config_params()
        self._load_data()
        self._init_descriptors()

    def _load_config(self, config_file):
        """
        Resolve and parse the JSON configuration file.  Sets self.config_file
        and self.config_parameters (as a Map for dot-notation access).
        """
        if not isinstance(config_file, str):
            raise TypeError(
                f'JSON config file must be a filepath of type string, got type {type(config_file)}.'
            )

        #append extension if only filename given without one
        if os.path.splitext(config_file)[1] == '':
            config_file = config_file + '.json'

        self.config_file = config_file

        config_filepath = ""
        _config_path = Path(self.config_file).expanduser()
        if _config_path.is_absolute() and _config_path.is_file():
            config_filepath = str(_config_path)
        elif _config_path.is_file():
            config_filepath = str(_config_path.resolve())
        elif (Path('config') / self.config_file).is_file():
            config_filepath = str((Path('config') / self.config_file).resolve())
        elif (Path(__file__).parent.parent / 'config' / self.config_file).is_file():
            config_filepath = str(
                (Path(__file__).parent.parent / 'config' / self.config_file).resolve()
            )
        else:
            raise OSError(
                f'JSON config file {self.config_file!r} not found. '
                f'Checked: current directory, config/, and package config/.'
            )

        try:
            with open(config_filepath) as f:
                self.config_parameters = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'Error parsing config JSON file: {config_filepath}.'
            ) from exc

        #create instance of Map class so parameters can be accessed via dot notation
        self.config_parameters = Map(self.config_parameters)

    def _extract_config_params(self):
        """
        Extract dataset, model and DSP parameters from self.config_parameters,
        with **kwargs values taking precedence over the config file values.
        """
        kw = self.kwargs
        cfg = self.config_parameters

        #dataset parameters
        self.dataset = kw.get('dataset') if 'dataset' in kw else cfg.dataset["dataset"]
        self.sequence_col = (
            kw.get('sequence_col') if 'sequence_col' in kw else cfg.dataset["sequence_col"]
        )
        self.activity_col = (
            kw.get('activity_col') if 'activity_col' in kw else cfg.dataset["activity"]
        )

        #model parameters
        self.model_parameters = (
            kw.get('model_parameters') if 'model_parameters' in kw else cfg.model["parameters"]
        )
        self.algorithm = kw.get('algorithm') if 'algorithm' in kw else cfg.model["algorithm"]
        self.test_split = (
            kw.get('test_split') if 'test_split' in kw else cfg.model["test_split"]
        )

        #pyDSP parameters - use_dsp, spectrum, window function, window filter
        self.use_dsp = kw.get('use_dsp') if 'use_dsp' in kw else cfg.pyDSP["use_dsp"]
        self.dsp_parameters = kw.get('dsp_parameters') if 'dsp_parameters' in kw else cfg.pyDSP
        self.filter_parameters = (
            kw.get('filter_parameters') if 'filter_parameters' in kw else self.dsp_parameters["filter"]
        )
        self.spectrum = kw.get('spectrum') if 'spectrum' in kw else cfg.pyDSP["spectrum"]
        self.window_type = (
            kw.get('window_type') if 'window_type' in kw else cfg.pyDSP["window"]["type"]
        )
        self.filter_type = (
            kw.get('filter_type') if 'filter_type' in kw else cfg.pyDSP["filter"]["type"]
        )

        #automatically enable DSP if any DSP kwargs are provided
        if any(k in kw for k in ('spectrum', 'window_type', 'filter_type')):
            self.use_dsp = True

    def _load_data(self):
        """
        Read the dataset, populate sequence/activity arrays, and run preprocessing.
        Sets self.data, self.sequences, self.activity, self.num_seqs,
        self.sequence_length.
        """
        self.data = self.read_data()
        # Use a placeholder if the column doesn't exist yet; preprocessing() will
        # fuzzy-match the column name and reload sequences/activity from the data.
        self.sequences = (
            self.data[self.sequence_col]
            if self.sequence_col in self.data.columns
            else pd.Series(dtype=str)
        )
        self.activity = (
            self.data[self.activity_col]
            if self.activity_col in self.data.columns
            else pd.Series(dtype=float)
        )
        self.preprocessing()
        self.num_seqs = len(self.sequences)
        self.sequence_length = len(max(self.sequences, key=len))

    def _init_descriptors(self):
        """Create the shared Descriptors instance used across all encode methods."""
        self.descriptor = Descriptors(
            self.config_file, protein_seqs=self.sequences, **self.kwargs
        )

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
            raise OSError(f'Dataset filepath is not correct: {self.dataset}.')

        #read in dataset csv
        try:
            data = pd.read_csv(self.dataset, sep=",", header=0)
            return data
        except Exception as e:
            raise OSError(f'Error opening dataset file: {self.dataset}.') from e

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
        #require exact match for sequence column; fall back to fuzzy match with a warning
        if self.sequence_col in self.data.columns:
            pass  # exact match found; nothing to do
        else:
            sequence_col_matches = get_close_matches(self.sequence_col, self.data.columns, cutoff=0.8)
            if sequence_col_matches:
                matched = sequence_col_matches[0]
                warnings.warn(
                    f"Sequence column '{self.sequence_col}' not found exactly; "
                    f"using closest match '{matched}'.",
                    UserWarning, stacklevel=3
                )
                self.sequence_col = matched
            else:
                raise ValueError(f'Sequence column ({self.sequence_col}) not present in dataset columns:\n{self.data.columns}.')

        # Always reload sequences from data with the (possibly fuzzy-matched) column name
        # so that a placeholder set in _load_data is replaced with the real data.
        self.sequences = self.data[self.sequence_col]
        #remove any gaps found in sequences in dataset
        self.sequences = remove_gaps(self.sequences)

        #verify no invalid amino acids found in sequences, if so then raise error
        invalid_seqs = valid_sequence(self.sequences)
        if invalid_seqs is not None:
            raise ValueError(f'Invalid amino acids found in protein sequence dataset: {invalid_seqs}.')

        #require exact match for activity column; fall back to fuzzy match with a warning
        if self.activity_col in self.data.columns:
            pass  # exact match found; nothing to do
        else:
            activity_matches = get_close_matches(self.activity_col, self.data.columns, cutoff=0.8)
            if activity_matches:
                matched = activity_matches[0]
                warnings.warn(
                    f"Activity column '{self.activity_col}' not found exactly; "
                    f"using closest match '{matched}'.",
                    UserWarning, stacklevel=3
                )
                self.activity_col = matched
            else:
                raise ValueError(f'Activity column ({self.activity_col}) not present in dataset columns:\n{list(self.data.columns)}.')

        #coerce to numeric so mixed-type columns (strings mixed with numbers) become NaN rather than
        #silently passing through and causing a cryptic sklearn error at fit time
        self.data[self.activity_col] = pd.to_numeric(self.data[self.activity_col], errors='coerce')
        #remove any +/- infinity values or any Null/NAN's from activity values
        nan_count = self.data[self.activity_col].replace([np.inf, -np.inf], np.nan).isna().sum()
        if nan_count > 0:
            warnings.warn(
                f'{nan_count} missing/infinite activity value(s) in column "{self.activity_col}" '
                f'replaced with 0. Consider reviewing or dropping these rows.',
                UserWarning, stacklevel=2
            )
        self.data[self.activity_col] = (
            self.data[self.activity_col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        #refresh self.activity to reflect the updated (NaN-replaced) column
        self.activity = self.data[self.activity_col]

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
        if aai_indices is None or aai_indices == "":
            raise ValueError(f'AAI indices input parameter cannot be None or empty: {aai_indices}.')

        #check input indices is of correct type (str/list), if not raise type error
        if (not isinstance(aai_indices, str) and (not isinstance(aai_indices, list))):
            raise TypeError(f"Input indices parameter must be a string or list, got {type(aai_indices)}.")

        #cast index string to list, split multiple indices using comma
        if isinstance(aai_indices, str):
            if ',' in aai_indices:
                aai_indices = aai_indices.split(',')  #split on ',' just in case multiple indices passed in as str
            else:
                aai_indices = [aai_indices]

        #accumulate each index's encoding into a list, then concatenate once
        encoded_parts = []

        #if multiple indices used then calculate AAI index encoding for each and concatenate after each calculation
        for index in range(0, len(aai_indices)):

            #get values from aaindex record using its accession number and the aaindex package
            encoded_aai = aaindex1[aai_indices[index]].values

            #build a lookup array indexed by ASCII ordinal for fast vectorized encoding
            lookup = np.zeros(128, dtype="float32")
            for aa_char, aa_val in encoded_aai.items():
                ordinal = ord(aa_char)
                if ordinal < 128:
                    lookup[ordinal] = float(aa_val) if aa_val is not None else 0.0

            #vectorize: convert each sequence string to an array of ASCII codes, then index into lookup
            encoded_rows = []
            for seq in self.sequences:
                char_codes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8).astype(np.int32)
                encoded_rows.append(lookup[char_codes])

            #zero-pad encoding list so that sequences are all the same length
            temp_all_seqs = zero_padding(encoded_rows)

            #convert list of lists into array
            temp_all_seqs = np.array(temp_all_seqs, dtype="float32")

            encoded_parts.append(temp_all_seqs)

        #concatenate all per-index encodings along the feature axis
        if len(encoded_parts) == 1:
            encoded_aai_ = encoded_parts[0]
        else:
            encoded_aai_ = np.concatenate(encoded_parts, axis=1)

        return encoded_aai_

    def _log(self, message: str, level: int = logging.INFO) -> None:
        """Log to the configured logger or fall back to print."""
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message)

    def _fit_and_evaluate(self, X, random_state=None, cv=None):
        """
        Shared pipeline used by encode_aai, encode_descriptor and encode_aai_descriptor.
        Builds a model from feature matrix *X*, splits, fits, predicts and evaluates.

        Sets self.feature_space and self.model as side effects.

        Parameters
        ==========
        :X: pd.DataFrame
            Feature matrix to train on.
        :random_state: int or None (default=None)
            Seed for the train/test split for reproducibility.
        :cv: int or None (default=None)
            When set, runs k-fold cross-validation and logs the mean CV R2.

        Returns
        =======
        :(evaluation, Y_test, Y_pred): tuple
            evaluation – Evaluate instance with all metrics.
            Y_test     – held-out labels (np.ndarray).
            Y_pred     – model predictions (np.ndarray).
        """
        self.feature_space = X.shape
        self.model = Model(X, self.activity, self.algorithm, parameters=self.model_parameters)
        self.model.train_test_split(test_split=self.test_split, random_state=random_state)
        self.model.fit()
        Y_pred = self.model.predict()
        Y_test = self.model.Y_test
        if cv is not None:
            cv_scores = self.model.cv_score(cv=cv)
            self._log(f'# CV R2 (k={cv}): mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}')
        return Evaluate(Y_test, Y_pred), Y_test, Y_pred

    def encode_aai(self, aai_indices=None, show_plot=False, print_results=True, output_folder="",
                   random_state=None, cv=None):
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
        if aai_indices is None or aai_indices == "" or aai_indices == []:
            raise ValueError(f'AAI indices input parameter cannot be None or empty: {aai_indices}.')

        #check input indices is of correct type (str/list), if not raise type error
        if ((not isinstance(aai_indices, str)) and (not isinstance(aai_indices, list))):
            raise TypeError(f"Input indices parameter must be a string or list, got {type(aai_indices)}.")

        self.aai_indices = aai_indices

        #if list of one element with multiple indices, split them into list of individual elements
        if isinstance(self.aai_indices, list) and len(self.aai_indices) == 1:
            self.aai_indices = self.aai_indices[0].replace(' ', '').split(',')

        #convert string indices into comma seperated list, remove whitespace
        if isinstance(self.aai_indices, str):
            self.aai_indices = self.aai_indices.replace(' ', '').split(',')

        #sort list of indices into alphabetical order
        self.aai_indices.sort()

        #record encoding strategy for use by predict_activity() and save_session()
        self._encoding_type = 'aai'
        self._encoding_aai_indices = list(self.aai_indices)
        self._encoding_descriptors = None

        #accumulate per-index DataFrames into a list, then concatenate once to avoid O(n²) copies
        aai_encoding_frames = []

        #iterate over each index, calculate its encoding, apply DSP functionality if applicable
        for index in self.aai_indices:

            #get AAI index encodings specified by indices input parameter
            encoded_seqs = self.get_aai_encoding(index)

            #if use_dsp true then get protein spectra from encoded sequences via the AAI indices using PyDSP class,
            #else use the AAI indices encoding's themselves as the feature/training data (X)
            if self.use_dsp:
                #if input spectrum is none or empty, raise error.
                if self.spectrum is None or self.spectrum == "":
                    raise ValueError(f'Spectrum cannot be None or empty: {self.spectrum}.')
                pyDSP = PyDSP(self.config_file, protein_seqs=encoded_seqs,
                              spectrum=self.spectrum, window_type=self.window_type,
                              filter_type=self.filter_type)
                X = pd.DataFrame(pyDSP.spectrum_encoding) #set training data to FFT spectrum encoding
            else:
                X = pd.DataFrame(encoded_seqs)  #no DSP applied to encoded sequences

            aai_encoding_frames.append(X)

        #single pd.concat eliminates O(n²) intermediate copies from the previous loop pattern
        aai_encoding_df = pd.concat(aai_encoding_frames, axis=1)

        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(aai_encoding_df.columns) + 1)]
        aai_encoding_df.columns = columns

        #build model, fit, predict and evaluate using shared helper
        evaluation, Y_test, Y_pred = self._fit_and_evaluate(aai_encoding_df, random_state=random_state, cv=cv)

        #get categories for all indices in self.aai_indices
        index_cat = []
        if isinstance(self.aai_indices, list):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]

        #create comma seperated list of categories
        index_cat = ', '.join(index_cat)

        #create output dataframe, set first row to attribute/metric values
        aai_df = pd.DataFrame(columns=['Index', 'Category', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])
        aai_df.loc[0] = [', '.join(self.aai_indices), str(index_cat).strip(), evaluation.r2, evaluation.rmse, evaluation.mse, evaluation.mae, evaluation.rpd, evaluation.explained_var]

        #convert index and category from default Object type -> String datatypes
        aai_df['Index'] = aai_df['Index'].astype(pd.StringDtype())
        aai_df['Category'] = aai_df['Category'].astype(pd.StringDtype())

        #print out results from encoding
        if (print_results):
            self.output_results(aai_df)

        #plot regression plot for predictive model
        _ts = _get_current_datetime()
        plot_reg(Y_test, Y_pred, evaluation.r2, output_folder, show_plot, timestamp=_ts)

        #save results of encoding to output folder specified by input param
        save_results(aai_df, 'aai_results', output_folder=output_folder, timestamp=_ts)

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
        if descriptors is None or descriptors == "" or descriptors == []:
            raise ValueError(f'Descriptors input parameter cannot be None or empty: {descriptors}.')
        
        #check input descriptor is of correct type str or list, if not raise type error
        if (not isinstance(descriptors, str) and (not isinstance(descriptors, list))):
            raise TypeError(f"Input descriptor parameter must be a str or list, got {type(descriptors)}.")

        #cast descriptors parameter to a list if it is a str by creating comma seperated list
        if (isinstance(descriptors, str)):
            descriptors = descriptors.split(',')

        #remove any leading or trailing whitespace from descriptors
        descriptors = [de.strip() for de in descriptors]

        #reuse cached Descriptors instance created in __init__
        descr = self.descriptor
        
        #store list of correct descriptor names from ones user input using the difflib library
        temp_descriptors = []

        #get closest valid available descriptor name from input descriptor parameter,
        #if a list of descriptors passed in as the input parameter then get
        #all valid descriptors in list
        for de in range(0, len(descriptors)):
            desc_matches = get_close_matches(descriptors[de],
                descr.valid_descriptors, cutoff=0.6)
            if (desc_matches == []):
                raise ValueError(f'No approximate descriptor found from one input: {de}.')
            descriptors[de] = desc_matches[0]
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
                raise ValueError(f'Descriptor cannot be empty or None: {descriptors[d]}.')
            encoded_desc_vals.append(encoded_desc_temp) #append to array of all descriptor values
            encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe for next iteration

        #concatenate dataframes of descriptors
        encoded_desc = pd.concat(encoded_desc_vals, axis=1)

        return encoded_desc

    def encode_descriptor(self, descriptors=None, show_plot=False, print_results=True, output_folder="",
                          random_state=None, cv=None):
        """
        
        Full pipeline for encoding the protein sequences in the dataset using protein 
        physicochemical, biochemical and or structural descriptors, using the Descriptors 
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
        :random_state: int or None (default=None)
            seed for the train/test split, enabling reproducible results.
        :cv: int or None (default=None)
            if set, performs k-fold cross-validation and logs CV R² mean and std.

        Returns
        =======
        :desc_df: pd.DataFrame
            pandas dataframe storing metrics and results of encoding.
        """
        #raise error if no descriptor specified in input
        if descriptors is None or descriptors == "" or descriptors == []:
            raise ValueError(f'Descriptors input parameter cannot be None or empty: {descriptors}.')

        #check input descriptor is of correct type (str or list), if not raise type error
        if not isinstance(descriptors, str) and not isinstance(descriptors, list):
            raise TypeError(f"Input descriptor parameter must be a string or list, got {type(descriptors)}.")

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

        #record encoding strategy for use by predict_activity()
        self._encoding_type = 'descriptor'
        self._encoding_aai_indices = None
        self._encoding_descriptors = list(self.descriptors)

        #reuse cached Descriptors instance created in __init__
        descr = self.descriptor

        #pandas dataframe to store all output results
        desc_df = pd.DataFrame(columns=['Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'MAE', 'RPD', 'Explained Variance'])

        #accumulate descriptor DataFrames, then concat once
        descriptor_frames = []

        #iterate over each input descriptor, calculate its encoding from its respective function
        for desc in range(0, len(self.descriptors)):

            #get closest matching descriptor from descriptor input parameter using difflib library
            desc_matches = get_close_matches(self.descriptors[desc], descr.valid_descriptors, cutoff=0.6)
            if desc_matches != []:
                self.descriptors[desc] = desc_matches[0]
            else:
                raise ValueError('Could not find a match for the input descriptor ({}) in list of valid descriptors:\n{}.'.
                    format(self.descriptors[desc], descr.valid_descriptors))

            descriptor_frames.append(self.get_descriptor_encoding(descriptors=self.descriptors[desc]))

        descriptor_encoding_df = pd.concat(descriptor_frames, axis=1)

        #build model, fit, predict and evaluate using shared helper
        evaluation, Y_test, Y_pred = self._fit_and_evaluate(descriptor_encoding_df, random_state=random_state, cv=cv)

        #get groups for all descriptors in self.desciptors, put multiple descriptor groups into comma seperated list
        if isinstance(self.descriptors, list):
            desc_group = []
            for desc_ in self.descriptors:
                desc_group.append(descr.descriptor_groups[desc_])
            desc_group = ', '.join(desc_group)
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #add metric values to output dataframe
        desc_df.loc[0] = [', '.join(self.descriptors), desc_group, evaluation.r2, evaluation.rmse, evaluation.mse, evaluation.mae, evaluation.rpd, evaluation.explained_var]

        #convert Descriptor and Group from default Object type -> String datatypes
        desc_df['Descriptor'] = desc_df['Descriptor'].astype(pd.StringDtype())
        desc_df['Group'] = desc_df['Group'].astype(pd.StringDtype())

        #ensure aai indices attribute doesn't show up in output results
        if self.aai_indices is not None:
            self.aai_indices = None

        #print out results from encoding
        if (print_results):
            self.output_results(desc_df)

        #plot regression plot for predictive model
        _ts = _get_current_datetime()
        plot_reg(Y_test, Y_pred, evaluation.r2, output_folder, show_plot, timestamp=_ts)

        #save results of encoding to output folder
        save_results(desc_df, 'desc_results', output_folder=output_folder, timestamp=_ts)

        return desc_df

    def encode_aai_descriptor(self, aai_indices=None, descriptors=None, show_plot=False, print_results=True, output_folder="",
                              random_state=None, cv=None):
        """
        Encode using both AAI indices and the physicochemical/structural descriptors from
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
        :random_state: int or None (default=None)
            seed for the train/test split, enabling reproducible results.
        :cv: int or None (default=None)
            if set, performs k-fold cross-validation and logs CV R² mean and std.

        Returns
        =======
        :aai_desc_df : pd.Dataframe
            pandas dataframe storing metrics and results of encoding.
        """
        #validate AAI indices and Descriptors are present in the input parameters, return error if either is None
        if (descriptors is None or descriptors in ("", [])) or (aai_indices is None or aai_indices in ("", [])):
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

        #record encoding strategy for use by predict_activity()
        self._encoding_type = 'aai_descriptor'
        self._encoding_aai_indices = list(self.aai_indices)
        self._encoding_descriptors = list(self.descriptors)

        #create output results Dataframe
        aai_desc_df = pd.DataFrame(columns=['Index', 'Category', 'Descriptor', 'Group', 'R2', 'RMSE', \
            'MSE', 'MAE', 'RPD', 'Explained Variance'])

        #accumulate AAI index DataFrames then concat once (avoids O(n²) copies)
        aai_encoding_frames = []
        for index in self.aai_indices:
            aai_encoding_frames.append(pd.DataFrame(self.get_aai_encoding(index)))
        aai_encoding_df = pd.concat(aai_encoding_frames, axis=1)

        #renaming columns in format aai_X, where X is the amino acid number in the sequence
        columns = ["aai_" + str(x) for x in range(1, len(aai_encoding_df.columns) + 1)]
        aai_encoding_df.columns = columns

        #if AAI indices encoding is empty, raise error
        if aai_encoding_df.empty:
            raise ValueError(f'AAI Indices encoding cannot be empty or None: {aai_indices}.')

        #reuse cached Descriptors instance from __init__
        descr = self.descriptor

        #accumulate descriptor DataFrames then concat once
        descriptor_frames = []
        for desc in range(0, len(self.descriptors)):

            #get closest matching descriptor from descriptor input parameter using difflib library
            desc_matches = get_close_matches(self.descriptors[desc], descr.valid_descriptors, cutoff=0.6)
            if desc_matches != []:
                self.descriptors[desc] = desc_matches[0]
            else:
                raise ValueError('Could not find a match for the input descriptor ({}) in list of valid descriptors:\n{}.'.
                    format(self.descriptors[desc], descr.valid_descriptors))

            descriptor_frames.append(self.get_descriptor_encoding(descriptors=self.descriptors[desc]))

        descriptor_encoding_df = pd.concat(descriptor_frames, axis=1)

        #reset index for aai indices and in descriptors output dataframe
        aai_encoding_df.reset_index(inplace=True, drop=True)
        descriptor_encoding_df.reset_index(inplace=True, drop=True)

        #concatenate AAI index and Descriptor features to get training data (X)
        X = pd.concat([aai_encoding_df, descriptor_encoding_df], axis=1)

        #build model, fit, predict and evaluate using shared helper
        evaluation, Y_test, Y_pred = self._fit_and_evaluate(X, random_state=random_state, cv=cv)

        #get categories for all indices in self.aai_indices
        index_cat = []
        if isinstance(self.aai_indices, list):
            for i in range(0, len(self.aai_indices)):
                index_cat.append(aaindex1[self.aai_indices[i]].category)
        else:
            index_cat = [aaindex1[self.aai_indices].category]

        #seperate index categories into comma seperated string
        index_cat = ', '.join(index_cat)

        #get groups for all descriptors in self.desciptors, can be string or list of descriptors
        if isinstance(self.descriptors, list):
            desc_group = []
            for desc_ in self.descriptors:
                desc_group.append(descr.descriptor_groups[desc_])
            desc_group = ', '.join(desc_group)
        else:
            desc_group = descr.descriptor_groups[self.descriptors]

        #set output dataframe columns
        aai_desc_df.loc[0] = [', '.join(self.aai_indices), str(index_cat).strip(), ', '.join(self.descriptors), str(desc_group), evaluation.r2, 
            evaluation.rmse, evaluation.mse, evaluation.mae, evaluation.rpd, evaluation.explained_var]

        #convert Index, Category, Descriptor and Group from default Object type -> String datatypes
        aai_desc_df['Index'] = aai_desc_df['Index'].astype(pd.StringDtype())
        aai_desc_df['Category'] = aai_desc_df['Category'].astype(pd.StringDtype())
        aai_desc_df['Descriptor'] = aai_desc_df['Descriptor'].astype(pd.StringDtype())
        aai_desc_df['Group'] = aai_desc_df['Group'].astype(pd.StringDtype())

        #print out results from encoding
        if (print_results):
            self.output_results(aai_desc_df)

        #plot regression plot for predictive model
        _ts = _get_current_datetime()
        plot_reg(Y_test, Y_pred, evaluation.r2, output_folder, show_plot, timestamp=_ts)

        #save results of encoding to output folder
        save_results(aai_desc_df, 'aai_desc_results', output_folder=output_folder, timestamp=_ts)

        return aai_desc_df

    def predict_activity(self, sequences, return_uncertainty=False):
        """
        Predict the activity/fitness value(s) for one or more unseen protein sequences.

        The method re-encodes *sequences* using the same strategy that was applied
        during the most recent ``encode_aai()``, ``encode_descriptor()``, or
        ``encode_aai_descriptor()`` call, then applies the fitted scaler and
        regression model.

        Parameters
        ==========
        :sequences: str, list, or pd.Series
            One or more protein sequences (amino-acid strings) to predict.
        :return_uncertainty: bool (default=False)
            If True and the underlying model is a GaussianProcessRegressor, return
            a tuple ``(predictions, std)`` where ``std`` is the per-sequence standard
            deviation of the predictive distribution.  For all other model types this
            flag is ignored and only ``predictions`` is returned.

        Returns
        =======
        :predictions: np.ndarray
            Predicted activity values, one per input sequence.
        :(predictions, std): tuple of np.ndarray
            Only returned when ``return_uncertainty=True`` and the model is a
            GaussianProcessRegressor.

        Raises
        ======
        :RuntimeError
            If no encoding has been run yet (i.e. none of the ``encode_*`` methods
            have been called on this instance).
        """
        if not hasattr(self, '_encoding_type') or self._encoding_type is None:
            raise RuntimeError(
                'No encoding has been run yet. Call encode_aai(), encode_descriptor(), '
                'or encode_aai_descriptor() before predict_activity().'
            )
        if self.model is None or not self.model.model_fitted():
            raise RuntimeError('The model has not been fitted. Call an encode_* method first.')

        #normalise input to list of strings
        if isinstance(sequences, str):
            sequences = [sequences]
        elif isinstance(sequences, pd.Series):
            sequences = sequences.tolist()
        else:
            sequences = list(sequences)

        #remove gaps and validate input sequences
        sequences = remove_gaps(sequences)
        invalid = valid_sequence(sequences)
        if invalid is not None:
            raise ValueError(f'Invalid amino acids found in input sequences: {invalid}.')

        #build feature matrix using the stored encoding strategy
        encoding_type = self._encoding_type

        if encoding_type in ('aai', 'aai_descriptor'):
            #encode sequences using each stored AAI index and concatenate
            _saved_sequences = self.sequences
            _saved_num_seqs = self.num_seqs
            _saved_seq_len = self.sequence_length

            self.sequences = sequences
            self.num_seqs = len(sequences)
            self.sequence_length = len(max(sequences, key=len))

            aai_encoding_parts = []
            for index in self._encoding_aai_indices:
                encoded = self.get_aai_encoding(index)
                if self.use_dsp:
                    if self.spectrum is None or self.spectrum == "":
                        raise ValueError(f'Spectrum cannot be None or empty: {self.spectrum}.')
                    pyDSP = PyDSP(self.config_file, protein_seqs=encoded,
                                  spectrum=self.spectrum, window_type=self.window_type,
                                  filter_type=self.filter_type)
                    aai_encoding_parts.append(pd.DataFrame(pyDSP.spectrum_encoding))
                else:
                    aai_encoding_parts.append(pd.DataFrame(encoded))

            #restore original sequences state before concat so state is clean on any error
            self.sequences = _saved_sequences
            self.num_seqs = _saved_num_seqs
            self.sequence_length = _saved_seq_len

            aai_encoding_df = pd.concat(aai_encoding_parts, axis=1)
            aai_encoding_df.columns = ["aai_" + str(x) for x in range(1, len(aai_encoding_df.columns) + 1)]

        if encoding_type in ('descriptor', 'aai_descriptor'):
            #re-calculate descriptors for the new sequences using a temporary Descriptors instance.
            #A new instance is required (not self.descriptor) because get_descriptor_encoding
            #returns cached attributes computed from the training sequences.
            #Pass descriptors_csv="" to prevent loading a pre-computed descriptor cache that
            #belongs to the training data (different number of sequences).
            from .descriptors import Descriptors as _Descriptors
            tmp_descriptor = _Descriptors(self.config_file, protein_seqs=pd.Series(sequences), descriptors_csv="")
            desc_parts = []
            for desc_name in self._encoding_descriptors:
                desc_parts.append(tmp_descriptor.get_descriptor_encoding(desc_name))
            desc_df = pd.concat(desc_parts, axis=1)

        if encoding_type == 'aai':
            X = aai_encoding_df
        elif encoding_type == 'descriptor':
            X = desc_df
        else:  # aai_descriptor
            aai_encoding_df.reset_index(drop=True, inplace=True)
            desc_df.reset_index(drop=True, inplace=True)
            X = pd.concat([aai_encoding_df, desc_df], axis=1)

        X_values = X.to_numpy(dtype=float)

        # Align the feature count to match the training feature space (self.feature_space[1]).
        # Zero-padding of sequences during encoding pads to the max length of the current
        # batch, which may differ from the training batch's max length.
        if hasattr(self, 'feature_space') and self.feature_space is not None:
            n_train_features = self.feature_space[1]
            if X_values.shape[1] < n_train_features:
                X_values = np.pad(X_values, ((0, 0), (0, n_train_features - X_values.shape[1])))
            elif X_values.shape[1] > n_train_features:
                X_values = X_values[:, :n_train_features]

        #apply the stored scaler if one was fitted
        scaler = getattr(self.model, 'scaler', None)
        if scaler is not None:
            X_values = scaler.transform(X_values)

        #return uncertainty for GaussianProcessRegressor when requested
        if return_uncertainty:
            from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
            if isinstance(self.model.model_fit, _GPR):
                predictions, std = self.model.model_fit.predict(X_values, return_std=True)
                return predictions, std

        return self.model.model_fit.predict(X_values)

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

        self._log('##########################################################################################')
        self._log('###################################### Parameters ########################################\n')
        if (self.aai_indices is not None) and (len(self.aai_indices) <= 10):
            self._log(textwrap.fill('# AAI Indices: {}'.format(', '.join(self.aai_indices)), line_length))
            if (self.use_dsp):
                self._log('# DSP Parameters:\n  # Spectrum: {}\n  # Window Function: {} \
                \n  # Filter Function: {}'.format(self.spectrum, self.window_type, self.filter_type))
        if (self.descriptors is not None):
            self._log(textwrap.fill('# Descriptors: {}'.format(', '.join(self.descriptors)), line_length))
        self._log('# Configuration File: {}\n# Dataset: {}\n# Number of Sequences/Sequence Length: {} x {} \
            \n# Target Activity: {}'.format(os.path.basename(self.config_file), self.dataset, self.num_seqs, self.sequence_length, self.activity_col))
        self._log(f"# Algorithm: {repr(self.model)}")
        if (self.model_parameters == "" or self.model_parameters is None or self.model_parameters == {}):
            self._log(textwrap.fill(f'# Model Parameters: {self.model.model.get_params()}', line_length))
        else:
            self._log(textwrap.fill(f'# Model Parameters: {self.model_parameters}', line_length))
        self._log(f'# Test Split: {self.test_split}\n# Feature Space: {self.feature_space}')

        self._log('\n##########################################################################################')
        self._log('######################################## Results #########################################\n')
        self._log('# R2: {}'.format(results['R2'].values[0]))
        self._log('# RMSE: {} '.format(results['RMSE'].values[0]))
        self._log('# MSE: {} '.format(results['MSE'].values[0]))
        self._log('# MAE: {}'.format(results['MAE'].values[0]))
        self._log('# RPD {}'.format(results['RPD'].values[0]))
        self._log('# Explained Variance {}\n'.format(results['Explained Variance'].values[0]))
        self._log('##########################################################################################\n')

    def save_session(self, path: str) -> None:
        """
        Persist the fitted PySAR instance (model, scaler, encoding strategy, and all
        configuration attributes) to a pickle file so that it can be restored later
        with :meth:`load_session`.

        Parameters
        ==========
        :path: str
            Destination file path.  A ``.pkl`` extension is appended automatically
            if the path does not already have one.

        Returns
        =======
        None
        """
        if os.path.splitext(path)[1] == "":
            path = path + ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self._log(f"Session saved to {path!r}")

    @classmethod
    def load_session(cls, path: str, allow_pickle: bool = True) -> 'PySAR':
        """
        Restore a :class:`PySAR` instance that was previously saved with
        :meth:`save_session`.

        Parameters
        ==========
        :path: str
            Path to the ``.pkl`` file written by :meth:`save_session`.
        :allow_pickle: bool (default=True)
            Set to False to disable loading and raise a ValueError.  Provided as
            an opt-in safety gate; never load sessions from untrusted sources.

        Returns
        =======
        :instance: PySAR
            The restored PySAR instance.

        Raises
        ======
        :ValueError
            If *allow_pickle* is False.
        :FileNotFoundError
            If *path* does not exist.
        """
        if not allow_pickle:
            raise ValueError(
                "allow_pickle=False: loading pickle files is disabled. "
                "Only load sessions from trusted sources."
            )
        import warnings as _warnings
        _warnings.warn(
            "PySAR.load_session() deserializes a pickle file. "
            "Never load session files from untrusted sources.",
            UserWarning, stacklevel=2,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Session file not found: {path!r}")
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Loaded object is not a PySAR instance (got {type(instance).__name__!r})."
            )
        return instance

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

    def __str__(self):
        return (
            f"Instance of PySAR class with attributes: Config: {os.path.basename(self.config_file)}, "
            f"Dataset: {os.path.basename(self.dataset)}, Sequences: {self.num_seqs} x {self.sequence_length}, "
            f"Activity: {self.activity_col}, Algorithm: {self.algorithm}, Test Split: {self.test_split}."
        )

    def __repr__(self):
        return f"<PySAR: config={os.path.basename(self.config_file)!r}, algorithm={self.algorithm!r}>"