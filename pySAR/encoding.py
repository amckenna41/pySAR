################################################################################
#################                    Encoding                  #################
################################################################################

import pandas as pd
import os
import time
import itertools
import logging
import threading
import warnings
from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from tqdm import tqdm
import textwrap

from aaindex import aaindex1
from .model import Model
from .pyDSP import PyDSP
from .evaluate import Evaluate
from .pySAR import PySAR
from .utils import save_results
from .descriptors import Descriptors


class MetricKey(str, Enum):
    """ Enum for consistent metric and column naming across encoding results. """
    INDEX = 'Index'
    CATEGORY = 'Category'
    DESCRIPTOR = 'Descriptor'
    GROUP = 'Group'
    R2 = 'R2'
    RMSE = 'RMSE'
    MSE = 'MSE'
    MAE = 'MAE'
    RPD = 'RPD'
    EXPLAINED_VARIANCE = 'Explained Variance'


class SortKey(str, Enum):
    """ Enum for valid sort_by options when sorting encoding results. """
    R2 = MetricKey.R2.value
    RMSE = MetricKey.RMSE.value
    MSE = MetricKey.MSE.value
    MAE = MetricKey.MAE.value
    RPD = MetricKey.RPD.value
    EXPLAINED_VARIANCE = MetricKey.EXPLAINED_VARIANCE.value


@dataclass
class EncodingResult:
    """
    Structured container for the output of an encoding run.

    Instances are returned by the convenience method
    :meth:`Encoding.to_encoding_result` after any of the three encoding
    methods has been called.  The raw :class:`pandas.DataFrame` returned
    directly by those methods is also available as :attr:`metrics`.

    Attributes
    ==========
    :metrics: pd.DataFrame
        Full results dataframe sorted by *sort_by* (default R2 descending).
    :best_index: str
        The identifier of the best-performing encoding (AAI index name,
        descriptor name, or ``'index+descriptor'`` pair).
    :best_r2: float
        R2 score of the best-performing encoding.
    :best_model_path: str or None
        Absolute path to the saved best model pickle, or ``None`` if
        *export_best_model* was not requested.
    :elapsed_time: float
        Wall-clock seconds consumed by the encoding run.
    """
    metrics: pd.DataFrame
    best_index: str
    best_r2: float
    best_model_path: Optional[str] = None
    elapsed_time: float = 0.0

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       elapsed_time: float = 0.0,
                       best_model_path: Optional[str] = None) -> 'EncodingResult':
        """
        Construct an :class:`EncodingResult` from a sorted metrics DataFrame.

        The first column of *df* is used as the *best_index* identifier.

        Parameters
        ==========
        :df: pd.DataFrame
            Sorted encoding results dataframe (first row is the best model).
        :elapsed_time: float
            Elapsed wall-clock seconds for the encoding run.
        :best_model_path: str or None
            Path to the saved best model, or None.

        Returns
        =======
        :EncodingResult
        """
        if df.empty:
            return cls(metrics=df, best_index='', best_r2=float('nan'),
                       best_model_path=best_model_path, elapsed_time=elapsed_time)
        best_row = df.iloc[0]
        best_index = str(best_row.iloc[0])
        best_r2 = float(best_row.get(MetricKey.R2.value, float('nan')))
        return cls(metrics=df, best_index=best_index, best_r2=best_r2,
                   best_model_path=best_model_path, elapsed_time=elapsed_time)

class Encoding(PySAR):
    """
    The use-case of this class is when you have a dataset of protein sequences with
    a sought-after protein activity/fitness value and you want to measure this activity
    value for new and unseen sequences that have not had their activity value
    experimentally measured. Prior to protein sequences being passed into ML models, 
    the amino acids have to be numerically encoded. The encoding class allows for 
    evaluation of a variety of potential techniques at which to numerically encode the 
    protein sequences, allowing for the building of predictive regression ML models 
    that can ultimately predict the activity value of an unseen protein sequence by 
    mapping a relationship between sequence and activity/function. The strategies each 
    generate a huge number of potential models built an a plethora of available features 
    that you can then assess for performance and predictability, selecting the 
    best-performing model out of all those evaluated. This best-performing model should 
    then be used when you want to predict the activity/fitness value for new sequences.

    The encoding class inherits from the main PySAR module and allows for a
    dataset of protein sequences to be encoded through 3 main strategies: AAI Indices,
    protein Descriptors and AAI Indices + protein Descriptors. The encoding class
    and its methods differ from the PySAR class by allowing for the encoding using
    all available features, in comparison to the PySAR class which is mainly used for
    accessing individual or a small subset of features.
    
    To date, there are 566 indices supported in the AAI and pySAR supports 33 different 
    descriptors. The features can be encoded using different combinations, for example, 
    1, 2 or 3 descriptors can be used for the descriptor and AAI + Descriptor encoding 
    strategies. In total, this class supports over 410,000 possible ways at which to 
    numerically encode the protein sequences in the building of a predictive ML model 
    for mapping these sequences to a particular activity/function, known as a 
    Sequence-Activity-Relationship (SAR) or Sequence-Function-Relationship (SFR).

    Parameters
    ==========
    :config_file: (str)
        path to configuration file with all required parameters for the pySAR encoding
        pipeline.
    **kwargs: dict
        keyword arguments and values passed into constructor. The keywords should be 
        the same name and form of those in the configuration file. The keyword values
        input take precedence over those in the config files.

    Methods
    =======
    aai_encoding(aai_indices=None, sort_by='R2', output_folder=""):
        encoding protein sequences using indices from the AAI and aaindex package.
    descriptor_encoding(descriptors=None, desc_combo=1, sort_by='R2', output_folder=""):
        encoding protein sequences using protein descriptors from descriptors module and protpy package.
    aai_descriptor_encoding(aai_indices=None, descriptors=None, desc_combo=1, sort_by='R2', output_folder=""):
        encoding protein sequences using indices from the AAI in concatenation with 
        the protein descriptors from the descriptors module and protpy package.
    """
    def __init__(self,
                 config_file: str = "",
                 verbose: bool = True,
                 logger: Optional[logging.Logger] = None,
                 **kwargs: Any) -> None:

        self.config_file = config_file
        self.verbose = verbose
        self.logger = logger
        self._aai_feature_cache: Dict[str, pd.DataFrame] = {}
        self._descriptor_feature_cache: Dict[str, pd.DataFrame] = {}
        self._cache_lock: threading.Lock = threading.Lock()  # guards both caches and inflight maps
        # Per-key Future maps prevent redundant computation when concurrent threads miss the cache
        # for the same key simultaneously (TOCTOU).  The first thread creates the Future and
        # computes; subsequent threads wait on the existing Future instead of recomputing.
        self._aai_inflight: Dict[str, "Future[pd.DataFrame]"] = {}
        self._desc_inflight: Dict[str, "Future[pd.DataFrame]"] = {}

        #pass config file and kwargs into parent pySAR class
        super().__init__(self.config_file, **kwargs)

    def aai_encoding(self,
                     aai_indices: Optional[Union[str, List[str]]] = None,
                     sort_by: Union[str, 'SortKey'] = 'R2',
                     output_folder: str = "",
                     n_jobs: int = 1,
                     random_state: Optional[int] = None,
                     max_models: Optional[int] = None,
                     sample_mode: bool = False,
                     resume: bool = False,
                     resume_file: str = "",
                     export_best_model: bool = False) -> pd.DataFrame:
        """
        Encoding all protein sequences using each of the available indices in the
        AAI and aaindex package. The protein spectra of the AAI indices can be generated 
        if use_dsp is true when creating the Encoding instance, also utilized for the 
        DSP spectra are the instance attributes: spectrum, window and filter. If not true 
        then the encoded sequences from the AAI will directly be used - default. 
        
        Each encoding will be used as the feature data to build the predictive regression 
        ML models. To date, there are 566 indices in the AAI, therefore 566 total models 
        can be built using this encoding strategy. The metrics evaluated from the model 
        for each AAI encoding combination will be collated into a dataframe, saved and 
        returned, with the results sorted by R2 by default, this can be changed using 
        the sort_by parameter. You can sort the output dataframe via the other metrics, 
        including: RMSE, MSE, MAE, RPD and Explained Variance. 

        Parameters
        ==========
        :aai_indices: str/list (default=None)
            str/list of aai indices to use for encoding the predictive models, by default
            ALL AAI indices will be used if parameter remains as None.
        :sort_by: str (default=R2)
            sort output dataframe by specified column/metric value, results sorted by R2 
            score by default.
        :output_folder: str (default="")
            output folder to store results csv to, if empty then input will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        =======
        :aaindex_metrics_df: pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using indices in the AAI for the AAI encoding strategy. Output will 
            be of the shape X x 8, where X is the number of indices that can be used
            for the encoding and 8 is the results/metric columns. If no indices are 
            passed in then this shape will be 566 x 8.
        """
        all_indices = self.validate_inputs(aai_indices, aaindex1.record_codes(), "AAI")
        all_indices = self._apply_model_limit(all_indices, sample_mode=sample_mode, max_models=max_models)
        metrics_rows, completed_keys = self._load_resume(
            resume_file if resume else None,
            key_columns=[MetricKey.INDEX.value]
        )
                      
        #create text wrapper for amino acid indices and model parameters text
        line_length = 90

        #create temp Model object to access the models' parameter values for use in display text below
        temp_model_parameters = Model(
            X=[],
            Y=self.activity,
            algorithm=self.algorithm,
            parameters=self.model_parameters
        )
        
        self._log('\n##########################################################################################\n')
        self._log(f'# Encoding using {len(all_indices)} AAI combination(s) with the parameters:\n')
        #only output indices if there are 10 or less
        if (len(all_indices) <= 10):
            self._log(textwrap.fill(f"# AAI Indices: {', '.join(all_indices)}", line_length))
        else:
            self._log(f'# AAI Indices: {len(all_indices)}')
        if (self.use_dsp): 
            self._log(
                f'# DSP Parameters:\n#   Spectrum: {self.spectrum}\n#   Window Function: {self.window_type}\n#   Filter Function: {self.filter_type}'
            )
        self._log(
            f'# Configuration File: {os.path.basename(self.config_file)}\n'
            f'# Dataset: {os.path.basename(self.dataset)}\n'
            f'# Number of Sequences/Sequence Length: {self.num_seqs} x {self.sequence_length}\n'
            f'# Target Activity: {self.activity_col}\n'
            f'# Algorithm: {repr(temp_model_parameters)}'
        )
        if not isinstance(self.model_parameters, dict) or not self.model_parameters:
            self._log(textwrap.fill(f'# Model Parameters: {temp_model_parameters.model.get_params()}', line_length))
        else:
            self._log(textwrap.fill(f'# Model Parameters: {self.model_parameters}', line_length))
        self._log(f'# Test Split: {self.test_split}')
        self._log('\n##########################################################################################\n')

        '''
        1.) Get AAI index encoding of protein sequences, if using DSP (use_dsp = True),
        create instance of pyDSP class and generate protein spectra from the AAI
        indices, according to instance parameters: spectrum, window and filter.
        2.) Build model using encoded AAI indices or protein spectra as features.
        3.) Predict and evaluate the model using the test data.
        4.) Append index, its category and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all indices.
        6.) Output results into a final dataframe, save to OUTPUT_DIR and return.
        '''
        #start time counter
        start = time.time() 

        #create list of pending indices to process, if resume is true then this will remove any completed indices from the list, otherwise it will be the same as all_indices. If 5 or less indices to process then disable tqdm progress bar.
        pending_indices = [idx for idx in all_indices if (idx,) not in completed_keys]
        tqdm_disable = len(pending_indices) <= 5

        def _run_index(index: str) -> Dict[str, Any]:
            """ Helper function to run encoding, model building and evaluation for a single AAI index. """
            X = self.build_features(feature_type="aai", index=index)
            eval_metrics = self.run_model(X, self.activity, random_state=random_state)
            return {
                MetricKey.INDEX.value: index,
                MetricKey.CATEGORY.value: aaindex1[index].category,
                MetricKey.R2.value: eval_metrics.r2,
                MetricKey.RMSE.value: eval_metrics.rmse,
                MetricKey.MSE.value: eval_metrics.mse,
                MetricKey.MAE.value: eval_metrics.mae,
                MetricKey.RPD.value: eval_metrics.rpd,
                MetricKey.EXPLAINED_VARIANCE.value: eval_metrics.explained_var,
            }

        #run encoding, model building and evaluation for each index in the AAI, using parallel processing with n_jobs threads, and append results to metrics_rows list. If resume is true then this will only run for indices that have not been completed yet according to the resume file.
        new_rows = self._execute_jobs(
            items=pending_indices,
            task_fn=_run_index,
            n_jobs=n_jobs,
            tqdm_desc="AAI Indices",
            tqdm_unit="indices",
            tqdm_disable=tqdm_disable
        )
        #append new rows to metrics_rows and save checkpoint if resume is true
        metrics_rows.extend(new_rows)
        self._save_resume_checkpoint(metrics_rows, resume_file if resume else None)

        #stop time counter, calculate elapsed time
        end = time.time()      
        elapsed = end - start

        self._log(f'\nElapsed time for AAI Encoding: {elapsed:.2f} seconds.')
        self._log('\n##########################################################################################')

        # format results into dataframe, save to OUTPUT_DIR and return, sorting by sort_by parameter and using the appropriate filename for the AAI encoding results. If resume is true then this will also save the resume checkpoint with the results.
        return self.format_and_save_results(
            metrics_rows=metrics_rows,
            columns=[
                MetricKey.INDEX.value,
                MetricKey.CATEGORY.value,
                MetricKey.R2.value,
                MetricKey.RMSE.value,
                MetricKey.MSE.value,
                MetricKey.MAE.value,
                MetricKey.RPD.value,
                MetricKey.EXPLAINED_VARIANCE.value,

            ],
            sort_by=sort_by,
            save_filename='aaindex_results',
            output_folder=output_folder,
            string_columns=[MetricKey.INDEX.value, MetricKey.CATEGORY.value],
            resume_file=resume_file if resume else None,
            export_best_model=export_best_model,
            best_model_feature_fn=lambda best_row: self.build_features(
                feature_type="aai", index=best_row[MetricKey.INDEX.value]
            ),
            random_state=random_state,
        )

    def descriptor_encoding(self,
                            descriptors: Optional[Union[str, List[str]]] = None,
                            desc_combo: int = 1,
                            sort_by: Union[str, 'SortKey'] = 'R2',
                            output_folder: str = "",
                            n_jobs: int = 1,
                            random_state: Optional[int] = None,
                            max_models: Optional[int] = None,
                            sample_mode: bool = False,
                            resume: bool = False,
                            resume_file: str = "",
                            export_best_model: bool = False) -> pd.DataFrame:
        """
        Encoding all protein sequences using the available physicochemical, biochemical
        and structural descriptors from the custom-built protpy package. The sequences 
        can be encoded using combinations of 1, 2 or 3 of these descriptors, dictated 
        by the desc_combo input parameter: set this to 1, 2 or 3 for what encoding 
        combination to use, default is 1. 
        
        Each descriptor encoding will be used as the feature data to build the predictive 
        regression ML models. These models can then be used to predict the sought-after
        activity/fitness value for unseen test sequences. With 33 descriptors supported 
        by pySAR & protpy this means there can be 33, 528 and 5456 total predictive models 
        built for 1, 2 or 3 descriptors, respectively. These totals may vary depending on 
        the meta-parameters on some of the descriptors e.g the lag or lambda for the 
        autocorrelation and pseudo amino acid descriptors, respectively. The metrics 
        evaluated from the model for each descriptor encoding combination will be collated 
        into a dataframe and saved and returned, with the results sorted by the R2 score 
        by default, this can be changed using the sort_by parameter.

        Parameters
        ==========
        :descriptors: str/list (default=None)
            str/list of descriptors to use for encoding, by default all available descriptors
            in the protpy package will be used for the encoding.
        :desc_combo: int (default=1)
            combination of descriptors to use, default of 1.
        :sort_by: str (default=R2)
            sort output dataframe by specified column/metric value, results sorted by R2 
            score by default.
        :output_folder: str (default="")
            output folder to store results csv to, if parameter not set then output will 
            be stored in the OUTPUT_FOLDER global var.

        Returns
        =======
        :desc_metrics_df_: pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using all or selected input descriptors for the descriptors 
            encoding strategy. Output will be of the shape X x 8, where X is the 
            number of descriptors input and 8 is the results/metric columns. By 
            default the output shape will be 33 x 8, but with a desc_combo of 2 
            and 3, the shape will be 528 x 8 and 5456 x 8, respectively.
        """
        #create instance of descriptors class using config file and any kwargs
        desc = self.descriptor
        self.validate_desc_combo(desc_combo)
        all_descriptor_names = self.validate_inputs(descriptors, desc.valid_descriptors, "Descriptor")

        #validate input descriptors and get list of descriptor names to use for encoding
        if desc_combo == 1:
            all_descriptors: List[Union[str, Tuple[str, ...]]] = all_descriptor_names
        else:
            all_descriptors = list(itertools.combinations(all_descriptor_names, desc_combo))

        # apply model limit if specified in config file, this will limit the number of models built and evaluated by taking a random sample of the total combinations of descriptors
        all_descriptors = self._apply_model_limit(all_descriptors, sample_mode=sample_mode, max_models=max_models)
        metrics_rows, completed_keys = self._load_resume(
            resume_file if resume else None,
            key_columns=[MetricKey.DESCRIPTOR.value]
        )

        # prime descriptor cache once; improves repeated and parallel usage
        for descriptor_name in all_descriptor_names:
            self._get_descriptor_features(descriptor_name, desc)

        #create text wrapper for descriptors and model parameters text
        line_length = 90

        #create temp Model object to access the models' parameter values for use in display text below
        temp_model_parameters = Model(
            X=[],
            Y=self.activity,
            algorithm=self.algorithm,
            parameters=self.model_parameters
        )

        self._log('\n##########################################################################################\n')
        descriptor_display = [
            '+'.join(descriptor_set) if isinstance(descriptor_set, tuple) else descriptor_set
            for descriptor_set in all_descriptors
        ]
        self._log(f'# Encoding using {len(all_descriptors)} descriptor combination(s) with the parameters:\n')
        self._log(textwrap.fill(f"# Descriptors: {', '.join(descriptor_display)}", line_length))
        self._log(
            f'# Configuration File: {os.path.basename(self.config_file)}\n'
            f'# Dataset: {os.path.basename(self.dataset)}\n'
            f'# Number of Sequences/Sequence Length: {len(self.data)} x {self.data[self.sequence_col].str.len().max()}\n'
            f'# Target Activity: {self.activity_col}\n'
            f'# Algorithm: {repr(temp_model_parameters)}'
        )
        if not isinstance(self.model_parameters, dict) or not self.model_parameters:
            self._log(f'# Model Parameters: {temp_model_parameters.model.get_params()}')
        else:
            self._log(f'# Model Parameters: {self.model_parameters}')
        self._log(f'# Test Split: {self.test_split}')
        self._log('\n##########################################################################################')

        #start counter
        start = time.time()     

        '''
        1.) Get current descriptor value or combination of descriptors from all_descriptors list for 
            dataset of protein sequences.
        2.) Build model using calculated descriptor features from current descriptor(s).
        3.) Predict and evaluate the model using the test data protein sequences.
        4.) Append descriptor(s) and calculated metrics to lists.
        5.) Repeat steps 1 - 4 for all descriptors.
        6.) Output results into a final dataframe, save it and return, sorting by sort_by parameter.
        '''
        #create list of pending descriptors to process, if resume is true then this will remove any completed descriptors from the list
        pending_descriptors = [
            descriptor_entry for descriptor_entry in all_descriptors
            if ('+'.join(list(descriptor_entry) if isinstance(descriptor_entry, tuple) else [descriptor_entry]),) not in completed_keys
        ]
        tqdm_disable = len(pending_descriptors) <= 3

        def _run_descriptor(descriptor_entry: Union[str, Tuple[str, ...]]) -> Dict[str, Any]:
            """ Helper function to run encoding, model building and evaluation for a single descriptor. """
            X = self.build_features(
                feature_type="descriptor",
                descriptor_entry=descriptor_entry,
                desc_instance=desc
            )
            # run model and get evaluation metrics for current descriptor(s)
            eval_metrics = self.run_model(X, self.activity, random_state=random_state)

            # create descriptor label and group label for results display and dataframe output
            descriptor_names = list(descriptor_entry) if isinstance(descriptor_entry, tuple) else [descriptor_entry]
            descriptor_label = '+'.join(descriptor_names)
            group_label = ','.join([desc.descriptor_groups[name] for name in descriptor_names])
            return {
                MetricKey.DESCRIPTOR.value: descriptor_label,
                MetricKey.GROUP.value: group_label,
                MetricKey.R2.value: eval_metrics.r2,
                MetricKey.RMSE.value: eval_metrics.rmse,
                MetricKey.MSE.value: eval_metrics.mse,
                MetricKey.MAE.value: eval_metrics.mae,
                MetricKey.RPD.value: eval_metrics.rpd,
                MetricKey.EXPLAINED_VARIANCE.value: eval_metrics.explained_var,
            }

        #run encoding, model building and evaluation for each descriptor or combination of descriptors
        new_rows = self._execute_jobs(
            items=pending_descriptors,
            task_fn=_run_descriptor,
            n_jobs=n_jobs,
            tqdm_desc="Descriptors",
            tqdm_unit="descriptor",
            tqdm_disable=tqdm_disable
        )
        metrics_rows.extend(new_rows)
        self._save_resume_checkpoint(metrics_rows, resume_file if resume else None)

        #stop counter and calculate elapsed time
        end = time.time()           
        elapsed = end - start

        self._log(f'\nElapsed time for Descriptor Encoding: {elapsed:.2f} seconds.\n')
        self._log('\n##########################################################################################')

        if (desc_combo == 2):
            save_filename = 'desc_combo2_results'
        elif (desc_combo == 3):
            save_filename = 'desc_combo3_results'
        else:
            save_filename = 'desc_results'

        return self.format_and_save_results(
            metrics_rows=metrics_rows,
            columns=[
                MetricKey.DESCRIPTOR.value,
                MetricKey.GROUP.value,
                MetricKey.R2.value,
                MetricKey.RMSE.value,
                MetricKey.MSE.value,
                MetricKey.MAE.value,
                MetricKey.RPD.value,
                MetricKey.EXPLAINED_VARIANCE.value,

            ],
            sort_by=sort_by,
            save_filename=save_filename,
            output_folder=output_folder,
            string_columns=[MetricKey.DESCRIPTOR.value, MetricKey.GROUP.value],
            resume_file=resume_file if resume else None,
            export_best_model=export_best_model,
            best_model_feature_fn=lambda best_row: self.build_features(
                feature_type="descriptor",
                descriptor_entry=tuple(best_row[MetricKey.DESCRIPTOR.value].split('+'))
                    if '+' in best_row[MetricKey.DESCRIPTOR.value]
                    else best_row[MetricKey.DESCRIPTOR.value],
                desc_instance=desc,
            ),
            random_state=random_state,
        )

    def aai_descriptor_encoding(self,
                                aai_indices: Optional[Union[str, List[str]]] = None,
                                descriptors: Optional[Union[str, List[str]]] = None,
                                desc_combo: int = 1,
                                sort_by: Union[str, 'SortKey'] = 'R2',
                                output_folder: str = "",
                                n_jobs: int = 1,
                                random_state: Optional[int] = None,
                                max_models: Optional[int] = None,
                                sample_mode: bool = False,
                                resume: bool = False,
                                resume_file: str = "",
                                export_best_model: bool = False) -> pd.DataFrame:
        """
        Encoding all protein sequences using each of the available indices in the AAI and
        aaindex package in concatenation with the protein descriptors available via the 
        protpy package. The sequences can be encoded using 1 AAI + 1 Descriptor, 2 
        Descriptors or 3 Descriptors, dictated by the desc_combo input parameter: set 
        this to 1, 2 or 3 for what encoding combination to use, default is 1. The protein 
        spectra of the AAI indices will be generated if the config param use_dsp is true, 
        also utilised for the DSP transformation is the class attributes: spectrum, window 
        and filter. 
        
        Each numerical encoding will be used as the feature data to build the predictive 
        regression ML models. To date, there are 566 indices and pySAR/protpy supports 
        33 descriptors so the encoding process will generate 18678, ~298000 and ~3.1M 
        models, when using 1, 2 or 3 descriptors + AAI indices, respectively. These values 
        may vary depending on the meta-parameters on some of the descriptors such as the 
        lag or lambda for the autocorrelation and pseudo amino acid descriptors, respectively. 
        The metrics evaluated from the model, accessing its accuracy and predictability for 
        each AAI + Descriptor encoding combination will be collated into a dataframe and saved 
        and returned, sorted by the R2 score by default.

        Parameters
        ==========        
        :aai_indices: str/list (default=None)
            str/list of aai indices to use for encoding the predictive models, by default
            ALL AAI indices will be used.
        :descriptors: list (default=None)
            str/list of descriptors to use for encoding, by default all available descriptors
            in the protpy package will be used for the encoding.
        :desc_combo: int (default=1)
            combination of descriptors to use.
        :sort_by: str (default=R2)
            sort output dataframe by specified column/metric value, results sorted by R2 
            score by default.
        :output_folder: str (default="")
            output folder to store results csv to, if empty input it will be stored in 
            the OUTPUT_FOLDER global var.

        Returns
        =======
        :aai_desc_metrics_df_: pd.DataFrame
            dataframe of calculated metric values from generated predictive models
            encoded using AAI indices + descriptors encoding strategy. The output will
            be of shape (X * Y) x 10, where X is the number of AAI indices input, Y is
            the number of descriptors input and 10 is the results/metrics columns of 
            the output dataframe. Using the default values and desc_combo of 1, 2 and 
            3, the output shapes will be (566 * 15) x 10, (566 * 105) x 10, or 
            (566 * 455) x 10.
        """
        all_indices = self.validate_inputs(aai_indices, aaindex1.record_codes(), "AAI")
        self.validate_desc_combo(desc_combo)
           
        #reuse cached Descriptors instance from PySAR.__init__
        desc = self.descriptor

        #validate input descriptors and get list of descriptor names to use for encoding
        all_descriptor_names = self.validate_inputs(descriptors, desc.valid_descriptors, "Descriptor")
        if desc_combo == 1:
            all_descriptors: List[Union[str, Tuple[str, ...]]] = all_descriptor_names
        else:
            all_descriptors = list(itertools.combinations(all_descriptor_names, desc_combo))

        # create list of all possible pairs of AAI indices and descriptors, then apply model limit if specified in config file
        all_pairs: List[Tuple[str, Union[str, Tuple[str, ...]]]] = list(itertools.product(all_indices, all_descriptors))
        all_pairs = self._apply_model_limit(all_pairs, sample_mode=sample_mode, max_models=max_models)

        # if resume is true then load resume file and get list of completed keys to skip already completed pairs of AAI indices and descriptors
        metrics_rows, completed_keys = self._load_resume(
            resume_file if resume else None,
            key_columns=[MetricKey.INDEX.value, MetricKey.DESCRIPTOR.value]
        )

        # prime descriptor cache once; improves repeated and parallel usage
        for descriptor_name in all_descriptor_names:
            self._get_descriptor_features(descriptor_name, desc)

        #create text wrapper for amino acid indices and descriptors text, split to newline if surpasses line length
        line_length = 90

        #create temp Model object to access the models' parameter values for use in display text below
        temp_model_parameters = Model(
            X=[],
            Y=self.activity,
            algorithm=self.algorithm,
            parameters=self.model_parameters
        )

        self._log('\n###########################################################################\n')
        self._log(
            f'# Encoding using {len(all_indices)} AAI and {len(all_descriptors)} descriptor combination(s) with the parameters:\n'
        )
        #only output indices if there are 10 or less
        if (len(all_indices) <= 10):
            self._log(textwrap.fill(f"# AAI Indices: {', '.join(all_indices)}", line_length))
        else:
            self._log(f'# AAI Indices: {len(all_indices)}')
        if (self.use_dsp):
            self._log(
                f'# DSP Parameters:\n#   Spectrum: {self.spectrum}\n#   Window Function: {self.window_type}\n#   Filter Function: {self.filter_type}'
            )
        descriptor_display = [
            '+'.join(descriptor_set) if isinstance(descriptor_set, tuple) else descriptor_set
            for descriptor_set in all_descriptors
        ]
        self._log(textwrap.fill(f"# Descriptors: {', '.join(descriptor_display)}", line_length))
        self._log(
            f'# Configuration File: {os.path.basename(self.config_file)}\n'
            f'# Dataset: {os.path.basename(self.dataset)}\n'
            f'# Number of Sequences/Sequence Length: {len(self.data)} x {self.data[self.sequence_col].str.len().max()}\n'
            f'# Target Activity: {self.activity_col}\n'
            f'# Algorithm: {repr(temp_model_parameters)}'
        )
        if not isinstance(self.model_parameters, dict) or not self.model_parameters:
            self._log(f'# Model Parameters: {temp_model_parameters.model.get_params()}')
        else:
            self._log(f'# Model Parameters: {self.model_parameters}')
        self._log(f'# Test Split : {self.test_split}')
        self._log('\n###########################################################################')

        #start counter
        start = time.time() 

        '''
        1.) Get AAI index encoding of protein sequences. If using DSP, create instance
            of pyDSP class and generate protein spectra from the AAI indices, according to
            instance parameters: spectrum, window and filter.
        2.) Get all descriptor values and concatenate to AAI encoding features.
        3.) Build model using concatenated AAI and Descriptor features as the training data.
        4.) Predict and evaluate the model using the test data unseen protein sequences.
        5.) Append index, descriptor and calculated metrics to lists.
        6.) Repeat steps 1 - 5 for all indices in the AAI.
        7.) Output results into a final dataframe, save it and return, sort by sort_by parameter.
        '''

        #create list of pending pairs of AAI indices and descriptors to process
        pending_pairs = [
            pair for pair in all_pairs
            if (str(pair[0]), '+'.join(list(pair[1]) if isinstance(pair[1], tuple) else [pair[1]])) not in completed_keys
        ]
        tqdm_disable = len(pending_pairs) <= 2
        
        def _run_pair(pair: Tuple[str, Union[str, Tuple[str, ...]]]) -> Dict[str, Any]:
            """ Helper function to run encoding, model building and evaluation for a single pair of AAI index and descriptor(s). """
            index, descriptor_entry = pair
            X = self.build_features(
                feature_type="aai_descriptor",
                index=index,
                descriptor_entry=descriptor_entry,
                desc_instance=desc
            )
            # run model and get evaluation metrics for current pair of AAI index and descriptor(s)
            eval_metrics = self.run_model(X, self.activity, random_state=random_state)

            # create descriptor label and group label for results display and dataframe output
            descriptor_names = list(descriptor_entry) if isinstance(descriptor_entry, tuple) else [descriptor_entry]
            descriptor_label = '+'.join(descriptor_names)
            group_label = ','.join([desc.descriptor_groups[name] for name in descriptor_names])
            return {
                MetricKey.INDEX.value: index,
                MetricKey.CATEGORY.value: aaindex1[index].category,
                MetricKey.DESCRIPTOR.value: descriptor_label,
                MetricKey.GROUP.value: group_label,
                MetricKey.R2.value: eval_metrics.r2,
                MetricKey.RMSE.value: eval_metrics.rmse,
                MetricKey.MSE.value: eval_metrics.mse,
                MetricKey.MAE.value: eval_metrics.mae,
                MetricKey.RPD.value: eval_metrics.rpd,
                MetricKey.EXPLAINED_VARIANCE.value: eval_metrics.explained_var,
            }

        #run encoding, model building and evaluation for each pair of AAI index and descriptor(s) using parallel processing with n_jobs threads
        new_rows = self._execute_jobs(
            items=pending_pairs,
            task_fn=_run_pair,
            n_jobs=n_jobs,
            tqdm_desc="AAI+Descriptors",
            tqdm_unit="pair",
            tqdm_disable=tqdm_disable
        )
        #append new rows to metrics_rows and save checkpoint if resume is true
        metrics_rows.extend(new_rows)
        self._save_resume_checkpoint(metrics_rows, resume_file if resume else None)

        #stop counter and calculate elapsed time
        end = time.time()           
        elapsed = end - start

        self._log(f'Elapsed time for AAI + Descriptor Encoding: {elapsed:.2f} seconds.')
        self._log('\n###########################################################################')

        if (desc_combo == 2):
            save_filename = 'aai_desc_combo2_results'
        elif (desc_combo == 3):
            save_filename = 'aai_desc_combo3_results'
        else:
            save_filename = 'aai_desc_results'

        return self.format_and_save_results(
            metrics_rows=metrics_rows,
            columns=[
                MetricKey.INDEX.value,
                MetricKey.CATEGORY.value,
                MetricKey.DESCRIPTOR.value,
                MetricKey.GROUP.value,
                MetricKey.R2.value,
                MetricKey.RMSE.value,
                MetricKey.MSE.value,
                MetricKey.MAE.value,
                MetricKey.RPD.value,
                MetricKey.EXPLAINED_VARIANCE.value,

            ],
            sort_by=sort_by,
            save_filename=save_filename,
            output_folder=output_folder,
            string_columns=[
                MetricKey.INDEX.value,
                MetricKey.CATEGORY.value,
                MetricKey.DESCRIPTOR.value,
                MetricKey.GROUP.value,
            ],
            resume_file=resume_file if resume else None,
            export_best_model=export_best_model,
            best_model_feature_fn=lambda best_row: self.build_features(
                feature_type="aai_descriptor",
                index=best_row[MetricKey.INDEX.value],
                descriptor_entry=tuple(best_row[MetricKey.DESCRIPTOR.value].split('+'))
                    if '+' in best_row[MetricKey.DESCRIPTOR.value]
                    else best_row[MetricKey.DESCRIPTOR.value],
                desc_instance=desc,
            ),
            random_state=random_state,
        )

    def _log(self, message: str, level: int = logging.INFO) -> None:
        """ Log to provided logger or fallback to print when verbose is enabled. """
        if self.logger is not None:
            self.logger.log(level, message)
            return
        if self.verbose:
            print(message)

    def validate_inputs(self,
                        input_values: Optional[Union[str, List[str]]],
                        valid_values: Sequence[str],
                        input_name: str) -> List[str]:
        """ Validate list/string inputs and normalize to sorted unique list of strings. """
        if input_values in (None, [], ""):
            values = list(valid_values)
        elif isinstance(input_values, str):
            if ',' in input_values:
                values = input_values.replace(' ', '').split(',')
            else:
                values = [input_values.strip()]
        elif isinstance(input_values, list):
            if not all(isinstance(item, str) for item in input_values):
                raise TypeError(f"Input {input_name} values must be strings.")
            values = input_values
        else:
            raise TypeError(f"Input {input_name} parameter is not of type list or str, got {type(input_values)}.")

        # Remove duplicates and sort values, then check for any invalid entries against valid_values list
        values = sorted(list(set(values)))
        invalid_values = [value for value in values if value not in valid_values]
        if invalid_values:
            # Attempt fuzzy correction for each invalid value using difflib.
            corrected = []
            unfixable = []
            for bad_val in invalid_values:
                matches = get_close_matches(bad_val, valid_values, n=1, cutoff=0.6)
                if matches:
                    corrected.append((bad_val, matches[0]))
                else:
                    unfixable.append(bad_val)
            if unfixable:
                raise ValueError(f"Invalid {input_name} value(s) found: {unfixable}.")
            # Apply corrections: replace bad value with closest match
            for bad_val, good_val in corrected:
                idx = values.index(bad_val)
                values[idx] = good_val
                warnings.warn(
                    f"{input_name} value '{bad_val}' not found exactly; using '{good_val}' instead.",
                    UserWarning, stacklevel=2
                )
            # Re-deduplicate after fuzzy correction
            values = sorted(list(set(values)))

        return values

    def validate_desc_combo(self, desc_combo: int) -> None:
        """ Validate descriptor combination size. """
        if desc_combo not in {1, 2, 3}:
            raise ValueError(f"Invalid desc_combo value '{desc_combo}'. Expected one of: 1, 2, 3.")

    def _apply_model_limit(self,
                           entries: List[Union[str, Tuple[str, ...]]],
                           sample_mode: bool,
                           max_models: Optional[int]) -> List[Union[str, Tuple[str, ...]]]:
        """ Limit model count for smoke runs or explicit truncation. """
        if not entries:
            return entries

        # If max_models is set, take the first N entries (deterministic slice)
        if max_models is not None:
            if max_models <= 0:
                raise ValueError(f"max_models must be > 0, got {max_models}.")
            return entries[:max_models]

        # If sample_mode is True, take the first 10 entries for quick testing
        if sample_mode:
            return entries[:min(10, len(entries))]

        return entries

    def _load_resume(self,
                     resume_file: Optional[str],
                     key_columns: Sequence[str]) -> Tuple[List[Dict[str, Any]], set]:
        """ Load existing partial results and return rows + completed key set. """
        if not resume_file:
            return [], set()

        # Load the resume file if it exists
        resume_path = Path(resume_file)
        if not resume_path.exists():
            return [], set()

        # Read existing results and build a set of completed keys based on specified key columns
        try:
            existing_df = pd.read_csv(resume_path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            warnings.warn(
                f"Could not read resume file {resume_path!r}; starting fresh.",
                UserWarning, stacklevel=3
            )
            return [], set()
        if existing_df.empty:
            return [], set()

        missing_cols = [col for col in key_columns if col not in existing_df.columns]
        if missing_cols:
            warnings.warn(
                f"Resume file {resume_path!r} is missing required columns {missing_cols}; starting fresh.",
                UserWarning, stacklevel=3
            )
            return [], set()

        # Convert existing rows to dicts and create a set of completed keys for quick lookup
        existing_rows = existing_df.to_dict(orient='records')
        completed_keys = {
            tuple(str(row[col]) for col in key_columns)
            for row in existing_rows if all(col in row for col in key_columns)
        }
        return existing_rows, completed_keys

    def _save_resume_checkpoint(self, metrics_rows: List[Dict[str, Any]], resume_file: Optional[str]) -> None:
        """ Persist current progress to a resume checkpoint file. """
        if not resume_file:
            return
        pd.DataFrame(metrics_rows).to_csv(resume_file, index=False)

    def _get_aai_features(self, index: str) -> pd.DataFrame:
        """ Return cached AAI features for an index, computing on first use.

        Uses a per-key Future to prevent TOCTOU races under concurrent access:
        the first thread to see a cache miss creates the Future and computes;
        any racing thread waits on the same Future instead of recomputing.
        """
        with self._cache_lock:
            if index in self._aai_feature_cache:
                return self._aai_feature_cache[index]
            if index in self._aai_inflight:
                fut = self._aai_inflight[index]
                should_compute = False
            else:
                fut: Future[pd.DataFrame] = Future()
                self._aai_inflight[index] = fut
                should_compute = True

        if not should_compute:
            # Another thread is already computing this index — wait for its result.
            return fut.result()

        try:
            encoded_seqs = self.get_aai_encoding(index)
            if self.use_dsp:
                py_dsp = PyDSP(
                    self.config_file,
                    protein_seqs=encoded_seqs,
                    spectrum=self.spectrum,
                    window_type=self.window_type,
                    filter_type=self.filter_type
                )
                features = pd.DataFrame(py_dsp.spectrum_encoding)
            else:
                features = pd.DataFrame(encoded_seqs)

            features.columns = [f"aai_{i}" for i in range(1, len(features.columns) + 1)]
            fut.set_result(features)
        except Exception as exc:
            fut.set_exception(exc)
            with self._cache_lock:
                self._aai_inflight.pop(index, None)
            raise

        with self._cache_lock:
            self._aai_feature_cache[index] = features
            self._aai_inflight.pop(index, None)
        return features

    def _get_descriptor_features(self, descriptor_name: str, desc_instance: Descriptors) -> pd.DataFrame:
        """ Return cached descriptor features, computing on first use.

        Uses the same per-key Future pattern as _get_aai_features to prevent
        redundant concurrent computation for the same descriptor key.
        """
        with self._cache_lock:
            if descriptor_name in self._descriptor_feature_cache:
                return self._descriptor_feature_cache[descriptor_name]
            if descriptor_name in self._desc_inflight:
                fut = self._desc_inflight[descriptor_name]
                should_compute = False
            else:
                fut: Future[pd.DataFrame] = Future()
                self._desc_inflight[descriptor_name] = fut
                should_compute = True

        if not should_compute:
            return fut.result()

        try:
            descriptor_df = desc_instance.get_descriptor_encoding(descriptor_name)
            fut.set_result(descriptor_df)
        except Exception as exc:
            fut.set_exception(exc)
            with self._cache_lock:
                self._desc_inflight.pop(descriptor_name, None)
            raise

        with self._cache_lock:
            self._descriptor_feature_cache[descriptor_name] = descriptor_df
            self._desc_inflight.pop(descriptor_name, None)
        return descriptor_df

    def build_features(self,
                       feature_type: str,
                       index: Optional[str] = None,
                       descriptor_entry: Optional[Union[str, Tuple[str, ...]]] = None,
                       desc_instance: Optional[Descriptors] = None) -> pd.DataFrame:
        """ Build feature matrix for AAI, descriptor, or combined encodings. """
        if feature_type == "aai":
            if index is None:
                raise ValueError("index must be provided for AAI feature building.")
            return self._get_aai_features(index)

        if feature_type == "descriptor":
            if desc_instance is None:
                raise ValueError("desc_instance must be provided for descriptor feature building.")
            if descriptor_entry is None:
                raise ValueError("descriptor_entry must be provided for descriptor feature building.")

            descriptor_names: List[str]
            if isinstance(descriptor_entry, tuple):
                descriptor_names = list(descriptor_entry)
            else:
                descriptor_names = [descriptor_entry]

            descriptor_frames: List[pd.DataFrame] = [
                self._get_descriptor_features(name, desc_instance)
                for name in descriptor_names
            ]
            if len(descriptor_frames) == 1:
                return descriptor_frames[0].reset_index(drop=True)
            # Align on row position to avoid index-based concat errors when source indexes are non-unique.
            return pd.concat([frame.reset_index(drop=True) for frame in descriptor_frames], axis=1)

        if feature_type == "aai_descriptor":
            if index is None:
                raise ValueError("index must be provided for combined feature building.")
            aai_features = self.build_features(feature_type="aai", index=index).reset_index(drop=True)
            descriptor_features = self.build_features(
                feature_type="descriptor",
                descriptor_entry=descriptor_entry,
                desc_instance=desc_instance
            ).reset_index(drop=True)
            # Concatenate descriptor and AAI features by row position for consistent model input shape.
            return pd.concat([descriptor_features, aai_features], axis=1)

        raise ValueError(f"Unknown feature_type '{feature_type}'.")

    def run_model(self,
                  X: pd.DataFrame,
                  Y: pd.Series,
                  random_state: Optional[int] = None) -> Evaluate:
        """Train model for current configuration and return evaluated metrics."""
        model_parameters = self.model_parameters if isinstance(self.model_parameters, dict) else {}
        if X.shape[1] == 1 and self.algorithm.lower() == "plsregression":
            model_parameters = dict(model_parameters)
            model_parameters['n_components'] = 1

        # Convert pandas containers to numpy arrays to avoid pandas dtype deprecation warnings in sklearn validation.
        X_values = X.to_numpy(dtype=float, copy=False) if isinstance(X, pd.DataFrame) else X
        Y_values = Y.to_numpy(copy=False) if isinstance(Y, pd.Series) else Y

        model = Model(
            X=X_values,
            Y=Y_values,
            algorithm=self.algorithm,
            parameters=model_parameters,
            test_split=self.test_split
        )
        _, _, _, y_test = model.train_test_split(test_split=self.test_split, random_state=random_state)
        model.fit()
        y_pred = model.predict()
        return Evaluate(y_test, y_pred)

    def _execute_jobs(self,
                      items: Sequence[Any],
                      task_fn: Callable[[Any], Union[Dict[str, Any], List[Dict[str, Any]], None]],
                      n_jobs: int,
                      tqdm_desc: str,
                      tqdm_unit: str,
                      tqdm_disable: bool) -> List[Dict[str, Any]]:
        """ Execute independent tasks sequentially or in parallel and collect rows. """
        if n_jobs <= 1:
            collected: List[Dict[str, Any]] = []
            for item in tqdm(items, desc=tqdm_desc, unit=tqdm_unit, disable=tqdm_disable, ncols=90):
                result = task_fn(item)
                if result is None:
                    continue
                if isinstance(result, list):
                    collected.extend(result)
                else:
                    collected.append(result)
            return collected

        collected_parallel: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(task_fn, item): item for item in items}
            iterator = as_completed(futures)
            if not tqdm_disable:
                iterator = tqdm(iterator, total=len(futures), desc=tqdm_desc, unit=tqdm_unit, ncols=90)

            for future in iterator:
                try:
                    result = future.result()
                except Exception as exc:
                    item = futures[future]
                    warnings.warn(
                        f"Task failed for item {item!r}: {exc}",
                        RuntimeWarning, stacklevel=2
                    )
                    continue
                if result is None:
                    continue
                if isinstance(result, list):
                    collected_parallel.extend(result)
                else:
                    collected_parallel.append(result)

        return collected_parallel

    def format_and_save_results(self,
                                metrics_rows: List[Dict[str, Any]],
                                columns: List[str],
                                sort_by: str,
                                save_filename: str,
                                output_folder: str,
                                string_columns: List[str],
                                resume_file: Optional[str] = None,
                                export_best_model: bool = False,
                                best_model_feature_fn: Optional[Callable] = None,
                                random_state: Optional[int] = None) -> pd.DataFrame:
        """ Create, sort, save and return result dataframe from collected metric rows. """
        metrics_df = pd.DataFrame(metrics_rows, columns=columns)
        for column in string_columns:
            if column in metrics_df.columns:
                metrics_df[column] = metrics_df[column].astype(pd.StringDtype())

        # Normalise sort_by: accept SortKey enum instances or their string .value equivalents.
        if isinstance(sort_by, SortKey):
            sort_by = sort_by.value
        valid_sort_columns = [metric.value for metric in SortKey]
        if sort_by not in valid_sort_columns:
            sort_by = SortKey.R2.value

        sort_ascending = sort_by not in {SortKey.R2.value, SortKey.EXPLAINED_VARIANCE.value}
        metrics_df = metrics_df.sort_values(by=[sort_by], ascending=sort_ascending)
        save_results(metrics_df, save_filename, output_folder=output_folder)
        if resume_file:
            metrics_df.to_csv(resume_file, index=False)

        #optionally re-train on the best encoding and persist the model + scaler
        if export_best_model and not metrics_df.empty and best_model_feature_fn is not None:
            best_row = metrics_df.iloc[0].to_dict()
            X_best = best_model_feature_fn(best_row)
            model_parameters = self.model_parameters if isinstance(self.model_parameters, dict) else {}
            best_model = Model(
                X=X_best.to_numpy(dtype=float, copy=False),
                Y=self.activity.to_numpy(copy=False) if isinstance(self.activity, pd.Series) else self.activity,
                algorithm=self.algorithm,
                parameters=model_parameters,
                test_split=self.test_split,
            )
            best_model.train_test_split(test_split=self.test_split, random_state=random_state)
            best_model.fit()
            folder = output_folder if output_folder else 'outputs'
            os.makedirs(folder, exist_ok=True)
            best_model.save(folder, model_name='best_model.pkl')
            self._log(f'\nBest model saved to: {os.path.join(folder, "best_model.pkl")}')

        return metrics_df

    def __str__(self) -> str:
        return (
            f"Instance of Encoding Class with attribute values: Configuration File: {os.path.basename(self.config_file)}, "
            f"Dataset: {os.path.basename(self.dataset)}, Target Activity: {self.activity}, "
            f"Algorithm: {self.algorithm}, Model Parameters: {self.model_parameters}, Test Split: {self.test_split}."
        )

    def __repr__(self) -> str:
        return f"<{self}>"