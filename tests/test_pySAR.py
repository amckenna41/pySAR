################################################################################
##################             pySAR Module Tests             ##################
################################################################################

import pandas as pd
import numpy as np
import os
import shutil
import re
import glob
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import pySAR as pysar_
import pySAR.pySAR as pysar
import pySAR.globals_ as _globals

# @unittest.skip("")
class PySARTests(unittest.TestCase):
    """
    Test suite for testing main pySAR module and functionality 
    in pySAR package. 

    Test Cases
    ==========
    test_pySAR_metadata:
        testing correct pysar software metadata.
    test_pySAR:
        testing overall pysar encoding functionality.
    test_sequences:
        testing correct sequences pysar encoding functionality.
    test_activity:
        testing correct activity pysar encoding functionality.
    test_get_aai_encoding:
        testing correct aai pysar encoding functionality.
    test_aai_encoding:
        testing correct aai pysar encoding functionality.
    test_get_desc_encoding:
        testing correct descriptor pysar encoding functionality.
    test_desc_encoding:
        testing correct descriptor pysar encoding functionality.
    test_aai_desc_encoding:
        testing correct aai + descriptor pysar encoding functionality.
    test_predict_activity:
        testing predict_activity() method for AAI, descriptor and combined encoding strategies.
    """
    def setUp(self):
        """ Import the 4 config files for each of the 4 datasets used for testing the pySAR methods. """
        #array of config files for each test dataset
        config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]

        #list of canonical amino acids
        self.amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", 
            "Q", "R", "S", "T", "V", "W", "Y"]

        # Create temporary unit test output folder to store any pysar assets and results
        self.test_output_folder = os.path.join("tests", "test_outputs")
        if not (os.path.isdir(self.test_output_folder)):
            os.makedirs(self.test_output_folder)
        
        # Validate config files exist before running tests
        missing_configs = [f for f in self.all_config_files if not os.path.isfile(f)]
        if missing_configs:
            raise FileNotFoundError(f"Test config files missing: {missing_configs}")

    # @unittest.skip("Skipping metadata tests.")
    def test_pySAR_metadata(self):
        """ Testing correct pySAR version and metadata. """
        self.assertEqual(pysar_.__version__, "2.5.3",
            f"pySAR version is not correct, expected 2.5.3, got {pysar_.__version__}.")
        self.assertEqual(pysar_.__name__, "pySAR", 
            f"pySAR software name is not correct, expected pySAR, got {pysar_.__name__}.")
        self.assertEqual(pysar_.__author__, "AJ McKenna: https://github.com/amckenna41", 
            f"pySAR author is not correct, expected AJ McKenna, got {pysar_.__author__}.")
        self.assertEqual(pysar_.__authorEmail__, "amckenna41@qub.ac.uk", 
            f"pySAR author email is not correct, expected amckenna41@qub.ac.uk, got {pysar_.__authorEmail__}.")
        self.assertEqual(pysar_.__url__, "https://github.com/amckenna41/pySAR", 
            f"pySAR repo URL is not correct, expected https://github.com/amckenna41/pySAR, got {pysar_.__url__}.")
        self.assertEqual(pysar_.__download_url__, "https://github.com/amckenna41/pySAR/archive/refs/heads/main.zip", 
            f"pySAR repo download URL is not correct, expected https://github.com/amckenna41/pySAR/archive/refs/heads/main.zip, got {pysar_.__download_url__}.")
        self.assertEqual(pysar_.__status__, "Production", 
            f"pySAR status is not correct, expected Production, got {pysar_.__status__}.")
        self.assertEqual(pysar_.__license__, "MIT", 
            f"pySAR license type is not correct, expected MIT, got {pysar_.__license__}.")
        self.assertEqual(pysar_.__maintainer__, "AJ McKenna", 
            f"pySAR maintainer is not correct, expected AJ McKenna, got {pysar_.__license__}.")
        self.assertEqual(pysar_.__keywords__, ["bioinformatics", "protein engineering", "python", \
            "pypi", "machine learning", "directed evolution", "drug discovery", "sequence activity relationships", \
            "SAR", "aaindex", "protpy", "protein descriptors"], f"pySAR keywords is not correct, got: {pysar_.__keywords__}.")

    def test_pySAR(self):
        """ Testing pySAR intialisation process and associated methods & attributes. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_thermostability.dataset, (os.path.join('tests', 'test_data', 'test_thermostability.txt')),
            f'Dataset attribute does not match expected, got {test_pySAR_thermostability.dataset}.')
        self.assertEqual(test_pySAR_thermostability.sequence_col, "sequence",
            f'Sequence column attribute is not correct, expected sequence, got {test_pySAR_thermostability.sequence_col}.')
        self.assertEqual(test_pySAR_thermostability.activity_col, "T50",
            f"Activity attribute name not correct, expected T50, got {test_pySAR_thermostability.activity_col}.")
        self.assertEqual(test_pySAR_thermostability.algorithm, "plsregression",
            f'Algorithm attribute not correct, expected plsregression, got {test_pySAR_thermostability.algorithm}.')
        self.assertEqual(test_pySAR_thermostability.test_split, 0.2,
            f'Test split not expected, expected 0.2, got {test_pySAR_thermostability.test_split}.')
        self.assertIsNone(test_pySAR_thermostability.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_thermostability.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_thermostability.model_parameters, {},
            f'Parameters attribute expected to be empty, got {test_pySAR_thermostability.model_parameters}.')
        self.assertIsInstance(test_pySAR_thermostability.data, pd.DataFrame,
            f'Data expected to be a DataFrame, got {type(test_pySAR_thermostability.data)}.')
        self.assertIsInstance(test_pySAR_thermostability.sequences, pd.Series,
            f'Sequences expected to be a pd.Series, got {type(test_pySAR_thermostability.sequences)}.')
        self.assertIsInstance(test_pySAR_thermostability.activity, pd.Series,
            f'Activity expected to be a pd.Series, got {type(test_pySAR_thermostability.activity)}.')
        self.assertEqual(test_pySAR_thermostability.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_thermostability.num_seqs, 261,
            f'Number of sequences expected to be 261, got {test_pySAR_thermostability.num_seqs}.')
        self.assertEqual(test_pySAR_thermostability.sequence_length, 466,
            f'Sequence length expected to be 466, got {test_pySAR_thermostability.sequence_length}.')
        self.assertEqual(test_pySAR_thermostability.feature_space, (),
            f'Feature space expected to be an empty tuple, got {test_pySAR_thermostability.feature_space}.')
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_enantioselectivity.dataset, (os.path.join('tests', 'test_data', 'test_enantioselectivity.txt')),
            f'Dataset attribute does not match expected, got {test_pySAR_enantioselectivity.dataset}.')
        self.assertEqual(test_pySAR_enantioselectivity.sequence_col, "sequence",
            f'Sequence column attribute is not correct, expected sequence, got {test_pySAR_enantioselectivity.sequence_col}.')
        self.assertEqual(test_pySAR_enantioselectivity.activity_col, "e-value",
            f"Activity attribute name not correct, expected e-value, got {test_pySAR_enantioselectivity.activity_col}.")
        self.assertEqual(test_pySAR_enantioselectivity.algorithm, "randomforestregressor",
            f'Algorithm attribute not correct, expected randomforestregressor, got {test_pySAR_enantioselectivity.algorithm}.')
        self.assertEqual(test_pySAR_enantioselectivity.test_split, 0.2,
            f'Test split not expected, expected 0.2, got {test_pySAR_enantioselectivity.test_split}.')
        self.assertIsNone(test_pySAR_enantioselectivity.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_enantioselectivity.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_enantioselectivity.model_parameters, {},
            f'Parameters attribute expected to be empty, got {test_pySAR_enantioselectivity.model_parameters}.')
        self.assertIsInstance(test_pySAR_enantioselectivity.data, pd.DataFrame,
            f'Data expected to be a DataFrame, got {type(test_pySAR_enantioselectivity.data)}.')
        self.assertIsInstance(test_pySAR_enantioselectivity.sequences, pd.Series,
            f'Sequences expected to be a pd.Series, got {type(test_pySAR_enantioselectivity.sequences)}.')
        self.assertIsInstance(test_pySAR_enantioselectivity.activity, pd.Series,
            f'Activity expected to be a pd.Series, got {type(test_pySAR_enantioselectivity.activity)}.')
        self.assertEqual(test_pySAR_enantioselectivity.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_enantioselectivity.num_seqs, 152,
            f'Number of sequences expected to be 152, got {test_pySAR_enantioselectivity.num_seqs}.')
        self.assertEqual(test_pySAR_enantioselectivity.sequence_length, 398,
            f'Sequence length expected to be 398, got {test_pySAR_enantioselectivity.sequence_length}.')
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (),
            f'Feature space expected to be an empty tuple, got {test_pySAR_enantioselectivity.feature_space}.')
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_absorption.dataset, (os.path.join('tests', 'test_data', 'test_absorption.txt')),
            f'Dataset attribute does not match expected, got {test_pySAR_absorption.dataset}.')
        self.assertEqual(test_pySAR_absorption.sequence_col, "sequence",
            f'Sequence column attribute is not correct, expected sequence, got {test_pySAR_absorption.sequence_col}.')
        self.assertEqual(test_pySAR_absorption.activity_col, "peak",
            f"Activity attribute name not correct, expected peak, got {test_pySAR_absorption.activity_col}.")
        self.assertEqual(test_pySAR_absorption.algorithm, "knn",
            f'Algorithm attribute not correct, expected knn, got {test_pySAR_absorption.algorithm}.')
        self.assertEqual(test_pySAR_absorption.test_split, 0.2,
            f'Test split not expected, expected 0.2, got {test_pySAR_absorption.test_split}.')
        self.assertIsNone(test_pySAR_absorption.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_absorption.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_absorption.model_parameters, {},
            f'Parameters attribute expected to be empty, got {test_pySAR_absorption.model_parameters}.')
        self.assertIsInstance(test_pySAR_absorption.data, pd.DataFrame,
            f'Data expected to be a DataFrame, got {type(test_pySAR_absorption.data)}.')
        self.assertIsInstance(test_pySAR_absorption.sequences, pd.Series,
            f'Sequences expected to be a pd.Series, got {type(test_pySAR_absorption.sequences)}.')
        self.assertIsInstance(test_pySAR_absorption.activity, pd.Series,
            f'Activity expected to be a pd.Series, got {type(test_pySAR_absorption.activity)}.')
        self.assertEqual(test_pySAR_absorption.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_absorption.num_seqs, 81,
            f'Number of sequences expected to be 81, got {test_pySAR_absorption.num_seqs}.')
        self.assertEqual(test_pySAR_absorption.sequence_length, 298,
            f'Sequence length expected to be 298, got {test_pySAR_absorption.sequence_length}.')
        self.assertEqual(test_pySAR_absorption.feature_space, (),
            f'Feature space expected to be an empty tuple, got {test_pySAR_absorption.feature_space}.')
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_localization.dataset, (os.path.join('tests', 'test_data', 'test_localization.txt')),
            f'Dataset attribute does not match expected, got {test_pySAR_localization.dataset}.')
        self.assertEqual(test_pySAR_localization.sequence_col, "sequence",
            f'Sequence column attribute is not correct, expected sequence, got {test_pySAR_localization.sequence_col}.')
        self.assertEqual(test_pySAR_localization.activity_col, "log_GFP",
            f"Activity attribute name not correct, expected log_GFP, got {test_pySAR_localization.activity_col}.")
        self.assertEqual(test_pySAR_localization.algorithm, "adaboostregressor",
            f'Algorithm attribute not correct, expected adaboostregressor, got {test_pySAR_localization.algorithm}.')
        self.assertEqual(test_pySAR_localization.test_split, 0.2,
            f'Test split not expected, expected 0.2, got {test_pySAR_localization.test_split}.')
        self.assertIsNone(test_pySAR_localization.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_localization.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_localization.model_parameters, {},
            f'Parameters attribute expected to be empty, got {test_pySAR_localization.model_parameters}.')
        self.assertIsInstance(test_pySAR_localization.data, pd.DataFrame,
            f'Data expected to be a DataFrame, got {type(test_pySAR_localization.data)}.')
        self.assertIsInstance(test_pySAR_localization.sequences, pd.Series,
            f'Sequences expected to be a pd.Series, got {type(test_pySAR_localization.sequences)}.')
        self.assertIsInstance(test_pySAR_localization.activity, pd.Series,
            f'Activity expected to be a pd.Series, got {type(test_pySAR_localization.activity)}.')
        self.assertEqual(test_pySAR_localization.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_localization.num_seqs, 254,
            f'Number of sequences expected to be 254, got {test_pySAR_localization.num_seqs}.')
        self.assertEqual(test_pySAR_localization.sequence_length, 361,
            f'Sequence length expected to be 361, got {test_pySAR_localization.sequence_length}.')
        self.assertEqual(test_pySAR_localization.feature_space, (),
            f'Feature space expected to be an empty tuple, got {test_pySAR_localization.feature_space}.')
#5.)
        #validate that if errorneous input parameters are input, that errors are raised
        with self.assertRaises(OSError, msg='OS Error raised, config file not found.'):
            pysar.PySAR(config_file="blahblahblah")
            pysar.PySAR(config_file="test_data/nothing.json")
#6.)
        with self.assertRaises(TypeError, msg='Type Error raised, config file parameter not correct data type.'):
            pysar.PySAR(config_file=101)
            pysar.PySAR(config_file=False)

    def test_preprocessing_fuzzy_column_matching(self):
        """ Fuzzy column matching: close name emits UserWarning; no match raises ValueError. """
        import warnings
        # 'sequences' is close enough to 'sequence' (the real column name) — should match with a warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p = pysar.PySAR(config_file=self.all_config_files[0], sequence_col="sequences")
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(len(user_warnings) > 0,
                "Expected a UserWarning when a fuzzy column match is found.")
        self.assertEqual(p.sequence_col, "sequence",
            f"Expected sequence_col to be resolved to 'sequence', got '{p.sequence_col}'.")
        self.assertIsNotNone(p.sequences,
            "Expected sequences to be loaded after fuzzy column resolution.")
        # Completely unrecognisable name should raise ValueError (no fuzzy match possible)
        with self.assertRaises(ValueError,
                msg='ValueError expected when sequence_col has no close match.'):
            pysar.PySAR(config_file=self.all_config_files[0], sequence_col="completelywrongcolumn")

    def test_sequences(self):
        """ Testing getting the protein sequences from the dataset. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability 
        test_seqs = test_pySAR_thermostability.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_thermostability._num_seqs, ),
            f'Shape of the sequences not correct, expected {test_seqs.shape}, got {(test_pySAR_thermostability._num_seqs, )}.')
        self.assertIsInstance(test_seqs, pd.Series,
            f'Sequences not of correct type, expected {pd.Series}, got {type(test_seqs)}.')
        self.assertTrue(test_seqs[0].startswith("MTIKEMPQPK"),
            'Error in first seqeuence, expected it to start with MTIKEMPQPK.')
        self.assertTrue(pd.api.types.is_string_dtype(test_seqs),
            f'Sequence Series expected to have a string dtype, got {test_seqs.dtype}.')
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_seqs = test_pySAR_enantioselectivity.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_enantioselectivity._num_seqs, ),
            f'Shape of the sequences not correct, expected {test_seqs.shape}, got {(test_pySAR_enantioselectivity._num_seqs, )}.')
        self.assertIsInstance(test_seqs, pd.Series,
            f'Sequences not of correct type, expected {pd.Series}, got {type(test_seqs)}.')
        self.assertTrue(test_seqs[0].startswith("MSAPFAKF"),
            'Error in second seqeuence expected it to start with MSAPFAKF.')
        self.assertTrue(pd.api.types.is_string_dtype(test_seqs),
            f'Sequence Series expected to have a string dtype, got {test_seqs.dtype}.')
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        test_seqs = test_pySAR_absorption.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_absorption._num_seqs, ),
            f'Shape of the sequences not correct, expected {test_seqs.shape}, got {(test_pySAR_absorption._num_seqs, )}.')
        self.assertIsInstance(test_seqs, pd.Series,
            f'Sequences not of correct type, expected {pd.Series}, got {type(test_seqs)}.')
        self.assertTrue(test_seqs[0].startswith("MLMTVFSSAP"),
            'Error in third seqeuence expected it to start with MLMTVFSSAP.')
        self.assertTrue(pd.api.types.is_string_dtype(test_seqs),
            f'Sequence Series expected to have a string dtype, got {test_seqs.dtype}.')
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        test_seqs = test_pySAR_localization.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_localization._num_seqs, ),
            f'Shape of the sequences not correct, expected {test_seqs.shape}, got {(test_pySAR_localization._num_seqs, )}.')
        self.assertIsInstance(test_seqs, pd.Series,
            f'Sequences not of correct type, expected {pd.Series}, got {type(test_seqs)}.')
        self.assertTrue(test_seqs[0].startswith("MSRLVAASWL"),
            'Error in third seqeuence expected it to start with MSRLVAASWL.')
        self.assertTrue(pd.api.types.is_string_dtype(test_seqs),
            f'Sequence Series expected to have a string dtype, got {test_seqs.dtype}.')
    
    def test_activity(self):
        """ Testing function that gets activity from dataset. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        activity = test_pySAR_thermostability.activity

        self.assertIsInstance(activity, pd.Series, 
            f'Output should be a Series, got {type(activity)}.')
        self.assertEqual(activity.shape, (test_pySAR_thermostability.num_seqs,), 
            f'Output expected to be shape {(test_pySAR_thermostability.num_seqs,)}, got {activity.shape}.')
        self.assertTrue((activity[:10] == np.array([55.0, 43.0, 49.0, 39.8, 52.9, 48.8, 45.0, 48.3, 61.5, 54.6])).all(),
                f"First 10 elements of activity don't match expected output:\n{activity[:10]}.")
        self.assertEqual(activity.name, "T50", 
            f"Expected T50 column name for Series, got {activity.name}.")
        self.assertTrue(activity.dtypes == np.float64, 
            f"Column datatypes should be np.float64, got {activity.dtypes}.")
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        activity_enantioselectivity = test_pySAR_enantioselectivity.activity

        self.assertIsInstance(activity_enantioselectivity, pd.Series, 
            f'Output should be a Series, got {type(activity_enantioselectivity)}.')
        self.assertEqual(activity_enantioselectivity.shape, (test_pySAR_enantioselectivity.num_seqs,), 
            f'Output expected to be shape {(test_pySAR_enantioselectivity.num_seqs,)}, got {activity_enantioselectivity.shape}.')
        self.assertTrue((activity_enantioselectivity[:10] == np.array([5.0, 23.0, 10.0, 9.0, 12.0, 11.0, 11.0, 21.0, 18.0, 17.0])).all(),
                f"First 10 elements of activity don't match expected output:\n{activity_enantioselectivity[:10]}.")
        self.assertEqual(activity_enantioselectivity.name, "e-value", 
            f"Expected e-value column name for Series, got {activity_enantioselectivity.name}.")
        self.assertTrue(np.float64 == activity_enantioselectivity.dtypes, 
            f"Column datatypes should be np.float64, got {activity_enantioselectivity.dtypes}.")
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        activity_absorption = test_pySAR_absorption.activity

        self.assertIsInstance(activity_absorption, pd.Series, 
            f'Output should be a Series, got {type(activity_absorption)}.')
        self.assertEqual(activity_absorption.shape, (test_pySAR_absorption.num_seqs,), 
            f'Output expected to be shape {(test_pySAR_absorption.num_seqs,)}, got {activity_absorption.shape}.')
        self.assertTrue((activity_absorption[:10] == np.array([539, 510, 510, 519, 525, 528, 528, 534, 528, 510])).all(),
                f"First 10 elements of activity don't match expected output:\n{activity_absorption[:10]}.")
        self.assertEqual(activity_absorption.name, "peak", 
            f"Expected peak column name for Series, got {activity_absorption.name}.")
        self.assertTrue(np.int64 == activity_absorption.dtypes, 
            f"Column datatypes should be np.float64, got {activity_absorption.dtypes}.")
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        activity_localization = test_pySAR_localization.activity

        self.assertIsInstance(activity_localization, pd.Series, 
            f'Output should be a Series, got {type(activity_localization)}.')
        self.assertEqual(activity_localization.shape, (test_pySAR_localization.num_seqs,), 
            f'Output expected to be shape {(test_pySAR_localization.num_seqs,)}, got {activity_localization.shape}.')
        # self.assertTrue((activity_localization[:10] == np.array([-4.626936, -5.599110, -5.715788, -5.335352, -4.187052, -6.732491, -7.135846, -6.128409, -5.319843, -5.092067])).all(),
        #         f"First 10 elements of activity don't match expected output:\n{activity_localization[:10]}.")
        self.assertEqual(activity_localization.name, "log_GFP", 
            f"Expected log_GFP column name for Series, got {activity_localization.name}.")
        self.assertTrue(activity_localization.dtypes == np.float64, 
            f"Column datatypes should be np.float64, got {activity_localization.dtypes}.")
    
    def test_get_aai_encoding(self):
        """ Testing getting the AAI encoding from the database for specific indices. """
        aa_indices = ["CHAM810101", "ISOY800103"]
        aa_indices1 = "NAKH920102"
        aa_indices2 = "ZIMJ680105"
        aa_indices3 = "PALJ810102, RACS820112"
        error_aaindices = ["ABCD1234", "ABCD12345"]
        error_aaindices1 = "XYZ4567"
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        aai_encoding_thermostability = test_pySAR_thermostability.get_aai_encoding(aa_indices)

        self.assertIsInstance(aai_encoding_thermostability, np.ndarray,
            f'AAI Encoding output expected to be a numpy array, got datatype {type(aai_encoding_thermostability)}.')
        self.assertEqual(aai_encoding_thermostability.shape[0], test_pySAR_thermostability.num_seqs,
            f'The number of sequences in the dataset expected to be {test_pySAR_thermostability.num_seqs}, got {aai_encoding_thermostability.shape[0]}.')
        self.assertEqual(aai_encoding_thermostability.shape[1], test_pySAR_thermostability.sequence_length * len(aa_indices),
            f'The length of the sequences expected to be {(test_pySAR_thermostability.sequence_length * len(aa_indices))}, got {str(aai_encoding_thermostability.shape[1])}.')
        self.assertEqual(aai_encoding_thermostability.dtype, np.float32,
            f'Datatype of elements in numpy array expected to be dtype np.float32, got {aai_encoding_thermostability.dtype}.')
        self.assertTrue((np.array([0.78, 0.5, 1.02, 0.68, 0.68, 0.78, 0.36, 0.68, 0.36, 0.68], 
            dtype=np.float32) == aai_encoding_thermostability[0][:10]).all(),
                f'The first 10 elements of the 1st sequence in encoding do not match what was expected:\n{aai_encoding_thermostability[0][:10]}.')
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        aai_encoding_enantioselectivity = test_pySAR_enantioselectivity.get_aai_encoding(aa_indices1)

        self.assertIsInstance(aai_encoding_enantioselectivity, np.ndarray,
            f'AAI Encoding output expected to be a numpy array, got datatype {type(aai_encoding_enantioselectivity)}.')
        self.assertEqual(aai_encoding_enantioselectivity.shape[0], test_pySAR_enantioselectivity.num_seqs,
            f'The number of sequences in the dataset expected to be {test_pySAR_enantioselectivity.num_seqs}, got {aai_encoding_enantioselectivity.shape[0]}.')
        self.assertEqual(aai_encoding_enantioselectivity.shape[1], test_pySAR_enantioselectivity.sequence_length,
            f'The length of the sequences expected to be {test_pySAR_enantioselectivity.sequence_length}, got {str(aai_encoding_enantioselectivity.shape[1])}.')
        self.assertEqual(aai_encoding_enantioselectivity.dtype, np.float32,
            f'Datatype of elements in numpy array should be of dtype np.float32, got {aai_encoding_enantioselectivity.dtype}.')
        self.assertTrue((np.array([3.79, 7.25, 10.88, 7.21, 2.93, 10.88, 6.11, 2.93, 7.21, 7.25],
            dtype=np.float32) == aai_encoding_enantioselectivity[0][:10]).all(),
                f'The first 10 elements of the 1st sequence do not match what was expected:\n{aai_encoding_enantioselectivity[0][:10]}.')
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        aai_encoding_absorption = test_pySAR_absorption.get_aai_encoding(aa_indices2)

        self.assertIsInstance(aai_encoding_absorption, np.ndarray,
            f'AAI Encoding output expected to be a numpy array, got datatype {type(aai_encoding_absorption)}.')
        self.assertEqual(aai_encoding_absorption.shape[0], test_pySAR_absorption.num_seqs,
            f'The number of sequences in the dataset expected to be {test_pySAR_absorption.num_seqs}, got {aai_encoding_absorption.shape[0]}.')
        self.assertEqual(aai_encoding_absorption.shape[1], test_pySAR_absorption.sequence_length,
            f'The length of the sequences expected to be {test_pySAR_absorption.sequence_length}, got {str(aai_encoding_absorption.shape[1])}.')
        self.assertEqual(aai_encoding_absorption.dtype, np.float32,
            f'Datatype of elements in numpy array should be of dtype np.float32, got {aai_encoding_absorption.dtype}.')
        self.assertTrue((np.array([14.9, 17.6, 14.9, 9.5, 14.3, 18.8, 6.9, 6.9, 9.9, 14.8],
            dtype=np.float32)==aai_encoding_absorption[0][:10]).all(),
                f'The first 10 elements of the 1st sequence do not match what was expected:\n{aai_encoding_absorption[0][:10]}.')
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        aai_encoding_localization = test_pySAR_localization.get_aai_encoding(aa_indices3)

        self.assertIsInstance(aai_encoding_localization, np.ndarray,
            f'AAI Encoding output expected to be a numpy array, got datatype {type(aai_encoding_localization)}.')
        self.assertEqual(aai_encoding_localization.shape[0], test_pySAR_localization.num_seqs,
            f'The number of sequences in the dataset expected to be {test_pySAR_localization.num_seqs}, got {aai_encoding_localization.shape[0]}.')
        self.assertEqual(aai_encoding_localization.shape[1], test_pySAR_localization.sequence_length * 2,
            f'The length of the sequences expected to be {test_pySAR_localization.sequence_length}, got {str(aai_encoding_localization.shape[1])}.')
        self.assertEqual(aai_encoding_localization.dtype, np.float32,
            f'Datatype of elements in numpy array should be of dtype np.float32, got {aai_encoding_localization.dtype}.')
        self.assertTrue((np.array([1.47, 0.77, 1.04, 1.22, 1.05, 1.32, 1.32, 0.77, 1.02, 1.22],
            dtype=np.float32)==aai_encoding_localization[0][:10]).all(),
                f'The first 10 elements of sequence 0 do not match what was expected:\n{aai_encoding_localization[0][:10]}.')
#4.)    
        with self.assertRaises(ValueError, msg='ValueError: Errorneous indices have been input.'):
            test_pySAR_thermostability.get_aai_encoding(error_aaindices)
            test_pySAR_thermostability.get_aai_encoding(error_aaindices1)
#5.)
        with self.assertRaises(TypeError, msg='TypeError: Errorneous indices datatypes have been input.'):
            test_pySAR_enantioselectivity.get_aai_encoding(1235)
            test_pySAR_localization.get_aai_encoding(40.89)
            test_pySAR_absorption.get_aai_encoding(False)
    
    def test_aai_encoding(self): 
        """ Testing AAI encoding pipeline. """ 
        aa_indices_1 = "NAKH920102"
        aa_indices_2 = "CHOP780207, GEIM800104"
        aa_indices_3 = ["CHAM810101, ISOY800103"]
        aa_indices_4 = ["PTIO830101", "QIAN880136", "RACS820110"]
        error_aaindices = ["ABCD1234", "ABCD12345"]
        error_aaindices1 = "XYZ4567"
        expected_output_cols = ['Index', 'Category', 'R2', 'RMSE', 'MSE', 
            'RPD', 'MAE', 'Explained Variance']
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        test_aai_thermostability = test_pySAR_thermostability.encode_aai(aai_indices=aa_indices_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_aai_thermostability, pd.DataFrame, 
            f'Expected output to be a DataFrame, got {type(test_aai_thermostability)}.')
        self.assertEqual(len(test_aai_thermostability.columns), 8, 
            f"Expected 8 columns in dataframe output, got {len(test_aai_thermostability.columns)}.")
        self.assertEqual(test_aai_thermostability['Index'].values[0], "NAKH920102", 
            f"Index codes in ouput dataframe don't match expected: {test_aai_thermostability['Index']}.")
        self.assertEqual(test_aai_thermostability['Category'].values[0], "composition",
            f"Category names in ouput dataframe don't match expected: {test_aai_thermostability['Category']}.")
        self.assertEqual(test_pySAR_thermostability.feature_space, (261, 466),
            f"Expected feature space dimensions to be 261 x 466, got {test_pySAR_thermostability.feature_space}.")  
        for col in test_aai_thermostability.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_thermostability[col])}.")  
        _aai_output_dirs = sorted(glob.glob(self.test_output_folder + "_*"))
        self.assertTrue(len(_aai_output_dirs) > 0,
            f"Output dir storing encoding results not found (pattern: {self.test_output_folder}_*).")
        _aai_output_dir = _aai_output_dirs[-1]
        self.assertTrue(os.path.isfile(os.path.join(_aai_output_dir, "aai_results.csv")),
            f"Output csv storing encoding results not found in: {_aai_output_dir}.")
        self.assertTrue(os.path.isfile(os.path.join(_aai_output_dir, "model_regression_plot.png")),
            f"Output regression plot not found in: {_aai_output_dir}.")
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_aai_enantioselectivity = test_pySAR_enantioselectivity.encode_aai(aai_indices=aa_indices_2, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_enantioselectivity, pd.DataFrame, 
            f'Output should be a DataFrame, got {type(test_aai_enantioselectivity)}.')
        self.assertEqual(len(test_aai_enantioselectivity.columns), 8, 
            f"Expected 8 columns in dataframe output, got {len(test_aai_enantioselectivity.columns)}.")
        self.assertEqual(test_aai_enantioselectivity['Index'].values[0], "CHOP780207, GEIM800104", 
            f"Index codes in ouput dataframe don't match expected: {test_aai_enantioselectivity['Index']}.")
        self.assertEqual(test_aai_enantioselectivity['Category'].values[0], "sec_struct, sec_struct",
            f"Category names in ouput dataframe don't match expected: {test_aai_enantioselectivity['Category']}.")
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 796),
            f"Expected feature space dimensions to be 152 x 796, got {test_pySAR_enantioselectivity.feature_space}.")  
        for col in test_aai_enantioselectivity.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_enantioselectivity[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_enantioselectivity[col])}.")  
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[2]) #localiztion
        test_aai_localization = test_pySAR_localization.encode_aai(aai_indices=aa_indices_3, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_localization, pd.DataFrame, 
            f'Output should be a DataFrame, got {type(test_aai_localization)}.')
        self.assertEqual(len(test_aai_localization.columns), 8, 
            f"Expected 8 columns in dataframe output, got {len(test_aai_localization.columns)}.")
        self.assertEqual(test_aai_localization['Index'].values[0], "CHAM810101, ISOY800103", 
            f"Index codes in ouput dataframe don't match expected: {test_aai_localization['Index']}.")
        self.assertEqual(test_aai_localization['Category'].values[0], "geometry, sec_struct",
            f"Category names in ouput dataframe don't match expected: {test_aai_localization['Category']}.")
        self.assertEqual(test_pySAR_localization.feature_space, (81, 596),
            f"Expected feature space dimensions to be 81 x 596, got {test_pySAR_localization.feature_space}.")  
        for col in test_aai_localization.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_localization[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_localization[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_localization[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_localization[col])}.")  
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[3]) #absorption
        test_aai_absorption = test_pySAR_absorption.encode_aai(aai_indices=aa_indices_4, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_absorption, pd.DataFrame, 
            f'Output should be a DataFrame, got {type(test_aai_absorption)}.')
        self.assertEqual(len(test_aai_absorption.columns), 8, 
            f"Expected 8 columns in dataframe output, got {len(test_aai_absorption.columns)}.")
        self.assertEqual(test_aai_absorption['Index'].values[0], "PTIO830101, QIAN880136, RACS820110", 
            f"Index codes in ouput dataframe don't match expected: {test_aai_absorption['Index']}.")
        self.assertEqual(test_aai_absorption['Category'].values[0], "sec_struct, sec_struct, geometry",
            f"Category names in ouput dataframe don't match expected: {test_aai_absorption['Category']}.")
        self.assertEqual(test_pySAR_absorption.feature_space, (254, 1083),
            f"Expected feature space dimensions to be 254 x 1083, got {test_pySAR_absorption.feature_space}.")  
        for col in test_aai_absorption.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_absorption[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_absorption[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_absorption[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_absorption[col])}.")           
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Indices parameter cannot be None, an empty string or an invalid AAI record number.'):
            test_pySAR_thermostability.encode_aai(aai_indices=None)
            test_pySAR_thermostability.encode_aai(aai_indices="")
            test_pySAR_enantioselectivity.encode_aai()
            test_pySAR_enantioselectivity.encode_aai(aai_indices=error_aaindices)
            test_pySAR_localization.encode_aai(aai_indices=error_aaindices1)
#6.)
        with self.assertRaises(TypeError, msg='TypeError: Indices must be lists or strings.'):
            test_pySAR_localization.encode_aai(aai_indices=123)
            test_pySAR_localization.encode_aai(aai_indices=0.90)
            test_pySAR_absorption.encode_aai(aai_indices=False)
            test_pySAR_absorption.encode_aai(aai_indices=9000)
    
    def test_get_desc_encoding(self):
        """ Testing Descriptor encoding functionality. """
        desc_1 = "dipeptide_composition"
        desc_2 = "ctd_transition"
        desc_3 = "moranauto, quasi_seq_order"
        all_desc = [desc_1, desc_2, "geary_auto", "sequence_order_coupling_number"]
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        desc_encoding_thermostability = test_pySAR_thermostability.get_descriptor_encoding(desc_1)

        self.assertIsInstance(desc_encoding_thermostability, pd.DataFrame,
            f'Expected encoding output to be a DataFrame, got datatype {type(desc_encoding_thermostability)}.')
        self.assertEqual(desc_encoding_thermostability.shape, (test_pySAR_thermostability.num_seqs, 400),
            f'Expected shape of descriptor encoding expected to be {(test_pySAR_thermostability.num_seqs, 400)}, but got {desc_encoding_thermostability.shape}.')
        for col in list(desc_encoding_thermostability.columns):
            #check all columns follow pattern of XY where x & y are amino acids 
            self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), 
                f"Column doesn't follow correct naming convention: {col}.")
            self.assertIn(col[0], self.amino_acids, 
                f"1st half of column name {col[0]} is not a valid amino acid.")
            self.assertIn(col[1], self.amino_acids, 
                f"2nd half of column name {col[0]} is not a valid amino acid.")
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_thermostability.dtypes)), 
            f"Descriptor values expected to be type np.float64, got:\n{list(desc_encoding_thermostability.dtypes)}.")
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[0]) #enantioselectivity
        desc_encoding_enantioselectivity = test_pySAR_enantioselectivity.get_descriptor_encoding(desc_2)
    
        self.assertIsInstance(desc_encoding_enantioselectivity, pd.DataFrame,
            f'Expected descriptor encoding output to be a DataFrame, got datatype {type(desc_encoding_enantioselectivity)}.')
        self.assertEqual(desc_encoding_enantioselectivity.shape, (test_pySAR_enantioselectivity.num_seqs, 3),
            f'Shape of descriptor encoding expected to be {(test_pySAR_enantioselectivity.num_seqs, 3)}, but got {desc_encoding_enantioselectivity.shape}.')
        for col in list(desc_encoding_enantioselectivity.columns):
            #check all column names follow pattern for CTD descriptor
            self.assertTrue((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_hydrophobicity", col))), 
                f"Column doesn't follow correct naming convention: {col}.")
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_enantioselectivity.dtypes)), 
            f"Descriptor values expected to be type np.float64, got:\n{list(desc_encoding_enantioselectivity.dtypes)}.")
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[0]) #localization
        desc_encoding_localization = test_pySAR_localization.get_descriptor_encoding(desc_3)

        self.assertIsInstance(desc_encoding_localization, pd.DataFrame,
            f'Expected descriptor encoding output to be a DataFrame, got datatype {type(desc_encoding_localization)}.')
        self.assertEqual(desc_encoding_localization.shape, (test_pySAR_localization.num_seqs, 240+50), #MAuto dim + QSO dim
            f'Shape of descriptor encoding expected to be {(test_pySAR_localization.num_seqs, 240+50)}, but got {desc_encoding_localization.shape}.')
        #check all column names follow pattern for MAuto + QSO descriptors
        for col in list(desc_encoding_localization.columns):
            self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r"QSO_SW[0-9]", col)) \
                or bool(re.match(r"QSO_SW[0-9][0-9]", col)), 
                    f"Column doesn't follow correct naming convention: {col}.")
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_localization.dtypes)), 
            f"Descriptor values expected to be type np.float64, got:\n{list(desc_encoding_localization.dtypes)}.")
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[0]) #absorption
        desc_encoding_absorption = test_pySAR_absorption.get_descriptor_encoding(all_desc)

        self.assertIsInstance(desc_encoding_absorption, pd.DataFrame,
            f'Expected descriptor encoding output to be a DataFrame, got datatype {type(desc_encoding_absorption)}.')
        self.assertEqual(desc_encoding_absorption.shape, (test_pySAR_absorption.num_seqs, 400+3+240+30), #DPComp dim + CTD_T dim + Gauto dim + QSO dim
            f'Shape of descriptor encoding expected to be {(test_pySAR_absorption.num_seqs, 400+3+240+30)}, but got {desc_encoding_absorption.shape}.')
        #check all column names follow pattern for DPComp + CTD_T + Gauto + QSO descriptors
        for col in list(desc_encoding_absorption.columns): 
            self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r'^[A-Z]{2}$', col)) or 
                bool(re.match(r"SOCN_SW[0-9]", col)) or bool(re.match(r"QSO_SW[0-9][0-9]", col)) or                 
                    bool(re.match(r"CTD_T_[0-9]_hydrophobicity", col)) or bool(re.match(r"CTD_T_[0-9]{2}_hydrophobicity", col)),  
                        f"Column doesn't follow correct naming convention: {col}.")
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_absorption.dtypes)), 
            f"Descriptor values expected to be type np.float64, got:\n{list(desc_encoding_absorption.dtypes)}.")
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor input parameter cannot be None.'):
            test_pySAR_thermostability.get_descriptor_encoding(descriptors=None)
            test_pySAR_enantioselectivity.get_descriptor_encoding(descriptors="")
            test_pySAR_enantioselectivity.get_descriptor_encoding(descriptors=[])
#6.)
        with self.assertRaises(TypeError, msg='ValueError: Descriptor input parameter cannot be an invalid descriptor name.'):
            test_pySAR_localization.get_descriptor_encoding(descriptor=123)
            test_pySAR_localization.get_descriptor_encoding(descriptor=0.90)
            test_pySAR_absorption.get_descriptor_encoding(descriptor=False)
            test_pySAR_absorption.get_descriptor_encoding(descriptor=9000)

    def test_desc_encoding(self):  #*rewrite and exapnd tests
        """ Testing Descriptor encoding pipeline. """
        desc_1 = "dipeptide_composition"
        desc_2 = "ctd_distribution"
        desc_3 = "seq_order_coupling_number"
        desc_4 = "moranauto, quasi_seq_order"
        expected_output_cols = ['Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 'RPD', 'MAE', 'Explained Variance']
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        test_desc_thermostability = test_pySAR_thermostability.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_thermostability, pd.DataFrame, f'Expected output to be a DataFrame, got {type(test_desc_thermostability)}.')
        self.assertEqual(len(test_desc_thermostability), 1, f"Expected 1 row in encoding output, got {len(test_desc_thermostability)}.")
        for col in test_desc_thermostability.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.") 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_desc_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_desc_thermostability[col])}.") 
        self.assertEqual(test_pySAR_thermostability.feature_space, (261, 400),
            f"Expected feature space dimensions to be 261 x 466, got {test_pySAR_thermostability.feature_space}.")  
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_desc_enantioselectivity = test_pySAR_enantioselectivity.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_enantioselectivity, pd.DataFrame, f'Expected output to be a DataFrame, got {type(test_desc_enantioselectivity)}.')
        self.assertEqual(len(test_desc_enantioselectivity), 1, f"Expected 1 row in encoding output, got {len(test_desc_enantioselectivity)}.")
        for col in test_desc_enantioselectivity.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.") 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type string got {type(test_desc_enantioselectivity[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_desc_enantioselectivity[col])}.") 
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 400),
            f"Expected feature space dimensions to be 152 x 400, got {test_pySAR_enantioselectivity.feature_space}.")  
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        test_desc_absorption = test_pySAR_absorption.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_absorption, pd.DataFrame, f'Expected output to be a DataFrame, got {type(test_desc_absorption)}.')
        self.assertEqual(len(test_desc_absorption), 1, f"Expected 1 row in encoding output, got {len(test_desc_absorption)}.")
        for col in test_desc_absorption.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.") 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_absorption[col].values)),
                    f"Column {col} expected to be of type string got {type(test_desc_absorption[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_absorption[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_desc_absorption[col])}.") 
        self.assertEqual(test_pySAR_absorption.feature_space, (81, 400),
            f"Expected feature space dimensions to be 81 x 400, got {test_pySAR_absorption.feature_space}.") 
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        test_desc_localization = test_pySAR_localization.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_localization, pd.DataFrame, f'Expected output to be a DataFrame, got {type(test_desc_localization)}.')
        self.assertEqual(len(test_desc_localization), 1, f"Expected 1 row in encoding output, got {len(test_desc_localization)}.")
        for col in test_desc_localization.columns:
            self.assertIn(col, expected_output_cols, 
                f"Col {col} not found in list of expected columns:\n{expected_output_cols}.") 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_localization[col].values)),
                    f"Column {col} expected to be of type string got {type(test_desc_localization[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_localization[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_desc_localization[col])}.") 
        self.assertEqual(test_pySAR_localization.feature_space, (254, 400),
            f"Expected feature space dimensions to be 254 x 400, got {test_pySAR_localization.feature_space}.") 

        _desc_output_dirs = sorted(glob.glob(self.test_output_folder + "_*"))
        self.assertTrue(len(_desc_output_dirs) > 0,
            f"Output dir storing encoding results not found (pattern: {self.test_output_folder}_*).")
        _desc_output_dir = _desc_output_dirs[-1]
        self.assertTrue(os.path.isfile(os.path.join(_desc_output_dir, "desc_results.csv")),
            f"Output csv storing encoding results not found in: {_desc_output_dir}.")
        self.assertTrue(os.path.isfile(os.path.join(_desc_output_dir, "model_regression_plot.png")),
            f"Output regression plot not found in: {_desc_output_dir}.")
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor parameter cannot be None or an empty string.'):
            test_pySAR_thermostability.encode_descriptor(descriptors=None)
            test_pySAR_thermostability.encode_descriptor(descriptors="")
            test_pySAR_enantioselectivity.encode_descriptor(descriptors="invalid_descriptor")
            test_pySAR_enantioselectivity.encode_descriptor(descriptors="blahblahblah")
#6.)    
        with self.assertRaises(TypeError, msg='TypeError: Descriptor parameter has to be a strong or list.'):
            test_pySAR_absorption.encode_descriptor(descriptors=123)
            test_pySAR_absorption.encode_descriptor(descriptors=0.90)
            test_pySAR_localization.encode_descriptor(descriptors=False)
            test_pySAR_localization.encode_descriptor(descriptors=9000)

    def test_aai_desc_encoding(self):
        """ Testing AAI + Descriptor encoding functionality. """
        aa_indices_1 = "CHAM810101"
        aa_indices_2 = "NAKH920102"
        aa_indices_3 = "LIFS790103"
        aa_indices_4 = ["PTIO830101", "QIAN880136", "RACS820110"]
        desc_1 = "amino_acid_composition"
        desc_2 = "ctd_distribution"
        desc_3 = "conjoint_triad"
        desc_4 = ["moran_auto", "quasi_seq_order"]
        expected_output_cols = ['Descriptor', 'Group', 'Index', 'Category', 'R2', 'RMSE', 'MSE', 
            'RPD', 'MAE', 'Explained Variance']
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        test_aai_desc = test_pySAR_thermostability.encode_aai_descriptor(descriptors=desc_1, aai_indices=aa_indices_1, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 
            f'Expected output to be a DataFrame, got {type(test_aai_desc)}.')
        self.assertEqual(len(test_aai_desc.columns), 10, 
            f"Expected 10 columns in output dataframe, got {len(test_aai_desc)}.")
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                f"Column {col} not found in list of expected columns:\n{expected_output_cols}")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_desc[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_desc[col])}.")    
            self.assertEqual(test_pySAR_thermostability.feature_space, (261, 486),
                f"Expected feature space dimensions to be 261 x 486, got {test_pySAR_thermostability.feature_space}.")  
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_aai_desc = test_pySAR_enantioselectivity.encode_aai_descriptor(descriptors=desc_2, aai_indices=aa_indices_2, print_results=0, output_folder=self.test_output_folder)
 
        self.assertIsInstance(test_aai_desc, pd.DataFrame,   #**add more tests , directly testing output of columns 
            f'Output expected to be a DataFrame, got {type(test_aai_desc)}.')
        self.assertEqual(len(test_aai_desc.columns), 10, 
            f"Expected 10 columns in output dataframe, got {len(test_aai_desc)}.")
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                f"Column {col} not found in list of expected columns:\n{expected_output_cols}")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_desc[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_desc[col])}.")    
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 413),
            f"Expected test_pySAR_enantioselectivity space dimensions to be 152 x 413, got {test_pySAR_enantioselectivity.feature_space}.")  
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[2]) #localization
        test_aai_desc = test_pySAR_localization.encode_aai_descriptor(descriptors=desc_3, aai_indices=aa_indices_3, print_results=0, output_folder=self.test_output_folder)
      
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 
            f'Output expected to be a DataFrame, got {type(test_aai_desc)}.')
        self.assertEqual(len(test_aai_desc.columns), 10, 
            f"Expected 10 columns in output dataframe, got {len(test_aai_desc)}.")
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                f"Column {col} not found in list of expected columns:\n{expected_output_cols}")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_desc[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_desc[col])}.")    
        self.assertEqual(test_pySAR_localization.feature_space, (81, 641),
            f"Expected feature space dimensions to be 81 x 641, got {test_pySAR_localization.feature_space}.")  
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[3]) #absorption
        test_aai_desc = test_pySAR_absorption.encode_aai_descriptor(descriptors=desc_4, aai_indices=aa_indices_4, print_results=0, output_folder=self.test_output_folder)
       
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 
            f'Output expected to be a DataFrame, got {type(test_aai_desc)}.')
        self.assertEqual(len(test_aai_desc.columns), 10, 
            f"Expected 10 columns in output dataframe, got {len(test_aai_desc)}.")
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                f"Column {col} not found in list of expected columns:\n{expected_output_cols}")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type string got {type(test_aai_desc[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_aai_desc[col])}.")     
        _aai_desc_output_dirs = sorted(glob.glob(self.test_output_folder + "_*"))
        self.assertTrue(len(_aai_desc_output_dirs) > 0,
            f"Output dir storing encoding results not found (pattern: {self.test_output_folder}_*).")
        _aai_desc_output_dir = _aai_desc_output_dirs[-1]
        self.assertTrue(os.path.isfile(os.path.join(_aai_desc_output_dir, "aai_desc_results.csv")),
            f"Output csv storing encoding results not found in: {_aai_desc_output_dir}.")
        self.assertTrue(os.path.isfile(os.path.join(_aai_desc_output_dir, "model_regression_plot.png")),
            f"Output regression plot not found in: {_aai_desc_output_dir}.")
        self.assertEqual(test_pySAR_absorption.feature_space, (254, 1373),
            f"Expected feature space dimensions to be 254 x 1373, got {test_pySAR_absorption.feature_space}.")  
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor and indices parameter cannot both be None or an empty string.'):
            test_pySAR_thermostability.encode_aai_descriptor(descriptors=None)
            test_pySAR_thermostability.encode_aai_descriptor(aai_indices=None)
            test_pySAR_thermostability.encode_aai_descriptor(descriptors="aa_comp")
            test_pySAR_thermostability.encode_aai_descriptor(aai_indices="LIFS790103")
            test_pySAR_enantioselectivity.encode_aai_descriptor(aai_indices=None, descriptors=None)
            test_pySAR_enantioselectivity.encode_aai_descriptor(aai_indices="", descriptors="")
            test_pySAR_enantioselectivity.encode_aai_descriptor(descriptors="invalid_descriptor")
            test_pySAR_enantioselectivity.encode_aai_descriptor(aai_indices="invalid_value")
            test_pySAR_localization.encode_aai_descriptor(descriptors="descriptor not found")
            test_pySAR_localization.encode_aai_descriptor(aai_indices="blahblahblah")
#6.)
        with self.assertRaises(TypeError, msg='ValueError: Descriptor and indices must be lists or strings.'):
            test_pySAR_localization.encode_aai_descriptor(descriptors=123, aai_indices=123)
            test_pySAR_localization.encode_aai_descriptor(descriptors=0000, aai_indices=0.90)
            test_pySAR_absorption.encode_aai_descriptor(descriptors=False, aai_indices=True)
            test_pySAR_absorption.encode_aai_descriptor(descriptors=2.9, aai_indices=9000)

    def test_predict_activity(self):
        """Testing predict_activity() after each of the three encoding strategies."""
        import warnings
#1.)    predict_activity() raises RuntimeError before any encode_* call
        fresh = pysar.PySAR(config_file=self.all_config_files[0])
        with self.assertRaises(RuntimeError,
                msg='RuntimeError expected when predict_activity called before any encode_* method.'):
            fresh.predict_activity("ACDEFGHIKLMNPQRSTVWY")
#2.)    AAI encoding strategy
        test_aai = pysar.PySAR(config_file=self.all_config_files[0])
        test_aai.encode_aai(aai_indices="NAKH920102", print_results=0,
                            output_folder=self.test_output_folder)
        # Use the first two sequences from the dataset as unseen inputs
        test_seqs = list(test_aai.sequences[:2])
        preds = test_aai.predict_activity(test_seqs)
        self.assertIsInstance(preds, np.ndarray,
            f"predict_activity() should return np.ndarray, got {type(preds)}.")
        self.assertEqual(len(preds), 2,
            f"Expected 2 predictions, got {len(preds)}.")
#3.)    Single string input returns length-1 array
        single_pred = test_aai.predict_activity(test_seqs[0])
        self.assertIsInstance(single_pred, np.ndarray,
            "predict_activity() with a single string should still return np.ndarray.")
        self.assertEqual(len(single_pred), 1,
            f"Expected 1 prediction for single-sequence input, got {len(single_pred)}.")
#4.)    Descriptor encoding strategy
        test_desc = pysar.PySAR(config_file=self.all_config_files[0])
        test_desc.encode_descriptor(descriptors="amino_acid_composition", print_results=0,
                                    output_folder=self.test_output_folder)
        desc_seqs = list(test_desc.sequences[:2])
        preds_desc = test_desc.predict_activity(desc_seqs)
        self.assertIsInstance(preds_desc, np.ndarray,
            f"predict_activity() (descriptor) should return np.ndarray, got {type(preds_desc)}.")
        self.assertEqual(len(preds_desc), 2,
            f"Expected 2 predictions from descriptor strategy, got {len(preds_desc)}.")
#5.)    Invalid sequences raise ValueError
        with self.assertRaises(ValueError,
                msg='ValueError expected for sequences containing invalid amino acids.'):
            test_aai.predict_activity(["ZZZZZZZZZZ"])

    def test_predict_activity_uncertainty(self):
        """predict_activity(return_uncertainty=True) returns (preds, std) for GPR models."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        # Build a PySAR instance with a GPR model by overriding algorithm
        test_gpr = pysar.PySAR(config_file=self.all_config_files[0])
        # Override algorithm to GaussianProcessRegressor
        test_gpr.algorithm = 'GaussianProcessRegressor'
        test_gpr.encode_aai(aai_indices="NAKH920102", print_results=0,
                            output_folder=self.test_output_folder)
        # Verify model is a GPR
        if not isinstance(test_gpr.model.model_fit, GaussianProcessRegressor):
            self.skipTest("Model is not a GaussianProcessRegressor — test N/A for this config.")
        test_seqs = list(test_gpr.sequences[:2])
        result = test_gpr.predict_activity(test_seqs, return_uncertainty=True)
        self.assertIsInstance(result, tuple,
            "return_uncertainty=True with GPR should return a tuple (preds, std).")
        preds, std = result
        self.assertIsInstance(preds, np.ndarray, "Predictions should be np.ndarray.")
        self.assertIsInstance(std, np.ndarray, "Uncertainty std should be np.ndarray.")
        self.assertEqual(len(preds), 2, f"Expected 2 predictions, got {len(preds)}.")
        self.assertEqual(len(std), 2, f"Expected 2 std values, got {len(std)}.")
        self.assertTrue(np.all(std >= 0), "Standard deviations must be non-negative.")

    def test_encode_aai_random_state_cv(self):
        """encode_aai() accepts random_state and cv parameters without raising errors."""
        test_obj = pysar.PySAR(config_file=self.all_config_files[0])
        result = test_obj.encode_aai(aai_indices="NAKH920102", print_results=0,
                                     output_folder=self.test_output_folder,
                                     random_state=42, cv=3)
        self.assertIsInstance(result, pd.DataFrame,
            "encode_aai() with random_state/cv should still return a DataFrame.")

    def test_encode_descriptor_random_state_cv(self):
        """encode_descriptor() accepts random_state and cv parameters without raising errors."""
        test_obj = pysar.PySAR(config_file=self.all_config_files[0])
        result = test_obj.encode_descriptor(descriptors="amino_acid_composition",
                                            print_results=0,
                                            output_folder=self.test_output_folder,
                                            random_state=42, cv=3)
        self.assertIsInstance(result, pd.DataFrame,
            "encode_descriptor() with random_state/cv should still return a DataFrame.")

    def test_encode_aai_descriptor_random_state_cv(self):
        """encode_aai_descriptor() accepts random_state and cv parameters without raising errors."""
        test_obj = pysar.PySAR(config_file=self.all_config_files[0])
        result = test_obj.encode_aai_descriptor(aai_indices="NAKH920102",
                                                descriptors="amino_acid_composition",
                                                print_results=0,
                                                output_folder=self.test_output_folder,
                                                random_state=42, cv=3)
        self.assertIsInstance(result, pd.DataFrame,
            "encode_aai_descriptor() with random_state/cv should still return a DataFrame.")

    def test_logger_parameter(self):
        """PySAR accepts a custom logger; output_results uses it instead of print."""
        import logging
        import io
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("test_pysar_logger")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        logger.addHandler(handler)
        test_obj = pysar.PySAR(config_file=self.all_config_files[0], logger=logger)
        self.assertIs(test_obj.logger, logger,
            "PySAR should store the logger passed to __init__.")
        # Run encoding so output_results is called
        test_obj.encode_aai(aai_indices="NAKH920102", print_results=True,
                            output_folder=self.test_output_folder)
        captured = log_stream.getvalue()
        self.assertIn("R2", captured,
            "Logger should have captured 'R2' from output_results.")

    def test_save_and_load_session(self):
        """save_session() and load_session() round-trip a fitted PySAR instance."""
        import tempfile
        test_obj = pysar.PySAR(config_file=self.all_config_files[0])
        test_obj.encode_aai(aai_indices="NAKH920102", print_results=0,
                            output_folder=self.test_output_folder)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            session_path = tmp.name
        try:
            # Save session
            test_obj.save_session(session_path)
            self.assertTrue(os.path.isfile(session_path),
                "save_session() should create a .pkl file.")
            # Load session
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                loaded = pysar.PySAR.load_session(session_path)
            self.assertIsInstance(loaded, pysar.PySAR,
                "load_session() should return a PySAR instance.")
            # Predictions from loaded session should match original
            test_seqs = list(test_obj.sequences[:2])
            preds_orig = test_obj.predict_activity(test_seqs)
            preds_loaded = loaded.predict_activity(test_seqs)
            np.testing.assert_array_almost_equal(preds_orig, preds_loaded,
                err_msg="Predictions from loaded session should match original.")
        finally:
            if os.path.isfile(session_path):
                os.remove(session_path)

    def test_load_session_allow_pickle_false(self):
        """load_session(allow_pickle=False) raises ValueError."""
        with self.assertRaises(ValueError,
                msg="load_session(allow_pickle=False) should raise ValueError."):
            pysar.PySAR.load_session("dummy.pkl", allow_pickle=False)

    def test_load_session_missing_file(self):
        """load_session() raises FileNotFoundError for non-existent path."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises(FileNotFoundError,
                    msg="load_session() should raise FileNotFoundError for missing file."):
                pysar.PySAR.load_session("this_file_definitely_does_not_exist.pkl")

    def tearDown(self):
        """Clean up test outputs and temporary files created during test case."""
        # Remove the main test output folder created in setUp
        if os.path.isdir(self.test_output_folder):
            shutil.rmtree(self.test_output_folder, ignore_errors=False, onerror=None)
        
        # Remove any timestamped output folders created by pySAR
        for _ts_dir in glob.glob(self.test_output_folder + "_*"):
            shutil.rmtree(_ts_dir, ignore_errors=True)
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)