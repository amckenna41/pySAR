################################################################################
##################             pySAR Module Tests             ##################
################################################################################

import pandas as pd
import numpy as np
import os
import shutil
import re
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import pySAR as pysar_
import pySAR.pySAR as pysar
import pySAR.globals_ as _globals

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

        #create temporary unit test output folder to store any pysar assets and results
        self.test_output_folder = os.path.join("tests", "test_outputs")
        if not (os.path.isdir(self.test_output_folder)):
            os.makedirs(self.test_output_folder)

    @unittest.skip("Skipping metadata tests.")
    def test_pySAR_metadata(self):
        """ Testing correct pySAR version and metadata. """
        self.assertEqual(pysar_.__version__, "2.4.1", 
            "pySAR version is not correct, got: {}.".format(pysar_.__version__))
        self.assertEqual(pysar_.__name__, "pySAR", 
            "pySAR software name is not correct, got: {}.".format(pysar_.__name__))
        self.assertEqual(pysar_.__author__, "AJ McKenna, https://github.com/amckenna41", 
            "pySAR author is not correct, got: {}.".format(pysar_.__author__))
        self.assertEqual(pysar_.__authorEmail__, "amckenna41@qub.ac.uk", 
            "pySAR author email is not correct, got: {}.".format(pysar_.__authorEmail__))
        self.assertEqual(pysar_.__url__, "https://github.com/amckenna41/pySAR", 
            "pySAR repo URL is not correct, got: {}.".format(pysar_.__url__))
        self.assertEqual(pysar_.__download_url__, "https://github.com/amckenna41/pySAR/archive/refs/heads/main.zip", 
            "pySAR repo download URL is not correct, got: {}.".format(pysar_.__download_url__))
        self.assertEqual(pysar_.__status__, "Production", 
            "pySAR status is not correct, got: {}.".format(pysar_.__status__))
        self.assertEqual(pysar_.__license__, "MIT", 
            "pySAR license type is not correct, got: {}.".format(pysar_.__license__))
        self.assertEqual(pysar_.__maintainer__, "AJ McKenna", 
            "pySAR maintainer is not correct, got: {}.".format(pysar_.__license__))
        self.assertEqual(pysar_.__keywords__, ["bioinformatics", "protein engineering", "python", \
            "pypi", "machine learning", "directed evolution", "drug discovery", "sequence activity relationships", \
            "SAR", "aaindex", "protein descriptors"], "pySAR keywords is not correct, got: {}.".format(pysar_.__keywords__))

    def test_pySAR(self):
        """ Testing pySAR intialisation process and associated methods & attributes. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_thermostability.dataset, (os.path.join('tests', 'test_data', 'test_thermostability.txt')),
            'Dataset attribute does not match expected, got {}.'.format(test_pySAR_thermostability.dataset))
        self.assertEqual(test_pySAR_thermostability.sequence_col, "sequence",
            'Sequence column attribute is not correct, expected sequence, got {}.'.format(test_pySAR_thermostability.sequence_col))
        self.assertEqual(test_pySAR_thermostability.activity_col, "T50",
            "Activity attribute name not correct, expected T50, got {}.".format(test_pySAR_thermostability.activity_col))
        self.assertEqual(test_pySAR_thermostability.algorithm, "plsregression",
            'Algorithm attribute not correct, expected plsregression, got {}.'.format(test_pySAR_thermostability.algorithm))
        self.assertEqual(test_pySAR_thermostability.test_split, 0.2,
            'Test split not expected, expected 0.2, got {}.'.format(test_pySAR_thermostability.test_split))
        self.assertIsNone(test_pySAR_thermostability.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_thermostability.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_thermostability.model_parameters, {},
            'Parameters attribute expected to be empty, got {}.'.format(test_pySAR_thermostability.model_parameters))
        self.assertIsInstance(test_pySAR_thermostability.data, pd.DataFrame,
            'Data expected to be a DataFrame, got {}.'.format(type(test_pySAR_thermostability.data)))
        self.assertIsInstance(test_pySAR_thermostability.sequences, pd.Series,
            'Sequences expected to be a pd.Series, got {}.'.format(type(test_pySAR_thermostability.sequences)))
        self.assertIsInstance(test_pySAR_thermostability.activity, pd.Series,
            'Activity expected to be a pd.Series, got {}.'.format(type(test_pySAR_thermostability.activity)))
        self.assertEqual(test_pySAR_thermostability.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_thermostability.num_seqs, 261,
            'Number of sequences expected to be 261, got {}.'.format(test_pySAR_thermostability.num_seqs))
        self.assertEqual(test_pySAR_thermostability.sequence_length, 466,
            'Sequence length expected to be 466, got {}.'.format(test_pySAR_thermostability.sequence_length))
        self.assertEqual(test_pySAR_thermostability.feature_space, (),
            'Feature space expected to be an empty tuble, got {}.'.format(test_pySAR_thermostability.feature_space))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_enantioselectivity.dataset, (os.path.join('tests', 'test_data', 'test_enantioselectivity.txt')),
            'Dataset attribute does not match expected, got {}.'.format(test_pySAR_enantioselectivity.dataset))
        self.assertEqual(test_pySAR_enantioselectivity.sequence_col, "sequence",
            'Sequence column attribute is not correct, expected sequence, got {}.'.format(test_pySAR_enantioselectivity.sequence_col))
        self.assertEqual(test_pySAR_enantioselectivity.activity_col, "e-value",
            "Activity attribute name not correct, expected e-value, got {}.".format(test_pySAR_enantioselectivity.activity_col))
        self.assertEqual(test_pySAR_enantioselectivity.algorithm, "randomforestregressor",
            'Algorithm attribute not correct, expected randomforestregressor, got {}.'.format(test_pySAR_enantioselectivity.algorithm))
        self.assertEqual(test_pySAR_enantioselectivity.test_split, 0.2,
            'Test split not expected, expected 0.2, got {}.'.format(test_pySAR_enantioselectivity.test_split))
        self.assertIsNone(test_pySAR_enantioselectivity.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_enantioselectivity.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_enantioselectivity.model_parameters, {},
            'Parameters attribute expected to be empty, got {}.'.format(test_pySAR_enantioselectivity.model_parameters))
        self.assertIsInstance(test_pySAR_enantioselectivity.data, pd.DataFrame,
            'Data expected to be a DataFrame, got {}.'.format(type(test_pySAR_enantioselectivity.data)))
        self.assertIsInstance(test_pySAR_enantioselectivity.sequences, pd.Series,
            'Sequences expected to be a pd.Series, got {}.'.format(type(test_pySAR_enantioselectivity.sequences)))
        self.assertIsInstance(test_pySAR_enantioselectivity.activity, pd.Series,
            'Activity expected to be a pd.Series, got {}.'.format(type(test_pySAR_enantioselectivity.activity)))
        self.assertEqual(test_pySAR_enantioselectivity.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_enantioselectivity.num_seqs, 152,
            'Number of sequences expected to be 152, got {}.'.format(test_pySAR_enantioselectivity.num_seqs))
        self.assertEqual(test_pySAR_enantioselectivity.sequence_length, 398,
            'Sequence length expected to be 398, got {}.'.format(test_pySAR_enantioselectivity.sequence_length))
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (),
            'Feature space expected to be an empty tuble, got {}.'.format(test_pySAR_enantioselectivity.feature_space))
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_absorption.dataset, (os.path.join('tests', 'test_data', 'test_absorption.txt')),
            'Dataset attribute does not match expected, got {}.'.format(test_pySAR_absorption.dataset))
        self.assertEqual(test_pySAR_absorption.sequence_col, "sequence",
            'Sequence column attribute is not correct, expected sequence, got {}.'.format(test_pySAR_absorption.sequence_col))
        self.assertEqual(test_pySAR_absorption.activity_col, "peak",
            "Activity attribute name not correct, expected peak, got {}.".format(test_pySAR_absorption.activity_col))
        self.assertEqual(test_pySAR_absorption.algorithm, "knn",
            'Algorithm attribute not correct, expected knn, got {}.'.format(test_pySAR_absorption.algorithm))
        self.assertEqual(test_pySAR_absorption.test_split, 0.2,
            'Test split not expected, expected 0.2, got {}.'.format(test_pySAR_absorption.test_split))
        self.assertIsNone(test_pySAR_absorption.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_absorption.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_absorption.model_parameters, {},
            'Parameters attribute expected to be empty, got {}.'.format(test_pySAR_absorption.model_parameters))
        self.assertIsInstance(test_pySAR_absorption.data, pd.DataFrame,
            'Data expected to be a DataFrame, got {}.'.format(type(test_pySAR_absorption.data)))
        self.assertIsInstance(test_pySAR_absorption.sequences, pd.Series,
            'Sequences expected to be a pd.Series, got {}.'.format(type(test_pySAR_absorption.sequences)))
        self.assertIsInstance(test_pySAR_absorption.activity, pd.Series,
            'Activity expected to be a pd.Series, got {}.'.format(type(test_pySAR_absorption.activity)))
        self.assertEqual(test_pySAR_absorption.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_absorption.num_seqs, 81,
            'Number of sequences expected to be 81, got {}.'.format(test_pySAR_absorption.num_seqs))
        self.assertEqual(test_pySAR_absorption.sequence_length, 298,
            'Sequence length expected to be 298, got {}.'.format(test_pySAR_absorption.sequence_length))
        self.assertEqual(test_pySAR_absorption.feature_space, (),
            'Feature space expected to be an empty tuble, got {}.'.format(test_pySAR_absorption.feature_space))
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization

        #testing attribute values, including default values
        self.assertEqual(test_pySAR_localization.dataset, (os.path.join('tests', 'test_data', 'test_localization.txt')),
            'Dataset attribute does not match expected, got {}.'.format(test_pySAR_localization.dataset))
        self.assertEqual(test_pySAR_localization.sequence_col, "sequence",
            'Sequence column attribute is not correct, expected sequence, got {}.'.format(test_pySAR_localization.sequence_col))
        self.assertEqual(test_pySAR_localization.activity_col, "log_GFP",
            "Activity attribute name not correct, expected log_GFP, got {}.".format(test_pySAR_localization.activity_col))
        self.assertEqual(test_pySAR_localization.algorithm, "adaboostregressor",
            'Algorithm attribute not correct, expected adaboostregressor, got {}.'.format(test_pySAR_localization.algorithm))
        self.assertEqual(test_pySAR_localization.test_split, 0.2,
            'Test split not expected, expected 0.2, got {}.'.format(test_pySAR_localization.test_split))
        self.assertIsNone(test_pySAR_localization.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR_localization.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR_localization.model_parameters, {},
            'Parameters attribute expected to be empty, got {}.'.format(test_pySAR_localization.model_parameters))
        self.assertIsInstance(test_pySAR_localization.data, pd.DataFrame,
            'Data expected to be a DataFrame, got {}.'.format(type(test_pySAR_localization.data)))
        self.assertIsInstance(test_pySAR_localization.sequences, pd.Series,
            'Sequences expected to be a pd.Series, got {}.'.format(type(test_pySAR_localization.sequences)))
        self.assertIsInstance(test_pySAR_localization.activity, pd.Series,
            'Activity expected to be a pd.Series, got {}.'.format(type(test_pySAR_localization.activity)))
        self.assertEqual(test_pySAR_localization.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR_localization.num_seqs, 254,
            'Number of sequences expected to be 254, got {}.'.format(test_pySAR_localization.num_seqs))
        self.assertEqual(test_pySAR_localization.sequence_length, 361,
            'Sequence length expected to be 361, got {}.'.format(test_pySAR_localization.sequence_length))
        self.assertEqual(test_pySAR_localization.feature_space, (),
            'Feature space expected to be an empty tuble, got {}.'.format(test_pySAR_localization.feature_space))
#5.)
        #validate that if errorneous input parameters are input, that errors are raised
        with self.assertRaises(OSError, msg='OS Error raised, config file not found.'):
            pysar.PySAR(config_file="blahblahblah")
            pysar.PySAR(config_file="test_data/nothing.json")
#6.)
        with self.assertRaises(TypeError, msg='Type Error raised, config file parameter not correct data type.'):
            pysar.PySAR(config_file=101)
            pysar.PySAR(config_file=False)

    def test_sequences(self):
        """ Testing getting the protein sequences from the dataset. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability 
        test_seqs = test_pySAR_thermostability.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_thermostability._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_thermostability._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not of correct type, expected {}, got {}.'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MTIKEMPQPK"),
            'Error in first seqeuence, expected it to start with MTIKEMPQPK.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_seqs = test_pySAR_enantioselectivity.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_enantioselectivity._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_enantioselectivity._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not of correct type, expected {}, got {}.'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MSAPFAKF"),
            'Error in second seqeuence expected it to start with MSAPFAKF.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        test_seqs = test_pySAR_absorption.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_absorption._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_absorption._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not of correct type, expected {}, got {}.'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MLMTVFSSAP"),
            'Error in third seqeuence expected it to start with MLMTVFSSAP.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        test_seqs = test_pySAR_localization.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_localization._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_localization._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not of correct type, expected {}, got {}.'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MSRLVAASWL"),
            'Error in third seqeuence expected it to start with MSRLVAASWL.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
    
    def test_activity(self):
        """ Testing function that gets activity from dataset. """
#1.)
        test_pySAR_thermostability = pysar.PySAR(config_file=self.all_config_files[0]) #thermostability
        activity = test_pySAR_thermostability.activity

        self.assertIsInstance(activity, pd.Series, 
            'Output should be a Series, got {}.'.format(type(activity)))
        self.assertEqual(activity.shape, (test_pySAR_thermostability.num_seqs,), 
            'Output expected to be shape {}, got {}.'.format((test_pySAR_thermostability.num_seqs,), activity.shape))
        self.assertTrue((activity[:10] == np.array([55.0, 43.0, 49.0, 39.8, 52.9, 48.8, 45.0, 48.3, 61.5, 54.6])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity[:10]))
        self.assertEqual(activity.name, "T50", 
            "Expected T50 column name for Series, got {}.".format(activity.name))
        self.assertTrue(activity.dtypes == np.float64, 
            "Column datatypes should be np.float64, got {}.".format(activity.dtypes))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        activity_enantioselectivity = test_pySAR_enantioselectivity.activity

        self.assertIsInstance(activity_enantioselectivity, pd.Series, 
            'Output should be a Series, got {}.'.format(type(activity_enantioselectivity)))
        self.assertEqual(activity_enantioselectivity.shape, (test_pySAR_enantioselectivity.num_seqs,), 
            'Output expected to be shape ({}), got {}.'.format((test_pySAR_enantioselectivity.num_seqs,), activity_enantioselectivity.shape))
        self.assertTrue((activity_enantioselectivity[:10] == np.array([5.0, 23.0, 10.0, 9.0, 12.0, 11.0, 11.0, 21.0, 18.0, 17.0])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity_enantioselectivity[:10]))
        self.assertEqual(activity_enantioselectivity.name, "e-value", 
            "Expected e-value column name for Series, got {}.".format(activity_enantioselectivity.name))
        self.assertTrue(np.float64 == activity_enantioselectivity.dtypes, 
            "Column datatypes should be np.float64, got {}.".format(activity_enantioselectivity.dtypes))
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        activity_absorption = test_pySAR_absorption.activity

        self.assertIsInstance(activity_absorption, pd.Series, 
            'Output should be a Series, got {}.'.format(type(activity_absorption)))
        self.assertEqual(activity_absorption.shape, (test_pySAR_absorption.num_seqs,), 
            'Output expected to be shape ({}), got {}.'.format((test_pySAR_absorption.num_seqs,), activity_absorption.shape))
        self.assertTrue((activity_absorption[:10] == np.array([539, 510, 510, 519, 525, 528, 528, 534, 528, 510])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity_absorption[:10]))
        self.assertEqual(activity_absorption.name, "peak", 
            "Expected peak column name for Series, got {}.".format(activity_absorption.name))
        self.assertTrue(np.int64 == activity_absorption.dtypes, 
            "Column datatypes should be np.float64, got {}.".format(activity_absorption.dtypes))
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        activity_localization = test_pySAR_localization.activity

        self.assertIsInstance(activity_localization, pd.Series, 
            'Output should be a Series, got {}.'.format(type(activity_localization)))
        self.assertEqual(activity_localization.shape, (test_pySAR_localization.num_seqs,), 
            'Output expected to be shape ({}), got {}.'.format((test_pySAR_localization.num_seqs,), activity_localization.shape))
        # self.assertTrue((activity_localization[:10] == np.array([-4.626936, -5.599110, -5.715788, -5.335352, -4.187052, -6.732491, -7.135846, -6.128409, -5.319843, -5.092067])).all(),
        #         "First 10 elements of activity don't match expected output:\n{}.".format(activity_localization[:10]))
        self.assertEqual(activity_localization.name, "log_GFP", 
            "Expected log_GFP column name for Series, got {}.".format(activity_localization.name))
        self.assertTrue(activity_localization.dtypes == np.float64, 
            "Column datatypes should be np.float64, got {}.".format(activity_localization.dtypes))
    
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
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_thermostability)))
        self.assertEqual(aai_encoding_thermostability.shape[0], test_pySAR_thermostability.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_thermostability.num_seqs, aai_encoding_thermostability.shape[0]))
        self.assertEqual(aai_encoding_thermostability.shape[1], test_pySAR_thermostability.sequence_length * len(aa_indices),
            'The length of the sequences expected to be {}, got {}.'.format((test_pySAR_thermostability.sequence_length * len(aa_indices)), str(aai_encoding_thermostability.shape[1])))
        self.assertEqual(aai_encoding_thermostability.dtype, np.float32,
            'Datatype of elements in numpy array expected to be dtype np.float32, got {}.'.format(aai_encoding_thermostability.dtype))
        self.assertTrue((np.array([0.78, 0.5, 1.02, 0.68, 0.68, 0.78, 0.36, 0.68, 0.36, 0.68], 
            dtype=np.float32) == aai_encoding_thermostability[0][:10]).all(),
                'The first 10 elements of the 1st sequence in encoding do not match what was expected:\n{}.'.format(aai_encoding_thermostability[0][:10]))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        aai_encoding_enantioselectivity = test_pySAR_enantioselectivity.get_aai_encoding(aa_indices1)

        self.assertIsInstance(aai_encoding_enantioselectivity, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_enantioselectivity)))
        self.assertEqual(aai_encoding_enantioselectivity.shape[0], test_pySAR_enantioselectivity.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_enantioselectivity.num_seqs, aai_encoding_enantioselectivity.shape[0]))
        self.assertEqual(aai_encoding_enantioselectivity.shape[1], test_pySAR_enantioselectivity.sequence_length,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_enantioselectivity.sequence_length, str(aai_encoding_enantioselectivity.shape[1])))
        self.assertEqual(aai_encoding_enantioselectivity.dtype, np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_enantioselectivity.dtype))
        self.assertTrue((np.array([3.79, 7.25, 10.88, 7.21, 2.93, 10.88, 6.11, 2.93, 7.21, 7.25],
            dtype=np.float32) == aai_encoding_enantioselectivity[0][:10]).all(),
                'The first 10 elements of the 1st sequence do not match what was expected:\n{}.'.format(aai_encoding_enantioselectivity[0][:10]))
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        aai_encoding_absorption = test_pySAR_absorption.get_aai_encoding(aa_indices2)

        self.assertIsInstance(aai_encoding_absorption, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_absorption)))
        self.assertEqual(aai_encoding_absorption.shape[0], test_pySAR_absorption.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_absorption.num_seqs, aai_encoding_absorption.shape[0]))
        self.assertEqual(aai_encoding_absorption.shape[1], test_pySAR_absorption.sequence_length,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_absorption.sequence_length, str(aai_encoding_absorption.shape[1])))
        self.assertEqual(aai_encoding_absorption.dtype, np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_absorption.dtype))
        self.assertTrue((np.array([14.9, 17.6, 14.9, 9.5, 14.3, 18.8, 6.9, 6.9, 9.9, 14.8],
            dtype=np.float32)==aai_encoding_absorption[0][:10]).all(),
                'The first 10 elements of the 1st sequence do not match what was expected:\n{}.'.format(aai_encoding_absorption[0][:10]))
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        aai_encoding_localization = test_pySAR_localization.get_aai_encoding(aa_indices3)

        self.assertIsInstance(aai_encoding_localization, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_localization)))
        self.assertEqual(aai_encoding_localization.shape[0], test_pySAR_localization.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_localization.num_seqs, aai_encoding_localization.shape[0]))
        self.assertEqual(aai_encoding_localization.shape[1], test_pySAR_localization.sequence_length * 2,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_localization.sequence_length, str(aai_encoding_localization.shape[1])))
        self.assertEqual(aai_encoding_localization.dtype, np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_localization.dtype))
        self.assertTrue((np.array([1.47, 0.77, 1.04, 1.22, 1.05, 1.32, 1.32, 0.77, 1.02, 1.22],
            dtype=np.float32)==aai_encoding_localization[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what was expected:\n{}.'.format(aai_encoding_localization[0][:10]))
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
            'Expected output to be a DataFrame, got {}.'.format(type(test_aai_thermostability)))
        self.assertEqual(len(test_aai_thermostability.columns), 8, 
            "Expected 8 columns in dataframe output, got {}.".format(len(test_aai_thermostability.columns)))
        self.assertEqual(test_aai_thermostability['Index'].values[0], "NAKH920102", 
            "Index codes in ouput dataframe don't match expected: {}.".format(test_aai_thermostability["Index"]))
        self.assertEqual(test_aai_thermostability['Category'].values[0], "composition",
            "Category names in ouput dataframe don't match expected: {}.".format(test_aai_thermostability["Category"]))
        self.assertEqual(test_pySAR_thermostability.feature_space, (261, 466),
            "Expected feature space dimensions to be 261 x 466, got {}.".format(test_pySAR_thermostability.feature_space))  
        for col in test_aai_thermostability.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_thermostability[col])))  
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found: {}.".format(self.test_output_folder + "_" + _globals.CURRENT_DATETIME))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_results.csv")),
            "Output csv storing encoding results not found: {}.".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")),
            "Output regression plot not found: {}.".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_aai_enantioselectivity = test_pySAR_enantioselectivity.encode_aai(aai_indices=aa_indices_2, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_enantioselectivity, pd.DataFrame, 
            'Output should be a DataFrame, got {}.'.format(type(test_aai_enantioselectivity)))
        self.assertEqual(len(test_aai_enantioselectivity.columns), 8, 
            "Expected 8 columns in dataframe output, got {}.".format(len(test_aai_enantioselectivity.columns)))
        self.assertEqual(test_aai_enantioselectivity['Index'].values[0], "CHOP780207, GEIM800104", 
            "Index codes in ouput dataframe don't match expected: {}.".format(test_aai_enantioselectivity["Index"]))
        self.assertEqual(test_aai_enantioselectivity['Category'].values[0], "sec_struct, sec_struct",
            "Category names in ouput dataframe don't match expected: {}.".format(test_aai_enantioselectivity["Category"]))
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 796),
            "Expected feature space dimensions to be 152 x 796, got {}.".format(test_pySAR_enantioselectivity.feature_space))  
        for col in test_aai_enantioselectivity.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_enantioselectivity[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_enantioselectivity[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_enantioselectivity[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_enantioselectivity[col])))  
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[2]) #localiztion
        test_aai_localization = test_pySAR_localization.encode_aai(aai_indices=aa_indices_3, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_localization, pd.DataFrame, 
            'Output should be a DataFrame, got {}.'.format(type(test_aai_localization)))
        self.assertEqual(len(test_aai_localization.columns), 8, 
            "Expected 8 columns in dataframe output, got {}.".format(len(test_aai_localization.columns)))
        self.assertEqual(test_aai_localization['Index'].values[0], "CHAM810101, ISOY800103", 
            "Index codes in ouput dataframe don't match expected: {}.".format(test_aai_localization["Index"]))
        self.assertEqual(test_aai_localization['Category'].values[0], "geometry, sec_struct",
            "Category names in ouput dataframe don't match expected: {}.".format(test_aai_localization["Category"]))
        self.assertEqual(test_pySAR_localization.feature_space, (81, 596),
            "Expected feature space dimensions to be 81 x 596, got {}.".format(test_pySAR_localization.feature_space))  
        for col in test_aai_localization.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_localization[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_localization[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_localization[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_localization[col])))  
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[3]) #absorption
        test_aai_absorption = test_pySAR_absorption.encode_aai(aai_indices=aa_indices_4, print_results=0, output_folder=self.test_output_folder)
        self.assertIsInstance(test_aai_absorption, pd.DataFrame, 
            'Output should be a DataFrame, got {}.'.format(type(test_aai_absorption)))
        self.assertEqual(len(test_aai_absorption.columns), 8, 
            "Expected 8 columns in dataframe output, got {}.".format(len(test_aai_absorption.columns)))
        self.assertEqual(test_aai_absorption['Index'].values[0], "PTIO830101, QIAN880136, RACS820110", 
            "Index codes in ouput dataframe don't match expected: {}.".format(test_aai_absorption["Index"]))
        self.assertEqual(test_aai_absorption['Category'].values[0], "sec_struct, sec_struct, geometry",
            "Category names in ouput dataframe don't match expected: {}.".format(test_aai_absorption["Category"]))
        self.assertEqual(test_pySAR_absorption.feature_space, (254, 1083),
            "Expected feature space dimensions to be 254 x 1083, got {}.".format(test_pySAR_absorption.feature_space))  
        for col in test_aai_absorption.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_absorption[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_absorption[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_absorption[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_absorption[col])))           
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
            'Expected encoding output to be a DataFrame, got datatype {}.'.format(type(desc_encoding_thermostability)))
        self.assertEqual(desc_encoding_thermostability.shape, (test_pySAR_thermostability.num_seqs, 400),
            'Expected shape of descriptor encoding expected to be {}, but got {}.'.format((test_pySAR_thermostability.num_seqs, 400), desc_encoding_thermostability.shape))
        for col in list(desc_encoding_thermostability.columns):
            #check all columns follow pattern of XY where x & y are amino acids 
            self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))
            self.assertIn(col[0], self.amino_acids, 
                "1st half of column name {} is not a valid amino acid.".format(col[0]))
            self.assertIn(col[1], self.amino_acids, 
                "2nd half of column name {} is not a valid amino acid.".format(col[0]))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_thermostability.dtypes)), 
            "Descriptor values expected to be type np.float64, got:\n{}.".format(list(desc_encoding_thermostability.dtypes)))
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[0]) #enantioselectivity
        desc_encoding_enantioselectivity = test_pySAR_enantioselectivity.get_descriptor_encoding(desc_2)
    
        self.assertIsInstance(desc_encoding_enantioselectivity, pd.DataFrame,
            'Expected descriptor encoding output to be a DataFrame, got datatype {}.'.format(type(desc_encoding_enantioselectivity)))
        self.assertEqual(desc_encoding_enantioselectivity.shape, (test_pySAR_enantioselectivity.num_seqs, 3),
            'Shape of descriptor encoding expected to be {}, but got {}.'.format((test_pySAR_enantioselectivity.num_seqs, 3), desc_encoding_enantioselectivity.shape))
        for col in list(desc_encoding_enantioselectivity.columns):
            #check all column names follow pattern for CTD descriptor
            self.assertTrue((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_hydrophobicity", col))), 
                "Column doesn't follow correct naming convention: {}.".format(col))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_enantioselectivity.dtypes)), 
            "Descriptor values expected to be type np.float64, got:\n{}.".format(list(desc_encoding_enantioselectivity.dtypes)))
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[0]) #localization
        desc_encoding_localization = test_pySAR_localization.get_descriptor_encoding(desc_3)

        self.assertIsInstance(desc_encoding_localization, pd.DataFrame,
            'Expected descriptor encoding output to be a DataFrame, got datatype {}.'.format(type(desc_encoding_localization)))
        self.assertEqual(desc_encoding_localization.shape, (test_pySAR_localization.num_seqs, 240+50), #MAuto dim + QSO dim
            'Shape of descriptor encoding expected to be {}, but got {}.'.format((test_pySAR_localization.num_seqs, 240+50), desc_encoding_localization.shape))
        #check all column names follow pattern for MAuto + QSO descriptors
        for col in list(desc_encoding_localization.columns):
            self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r"QSO_SW[0-9]", col)) \
                or bool(re.match(r"QSO_SW[0-9][0-9]", col)), 
                    "Column doesn't follow correct naming convention: {}.".format(col))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_localization.dtypes)), 
            "Descriptor values expected to be type np.float64, got:\n{}.".format(list(desc_encoding_localization.dtypes)))
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[0]) #absorption
        desc_encoding_absorption = test_pySAR_absorption.get_descriptor_encoding(all_desc)

        self.assertIsInstance(desc_encoding_absorption, pd.DataFrame,
            'Expected descriptor encoding output to be a DataFrame, got datatype {}.'.format(type(desc_encoding_absorption)))
        self.assertEqual(desc_encoding_absorption.shape, (test_pySAR_absorption.num_seqs, 400+3+240+30), #DPComp dim + CTD_T dim + Gauto dim + QSO dim
            'Shape of descriptor encoding expected to be {}, but got {}.'.format((test_pySAR_absorption.num_seqs, 400+3+240+30), desc_encoding_absorption.shape))
        #check all column names follow pattern for DPComp + CTD_T + Gauto + QSO descriptors
        for col in list(desc_encoding_absorption.columns): 
            self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r'^[A-Z]{2}$', col)) or 
                bool(re.match(r"SOCN_SW[0-9]", col)) or bool(re.match(r"QSO_SW[0-9][0-9]", col)) or                 
                    bool(re.match(r"CTD_T_[0-9]_hydrophobicity", col)) or bool(re.match(r"CTD_T_[0-9]{2}_hydrophobicity", col)),  
                        "Column doesn't follow correct naming convention: {}.".format(col))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding_absorption.dtypes)), 
            "Descriptor values expected to be type np.float64, got:\n{}.".format(list(desc_encoding_absorption.dtypes)))
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

        self.assertIsInstance(test_desc_thermostability, pd.DataFrame, 'Expected output to be a DataFrame, got {}.'.format(type(test_desc_thermostability)))
        self.assertEqual(len(test_desc_thermostability), 1, "Expected 1 row in encoding output, got {}.".format(len(test_desc_thermostability)))
        for col in test_desc_thermostability.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols)) 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_desc_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_desc_thermostability[col]))) 
        self.assertEqual(test_pySAR_thermostability.feature_space, (261, 400),
            "Expected feature space dimensions to be 261 x 466, got {}.".format(test_pySAR_thermostability.feature_space))  
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_desc_enantioselectivity = test_pySAR_enantioselectivity.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_enantioselectivity, pd.DataFrame, 'Expected output to be a DataFrame, got {}.'.format(type(test_desc_enantioselectivity)))
        self.assertEqual(len(test_desc_enantioselectivity), 1, "Expected 1 row in encoding output, got {}.".format(len(test_desc_enantioselectivity)))
        for col in test_desc_enantioselectivity.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols)) 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_enantioselectivity[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_desc_enantioselectivity[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_enantioselectivity[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_desc_enantioselectivity[col]))) 
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 400),
            "Expected feature space dimensions to be 152 x 400, got {}.".format(test_pySAR_enantioselectivity.feature_space))  
#3.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[2]) #absorption
        test_desc_absorption = test_pySAR_absorption.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_absorption, pd.DataFrame, 'Expected output to be a DataFrame, got {}.'.format(type(test_desc_absorption)))
        self.assertEqual(len(test_desc_absorption), 1, "Expected 1 row in encoding output, got {}.".format(len(test_desc_absorption)))
        for col in test_desc_absorption.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols)) 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_absorption[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_desc_absorption[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_absorption[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_desc_absorption[col]))) 
        self.assertEqual(test_pySAR_absorption.feature_space, (81, 400),
            "Expected feature space dimensions to be 81 x 400, got {}.".format(test_pySAR_absorption.feature_space)) 
#4.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[3]) #localization
        test_desc_localization = test_pySAR_localization.encode_descriptor(descriptors=desc_1, print_results=0, output_folder=self.test_output_folder)

        self.assertIsInstance(test_desc_localization, pd.DataFrame, 'Expected output to be a DataFrame, got {}.'.format(type(test_desc_localization)))
        self.assertEqual(len(test_desc_localization), 1, "Expected 1 row in encoding output, got {}.".format(len(test_desc_localization)))
        for col in test_desc_localization.columns:
            self.assertIn(col, expected_output_cols, 
                "Col {} not found in list of expected columns:\n{}.".format(col, expected_output_cols)) 
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_desc_localization[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_desc_localization[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_desc_localization[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_desc_localization[col]))) 
        self.assertEqual(test_pySAR_localization.feature_space, (254, 400),
            "Expected feature space dimensions to be 254 x 400, got {}.".format(test_pySAR_localization.feature_space)) 

        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found: {}.".format(self.test_output_folder + "_" + _globals.CURRENT_DATETIME))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "desc_results.csv")),
            "Output csv storing encoding results not found: {}.".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "desc_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")),
            "Output regression plot not found: {}.".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")))
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
            'Expected output to be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, 
            "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                "Column {} not found in list of expected columns:\n{}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_desc[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_desc[col])))    
            self.assertEqual(test_pySAR_thermostability.feature_space, (261, 486),
                "Expected feature space dimensions to be 261 x 486, got {}.".format(test_pySAR_thermostability.feature_space))  
#2.)
        test_pySAR_enantioselectivity = pysar.PySAR(config_file=self.all_config_files[1]) #enantioselectivity
        test_aai_desc = test_pySAR_enantioselectivity.encode_aai_descriptor(descriptors=desc_2, aai_indices=aa_indices_2, print_results=0, output_folder=self.test_output_folder)
 
        self.assertIsInstance(test_aai_desc, pd.DataFrame,   #**add more tests , directly testing output of columns 
            'Output expected to be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, 
            "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                "Column {} not found in list of expected columns:\n{}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_desc[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_desc[col])))    
        self.assertEqual(test_pySAR_enantioselectivity.feature_space, (152, 413),
            "Expected test_pySAR_enantioselectivity space dimensions to be 152 x 413, got {}.".format(test_pySAR_enantioselectivity.feature_space))  
#3.)
        test_pySAR_localization = pysar.PySAR(config_file=self.all_config_files[2]) #localization
        test_aai_desc = test_pySAR_localization.encode_aai_descriptor(descriptors=desc_3, aai_indices=aa_indices_3, print_results=0, output_folder=self.test_output_folder)
      
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 
            'Output expected to be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, 
            "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                "Column {} not found in list of expected columns:\n{}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_desc[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_desc[col])))    
        self.assertEqual(test_pySAR_localization.feature_space, (81, 641),
            "Expected feature space dimensions to be 81 x 641, got {}.".format(test_pySAR_localization.feature_space))  
#4.)
        test_pySAR_absorption = pysar.PySAR(config_file=self.all_config_files[3]) #absorption
        test_aai_desc = test_pySAR_absorption.encode_aai_descriptor(descriptors=desc_4, aai_indices=aa_indices_4, print_results=0, output_folder=self.test_output_folder)
       
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 
            'Output expected to be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, 
            "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, 
                "Column {} not found in list of expected columns:\n{}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_aai_desc[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_aai_desc[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_aai_desc[col])))     
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found: {}".format(self.test_output_folder + "_" + _globals.CURRENT_DATETIME))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
            "Output csv storing encoding results not found: {}".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")),
            "Output regression plot not found: {}".format(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "model_regression_plot.png")))
        self.assertEqual(test_pySAR_absorption.feature_space, (254, 1373),
            "Expected feature space dimensions to be 254 x 1373, got {}.".format(test_pySAR_absorption.feature_space))  
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

    def tearDown(self):
        """ Delete any temp files or folders created during test case. """
        #removing any of the temp files created such as the results files, if you want to verify the results files 
        # are actually being created thencomment out the below code block
        if (os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME)):
            shutil.rmtree(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, ignore_errors=False, onerror=None)
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)