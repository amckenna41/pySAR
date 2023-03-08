################################################################################
#################             pySAR Module Tests             #################
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
    ----------
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

        #set global vars to create temp test data folders
        _globals.OUTPUT_DIR = os.path.join('tests', _globals.OUTPUT_DIR)
        _globals.OUTPUT_FOLDER = os.path.join('tests', _globals.OUTPUT_FOLDER)

    def test_pySAR_metadata(self):
        """ Testing correct pySAR version and metadata. """
        self.assertEqual(pysar_.__version__, "2.2.2", 
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
            "pypi", "machine learning", "directed evolution", "sequence activity relationships", \
            "SAR", "aaindex", "protein descriptors"], 
            "pySAR keywords is not correct, got: {}.".format(pysar_.__keywords__))

    def test_pySAR(self):
        """ Testing pySAR intialisation process and associated methods & attributes. """
        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
#1.)
        #testing attribute values, including default values
        self.assertEqual(test_pySAR.dataset, (os.path.join('tests', 'test_data', 'test_thermostability.txt')),
            'Dataset attribute does not equal what was input, got {}.'.format(test_pySAR.dataset))
        self.assertEqual(test_pySAR.sequence_col, "sequence",
            'Sequence column attribute is not correct, got {}, expected {}.'.format(test_pySAR.sequence_col, "sequence"))
        self.assertEqual(test_pySAR.activity_col, "T50",
            "Activity attribute name not correct, expected {}, got {}.".format("T50", test_pySAR.activity_col))
        self.assertEqual(test_pySAR.algorithm, "PLSRegression",
            'Algorithm attribute not correct, expected {}, got {}.'.format("PLSRegression", test_pySAR.algorithm))
        self.assertEqual(test_pySAR.test_split, 0.2,
            'Test split not expected, got {}, expected 0.2.'.format(test_pySAR.test_split))
        self.assertIsNone(test_pySAR.aai_indices, 
            "AAI Indices attribute should be none on class initialisation.")
        self.assertIsNone(test_pySAR.descriptors,
            "Descriptors attribute should be none on class initialisation.")
        self.assertEqual(test_pySAR.model_parameters, {},
            'Parameters attribute expected to be empty, got {}.'.format(test_pySAR.model_parameters))
        self.assertIsInstance(test_pySAR.data, pd.DataFrame,
            'Data expected to be a DataFrame, got {}.'.format(type(test_pySAR.data)))
        self.assertIsInstance(test_pySAR.sequences, pd.Series,
            'Sequences expected to be a pd.Series, got {}.'.format(type(test_pySAR.sequences)))
        self.assertIsInstance(test_pySAR.activity, pd.Series,
            'Activity expected to be a pd.Series, got {}.'.format(type(test_pySAR.activity)))
        self.assertEqual(test_pySAR.data.isnull().sum().sum(), 0,
            'Expected there to be no NAN/null values in data dataframe.')
        self.assertEqual(test_pySAR.num_seqs, 261,
            'Number of sequences expected to be 261, got {}.'.format(test_pySAR.num_seqs))
        self.assertEqual(test_pySAR.seq_len, 466,
            'Sequence length expected to be 466, got {}.'.format(test_pySAR.seq_len))
        self.assertEqual((str(type(test_pySAR.model))), "<class 'pySAR.model.Model'>",
            'Model class attribute expected to be an instance of the Model class, got {}.'.format((str(type(test_pySAR.model)))))
        self.assertEqual(((type(test_pySAR.model.model).__name__)), "PLSRegression",
            'Model type expected to be of type PLSRegression, got {}.'.format((type(test_pySAR.model.model).__name__)))
#2.)
        #validate that if errorneous input parameters are input, that errors are raised
        with self.assertRaises(OSError, msg='OS Error raised, config file not found.'):
            test_pySAR1 = pysar.PySAR(config_file="blahblahblah")

        with self.assertRaises(TypeError, msg='Type Error raised, config file parameter not correct data type.'):
            test_pySAR2 = pysar.PySAR(config_file=101)

    def test_sequences(self):
        """ Testing getting the protein sequences from the dataset. """
#1.)
        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
        test_seqs = test_pySAR.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not correct type, expected {}, got {}.'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MTIKEMPQPK"),
            'Error in first seqeuence, expected it to start with MTIKEMPQPK.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
#2.)
        test_pySAR_2 = pysar.PySAR(config_file=self.all_config_files[1])
        test_seqs = test_pySAR_2.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_2._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_2._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not correct type, expected {}, got {}'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MSAPFAKF"),
            'Error in second seqeuence expected it to start with MSAPFAKF.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}.'.format(test_seqs.dtype))
#3.)
        test_pySAR_3 = pysar.PySAR(config_file=self.all_config_files[2])
        test_seqs = test_pySAR_3.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_3._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_3._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not correct type, expected {}, got {}'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MLMTVFSSAP"),
            'Error in third seqeuence expected it to start with MLMTVFSSAP.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}'.format(test_seqs.dtype))
#4.)
        test_pySAR_4 = pysar.PySAR(config_file=self.all_config_files[3])
        test_seqs = test_pySAR_4.sequences

        self.assertEqual(test_seqs.shape, (test_pySAR_4._num_seqs, ),
            'Shape of the sequences not correct, expected {}, got {}.'.format(test_seqs.shape, (test_pySAR_4._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series,
            'Sequences not correct type, expected {}, got {}'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MSRLVAASWL"),
            'Error in third seqeuence expected it to start with MSRLVAASWL.')
        self.assertEqual(test_seqs.dtype, object,
            'Sequence object expected to be of dtype object, got {}'.format(test_seqs.dtype))

    def test_activity(self):
        """ Testing function that gets activity from dataset. """
#1.)
        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
        activity = test_pySAR.activity

        self.assertIsInstance(activity, pd.Series, 'Output should be a Series, got {}.'.format(type(activity)))
        self.assertEqual(activity.shape, (test_pySAR.num_seqs,), 'Output should be a Series, got {}.'.format(type(activity)))
        self.assertTrue((activity[:10] == np.array([55.0, 43.0, 49.0, 39.8, 52.9, 48.8, 45.0, 48.3, 61.5, 54.6])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity[:10]))
        self.assertEqual(activity.name, "T50", "Expected T50 column name for Series, got {}.".format(activity.name))
        self.assertTrue(np.float64 == activity.dtypes, "Column datatypes should be np.float64, got {}.".format(activity.dtypes))
#2.)
        test_pySAR_2 = pysar.PySAR(config_file=self.all_config_files[1])
        activity_2 = test_pySAR_2.activity

        self.assertIsInstance(activity_2, pd.Series, 'Output should be a Series, got {}.'.format(type(activity_2)))
        self.assertEqual(activity_2.shape, (test_pySAR_2.num_seqs,), 'Output should be a Series, got {}.'.format(type(activity_2)))
        self.assertTrue((activity_2[:10] == np.array([5.0, 23.0, 10.0, 9.0, 12.0, 11.0, 11.0, 21.0, 18.0, 17.0])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity_2[:10]))
        self.assertEqual(activity_2.name, "e-value", "Expected e-value column name for Series, got {}.".format(activity_2.name))
        self.assertTrue(np.float64 == activity_2.dtypes, "Column datatypes should be np.float64, got {}.".format(activity_2.dtypes))
#3.)
        test_pySAR_3 = pysar.PySAR(config_file=self.all_config_files[2])
        activity_3 = test_pySAR_3.activity

        self.assertIsInstance(activity_3, pd.Series, 'Output should be a Series, got {}.'.format(type(activity_3)))
        self.assertEqual(activity_3.shape, (test_pySAR_3.num_seqs,), 'Output should be a Series, got {}.'.format(type(activity_3)))
        self.assertTrue((activity_3[:10] == np.array([539, 510, 510, 519, 525, 528, 528, 534, 528, 510])).all(),
                "First 10 elements of activity don't match expected output:\n{}.".format(activity_3[:10]))
        self.assertEqual(activity_3.name, "peak", "Expected peak column name for Series, got {}.".format(activity_3.name))
        self.assertTrue(np.int64 == activity_3.dtypes, "Column datatypes should be np.float64, got {}.".format(activity_3.dtypes))
#4.)
        test_pySAR_4 = pysar.PySAR(config_file=self.all_config_files[3])
        activity_4 = test_pySAR_4.activity

        self.assertIsInstance(activity_4, pd.Series, 'Output should be a Series, got {}.'.format(type(activity_4)))
        self.assertEqual(activity_4.shape, (test_pySAR_4.num_seqs,), 'Output should be a Series, got {}.'.format(type(activity_4)))
        # self.assertTrue((activity_4[:10] == np.array([-4.626936, -5.599110, -5.715788, -5.335352, -4.187052, -6.732491, -7.135846, -6.128409, -5.319843, -5.092067])).all(),
        #         "First 10 elements of activity don't match expected output:\n{}.".format(activity_4[:10]))
        self.assertEqual(activity_4.name, "log_GFP", "Expected log_GFP column name for Series, got {}.".format(activity_4.name))
        self.assertTrue(np.float64 == activity_4.dtypes, "Column datatypes should be np.float64, got {}.".format(activity_4.dtypes))

    def test_get_aai_encoding(self):
        """ Testing getting the AAI encoding from the database for specific indices. """
        aa_indices = ["CHAM810101", "ISOY800103"]
        aa_indices1 = "NAKH920102"
        aa_indices2 = "ZIMJ680105"
        aa_indices3 = "PALJ810102, RACS820112"
        error_aaindices = ["ABCD1234", "ABCD12345"]
        error_aaindices1 = "XYZ4567"
#1.)
        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
        aai_encoding = test_pySAR.get_aai_encoding(aa_indices)

        self.assertIsInstance(aai_encoding, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding)))
        self.assertEqual(aai_encoding.shape[0], test_pySAR.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR.num_seqs, aai_encoding.shape[0]))
        self.assertEqual(aai_encoding.shape[1], test_pySAR.seq_len * len(aa_indices),
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR.seq_len, str(aai_encoding.shape[1])))
        self.assertEqual(aai_encoding.dtype,np.float32,
            'Datatype of elements in numpy array expected be dtype np.float32, got {}.'.format(aai_encoding.dtype))
        self.assertTrue((np.array([0.78, 0.5, 1.02, 0.68, 0.68, 0.78, 0.36, 0.68, 0.36, 0.68], 
            dtype=np.float32) == aai_encoding[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what was expected:\n{}.'.format(aai_encoding[0][:10]))
#2.)
        test_pySAR_1 = pysar.PySAR(config_file=self.all_config_files[1])
        aai_encoding_1 = test_pySAR_1.get_aai_encoding(aa_indices1)

        self.assertIsInstance(aai_encoding_1, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_1)))
        self.assertEqual(aai_encoding_1.shape[0], test_pySAR_1.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_1.num_seqs, aai_encoding_1.shape[0]))
        self.assertEqual(aai_encoding_1.shape[1], test_pySAR_1.seq_len,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_1.seq_len, str(aai_encoding_1.shape[1])))
        self.assertEqual(aai_encoding_1.dtype, np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_1.dtype))
        self.assertTrue((np.array([3.79, 7.25, 10.88, 7.21, 2.93, 10.88, 6.11, 2.93, 7.21, 7.25],
            dtype=np.float32) == aai_encoding_1[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what was expected:\n{}.'.format(aai_encoding_1[0][:10]))
#3.)
        test_pySAR_2 = pysar.PySAR(config_file=self.all_config_files[2])
        aai_encoding_2 = test_pySAR_2.get_aai_encoding(aa_indices2)

        self.assertIsInstance(aai_encoding_2, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_2)))
        self.assertEqual(aai_encoding_2.shape[0], test_pySAR_2.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_2.num_seqs, aai_encoding_2.shape[0]))
        self.assertEqual(aai_encoding_2.shape[1], test_pySAR_2.seq_len,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_2.seq_len, str(aai_encoding_2.shape[1])))
        self.assertEqual(aai_encoding_2.dtype,np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_2.dtype))
        self.assertTrue((np.array([14.9, 17.6, 14.9, 9.5, 14.3, 18.8, 6.9, 6.9, 9.9, 14.8],
            dtype=np.float32)==aai_encoding_2[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what was expected:\n{}.'.format(aai_encoding_2[0][:10]))
#3.)
        test_pySAR_3 = pysar.PySAR(config_file=self.all_config_files[3])
        aai_encoding_3 = test_pySAR_2.get_aai_encoding(aa_indices3)

        self.assertIsInstance(aai_encoding_3, np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_3)))
        self.assertEqual(aai_encoding_3.shape[0], test_pySAR_2.num_seqs,
            'The number of sequences in the dataset expected to be {}, got {}.'.format(test_pySAR_2.num_seqs, aai_encoding_3.shape[0]))
        self.assertEqual(aai_encoding_3.shape[1], test_pySAR_2.seq_len * 2,
            'The length of the sequences expected to be {}, got {}.'.format(test_pySAR_2.seq_len, str(aai_encoding_3.shape[1])))
        self.assertEqual(aai_encoding_3.dtype,np.float32,
            'Datatype of elements in numpy array should be of dtype np.float32, got {}.'.format(aai_encoding_3.dtype))
        self.assertTrue((np.array([1.47, 1.22, 1.47, 0.86, 1.05, 1.1, 0.77, 0.77, 1.32, 0.57],
            dtype=np.float32)==aai_encoding_3[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what was expected:\n{}.'.format(aai_encoding_3[0][:10]))
#4.)    #testing errenous indices
        with self.assertRaises(ValueError, msg='ValueError: Errorneous indices have been input.'):
            aai_encoding = test_pySAR_1.get_aai_encoding(error_aaindices)
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Errorneous indices have been input.'):
            aai_encoding = test_pySAR_1.get_aai_encoding(error_aaindices1)
#6.)
        with self.assertRaises(TypeError, msg='TypeError: Errorneous indices datatypes have been input.'):
            aai_encoding = test_pySAR_1.get_aai_encoding(1235)
            aai_encoding = test_pySAR_1.get_aai_encoding(40.89)
            aai_encoding = test_pySAR_1.get_aai_encoding(False)

    def test_aai_encoding(self):
        """ Testing AAI encoding pipeline. """
        aa_indices = ["CHAM810101","ISOY800103"]
        aa_indices_1 = "NAKH920102"
        aa_indices_2 = "LIFS790103"
        aa_indices_3 = ["PTIO830101", "QIAN880136", "RACS820110"]
        all_indices = [aa_indices, aa_indices_1, aa_indices_2, aa_indices_3] #list of all idncies
        error_aaindices = ["ABCD1234","ABCD12345"]
        error_aaindices1 = "XYZ4567"
        expected_output_cols = ['Index', 'Category', 'R2', 'RMSE', 'MSE', 
            'RPD', 'MAE', 'Explained Variance']
        # expected_output_cols_dtypes = [str, str, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]   

        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
#1.)
        test_aai_ = test_pySAR.encode_aai(indices=aa_indices, print_results=0)
        self.assertIsInstance(test_aai_, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_)))
        self.assertEqual(len(test_aai_.columns), 8, "Expected 8 columns in Series, got {}.".format(len(test_aai_.columns)))
        for col in test_aai_.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(type(test_aai_[col]), str)
            else:
                self.assertTrue(type(test_aai_[col]), np.float64)
#2.)
        test_aai_ = test_pySAR.encode_aai(indices=aa_indices_1, print_results=0)
        self.assertIsInstance(test_aai_, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_)))
        self.assertEqual(len(test_aai_.columns), 8, "Expected 8 columns in Series, got {}.".format(len(test_aai_.columns)))
        for col in test_aai_.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category"):
                self.assertTrue(type(test_aai_[col]), str)
            else:
                self.assertTrue(type(test_aai_[col]), np.float64)
#3.)
        for index in range(0, len(all_indices)):
            test_aai_ = test_pySAR.encode_aai(indices=all_indices[index], print_results=0)
            self.assertIsInstance(test_aai_, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_)))
            self.assertEqual(len(test_aai_.columns), 8, "Expected 8 columns in Series, got {}.".format(len(test_aai_.columns)))
            for col in test_aai_.columns:
                self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
                if (col == "Index" or col == "Category"):
                    self.assertTrue(type(test_aai_[col]), str)
                else:
                    self.assertTrue(type(test_aai_[col]), np.float64)
#4.)
        with self.assertRaises(ValueError, msg='ValueError: Indices parameter cannot be None.'):
            test_aai_ = test_pySAR.encode_aai(indices=None)
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Indices parameter cannot be None.'):
            test_aai_ = test_pySAR.encode_aai()
#6.)
        with self.assertRaises(ValueError, msg='ValueError: Erroneous indices put into indices parameter.'):
            test_aai_ = test_pySAR.encode_aai(indices=error_aaindices)
#7.)
        with self.assertRaises(ValueError, msg='ValueError: Erroneous indices put into indices parameter.'):
            test_aai_ = test_pySAR.encode_aai(indices=error_aaindices1)
#8.)
        with self.assertRaises(TypeError, msg='TypeError: Indices must be lists or strings.'):
            test_aai_ = test_pySAR.encode_aai(indices=123)
            test_aai_ = test_pySAR.encode_aai(indices=0.90)
            test_aai_ = test_pySAR.encode_aai(indices=False)
            test_aai_ = test_pySAR.encode_aai(indices=9000)

    def test_get_desc_encoding(self):
        """ Testing Descriptor encoding functionality. """
        desc_1 = "dipeptide_composition"
        desc_2 = "ctd_transition"
        # desc_3 = "seq_order_coupling_number"
        desc_3 = "moranauto, quasi_seq_order"
        all_desc = [desc_1, desc_2, "geary_auto", "sequence_order_coupling_number"]
        error_desc = 123

        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
        desc_encoding = test_pySAR.get_descriptor_encoding(desc_1)
#1.)
        self.assertIsInstance(desc_encoding, pd.DataFrame,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(desc_encoding)))
        self.assertEqual(desc_encoding.shape, (test_pySAR.num_seqs, 400),
            'The number of sequences in the dataset expected to be {}, got {}.'.format((test_pySAR.num_seqs, 400), desc_encoding.shape))
        for col in list(desc_encoding.columns):
            #check all columns follow pattern of XY where x & y are amino acids 
            self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "Column doesn't follow correct naming convention: {}.".format(col))
            self.assertIn(col[0], self.amino_acids, "")
            self.assertIn(col[1], self.amino_acids, "")
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding.dtypes)), 
            "Descriptor values not of correct datatype: {}".format(list(desc_encoding.dtypes)))
#2.)
        desc_encoding = test_pySAR.get_descriptor_encoding(desc_2)

        self.assertIsInstance(desc_encoding, pd.DataFrame,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(desc_encoding)))
        self.assertEqual(desc_encoding.shape, (test_pySAR.num_seqs, 3),
            'The number of sequences in the dataset expected to be {}, got {}.'.format((test_pySAR.num_seqs, 3), desc_encoding.shape))
                #iterate over all columns, checking they follow naming convention using regex
        for col in list(desc_encoding.columns):
            self.assertTrue((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_hydrophobicity", col))), 
                    "Column name does not follow expected format: {}.".format(col)) 
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding.dtypes)), 
            "Descriptor values not of correct datatype: {}.".format(list(desc_encoding.dtypes)))
#3.)
        desc_encoding = test_pySAR.get_descriptor_encoding(desc_3)

        self.assertIsInstance(desc_encoding, pd.DataFrame,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(desc_encoding)))
        self.assertEqual(desc_encoding.shape, (test_pySAR.num_seqs, 240+50), #MAuto dim + QSO dim
            'The number of sequences in the dataset expected to be {}, got {}.'.format((test_pySAR.num_seqs, 240+50), desc_encoding.shape))
                #iterate over all columns, checking they follow naming convention using regex
        for col in list(desc_encoding.columns):
            self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r"QSO_SW[0-9]", col)) \
                or bool(re.match(r"QSO_SW[0-9][0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding.dtypes)), 
            "Descriptor values not of correct datatype: {}.".format(list(desc_encoding.dtypes)))
#4.)
        desc_encoding = test_pySAR.get_descriptor_encoding(all_desc)

        self.assertIsInstance(desc_encoding, pd.DataFrame,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(desc_encoding)))
        self.assertEqual(desc_encoding.shape, (test_pySAR.num_seqs, 400+3+240+30), #MAuto dim + QSO dim
            'The number of sequences in the dataset expected to be {}, got {}.'.format((test_pySAR.num_seqs, 240+50), desc_encoding.shape))
        #iterate over all columns, checking they follow naming convention using regex
        for col in list(desc_encoding.columns): 
            self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)) or bool(re.match(r'^[A-Z]{2}$', col)) or 
                bool(re.match(r"SOCN_SW[0-9]", col)) or bool(re.match(r"QSO_SW[0-9][0-9]", col)) or                 
                    bool(re.match(r"CTD_T_[0-9]_hydrophobicity", col)) or bool(re.match(r"CTD_T_[0-9]{2}_hydrophobicity", col)),  
                        "Column name doesn't match expected regex pattern: {}.".format(col))
        self.assertTrue(all(col == np.float64 for col in list(desc_encoding.dtypes)), 
            "Descriptor values not of correct datatype: {}.".format(list(desc_encoding.dtypes)))
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor input parameter cannot be None.'):
            test_desc = test_pySAR.get_descriptor_encoding(descriptors=None)
#6.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor input parameter cannot be an empty string.'):
            test_desc = test_pySAR.get_descriptor_encoding(descriptors="")
#7.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor input parameter cannot be an empty list.'):
            test_desc = test_pySAR.get_descriptor_encoding(descriptors=[])
#8.)
        with self.assertRaises(TypeError, msg='ValueError: Descriptor input parameter cannot be an invalid descriptor name.'):
            test_desc = test_pySAR.get_descriptor_encoding(descriptors=error_desc)

    def test_desc_encoding(self):
        """ Testing Descriptor encoding pipeline. """
        desc_1 = "dipeptide_composition"
        desc_2 = "ctd_distribution"
        desc_3 = "seq_order_coupling_number"
        desc_4 = "moranauto, quasi_seq_order"
        all_desc = [desc_1, desc_2, desc_3, "moranauto, quasi_seq_order"]
        error_desc = "blahblahblah"
        error_desc_1 = 123
        expected_output_cols = ['Descriptor', 'Group', 'R2', 'RMSE', 'MSE', 
            'RPD', 'MAE', 'Explained Variance']

        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
#1.)
        for de in range(0, len(all_desc)):
            test_desc = test_pySAR.encode_desc(descriptor=all_desc[de], print_results=0)
            self.assertIsInstance(test_desc, pd.DataFrame, 'Output should be a Series, got {}.'.format(type(test_desc)))
            self.assertEqual(len(test_desc), 1, "")
            for col in test_desc.columns:
                self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
                if (col == "Descriptor" or col == "Group"):
                    self.assertTrue(type(test_desc[col]), str)
                else:
                    self.assertTrue(type(test_desc[col]), np.float64)
#2.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor parameter cannot be None.'):
            test_desc = test_pySAR.encode_desc(descriptor=None)
#3.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor parameter cannot be None.'):
            test_desc = test_pySAR.encode_desc(descriptor=error_desc)
#4.)    
        with self.assertRaises(TypeError, msg='TypeError: Descriptor parameter has to be a strong or list.'):
            test_desc = test_pySAR.encode_desc(descriptor=error_desc_1)
            test_desc = test_pySAR.encode_desc(descriptor=123)
            test_desc = test_pySAR.encode_desc(descriptor=0.90)
            test_desc = test_pySAR.encode_desc(descriptor=False)
            test_desc = test_pySAR.encode_desc(descriptor=9000)

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

        test_pySAR = pysar.PySAR(config_file=self.all_config_files[0])
#1.)
        test_aai_desc = test_pySAR.encode_aai_desc(descriptors=desc_1, indices=aa_indices_1, print_results=0)
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 'Output should be a DataFrame, got {}\n.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(type(test_aai_desc[col]), str)
            else:
                self.assertTrue(type(test_aai_desc[col]), np.float64)    
#2.)
        test_aai_desc = test_pySAR.encode_aai_desc(descriptors=desc_2, indices=aa_indices_2, print_results=0)
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(type(test_aai_desc[col]), str)
            else:
                self.assertTrue(type(test_aai_desc[col]), np.float64)    
#3.)
        test_aai_desc = test_pySAR.encode_aai_desc(descriptors=desc_3, indices=aa_indices_3, print_results=0)
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(type(test_aai_desc[col]), str)
            else:
                self.assertTrue(type(test_aai_desc[col]), np.float64)    
#4.)
        test_aai_desc = test_pySAR.encode_aai_desc(descriptors=desc_4, indices=aa_indices_4, print_results=0)
        self.assertIsInstance(test_aai_desc, pd.DataFrame, 'Output should be a DataFrame, got {}.'.format(type(test_aai_desc)))
        self.assertEqual(len(test_aai_desc.columns), 10, "Expected 10 columns in output dataframe, got {}.".format(len(test_aai_desc)))
        for col in test_aai_desc.columns:
            self.assertIn(col, expected_output_cols, "Col {} not found in list of expected columns: {}".format(col, expected_output_cols))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(type(test_aai_desc[col]), str)
            else:
                self.assertTrue(type(test_aai_desc[col]), np.float64)    
#5.)
        with self.assertRaises(ValueError, msg='ValueError: Descriptor and indices parameter cannot both be None.'):
            test_desc = test_pySAR.encode_aai_desc(descriptors=None)
            test_desc = test_pySAR.encode_aai_desc(indices=None)
            test_desc = test_pySAR.encode_aai_desc(descriptors="aa_comp")
            test_desc = test_pySAR.encode_aai_desc(indices="LIFS790103")
#6.)
        with self.assertRaises(TypeError, msg='ValueError: Descriptor and indices must be lists or strings.'):
            test_desc = test_pySAR.encode_aai_desc(descriptors=123, indices=123)
            test_desc = test_pySAR.encode_aai_desc(descriptors=0000, indices=0.90)
            test_desc = test_pySAR.encode_aai_desc(descriptors=False, indices=True)
            test_desc = test_pySAR.encode_aai_desc(descriptors=2.9, indices=9000)
#7.)
        with self.assertRaises(ValueError, msg='TypeError: Descriptor not found in list of valid descriptors.'):
            test_aai_desc = test_pySAR.encode_aai_desc(descriptors="invalid_descriptor")
            test_aai_desc = test_pySAR.encode_aai_desc(indices="invalid_value")
            test_aai_desc = test_pySAR.encode_aai_desc(descriptors="descriptor not found")
            test_aai_desc = test_pySAR.encode_aai_desc(indices="blahblahblah")

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        if (os.path.isdir(_globals.OUTPUT_DIR)):
            shutil.rmtree(_globals.OUTPUT_DIR, ignore_errors=False, onerror=None)
        if (os.path.isdir(_globals.OUTPUT_FOLDER)):
            shutil.rmtree(_globals.OUTPUT_FOLDER, ignore_errors=False, onerror=None)

        # del _globals.OUTPUT_DIR
        # del _globals.OUTPUT_FOLDER
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)