################################################################################
#################             Utilities Module Tests           #################
################################################################################

import glob
import os
import shutil
import unittest
import numpy as np
#suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import pySAR.globals_ as _globals
import pySAR.utils as utils

class UtilsTest(unittest.TestCase):
    """
    Test suite for testing utilites module and functionality in pySAR package. 

    Test Cases
    ==========
    test_valid_sequence:
        testing correct utils.valid_sequence functionality.
    test_remove_gaps:
        testing correct utils.remove_gaps functionality.
    test_zero_padding:
        testing correct utils.zero_padding functionality.
    test_save_results:
        testing correct utils.save_results functionality.
    test_map:
        testing correct utils.Map class functionality.
    """
    def setUp(self):
        """ Import all test datasets from test_data folder. """
        self.test_dataset1 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_thermostability.txt'), sep=",", header=0)
        self.test_dataset2 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_enantioselectivity.txt'), sep=",", header=0)
        self.test_dataset3 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_localization.txt'), sep=",", header=0)
        self.test_dataset4 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_absorption.txt'), sep=",", header=0)

        #append all datasets to a list
        self.all_test_datasets = [self.test_dataset1, self.test_dataset2, 
                self.test_dataset3, self.test_dataset4]

        #create temporary unit test output folder
        self.test_output_folder = os.path.join("tests", "test_outputs")
        if not (os.path.isdir(self.test_output_folder)):
                os.makedirs(self.test_output_folder)

    def test_valid_sequence(self):
        """ Test Valid/Invalid Sequences utility function. """
        invalid_seqs = [["A", "B", "C", "D"], ["E", "F", "J"]]
        invalid_seqs1 = ["ZZZZZZ"]
        invalid_seqs2 = [["Z", 2, "Y", "X", 321]]
        invalid_seqs3 = "XXZXXZXXZ"

        #cache results to avoid redundant calls
        result_seqs = utils.valid_sequence(invalid_seqs)
        result_seqs1 = utils.valid_sequence(invalid_seqs1)
        result_seqs2 = utils.valid_sequence(invalid_seqs2)
        result_seqs3 = utils.valid_sequence(invalid_seqs3)
#1.)
        self.assertIsNotNone(result_seqs, "Valid sequence function should not return None.")
        self.assertIsNotNone(result_seqs1, "Valid sequence function should not return None.")
        self.assertIsNotNone(result_seqs2, "Valid sequence function should not return None.")
        self.assertIsNotNone(result_seqs3, "Valid sequence function should not return None.")
#2.)
        self.assertEqual(len(result_seqs), 2,
                f"Expected 2 outputs from from valid sequence function, got {len(result_seqs)}.")
        self.assertEqual(len(result_seqs1), 6,
                f"Expected 6 outputs from from valid sequence function, got {len(result_seqs1)}.")
        self.assertEqual(len(result_seqs2), 4,
                f"Expected 4 outputs from from valid sequence function, got {len(result_seqs2)}.")
        self.assertEqual(len(result_seqs3), 9,
                f"Expected 9 outputs from from valid sequence function, got {len(result_seqs3)}.")
#3.)
        self.assertIsInstance(result_seqs, list,
                f"Valid sequence function should return a list, got {type(result_seqs)}.")
        self.assertIsInstance(result_seqs1, list,
                f"Valid sequence function should return a list, got {type(result_seqs1)}.")
        self.assertIsInstance(result_seqs2, list,
                f"Valid sequence function should return a list, got {type(result_seqs2)}.")
        self.assertIsInstance(result_seqs3, list,
                f"Valid sequence function should return a list, got {type(result_seqs3)}.")
#4.)
        #testing with valid sequences
        random_seq = np.random.randint(0, len(self.test_dataset1))
        random_seq1 = np.random.randint(0, len(self.test_dataset2))
        random_seq2 = np.random.randint(0, len(self.test_dataset3))
        random_seq3 = np.random.randint(0, len(self.test_dataset4))

        self.assertIsNone(utils.valid_sequence(self.test_dataset1['sequence'][random_seq]), "Valid sequence function should return None.")
        self.assertIsNone(utils.valid_sequence(self.test_dataset2['sequence'][random_seq1]), "Valid sequence function should return None.")
        self.assertIsNone(utils.valid_sequence(self.test_dataset3['sequence'][random_seq2]), "Valid sequence function should return None.")
        self.assertIsNone(utils.valid_sequence(self.test_dataset4['sequence'][random_seq3]), "Valid sequence function should return None.")

    def test_remove_gaps(self):
        """ Test utility function that removes any gaps from sequences. """
        # Lists of individual characters: each character is a separate "sequence";
        # remove_gaps strips '-' from each element individually, so length is preserved.
        seq1 = ["A", "B", "C", "D", "-"]
        seq2 = ["A", "B", "C", "D", "-", "-", "-", "E", "F", "-"]
        seq3 = 'ABCDFSDJWD---'
        seq4 = "YUJBVFGHYJ---ASD"
#1.)    List of individual chars — length unchanged, '-' elements become empty strings
        seq1_test = utils.remove_gaps(seq1)
        self.assertEqual(len(seq1_test), 5, f"Expected length of output to be 5, got {len(seq1_test)}.")
        self.assertIsInstance(seq1_test, list, f"Expected output to be of type list, got {type(seq1_test)}.")
        self.assertNotIn('-', seq1_test, "Expected there to be no '-' elements in the result.")
#2.)    List of individual chars — length unchanged
        seq2_test = utils.remove_gaps(seq2)
        self.assertEqual(len(seq2_test), 10, f"Expected length of output to be 10, got {len(seq2_test)}.")
        self.assertIsInstance(seq2_test, list, f"Expected output to be of type list, got {type(seq2_test)}.")
        self.assertNotIn('-', seq2_test, "Expected there to be no '-' elements in the result.")
#3.)    String input — gaps stripped, length reduced
        seq3_test = utils.remove_gaps(seq3)
        self.assertEqual(len(seq3_test), 10, f"Expected length of output to be 10, got {len(seq3_test)}.")
        self.assertIsInstance(seq3_test, str, f"Expected output to be of type str, got {type(seq3_test)}.")
        self.assertNotIn('-', seq3_test, "Expected there to be no gaps (-) in the sequence.")
#4.)    String input — gaps stripped, length reduced
        seq4_test = utils.remove_gaps(seq4)
        self.assertEqual(len(seq4_test), 13, f"Expected length of output to be 13, got {len(seq4_test)}.")
        self.assertIsInstance(seq4_test, str, f"Expected output to be of type str, got {type(seq4_test)}.")
        self.assertNotIn('-', seq4_test, "Expected there to be no gaps (-) in the sequence.")
#5.)    List of full protein sequences — gaps removed from each sequence string
        seq5 = ["ACDE-FGH", "MNOP-QRS"]
        seq5_test = utils.remove_gaps(seq5)
        self.assertEqual(len(seq5_test), 2, f"Expected 2 sequences, got {len(seq5_test)}.")
        self.assertEqual(seq5_test[0], "ACDEFGH", f"Expected 'ACDEFGH', got '{seq5_test[0]}'.")
        self.assertEqual(seq5_test[1], "MNOPQRS", f"Expected 'MNOPQRS', got '{seq5_test[1]}'.")
        self.assertIsInstance(seq5_test, list, f"Expected output to be of type list, got {type(seq5_test)}.")
#6.)    numpy array of full protein sequences — gaps removed from each sequence
        seq6 = np.array(["AC--DE", "FG-HI"])
        seq6_test = utils.remove_gaps(seq6)
        self.assertNotIn('-', ''.join(seq6_test), "Expected no gaps in numpy-array input result.")
        self.assertEqual(seq6_test[0].replace('-', ''), "ACDE", f"Expected 'ACDE', got '{seq6_test[0]}'.")


    def test_zero_padding(self):
        """ Test zero padding utility function that pads an array or list with 0's. """
        seq1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8]], dtype=object)
        seq2 = [["A", "B", "C"], ["D", "E", "F", "G"], ["H"]]
        seq3 = np.random.randint(10,90,(4,5,2))
#1.)
        test_dataset3_padded = utils.zero_padding(self.test_dataset3['sequence'])

        #verify all sequences in dataset have been zero-padded to same length
        for seq in range(0, len(test_dataset3_padded)):
            self.assertEqual(len(test_dataset3_padded[seq]), 361,
                f"Expected length of output to be 361, got {len(test_dataset3_padded[seq])}.")
            self.assertIsInstance(test_dataset3_padded[seq], str,
                f"Expected output to be of type string, got {type(test_dataset3_padded[seq])}.")
            
        self.assertIsInstance(test_dataset3_padded, pd.Series,
                f"Expected output to be of type Series, got {type(test_dataset3_padded)}.")
        self.assertEqual(test_dataset3_padded.shape[0], 254,
                f"Expected number of sequences to be 254, got {test_dataset3_padded[0]}.")
#2.)
        padded_seqs1 = utils.zero_padding(seq1)
        self.assertEqual(len(padded_seqs1), 2, f"Expected length of output to be 2, got {len(padded_seqs1)}.")
        self.assertEqual(len(padded_seqs1[0]), 5, f"Expected length of output to be 5, got {len(padded_seqs1[0])}.")
        self.assertIsInstance(padded_seqs1, np.ndarray, f"Expected output to be of type numpy array, got {type(padded_seqs1)}.")
#3.)
        padded_seqs2 = utils.zero_padding(seq2)
        self.assertEqual(len(padded_seqs2), 3, f"Expected length of output to be 3, got {len(padded_seqs2)}.")
        self.assertEqual(len(padded_seqs2[0]), 4, f"Expected length of output to be 4, got {len(padded_seqs2[0])}.")
        self.assertIsInstance(padded_seqs2, list, f"Expected output to be of type list, got {type(padded_seqs2)}.")
#4.)
        padded_seqs3 = utils.zero_padding(seq3)
        self.assertEqual(len(padded_seqs3), 4, f"Expected length of output to be 4, got {len(padded_seqs3)}.")
        self.assertEqual(padded_seqs3.shape, (4,5,2), f"Expected output to be of shape (4,5,2), got {padded_seqs3.shape}.")
        self.assertIsInstance(padded_seqs3, np.ndarray, f"Expected output to be of type numpy array, got {type(padded_seqs3)}.")
        self.assertTrue(np.array_equal(padded_seqs3, seq3), "Expected original and padded sequences to have the same values.")
        
    def test_save_results(self):
        """ Testing save results utility function. """
#1.)
        #create dummy test results, save to csv and verify csv has been created & saved
        test_results = {'R2': 0.56, 'MSE': 0.34, 'RMSE': 0.89}
        utils.save_results(test_results, 'test_results', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results.csv')),
                f"Output results csv not found in output folder: {os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + '_' + _globals.CURRENT_DATETIME, 'test_results.csv')}.")
#2.)
        test_results1 = {'MAE': 2.10, 'MSE': 0.99, 'RPD': 1.28}
        utils.save_results(test_results1, 'test_results1', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results1.csv')),
                f"Output results csv not found in output folder: {os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + '_' + _globals.CURRENT_DATETIME, 'test_results1.csv')}.")
#3.)
        test_results2 = pd.DataFrame(np.random.randint(1,100, size=(5,3)), columns=['R2', 'MSE', 'RMSE'])
        utils.save_results(test_results2, 'test_results2', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results2.csv')),
                f"Output results csv not found in output folder: {os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + '_' + _globals.CURRENT_DATETIME, 'test_results2.csv')}.")
#4.)
        test_results3 = pd.Series(np.random.randint(1,100), index=['Col1', 'Col2', 'Col3', 'Col4'])
        utils.save_results(test_results3, 'test_results3',  output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results3.csv')),
                f"Output results csv not found in output folder: {os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + '_' + _globals.CURRENT_DATETIME, 'test_results3.csv')}.")
#5.)
        test_results4 = np.random.randint(1, 100, size=(2,6))
        with self.assertRaises(TypeError, msg='Type Error raised, invalid input parameter data type given.'):
            utils.save_results(test_results4, 'test_results4',  output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertFalse(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results4.csv')),
                f"Output results csv should not be found in output folder: {os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + '_' + _globals.CURRENT_DATETIME, 'test_results4.csv')}.")

    def test_map(self):
        """ Testing Map class which allows for a dict to be accessed via dot notation. """
#1.)
        test_map1 = utils.Map({"first_name":"Joe", "last_name":"Bloggs", "country":"Ireland", "city":"Dublin"})
        
        self.assertIsInstance(test_map1, dict, f"Expected instance to be of type dict, got {type(test_map1)}.")
        self.assertEqual(test_map1.first_name, "Joe", f"Expected Joe, got {test_map1.first_name}.")
        self.assertEqual(test_map1.last_name, "Bloggs", f"Expected Bloggs, got {test_map1.last_name}.")
        self.assertEqual(test_map1.country, "Ireland", f"Expected Ireland, got {test_map1.country}.")
        self.assertEqual(test_map1.city, "Dublin", f"Expected Dublin, got {test_map1.city}.")
        self.assertEqual(len(test_map1), 4, f"Expected output length to be 4, got {len(test_map1)}.")
#2.)
        test_map2 = utils.Map({"first_name":"John", "last_name":"Smith"}, country="Germany", city="Hanover")

        self.assertIsInstance(test_map2, dict, f"Expected instance to be of type dict, got {type(test_map2)}.")
        self.assertEqual(test_map2.first_name, "John", f"Expected John, got {test_map2.first_name}.")
        self.assertEqual(test_map2.last_name, "Smith", f"Expected Smith, got {test_map2.last_name}.")
        self.assertEqual(test_map2.country, "Germany", f"Expected Germany, got {test_map2.country}.")
        self.assertEqual(test_map2.city, "Hanover", f"Expected Hanover, got {test_map2.city}.")
        self.assertEqual(len(test_map2), 4, f"Expected output length to be 4, got {len(test_map2)}.")
#3.)
        test_map3 = utils.Map({})

        self.assertIsInstance(test_map3, dict, f"Expected instance to be of type dict, got {type(test_map3)}.")
        self.assertEqual(test_map3, {}, f"Expected an empty dict, got {test_map3}.")
        self.assertEqual(len(test_map3), 0, f"Expected output length to be 0, got {len(test_map3)}.")
#4.)    
        test_map1.language = "Python"
        test_map1["age"] = 42
        self.assertEqual(test_map1.language, "Python", f"Expected Python, got {test_map1.language}.")
        self.assertEqual(test_map1.age, 42, f"Expected 42, got {test_map1.age}.")
        self.assertEqual(len(test_map1), 6, f"Expected output length to be 6, got {len(test_map1)}.")

        test_map2.language = "C++"
        test_map2.age = 20
        self.assertEqual(test_map2.language, "C++", f"Expected C++, got {test_map2.language}.")
        self.assertEqual(test_map2.age, 20, f"Expected 20, got {test_map2.age}.")
        self.assertEqual(len(test_map2), 6, f"Expected output length to be 6, got {len(test_map2)}.")

        test_map3.language = "Ruby"
        test_map3.age = 99
        self.assertEqual(test_map3.language, "Ruby", f"Expected Ruby, got {test_map3.language}.")
        self.assertEqual(test_map3.age, 99, f"Expected 99, got {test_map3.age}.")
        self.assertEqual(len(test_map3), 2, f"Expected output length to be 2, got {len(test_map3)}.")
#5.)
        del test_map1.first_name
        self.assertEqual(len(test_map1), 5, f"Expected output length to be 5, got {len(test_map1)}.")
        del test_map1.country
        self.assertEqual(len(test_map1), 4, f"Expected output length to be 4, got {len(test_map1)}.")
        del test_map3.language
        self.assertEqual(len(test_map3), 1, f"Expected output length to be 1, got {len(test_map3)}.")
#6.)
        with self.assertRaises(TypeError):
                utils.Map(1245)
                utils.Map(10.4)
                utils.Map(False)
#7.)    missing attribute raises AttributeError (not returns None)
        m = utils.Map({"a": 1})
        with self.assertRaises(AttributeError,
                msg="Accessing a non-existent Map attribute should raise AttributeError."):
            _ = m.nonexistent_key
#8.)    getattr() fallback works correctly when key is absent
        fallback = getattr(m, "nonexistent_key", "default_value")
        self.assertEqual(fallback, "default_value",
            "getattr(map, missing_key, default) should return the default, not raise.")
#9.)    existing key still accessible via dot notation
        self.assertEqual(getattr(m, "a", None), 1,
            "getattr(map, 'a', None) should return the value 1.")

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4

        #remove main test output folder
        if os.path.isdir(self.test_output_folder):
            shutil.rmtree(self.test_output_folder, ignore_errors=False, onerror=None)

        #remove any timestamped output folders created by save_results
        for _ts_dir in glob.glob(self.test_output_folder + "_*"):
            shutil.rmtree(_ts_dir, ignore_errors=True)