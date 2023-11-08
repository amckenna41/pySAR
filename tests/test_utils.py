################################################################################
#################             Utilities Module Tests           #################
################################################################################

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
    test_flatten:
        testing correct utils.flatten functionality.
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
#1.)
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs), "Valid sequence function should not return None.")
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs1), "Valid sequence function should not return None.")
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs2), "Valid sequence function should not return None.")
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs3), "Valid sequence function should not return None.")
#2.)
        self.assertEqual(len(utils.valid_sequence(invalid_seqs)), 2,
                "Expected 2 outputs from from valid sequence function, got {}.".format(len(utils.valid_sequence(invalid_seqs))))
        self.assertEqual(len(utils.valid_sequence(invalid_seqs1)), 6,
                "Expected 6 outputs from from valid sequence function, got {}.".format(len(utils.valid_sequence(invalid_seqs1))))
        self.assertEqual(len(utils.valid_sequence(invalid_seqs2)), 4,
                "Expected 4 outputs from from valid sequence function, got {}.".format(len(utils.valid_sequence(invalid_seqs2))))
        self.assertEqual(len(utils.valid_sequence(invalid_seqs3)), 9,
                "Expected 9 outputs from from valid sequence function, got {}.".format(len(utils.valid_sequence(invalid_seqs3))))
#3.)
        self.assertIsInstance((utils.valid_sequence(invalid_seqs)), list,
                "Valid sequence function should return a list, got {}.".format(type(utils.valid_sequence(invalid_seqs))))
        self.assertIsInstance((utils.valid_sequence(invalid_seqs1)), list,
                "Valid sequence function should return a list, got {}.".format(type(utils.valid_sequence(invalid_seqs1))))
        self.assertIsInstance((utils.valid_sequence(invalid_seqs2)), list,
                "Valid sequence function should return a list, got {}.".format(type(utils.valid_sequence(invalid_seqs2))))
        self.assertIsInstance((utils.valid_sequence(invalid_seqs3)), list,
                "Valid sequence function should return a list, got {}.".format(type(utils.valid_sequence(invalid_seqs3))))
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
        seq1 = ["A", "B", "C", "D", "-"]
        seq2 = ["A", "B", "C", "D", "-", "-", "-", "E", "F", "-"]
        seq3 = 'ABCDFSDJWD---'
        seq4 = "YUJBVFGHYJ---ASD"
#1.)
        seq1_test = utils.remove_gaps(seq1)
        self.assertEqual(len(seq1_test), 1, "Expected length of output to be 1, got {}.".format(len(seq1_test)))
        self.assertEqual(len(seq1_test[0]), 4, "Expected length of output to be 4, got {}.".format(len(seq1_test[0])))
        self.assertIsInstance(seq1_test, list, "Expected output to be of type list, got {}.".format(type(seq1_test)))
        self.assertNotIn('-', seq1_test, "Expected there to be no gaps (-) in the sequence.")
#2.)
        seq2_test = utils.remove_gaps(seq2)
        self.assertEqual(len(seq2_test), 1, "Expected length of output to be 1, got {}.".format(len(seq2_test)))
        self.assertEqual(len(seq2_test[0]), 6, "Expected length of output to be 6, got {}.".format(len(seq2_test)))
        self.assertIsInstance(seq2_test, list, "Expected output to be of type list, got {}.".format(type(seq2_test)))
        self.assertNotIn('-', seq2_test, "Expected there to be no gaps (-) in the sequence.")
#3.)
        seq3_test = utils.remove_gaps(seq3)
        self.assertEqual(len(seq3_test), 10, "Expected length of output to be 10, got {}.".format(len(seq3_test)))
        self.assertIsInstance(seq3_test, str, "Expected output to be of type str, got {}.".format(len(seq3_test)))
        self.assertNotIn('-', seq3_test, "Expected there to be no gaps (-) in the sequence.")
#4.)
        seq4_test = utils.remove_gaps(seq4)
        self.assertEqual(len(seq4_test), 13, "Expected length of output to be 13, got {}.".format(len(seq4_test)))
        self.assertIsInstance(seq4_test, str, "Expected output to be of type str, got {}.".format(len(seq4_test)))
        self.assertNotIn('-', seq4_test, "Expected there to be no gaps (-) in the sequence.")

    def test_flatten(self):
        """ Test flatten utility function that flattens an array or list. """
        seq1 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        seq2 = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]], np.int32)
        seq3 = np.random.randint(10,90,(4,5,2))
        seq4 = ["A", "B", "C", "D", "E", "F"]
        seq5 = "TUVWXYZ"
#1.) 
        flattened_array = utils.flatten(seq1)
        self.assertEqual(flattened_array.shape, (6,1), "Expected output shape to be (6,1), got {}.".format(flattened_array.shape))
        self.assertIsInstance(flattened_array, np.ndarray, "Expected output to be of type np.ndarray, got {}.".format(type(flattened_array)))
        self.assertEqual(flattened_array.ndim, 2, "Expected 2 output dimensions, got {}.".format(flattened_array.ndim))
        self.assertTrue((np.array([[1],[2],[3],[4],[5],[6]]) == flattened_array).all(),
                        "Output array doesn't match expected:\n{}.".format(flattened_array))
#2.)
        flattened_array_2 = utils.flatten(seq2)
        self.assertEqual(flattened_array_2.shape, (9,1), "Expected output shape to be (9,1), got {}.".format(flattened_array_2.shape))
        self.assertIsInstance(flattened_array_2, np.ndarray, "Expected output to be of type np.ndarray, got {}.".format(type(flattened_array_2)))
        self.assertEqual(flattened_array_2.ndim, 2, "Expected 2 output dimensions, got {}.".format(flattened_array_2.ndim))
        self.assertTrue((np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]]) == flattened_array_2).all(),
                "Output array doesn't match expected:\n{}.".format(flattened_array_2))
#3.)
        flattened_array_3 = utils.flatten(seq3)
        self.assertEqual(flattened_array_3.shape, (40,1), "Expected output shape to be (40,1), got {}.".format(flattened_array_3.shape))
        self.assertIsInstance(flattened_array_3, np.ndarray, "Expected output to be of type np.ndarray, got {}.".format(type(flattened_array_3)))
        self.assertEqual(flattened_array_3.ndim, 2, "Expected 2 output dimensions, got {}.".format(flattened_array_3.ndim))
#4.)
        flattened_array_4 = utils.flatten(seq4)
        self.assertEqual(len(flattened_array_4), 6, "Expected length of output to be 6, got {}.".format(len(flattened_array_4)))
        self.assertIsInstance(flattened_array_4, list, "Expected output to be of type list, got {}.".format(type(flattened_array_4)))
        self.assertEqual(flattened_array_4, seq4, "Output doesn't match expected sequence {}.".format(seq4))
#5.)
        flattened_array_5 = utils.flatten(seq5)
        self.assertEqual(flattened_array_5, seq5, "Output doesn't match expected sequence {}.".format(seq5))
        self.assertIsInstance(flattened_array_5, str, "Expected output to be of type string, got {}.".format(type(flattened_array_5)))

    def test_zero_padding(self):
        """ Test zero padding utility function that pads an array or list with 0's. """
        seq1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8]], dtype=object)
        seq2 = [["A", "B", "C"], ["D", "E", "F", "G"], ["H"]]
        seq3 = np.random.randint(10,90,(4,5,2))
#1.)
        test_dataset3_padded = utils.zero_padding((self.test_dataset3['sequence']))

        #verify all sequences in dataset have been zero-padded to same length
        for seq in range(0, len(test_dataset3_padded)):
            self.assertEqual(len(test_dataset3_padded[seq]), 361,
                "Expected length of output to be 361, got {}.".format(len(test_dataset3_padded[seq])))
            self.assertIsInstance(test_dataset3_padded[seq], str,
                "Expected output to be of type string, got {}.".format(type(test_dataset3_padded[seq])))
            
        self.assertIsInstance(test_dataset3_padded, pd.Series,
                "Expected output to be of type Series, got {}.".format(type(test_dataset3_padded)))
        self.assertEqual(test_dataset3_padded.shape[0], 254,
                "Expected number of sequences to be 254, got {}.".format(test_dataset3_padded[0]))
#2.)
        padded_seqs1 = utils.zero_padding(seq1)
        self.assertEqual(len(padded_seqs1), 2, "Expected length of output to be 2, got {}.".format(len(padded_seqs1)))
        self.assertEqual(len(padded_seqs1[0]), 5, "Expected length of output to be 5, got {}.".format(len(padded_seqs1[0])))
        self.assertIsInstance(padded_seqs1, np.ndarray, "Expected output to be of type numpy array, got {}.".format(type(padded_seqs1)))
#3.)
        padded_seqs2 = utils.zero_padding(seq2)
        self.assertEqual(len(padded_seqs2), 3, "Expected length of output to be 3, got {}.".format(len(padded_seqs2)))
        self.assertEqual(len(padded_seqs2[0]), 4, "Expected length of output to be 4, got {}.".format(len(padded_seqs2[0])))
        self.assertIsInstance(padded_seqs2, list, "Expected output to be of type list, got {}.".format(type(padded_seqs2)))
#4.)
        padded_seqs3 = utils.zero_padding(seq3)
        self.assertEqual(len(padded_seqs3), 4, "Expected length of output to be 4, got {}.".format(len(padded_seqs3)))
        self.assertEqual(padded_seqs3.shape, (4,5,2), "Expected output to be of shape (4,5,2), got {}.".format(padded_seqs3.shape))
        self.assertIsInstance(padded_seqs3, np.ndarray, "Expected output to be of type numpy array, got {}.".format(type(padded_seqs3)))
        self.assertTrue(padded_seqs3.any() == seq3.any(), "Expected original and padded sequences to have the same values.")
        
    def test_save_results(self):
        """ Testing save results utility function. """
#1.)
        #create dummy test results, save to csv and verify csv has been created & saved
        test_results = {'R2': 0.56, 'MSE': 0.34, 'RMSE': 0.89}
        utils.save_results(test_results, 'test_results', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results.csv')),
                "Output results csv not found in output folder: {}.".format(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results.csv')))
#2.)
        test_results1 = {'MAE': 2.10, 'MSE': 0.99, 'RPD': 1.28}
        utils.save_results(test_results1, 'test_results1', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results1.csv')),
                "Output results csv not found in output folder: {}.".format(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results1.csv')))
#3.)
        test_results2 = pd.DataFrame(np.random.randint(1,100, size=(5,3)), columns=['R2', 'MSE', 'RMSE'])
        utils.save_results(test_results2, 'test_results2', output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results2.csv')),
                "Output results csv not found in output folder: {}.".format(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results2.csv')))
#4.)
        test_results3 = pd.Series(np.random.randint(1,100), index=['Col1', 'Col2', 'Col3', 'Col4'])
        utils.save_results(test_results3, 'test_results3',  output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results3.csv')),
                "Output results csv not found in output folder: {}.".format(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results3.csv')))
#5.)
        test_results4 = np.random.randint(1, 100, size=(2,6))
        with self.assertRaises(TypeError, msg='Type Error raised, invalid input parameter data type given.'):
            utils.save_results(test_results4, 'test_results4',  output_folder=os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder)))
        self.assertFalse(os.path.isfile(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results4.csv')),
                "Output results csv should not be found in output folder: {}.".format(os.path.join(self.test_output_folder, os.path.basename(self.test_output_folder) + "_" + _globals.CURRENT_DATETIME, 'test_results4.csv')))

    def test_map(self):
        """ Testing Map class which allows for a dict to be accessed via dot notation. """
#1.)
        test_map1 = utils.Map({"first_name":"Joe", "last_name":"Bloggs", "country":"Ireland", "city":"Dublin"})
        
        self.assertIsInstance(test_map1, dict, "Expected instance to be of type dict, got {}.".format(type(test_map1)))
        self.assertEqual(test_map1.first_name, "Joe", "Expected Joe, got {}.".format(test_map1.first_name))
        self.assertEqual(test_map1.last_name, "Bloggs", "Expected Bloggs, got {}.".format(test_map1.last_name))
        self.assertEqual(test_map1.country, "Ireland", "Expected Ireland, got {}.".format(test_map1.country))
        self.assertEqual(test_map1.city, "Dublin", "Expected Dublin, got {}.".format(test_map1.city))
        self.assertEqual(len(test_map1), 4, "Expected output length to be 4, got {}.".format(len(test_map1)))
#2.)
        test_map2 = utils.Map({"first_name":"John", "last_name":"Smith"}, country="Germany", city="Hanover")

        self.assertIsInstance(test_map2, dict, "Expected instance to be of type dict, got {}.".format(type(test_map2)))
        self.assertEqual(test_map2.first_name, "John", "Expected John, got {}.".format(test_map2.first_name))
        self.assertEqual(test_map2.last_name, "Smith", "Expected Smith, got {}.".format(test_map2.last_name))
        self.assertEqual(test_map2.country, "Germany", "Expected Germany, got {}.".format(test_map2.country))
        self.assertEqual(test_map2.city, "Hanover", "Expected Hanover, got {}.".format(test_map2.city))
        self.assertEqual(len(test_map2), 4, "Expected output length to be 4, got {}.".format(len(test_map2)))
#3.)
        test_map3 = utils.Map({})

        self.assertIsInstance(test_map3, dict, "Expected instance to be of type dict, got {}.".format(type(test_map3)))
        self.assertEqual(test_map3, {}, "Expected an empty dict, got {}.".format(test_map3))
        self.assertEqual(len(test_map3), 0, "Expected output length to be 0, got {}.".format(len(test_map3)))
#4.)    
        test_map1.language = "Python"
        test_map1["age"] = 42
        self.assertEqual(test_map1.language, "Python", "Expected Python, got {}.".format(test_map1.language))
        self.assertEqual(test_map1.age, 42, "Expected 42, got {}.".format(test_map1.age))
        self.assertEqual(len(test_map1), 6, "Expected output length to be 6, got {}.".format(len(test_map1)))

        test_map2.language = "C++"
        test_map2.age = 20
        self.assertEqual(test_map2.language, "C++", "Expected C++, got {}.".format(test_map2.language))
        self.assertEqual(test_map2.age, 20, "Expected 20, got {}.".format(test_map2.age))
        self.assertEqual(len(test_map2), 6, "Expected output length to be 6, got {}.".format(len(test_map2)))

        test_map3.language = "Ruby"
        test_map3.age = 99
        self.assertEqual(test_map3.language, "Ruby", "Expected Ruby, got {}.".format(test_map3.language))
        self.assertEqual(test_map3.age, 99, "Expected 99, got {}.".format(test_map3.age))
        self.assertEqual(len(test_map3), 2, "Expected output length to be 2, got {}.".format(len(test_map3)))
#5.)
        del test_map1.first_name
        self.assertEqual(len(test_map1), 5, "Expected output length to be 5, got {}.".format(len(test_map1)))
        del test_map1.country
        self.assertEqual(len(test_map1), 4, "Expected output length to be 4, got {}.".format(len(test_map1)))
        del test_map3.language
        self.assertEqual(len(test_map3), 1, "Expected output length to be 1, got {}.".format(len(test_map3)))
#6.)
        with self.assertRaises(TypeError):
                utils.Map(1245)
                utils.Map(10.4)
                utils.Map(False)

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4

        #removing any of the temp files created such as the results files/outputs
        shutil.rmtree(self.test_output_folder , ignore_errors=False, onerror=None)