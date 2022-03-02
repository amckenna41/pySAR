
################################################################################
#################             Utilities Module Tests           #################
################################################################################
import os
import shutil
import unittest
import numpy as np
# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
import pandas as pd

import pySAR.globals_ as _globals
import pySAR.utils as utils

class UtilsTest(unittest.TestCase):

    def setUp(self):
        """ Import all test datasets. """
        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_thermostability.txt'), sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1.')
        try:
            self.test_dataset2 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_enantioselectivity.txt'), sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset2.')
        try:
            self.test_dataset3 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_localization.txt'), sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset3.')
        try:
            self.test_dataset4 = pd.read_csv(os.path.join('tests', 'test_data',
                'test_absorption.txt'), sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset4.')

        #append all datasets to a list
        self.all_test_datasets = [self.test_dataset1, self.test_dataset2, 
                self.test_dataset3, self.test_dataset4]

        #set global vars to create temp test data folders
        _globals.OUTPUT_DIR = os.path.join('tests', _globals.OUTPUT_DIR)
        _globals.OUTPUT_FOLDER = os.path.join('tests', _globals.OUTPUT_FOLDER)

    def test_valid_sequence(self):
        """ Test Valid/Invalid Sequences utility function. """
        invalid_seqs = [["A", "B", "C", "D"],["E","F","J"]]
        invalid_seqs1 = ["ZZZZZZ"]
        invalid_seqs2 = [["Z", 2, "Y", "X", 321]]
        invalid_seqs3 = "XXZXXZXXZ"
#1.)
        #testing with invalid sequences
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs))
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs1))
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs2))
        self.assertIsNotNone(utils.valid_sequence(invalid_seqs3))
#2.)
        self.assertEqual(len(utils.valid_sequence(invalid_seqs)), 2)
        self.assertEqual(len(utils.valid_sequence(invalid_seqs1)), 6)
        self.assertEqual(len(utils.valid_sequence(invalid_seqs2)), 4)
        self.assertEqual(len(utils.valid_sequence(invalid_seqs3)), 9)
#3.)
        self.assertIsInstance((utils.valid_sequence(invalid_seqs)), list)
        self.assertIsInstance((utils.valid_sequence(invalid_seqs1)), list)
        self.assertIsInstance((utils.valid_sequence(invalid_seqs2)), list)
        self.assertIsInstance((utils.valid_sequence(invalid_seqs3)), list)
#4.)
        #testing with valid sequences
        random_seq = np.random.randint(0, len(self.test_dataset1))
        random_seq1 = np.random.randint(0, len(self.test_dataset1))
        random_seq2 = np.random.randint(0, len(self.test_dataset1))

        self.assertIsNone(utils.valid_sequence(self.test_dataset1['sequence'][random_seq]))
        self.assertIsNone(utils.valid_sequence(self.test_dataset1['sequence'][random_seq1]))
        self.assertIsNone(utils.valid_sequence(self.test_dataset1['sequence'][random_seq2]))

    def test_remove_gaps(self):
        """ Test utility function that removes any gaps from sequences. """
        seq1 = ["A", "B", "C", "D", "-"]
        seq2 = ["A", "B", "C", "D", "-", "-", "-", "E", "F", "-"]
        seq3 = 'ABCDFSDJWD---'
        seq4 = "YUJBVFGHYJ---ASD"
#1.)
        seq1_test = utils.remove_gaps(seq1)
        self.assertEqual(len(seq1_test), 1)
        self.assertEqual(len(seq1_test[0]), 4)
        self.assertIsInstance(seq1_test, list)
        self.assertNotIn('-', seq1_test)
#2.)
        seq2_test = utils.remove_gaps(seq2)
        self.assertEqual(len(seq2_test), 1)
        self.assertEqual(len(seq2_test[0]), 6)
        self.assertIsInstance(seq2_test, list)
        self.assertNotIn('-', seq2_test)
#3.)
        seq3_test = utils.remove_gaps(seq3)
        self.assertEqual(len(seq3_test), 10)
        self.assertIsInstance(seq3_test, str)
        self.assertNotIn('-', seq3_test)
#4.)
        seq4_test = utils.remove_gaps(seq4)
        self.assertEqual(len(seq4_test), 13)
        self.assertIsInstance(seq4_test, str)
        self.assertNotIn('-', seq4_test)

    def test_flatten(self):
        """ Test flatten utility function that flattens an array or list. """
        seq1 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        seq2 = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]], np.int32)
        seq3 = np.random.randint(10,90,(4,5,2))
        seq4 = ["A", "B", "C", "D", "E", "F"]
        seq5 = "TUVWXYZ"
#1.)
        flattened_array = utils.flatten(seq1)
        self.assertEqual(flattened_array.shape, (6,1))
        self.assertIsInstance(flattened_array, np.ndarray)
        self.assertEqual(flattened_array.ndim, 2)
        self.assertTrue((np.array([[1],[2],[3],[4],[5],[6]]) == flattened_array).all())
#2.)
        flattened_array_2 = utils.flatten(seq2)
        self.assertEqual(flattened_array_2.shape, (9,1))
        self.assertIsInstance(flattened_array_2, np.ndarray)
        self.assertEqual(flattened_array_2.ndim, 2)
        self.assertTrue((np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]]) == flattened_array_2).all())
#3.)
        flattened_array_3 = utils.flatten(seq3)
        self.assertEqual(flattened_array_3.shape, (40,1))
        self.assertIsInstance(flattened_array_3, np.ndarray)
        self.assertEqual(flattened_array_3.ndim, 2)
#4.)
        flattened_array_4 = utils.flatten(seq4)
        self.assertEqual(len(flattened_array_4), 6)
        self.assertIsInstance(flattened_array_4, list)
        self.assertEqual(flattened_array_4,seq4)
#5.)
        flattened_array_5 = utils.flatten(seq5)
        self.assertEqual(flattened_array_5, seq5)
        self.assertIsInstance(flattened_array_5, str)

    def test_zero_padding(self):
        """ Test zero padding utility function that pads an array or list with 0's. """
        seq1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8]], dtype=object)
        seq2 = [["A", "B", "C"], ["D", "E", "F", "G"],["H"]]
        seq3 = np.random.randint(10,90,(4,5,2))
#1.)
        test_dataset3_padded = utils.zero_padding((self.test_dataset3['sequence']))

        #verify all sequences in dataset have been zero-padded to same length
        for seq in range(0, len(test_dataset3_padded)):
            self.assertEqual(len(test_dataset3_padded[seq]), 361)
            self.assertIsInstance(test_dataset3_padded[seq], str)

        self.assertIsInstance(test_dataset3_padded, pd.Series)
        self.assertEqual(test_dataset3_padded.shape[0], 254)
#2.)
        padded_seqs1 = utils.zero_padding(seq1)
        self.assertEqual(len(padded_seqs1), 2)
        self.assertEqual(len(padded_seqs1[0]), 5)
        self.assertIsInstance(padded_seqs1, np.ndarray)
#3.)
        padded_seqs2 = utils.zero_padding(seq2)
        self.assertEqual(len(padded_seqs2), 3)
        self.assertEqual(len(padded_seqs2[0]), 4)
        self.assertIsInstance(padded_seqs2, list)
#4.)
        padded_seqs3 = utils.zero_padding(seq3)
        self.assertEqual(len(padded_seqs3), 4)
        self.assertEqual(padded_seqs3.shape, (4,5,2))
        self.assertIsInstance(padded_seqs3, np.ndarray)
        self.assertTrue(padded_seqs3.any() == seq3.any())

    @unittest.skip('')
    def test_create_output_dir(self):
        """ Testing create output directory utility function. """
        utils.create_output_dir()
#1.)
        #verify folders and directories have been created
        self.assertTrue(os.path.isdir(_globals.OUTPUT_DIR))
        self.assertTrue(os.path.isdir(_globals.OUTPUT_FOLDER))

    @unittest.skip('')
    def test_save_results(self):
        """ Testing save results utility function. """
        #create output dir to save results
        utils.create_output_dir()
#1.)
        #create dummy test results, save to csv and verify csv has been created & saved
        test_results = {'R2': 0.56, 'MSE': 0.34, 'RMSE': 0.89}
        utils.save_results(test_results, 'test_results')
        self.assertTrue(os.path.isfile(os.path.join(_globals.OUTPUT_FOLDER, 'test_results.csv')))
#2.)
        test_results1 = {'MAE': 2.10, 'MSE': 0.99, 'RPD': 1.28}
        utils.save_results(test_results1, 'test_results1')
        self.assertTrue(os.path.isfile(os.path.join(_globals.OUTPUT_FOLDER, 'test_results1.csv')))
#3.)
        test_results2 = pd.DataFrame(np.random.randint(1,100, size=(5,3)), columns=['R2','MSE','RMSE'])
        utils.save_results(test_results2, 'test_results2')
        self.assertTrue(os.path.isfile(os.path.join(_globals.OUTPUT_FOLDER, 'test_results2.csv')))
#4.)
        test_results3 = pd.Series(np.random.randint(1,100), index=['Col1','Col2','Col3','Col4'])
        utils.save_results(test_results3, 'test_results3')
        self.assertTrue(os.path.isfile(os.path.join(_globals.OUTPUT_FOLDER, 'test_results3.csv')))
#5.)
        test_results4 = np.random.randint(1,100, size=(2,6))
        with self.assertRaises(TypeError, msg='Type Error raised, invalid input parameter data type given.'):
            utils.save_results(test_results4, 'test_results4')
        self.assertFalse(os.path.isfile(os.path.join(_globals.OUTPUT_FOLDER, 'test_results4.csv')))

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4

        #removing any of the temp files created such as the results files, if
        #   you want to verify the results files are actually being created then
        #       comment out the below code block.
        if (os.path.isdir(_globals.OUTPUT_DIR)):
            shutil.rmtree(_globals.OUTPUT_DIR, ignore_errors=False, onerror=None)
        if (os.path.isdir(_globals.OUTPUT_FOLDER)):
            shutil.rmtree(_globals.OUTPUT_FOLDER, ignore_errors=False, onerror=None)
