
################################################################################
#################               Utils Module Tests             #################
################################################################################
import os
import sys
import unittest
import numpy as np
import pandas as pd

from aaindex import AAIndex
from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from utils import *

class UtilsTest(unittest.TestCase):

    def setUp(self):

        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests','test_data','test_thermostability.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1')
        try:
            self.test_dataset2 = pd.read_csv(os.path.join('tests','test_data','test_enantioselectivity.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset2')
        try:
            self.test_dataset3 = pd.read_csv(os.path.join('tests','test_data','test_localization.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset3')
        try:
            self.test_dataset4 = pd.read_csv(os.path.join('tests','test_data','test_absorption.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset4')

    def test_valid_sequence(self):



        pass

    def test_remove_gaps(self):
        pass
        # print('here')
        # gaps_test = utils.remove_gaps(desc.protein_seqs)
        # print('gaps',gaps_test)
        #
        # self.assertIsNone(gaps_test)


    def test_flatten(self):

        test_array = np.random.rand(100,3)

        flattened_array = utils.flatten(test_array)

        self.assertEqual(flattened_array.shape, (300,1))
        self.assertIsInstance(flattened_array, np.ndarray)
        self.assertEqual(flattened_array.ndim, 1)

        test_array = np.random.rand(42,720)

        flattened_array = utils.flatten(test_array)

        self.assertEqual(flattened_array.shape, (1,30240))
        self.assertIsInstance(flattened_array, np.ndarray)
        self.assertEqual(flattened_array.ndim, 1)

        test_array = np.random.rand(1,2)

        flattened_array = utils.flatten(test_array)

        self.assertEqual(flattened_array.shape, (1,2))
        self.assertIsInstance(flattened_array, np.ndarray)
        self.assertEqual(flattened_array.ndim, 1)


    def test_parse_json(self):
        pass

    def test_create_output_dir(self):





        pass
