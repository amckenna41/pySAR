
################################################################################
#################               Utils Module Tests             #################
################################################################################
import os
import sys
import unittest
import requests
import urllib.request

from aaindex import AAIndex

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
        pass

    def test_parse_json(self):
        pass

    def test_create_output_dir(self):
        pass
