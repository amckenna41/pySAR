################################################################################
#################              ProDSP Module Tests             #################
################################################################################

import os
import sys
from aaindex import AAIndex
import unittest
import requests
import urllib.request

class ProDSPTests(unittest.TestCase):

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

    def test_proDSP(self):
        #test general stuff like the input parameters etc

        pass
    def test_preprocessing(self):
        pass

    def test_fft(self):

        pass

    def test_protein_spectra(self):

        pass

    def test_max_freq(self):
        pass

    pass
