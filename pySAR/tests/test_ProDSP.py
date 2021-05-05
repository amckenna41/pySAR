################################################################################
#################              ProDSP Module Tests             #################
################################################################################

import os
import sys
import pandas as pd
import numpy as np
from aaindex import AAIndex
from proDSP import ProDSP
from pySAR import PySAR
import unittest
import requests
import urllib.request

class ProDSPTests(unittest.TestCase):

    def setUp(self):

        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests','test_data','test_thermostability.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1.')
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

        self.aaindex = AAIndex()
        self.pySAR = PySAR(algorithm="PLSReg",dataset=os.path.join('tests','test_data','test_enantioselectivity.txt'), activity="e-value")

    def test_proDSP(self):
        #test general stuff like the input parameters etc


        test_aaindices1 = "BURA740101"
        test_aaindices2 = "COHE430101"
        test_aaindices3 = ["FAUJ880108","NAKH900102","YUTK870103"]
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        proDSP = ProDSP(encoded_seq1)
        self.assertEqual(proDSP.spectrum, "power")
        self.assertEqual(proDSP.window_type, "hamming")
        self.assertIsInstance(proDSP.window, np.ndarray)
        self.assertIsNone(proDSP.filter_)
        self.assertEqual(proDSP.encoded_sequences.shape, (self.pySAR.num_seqs,self.pySAR.seq_len))
        self.assertEqual(proDSP.num_seqs, self.pySAR.num_seqs)
        self.assertEqual(proDSP.signal_len, self.pySAR.seq_len)
        self.assertEqual(proDSP.fft_power.shape, encoded_seq1.shape)
        self.assertEqual(proDSP.fft_real.shape, encoded_seq1.shape)
        self.assertEqual(proDSP.fft_abs.shape, encoded_seq1.shape)
        self.assertEqual(proDSP.fft_imag.shape, encoded_seq1.shape)
        # self.assertTrue(proDSP.spectrum_encoding == proDSP.fft_power)
        self.assertTrue(proDSP.spectrum_encoding.any() == proDSP.fft_power.any())
        self.assertEqual(proDSP.fft_freqs.shape,encoded_seq1.shape)
#2.)
        proDSP = ProDSP(encoded_seq1, window="notawindow")
        self.assertEqual(proDSP.window, 1)
#3.)
        with(self.assertRaises(ValueError)):
            proDSP = ProDSP(encoded_seq1, spectrum="blahblahblah")

        with(self.assertRaises(ValueError)):
            proDSP = ProDSP(encoded_seq1, spectrum=None)
#4.)
        proDSP = ProDSP(encoded_seq1, filter_="blahblahblah")
        self.assertEqual(proDSP.filter_, "")
#5.)
        #test window closeness function
        proDSP = ProDSP(encoded_seq1, window = "hamm")
        self.assertEqual(proDSP.window_type, "hamming")
        proDSP = ProDSP(encoded_seq1, window = "bart")
        self.assertEqual(proDSP.window_type, "bartlett")
        proDSP = ProDSP(encoded_seq1, window = "gausi")
        self.assertEqual(proDSP.window_type, "gaussian")
#6.)    #
        # proDSP = ProDSP(encoded_seq1, spectrum = "absolut")
        # self.assertEqual(proDSP.spectrum, "absolute")
        # proDSP = ProDSP(encoded_seq1, spectrum = "imagin")
        # self.assertEqual(proDSP.spectrum, "imaginary")

        #create array of Random 2D array, pass into FFT

    def test_preprocessing(self):
        """ """

        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        proDSP = ProDSP(encoded_seq1)
        proDSP.pre_processing()
        self.assertTrue(np.all((proDSP.fft_power==0)))
        self.assertTrue(np.all((proDSP.fft_real==0)))
        self.assertTrue(np.all((proDSP.fft_imag==0)))
        self.assertTrue(np.all((proDSP.fft_abs==0)))
        self.assertFalse(np.isnan(proDSP.encoded_sequences).any(),
            'Sequences contain null values.')
#2.)

    def test_protein_spectra(self):
        """ """

        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        proDSP = ProDSP(encoded_seq1)
        self.assertTrue(proDSP.fft_power.dtype, "complex128")
        self.assertTrue(proDSP.fft_real.dtype, "complex128")
        self.assertTrue(proDSP.fft_imag.dtype, "complex128")
        self.assertTrue(proDSP.fft_abs.dtype, "complex128")



    def test_max_freq(self):

        test_aaindices1 = "COHE430101"

        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        proDSP = ProDSP(encoded_seq1)

# proDSP.fft.dtype == dtype('complex128')
# proDSP.rfft.dtype == dtype('complex128')
