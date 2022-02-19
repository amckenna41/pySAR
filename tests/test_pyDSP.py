################################################################################
#################              PyDSP Module Tests             #################
################################################################################

import os
import pandas as pd
import numpy as np
import pySAR.aaindex as aaindex
import pySAR.pyDSP as pyDSP_
import pySAR.pySAR as pysar
import pySAR
import unittest
import urllib.request

class pyDSPTests(unittest.TestCase):

    def setUp(self):
        """  Import the 4 config files for each of the 4 datasets used for testing the pyDSP methods. """        
        #array of config files for each test dataset
        config_path = os.path.join('tests','test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]

    def test_pyDSP(self):
        """ Test class input parameters and attributes. """
#1.)
        # encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)            
        pyDSP = pyDSP_.pyDSP(dsp_config=self.all_config_files[0])
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter_)
        self.assertEqual(pyDSP.encoded_sequences.shape, (self.pySAR.num_seqs,self.pySAR.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pySAR.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pySAR.seq_len)
        # self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape)
        self.assertTrue(pyDSP.fft_power.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_real.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64')
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any())
        # self.assertEqual(pyDSP.fft_freqs.shape,encoded_seq1.shape)
#2.)
        pyDSP = pyDSP_.pyDSP(dsp_config=self.all_config_files[1])


#2.)
        pyDSP = pyDSP_.pyDSP(encoded_seq1, window="notawindow")
        self.assertEqual(pyDSP.window, 1)
#3.)
        with(self.assertRaises(ValueError)):
            pyDSP = pyDSP_.pyDSP(encoded_seq1, spectrum="blahblahblah")

        with(self.assertRaises(ValueError)):
            pyDSP = pyDSP_.pyDSP(encoded_seq1, spectrum=None)
#4.)
        pyDSP = pyDSP_.pyDSP(encoded_seq1, filter_="blahblahblah")
        self.assertEqual(pyDSP.filter_, "")
#5.)
        #test window closeness function
        pyDSP = pyDSP_.pyDSP(encoded_seq1, window = "hamm")
        self.assertEqual(pyDSP.window_type, "hamming")
        pyDSP = pyDSP_.pyDSP(encoded_seq1, window = "bart")
        self.assertEqual(pyDSP.window_type, "bartlett")
        pyDSP = pyDSP_.pyDSP(encoded_seq1, window = "gausi")
        self.assertEqual(pyDSP.window_type, "gaussian")
#6.)
        pyDSP = pyDSP_.pyDSP(encoded_seq1, spectrum = "absolute")
        self.assertEqual(pyDSP.spectrum, "absolute")
        pyDSP = pyDSP_.pyDSP(encoded_seq1, spectrum = "imaginary")
        self.assertEqual(pyDSP.spectrum, "imaginary")

        self.assertEqual(pyDSP.window_type, "gaussian")
#7.)
        pyDSP = pyDSP_.pyDSP(encoded_seq1, filter_ = "savgol")
        self.assertEqual(pyDSP.filter_, "savgol")

    def test_preprocessing(self):
        """ Testing preprocessing functionality of pyDSP class. """

        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.pyDSP(encoded_seq1)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)))
        self.assertTrue(np.all((pyDSP.fft_real==0)))
        self.assertTrue(np.all((pyDSP.fft_imag==0)))
        self.assertTrue(np.all((pyDSP.fft_abs==0)))
        self.assertFalse(np.isnan(pyDSP.encoded_sequences).any(),
            'Sequences contain null values.')
#2.)

    def test_protein_spectra(self):
        """ Testing getting protein spectra from encoded protein sequences. """

        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.pyDSP(encoded_seq1)
        self.assertTrue(pyDSP.fft_power.dtype, "complex128")
        self.assertTrue(pyDSP.fft_real.dtype, "complex128")
        self.assertTrue(pyDSP.fft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.fft_abs.dtype, "complex128")

    def test_max_freq(self):
        """ Testing max frequency functionality. """
#1.)
        test_aaindices1 = "COHE430101"
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.pyDSP(encoded_seq1)
