################################################################################
#################              PyDSP Module Tests             #################
################################################################################

import os
import pandas as pd
import numpy as np
import pySAR.aaindex as aaindex
import pySAR.pyDSP as pyDSP_
import pySAR.pySAR as pysar
import unittest

@unittest.skip("")
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
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0]) #test_thermostability
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter_)
        self.assertEqual(pyDSP.encoded_sequences.shape, (self.pySAR.num_seqs, self.pySAR.seq_len))
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
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[1]) #test_enantioselectivity
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

#3.) 
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[2], protein_seqs="") #test_absorption
        #pre-encode sequences
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter_)
        self.assertEqual(pyDSP.encoded_sequences.shape, (self.pySAR.num_seqs, self.pySAR.seq_len))
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
#4.) 
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[3]) #test_localization
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter_)
        self.assertEqual(pyDSP.encoded_sequences.shape, (self.pySAR.num_seqs, self.pySAR.seq_len))
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
#3.)
        with self.assertRaises(OSError, msg='OS Error raised, invalid config file path given.'):
            pyDSP = pyDSP_.PyDSP(dsp_config="blahblahblah")

        with self.assertRaises(TypeError, msg='Type Error raised, invalid config file path data type given.'):
            pyDSP = pyDSP_.PyDSP(dsp_config=4.21)

    def test_preprocessing(self):
        """ Testing preprocessing functionality of pyDSP class. """
        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.PyDSP(encoded_seq1)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)))
        self.assertTrue(np.all((pyDSP.fft_real==0)))
        self.assertTrue(np.all((pyDSP.fft_imag==0)))
        self.assertTrue(np.all((pyDSP.fft_abs==0)))
        self.assertFalse(np.isnan(pyDSP.encoded_sequences).any(), 'Sequences should not contain null values.')
#2.)

    def test_protein_spectra(self):
        """ Testing getting protein spectra from encoded protein sequences. """
        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.PyDSP(encoded_seq1)
        self.assertTrue(pyDSP.fft_power.dtype, "complex128")
        self.assertTrue(pyDSP.fft_real.dtype, "complex128")
        self.assertTrue(pyDSP.fft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.fft_abs.dtype, "complex128")

    def test_max_freq(self):
        """ Testing max frequency functionality. """
#1.)
        test_aaindices1 = "COHE430101"
        encoded_seq1 = self.pySAR.get_aai_enoding(test_aaindices1)
        pyDSP = pyDSP_.PyDSP(encoded_seq1)

