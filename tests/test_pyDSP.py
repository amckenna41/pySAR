################################################################################
#################              PyDSP Module Tests             #################
################################################################################

import os
import numpy as np
import pySAR.pyDSP as pyDSP_
import pySAR.pySAR as pySAR
import unittest

class pyDSPTests(unittest.TestCase):
    """
    Test suite for testing pyDSP module and functionality 
    in pySAR package. 

    Test Cases
    ----------
    test_pyDSP:
        testing correct overall pyDSP class and module functionality.
    test_preprocessing:
        testing correct pydsp pre processing functionality.
    test_protein_spectra:
        testing correct pydsp protein_spectra functionality.
    test_window:
        testing correct pydsp window functionality.
    test_filter:
        testing correct pydsp filter functionality.
    test_max_freq:
        testing correct max_freq pydsp functionality.
    """
    def setUp(self):
        """  Import the 4 config files for each of the 4 datasets used for testing the pyDSP methods. """        
        #array of config files for each test dataset
        config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]
        #create instance of pysar class using thermostability dataset & config
        self.pysar = pySAR.PySAR(config_file=self.all_config_files[0])

    def test_pyDSP(self):
        """ Test class input parameters and attributes. """
        aa_indices1 = "EISD860101"
        aa_indices2 = "GEIM800107"
        aa_indices3 = "NAKH900106"
        aa_indices4 = "QIAN880105"
#1.)   
        encoded_seq1 = self.pysar.get_aai_encoding(aa_indices1)            
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[0], protein_seqs=encoded_seq1) #test_thermostability
        
        self.assertEqual(pyDSP.spectrum, "power", 
            "Output spectrum should be power, got {}.".format(pyDSP.spectrum))
        self.assertEqual(pyDSP.window_type, "blackmanharris", 
            "Output window function should be blackmanharris, got {}.".format(pyDSP.window_type))
        self.assertIsInstance(pyDSP.window, np.ndarray, 
            "Output from window function should be a numpy array.")
        self.assertIsNone(pyDSP.filter, 
            "Filter function should be None on class initialisation.")
        self.assertIsNone(pyDSP.filter_type, 
            "Filter type expected to be None, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len),
            "Spectrum encoding shape expected to be ({}, {}), got {}.".format(self.pysar.num_seqs, self.pysar.seq_len, pyDSP.spectrum_encoding.shape))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs,
            "num_seqs attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.num_seqs))
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len,
            "signal_len attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.seq_len))
        self.assertEqual(pyDSP.fft_power.dtype, 'float64',
            "power spectrum expected to be of type float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape,
            "FFT encoding with power spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_real.dtype, 'float64',
            "real spectrum expected to be of type float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape,
            "FFT encoding with real spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64',
            "absolute spectrum expected to be of type float64, got {}.".format(pyDSP.fft_abs.dtype))
        self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape,
            "FFT encoding with absolute spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64',
            "imaginary spectrum expected to be of type float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape,
            "FFT encoding with imaginary spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_freqs.shape, encoded_seq1.shape,
            "FFT frequencies expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any(), 
            "Spectrum encoding attribute should equal that of chosen fft spectrum, power.")
        self.assertEqual(pyDSP.fft_power.dtype, "float64",
            "Data type of power spectrum should be float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_real.dtype, "float64",
            "Data type of real spectrum should be float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_imag.dtype, "float64",
            "Data type of imaginary spectrum should be float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_abs.dtype, "float64",
            "Data type of absolute spectrum should be float64, got {}.".format(pyDSP.fft_abs.dtype))
#2.)
        encoded_seq2 = self.pysar.get_aai_encoding(aa_indices2)            
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[1], protein_seqs=encoded_seq2) #test_enantioselectivity
        
        self.assertEqual(pyDSP.spectrum, "absolute", 
            "Output spectrum should be absolute, got {}.".format(pyDSP.spectrum))
        self.assertEqual(pyDSP.window_type, "blackman", 
            "Output window function should be blackman, got {}.".format(pyDSP.window_type))
        self.assertIsInstance(pyDSP.window, np.ndarray, 
            "Output from window function should be a numpy array.")
        self.assertIsNone(pyDSP.filter, 
            "Filter function should be None on class initialisation.")
        self.assertIsNone(pyDSP.filter_type, 
            "Filter type expected to be None, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len),
            "Spectrum encoding shape expected to be ({}, {}), got {}.".format(self.pysar.num_seqs, self.pysar.seq_len, pyDSP.spectrum_encoding.shape))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs,
            "num_seqs attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.num_seqs))
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len,
            "signal_len attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.seq_len))
        self.assertEqual(pyDSP.fft_power.dtype, 'float64',
            "power spectrum expected to be of type float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape,
            "FFT encoding with power spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_real.dtype, 'float64',
            "real spectrum expected to be of type float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape,
            "FFT encoding with real spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64',
            "absolute spectrum expected to be of type float64, got {}.".format(pyDSP.fft_abs.dtype))
        self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape,
            "FFT encoding with absolute spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64',
            "imaginary spectrum expected to be of type float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape,
            "FFT encoding with imaginary spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_freqs.shape, encoded_seq1.shape,
            "FFT frequencies expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any(), 
            "Spectrum encoding attribute should equal that of chosen fft spectrum, power.")
        self.assertEqual(pyDSP.fft_power.dtype, "float64",
            "Data type of power spectrum should be float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_real.dtype, "float64",
            "Data type of real spectrum should be float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_imag.dtype, "float64",
            "Data type of imaginary spectrum should be float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_abs.dtype, "float64",
            "Data type of absolute spectrum should be float64, got {}.".format(pyDSP.fft_abs.dtype))
#3.) 
        encoded_seq3 = self.pysar.get_aai_encoding(aa_indices3)            
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs=encoded_seq3) #test_absorption
        
        self.assertEqual(pyDSP.spectrum, "power", 
            "Output spectrum should be power, got {}.".format(pyDSP.spectrum))
        self.assertEqual(pyDSP.window_type, "hamming", 
            "Output window function should be hamming, got {}.".format(pyDSP.window_type))
        self.assertIsInstance(pyDSP.window, np.ndarray, 
            "Output from window function should be a numpy array.")
        self.assertIsNone(pyDSP.filter, 
            "Filter function should be None on class initialisation.")
        self.assertIsNone(pyDSP.filter_type, 
            "Filter type expected to be None, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len),
            "Spectrum encoding shape expected to be ({}, {}), got {}.".format(self.pysar.num_seqs, self.pysar.seq_len, pyDSP.spectrum_encoding.shape))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs,
            "num_seqs attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.num_seqs))
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len,
            "signal_len attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.seq_len))
        self.assertEqual(pyDSP.fft_power.dtype, 'float64',
            "power spectrum expected to be of type float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape,
            "FFT encoding with power spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_real.dtype, 'float64',
            "real spectrum expected to be of type float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape,
            "FFT encoding with real spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64',
            "absolute spectrum expected to be of type float64, got {}.".format(pyDSP.fft_abs.dtype))
        self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape,
            "FFT encoding with absolute spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64',
            "imaginary spectrum expected to be of type float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape,
            "FFT encoding with imaginary spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_freqs.shape, encoded_seq1.shape,
            "FFT frequencies expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any(), 
            "Spectrum encoding attribute should equal that of chosen fft spectrum, power.")
        self.assertEqual(pyDSP.fft_power.dtype, "float64",
            "Data type of power spectrum should be float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_real.dtype, "float64",
            "Data type of real spectrum should be float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_imag.dtype, "float64",
            "Data type of imaginary spectrum should be float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_abs.dtype, "float64",
            "Data type of absolute spectrum should be float64, got {}.".format(pyDSP.fft_abs.dtype))
#4.) 
        encoded_seq4 = self.pysar.get_aai_encoding(aa_indices4)            
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=encoded_seq4) #test_localization
        
        self.assertEqual(pyDSP.spectrum, "imaginary", 
            "Output spectrum should be imaginary, got {}.".format(pyDSP.spectrum))
        self.assertEqual(pyDSP.window_type, "bartlett", 
            "Output window function should be bartlett, got {}.".format(pyDSP.window_type))
        self.assertIsInstance(pyDSP.window, np.ndarray, 
            "Output from window function should be a numpy array.")
        self.assertIsNone(pyDSP.filter, 
            "Filter function should be None on class initialisation.")
        self.assertIsNone(pyDSP.filter_type, 
            "Filter type expected to be None, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len),
            "Spectrum encoding shape expected to be ({}, {}), got {}.".format(self.pysar.num_seqs, self.pysar.seq_len, pyDSP.spectrum_encoding.shape))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs,
            "num_seqs attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.num_seqs))
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len,
            "signal_len attribute in pyDSP class should equal that of pysar attribute: {}.".format(self.pysar.seq_len))
        self.assertEqual(pyDSP.fft_power.dtype, 'float64',
            "power spectrum expected to be of type float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape,
            "FFT encoding with power spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_real.dtype, 'float64',
            "real spectrum expected to be of type float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape,
            "FFT encoding with real spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64',
            "absolute spectrum expected to be of type float64, got {}.".format(pyDSP.fft_abs.dtype))
        self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape,
            "FFT encoding with absolute spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64',
            "imaginary spectrum expected to be of type float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape,
            "FFT encoding with imaginary spectrum expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertEqual(pyDSP.fft_freqs.shape, encoded_seq1.shape,
            "FFT frequencies expected to be same shape as encoded sequences: {}.".format(encoded_seq1.shape))
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any(), 
            "Spectrum encoding attribute should equal that of chosen fft spectrum, power.")
        self.assertEqual(pyDSP.fft_power.dtype, "float64",
            "Data type of power spectrum should be float64, got {}.".format(pyDSP.fft_power.dtype))
        self.assertEqual(pyDSP.fft_real.dtype, "float64",
            "Data type of real spectrum should be float64, got {}.".format(pyDSP.fft_real.dtype))
        self.assertEqual(pyDSP.fft_imag.dtype, "float64",
            "Data type of imaginary spectrum should be float64, got {}.".format(pyDSP.fft_imag.dtype))
        self.assertEqual(pyDSP.fft_abs.dtype, "float64",
            "Data type of absolute spectrum should be float64, got {}.".format(pyDSP.fft_abs.dtype))
#5.)
        with self.assertRaises(OSError, msg='OS Error raised, invalid config file path given.'):
            pyDSP = pyDSP_.PyDSP(config_file="blahblahblah")
#6.)
        with self.assertRaises(TypeError, msg='Type Error raised, invalid config file path data type given.'):
            pyDSP = pyDSP_.PyDSP(config_file=4.21)
#7.)
        with self.assertRaises(ValueError, msg='Value Error raised, protein sequences input parameter cant be none.'):
            pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=None)
#8.)
        with self.assertRaises(ValueError, msg='Value Error raised, protein sequences input parameter cant be a single str.'):
            pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs="ABCDEF")
    
    def test_preprocessing(self):
        """ Testing preprocessing functionality of pyDSP class. """
        test_aaindices1 = "COHE430101"
        test_aaindices2 = "LEVM780105"
        test_aaindices3 = "QIAN880107"
        test_aaindices4 = "ROSG850102"
#1.)
        encoded_seq1 = self.pysar.get_aai_encoding(test_aaindices1)
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[0], protein_seqs=encoded_seq1)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)), 
            "Power spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_real==0)), 
            "Real spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_imag==0)),
            "Imaginary spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_abs==0)),
            "Absolute spectrum should be initialised into zeros array on instantiation.")
        self.assertFalse(np.isnan(pyDSP.spectrum_encoding).any(), 
            'Sequences after pre-processing step should not contain null values.')
#2.)
        encoded_seq2 = self.pysar.get_aai_encoding(test_aaindices2)
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[1], protein_seqs=encoded_seq2)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)), 
            "Power spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_real==0)), 
            "Real spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_imag==0)),
            "Imaginary spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_abs==0)),
            "Absolute spectrum should be initialised into zeros array on instantiation.")
        self.assertFalse(np.isnan(pyDSP.spectrum_encoding).any(), 
            'Sequences after pre-processing step should not contain null values.')
#3.)
        encoded_seq3 = self.pysar.get_aai_encoding(test_aaindices3)
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs=encoded_seq3)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)), 
            "Power spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_real==0)), 
            "Real spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_imag==0)),
            "Imaginary spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_abs==0)),
            "Absolute spectrum should be initialised into zeros array on instantiation.")
        self.assertFalse(np.isnan(pyDSP.spectrum_encoding).any(), 
            'Sequences after pre-processing step should not contain null values.')
#4.)
        encoded_seq4 = self.pysar.get_aai_encoding(test_aaindices4)
        pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=encoded_seq4)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)), 
            "Power spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_real==0)), 
            "Real spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_imag==0)),
            "Imaginary spectrum should be initialised into zeros array on instantiation.")
        self.assertTrue(np.all((pyDSP.fft_abs==0)),
            "Absolute spectrum should be initialised into zeros array on instantiation.")
        self.assertFalse(np.isnan(pyDSP.spectrum_encoding).any(), 
            'Sequences after pre-processing step should not contain null values.')

    def test_max_freq(self):
        """ Testing max frequency functionality. """
#1.)
        test_aaindices1 = "COHE430101"
        encoded_seq1 = self.pysar.get_aai_encoding(test_aaindices1)
        for config in self.all_config_files:
            pyDSP = pyDSP_.PyDSP(config_file=config, protein_seqs=encoded_seq1)
            max_freq_, max_freq_index = pyDSP.max_freq(pyDSP.spectrum_encoding[0])
            self.assertIsInstance(max_freq_, float, "")
            self.assertIsInstance(max_freq_index, np.int64, "")
#2.)
        with self.assertRaises(ValueError):
            max_freq_, max_freq_index = pyDSP.max_freq(pyDSP.spectrum_encoding)