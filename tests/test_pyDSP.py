################################################################################
#################              PyDSP Module Tests             ##################
################################################################################

import os
import numpy as np
import pySAR.pyDSP as pyDSP_
import pySAR.pySAR as pySAR
import unittest
#suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore")

class pyDSPTests(unittest.TestCase):
    """
    Test suite for testing pyDSP module and functionality in pySAR package. 

    Test Cases
    ==========
    test_pyDSP:
        testing correct overall pyDSP class and module functionality.
    test_preprocessing:
        testing correct pyDSP pre processing functionality.
    test_protein_spectra:
        testing correct pyDSP protein_spectra functionality.
    test_consensus_freq:
        testing correct pyDSP consensus frequency functionality.
    test_max_freq:
        testing correct pyDSP max_freq functionality.
    """
    def setUp(self):
        """  Import the 4 config files for each of the 4 datasets used for testing the pyDSP methods. """        
        #array of config files for each test dataset
        self.config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(self.config_path, "test_thermostability.json"), 
            os.path.join(self.config_path, "test_enantioselectivity.json"),
            os.path.join(self.config_path, "test_absorption.json"), 
            os.path.join(self.config_path, "test_localization.json")
        ]

    def test_pyDSP(self):
        """ Test class input parameters and attributes. """
        aa_indices1 = "EISD860101"
        aa_indices2 = "GEIM800107"
        aa_indices3 = "NAKH900106"
        aa_indices4 = "QIAN880105"
#1.)   
        pysar_thermostability = pySAR.PySAR(config_file=self.all_config_files[0])  #thermostability
        encoded_seq_thermostability = pysar_thermostability.get_aai_encoding(aa_indices1)            
        pyDSP_thermostability = pyDSP_.PyDSP(config_file=self.all_config_files[0], protein_seqs=encoded_seq_thermostability) 
        
        self.assertEqual(pyDSP_thermostability.spectrum, "power", "Expected spectrum to be power, got {}.".format(pyDSP_thermostability.spectrum))
        self.assertEqual(pyDSP_thermostability.window_type, "hamming", "Expected window function to be hamming, got {}.".format(pyDSP_thermostability.window_type))
        self.assertIsInstance(pyDSP_thermostability.window, np.ndarray, "Expected window function to be a numpy array.")
        self.assertIsNone(pyDSP_thermostability.filter, "Expected filter function to be None on class initialisation.")
        self.assertIsNone(pyDSP_thermostability.filter_type, "Expected filter type to to be None, got {}.".format(pyDSP_thermostability.filter_type))
        self.assertEqual(pyDSP_thermostability.spectrum_encoding.shape, (pysar_thermostability.num_seqs, pysar_thermostability.sequence_length),
            "Expected spectrum encoding to be ({}, {}), got {}.".format(pysar_thermostability.num_seqs, pysar_thermostability.sequence_length, pyDSP_thermostability.spectrum_encoding.shape))
        self.assertEqual(pyDSP_thermostability.num_seqs, pysar_thermostability.num_seqs,
            "Expected num_seqs attribute in pyDSP_thermostability_thermostability class to be equal to that of pysar attribute: {}.".format(pysar_thermostability.num_seqs))
        self.assertEqual(pyDSP_thermostability.signal_len, pysar_thermostability.sequence_length,
            "Expectd signal_len attribute in pyDSP_thermostability_thermostability class to be equal that of pysar attribute: {}.".format(pysar_thermostability.sequence_length))
        self.assertEqual(pyDSP_thermostability.fft_power.dtype, 'float64', "Expected power spectrum to be of type float64, got {}.".format(pyDSP_thermostability.fft_power.dtype))
        self.assertEqual(pyDSP_thermostability.fft_power.shape, encoded_seq_thermostability.shape,
            "Expected FFT encoding with power spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_thermostability.shape))
        self.assertEqual(pyDSP_thermostability.fft_real.dtype, 'float64', "Expected real spectrum to be of type float64, got {}.".format(pyDSP_thermostability.fft_real.dtype))
        self.assertEqual(pyDSP_thermostability.fft_real.shape, encoded_seq_thermostability.shape,
            "Expected FFT encoding with real spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_thermostability.shape))
        self.assertEqual(pyDSP_thermostability.fft_abs.dtype, 'float64', "Expected absolute spectrum to be of type float64, got {}.".format(pyDSP_thermostability.fft_abs.dtype))
        self.assertEqual(pyDSP_thermostability.fft_abs.shape, encoded_seq_thermostability.shape,
            "Expected FFT encoding with absolute spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_thermostability.shape))
        self.assertEqual(pyDSP_thermostability.fft_imag.dtype, 'float64', "Expected imaginary spectrum to be of type float64, got {}.".format(pyDSP_thermostability.fft_imag.dtype))
        self.assertEqual(pyDSP_thermostability.fft_imag.shape, encoded_seq_thermostability.shape,
            "Expected FFT encoding with imaginary spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_thermostability.shape))
        self.assertEqual(pyDSP_thermostability.fft_freqs.shape, encoded_seq_thermostability.shape, "Expected FFT frequencies to be same shape as encoded sequences: {}.".format(encoded_seq_thermostability.shape))
        self.assertTrue(pyDSP_thermostability.spectrum_encoding.any() == pyDSP_thermostability.fft_power.any(), 
            "Expected spectrum encoding attribute to be equal to chosen fft spectrum, power.")
        self.assertEqual(pyDSP_thermostability.fft_power.dtype, "float64", "Expected data type of power spectrum to be float64, got {}.".format(pyDSP_thermostability.fft_power.dtype))
        self.assertEqual(pyDSP_thermostability.fft_real.dtype, "float64", "Expected data type of real spectrum to be float64, got {}.".format(pyDSP_thermostability.fft_real.dtype))
        self.assertEqual(pyDSP_thermostability.fft_imag.dtype, "float64", "Expected data type of imaginary spectrum to be float64, got {}.".format(pyDSP_thermostability.fft_imag.dtype))
        self.assertEqual(pyDSP_thermostability.fft_abs.dtype, "float64", "Expected data type of absolute spectrum to be float64, got {}.".format(pyDSP_thermostability.fft_abs.dtype))
#2.)
        pysar_enantioselectivity = pySAR.PySAR(config_file=self.all_config_files[1])  #enantioselectivity
        encoded_seq_enantioselectivity = pysar_enantioselectivity.get_aai_encoding(aa_indices2)            
        pyDSP_enantioselectivity = pyDSP_.PyDSP(config_file=self.all_config_files[1], protein_seqs=encoded_seq_enantioselectivity) 
        
        self.assertEqual(pyDSP_enantioselectivity.spectrum, "power", "Expected spectrum to be power, got {}.".format(pyDSP_enantioselectivity.spectrum))
        self.assertEqual(pyDSP_enantioselectivity.window_type, "hamming", "Expected window function to be hamming, got {}.".format(pyDSP_enantioselectivity.window_type))
        self.assertIsInstance(pyDSP_enantioselectivity.window, np.ndarray, "Expected window function to be a numpy array.")
        self.assertIsNone(pyDSP_enantioselectivity.filter, "Expected filter function to be None on class initialisation.")
        self.assertIsNone(pyDSP_enantioselectivity.filter_type, "Expected filter type to to be None, got {}.".format(pyDSP_enantioselectivity.filter_type))
        self.assertEqual(pyDSP_enantioselectivity.spectrum_encoding.shape, (pysar_enantioselectivity.num_seqs, pysar_enantioselectivity.sequence_length),
            "Expected spectrum encoding to be ({}, {}), got {}.".format(pysar_enantioselectivity.num_seqs, pysar_enantioselectivity.sequence_length, pyDSP_enantioselectivity.spectrum_encoding.shape))
        self.assertEqual(pyDSP_enantioselectivity.num_seqs, pysar_enantioselectivity.num_seqs,
            "Expected num_seqs attribute in pyDSP_enantioselectivity_thermostability class to be equal to that of pysar attribute: {}.".format(pysar_enantioselectivity.num_seqs))
        self.assertEqual(pyDSP_enantioselectivity.signal_len, pysar_enantioselectivity.sequence_length,
            "Expectd signal_len attribute in pyDSP_enantioselectivity_thermostability class to be equal that of pysar attribute: {}.".format(pysar_enantioselectivity.sequence_length))
        self.assertEqual(pyDSP_enantioselectivity.fft_power.dtype, 'float64', "Expected power spectrum to be of type float64, got {}.".format(pyDSP_enantioselectivity.fft_power.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_power.shape, encoded_seq_enantioselectivity.shape,
            "Expected FFT encoding with power spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_enantioselectivity.shape))
        self.assertEqual(pyDSP_enantioselectivity.fft_real.dtype, 'float64', "Expected real spectrum to be of type float64, got {}.".format(pyDSP_enantioselectivity.fft_real.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_real.shape, encoded_seq_enantioselectivity.shape,
            "Expected FFT encoding with real spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_enantioselectivity.shape))
        self.assertEqual(pyDSP_enantioselectivity.fft_abs.dtype, 'float64', "Expected absolute spectrum to be of type float64, got {}.".format(pyDSP_enantioselectivity.fft_abs.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_abs.shape, encoded_seq_enantioselectivity.shape,
            "Expected FFT encoding with absolute spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_enantioselectivity.shape))
        self.assertEqual(pyDSP_enantioselectivity.fft_imag.dtype, 'float64', "Expected imaginary spectrum to be of type float64, got {}.".format(pyDSP_enantioselectivity.fft_imag.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_imag.shape, encoded_seq_enantioselectivity.shape,
            "Expected FFT encoding with imaginary spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_enantioselectivity.shape))
        self.assertEqual(pyDSP_enantioselectivity.fft_freqs.shape, encoded_seq_enantioselectivity.shape, "Expected FFT frequencies to be same shape as encoded sequences: {}.".format(encoded_seq_enantioselectivity.shape))
        self.assertTrue(pyDSP_enantioselectivity.spectrum_encoding.any() == pyDSP_enantioselectivity.fft_power.any(), 
            "Expected spectrum encoding attribute to be equal to chosen fft spectrum, power.")
        self.assertEqual(pyDSP_enantioselectivity.fft_power.dtype, "float64", "Expected data type of power spectrum to be float64, got {}.".format(pyDSP_enantioselectivity.fft_power.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_real.dtype, "float64", "Expected data type of real spectrum to be float64, got {}.".format(pyDSP_enantioselectivity.fft_real.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_imag.dtype, "float64", "Expected data type of imaginary spectrum to be float64, got {}.".format(pyDSP_enantioselectivity.fft_imag.dtype))
        self.assertEqual(pyDSP_enantioselectivity.fft_abs.dtype, "float64", "Expected data type of absolute spectrum to be float64, got {}.".format(pyDSP_enantioselectivity.fft_abs.dtype))
#3.) 
        pysar_absorption = pySAR.PySAR(config_file=self.all_config_files[2])  #absorption
        encoded_seq_absorption = pysar_absorption.get_aai_encoding(aa_indices3)            
        pyDSP_absorption = pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs=encoded_seq_absorption) 
        
        self.assertEqual(pyDSP_absorption.spectrum, "power", "Expected spectrum to be power, got {}.".format(pyDSP_absorption.spectrum))
        self.assertEqual(pyDSP_absorption.window_type, "hamming", "Expected window function to be hamming, got {}.".format(pyDSP_absorption.window_type))
        self.assertIsInstance(pyDSP_absorption.window, np.ndarray, "Expected window function to be a numpy array.")
        self.assertIsNone(pyDSP_absorption.filter, "Expected filter function to be None on class initialisation.")
        self.assertIsNone(pyDSP_absorption.filter_type, "Expected filter type to to be None, got {}.".format(pyDSP_absorption.filter_type))
        self.assertEqual(pyDSP_absorption.spectrum_encoding.shape, (pysar_absorption.num_seqs, pysar_absorption.sequence_length),
            "Expected spectrum encoding to be ({}, {}), got {}.".format(pysar_absorption.num_seqs, pysar_absorption.sequence_length, pyDSP_absorption.spectrum_encoding.shape))
        self.assertEqual(pyDSP_absorption.num_seqs, pysar_absorption.num_seqs,
            "Expected num_seqs attribute in pyDSP_absorption_thermostability class to be equal to that of pysar attribute: {}.".format(pysar_absorption.num_seqs))
        self.assertEqual(pyDSP_absorption.signal_len, pysar_absorption.sequence_length,
            "Expectd signal_len attribute in pyDSP_absorption_thermostability class to be equal that of pysar attribute: {}.".format(pysar_absorption.sequence_length))
        self.assertEqual(pyDSP_absorption.fft_power.dtype, 'float64', "Expected power spectrum to be of type float64, got {}.".format(pyDSP_absorption.fft_power.dtype))
        self.assertEqual(pyDSP_absorption.fft_power.shape, encoded_seq_absorption.shape,
            "Expected FFT encoding with power spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertEqual(pyDSP_absorption.fft_real.dtype, 'float64', "Expected real spectrum to be of type float64, got {}.".format(pyDSP_absorption.fft_real.dtype))
        self.assertEqual(pyDSP_absorption.fft_real.shape, encoded_seq_absorption.shape,
            "Expected FFT encoding with real spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertEqual(pyDSP_absorption.fft_abs.dtype, 'float64', "Expected absolute spectrum to be of type float64, got {}.".format(pyDSP_absorption.fft_abs.dtype))
        self.assertEqual(pyDSP_absorption.fft_abs.shape, encoded_seq_absorption.shape,
            "Expected FFT encoding with absolute spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertEqual(pyDSP_absorption.fft_imag.dtype, 'float64', "Expected imaginary spectrum to be of type float64, got {}.".format(pyDSP_absorption.fft_imag.dtype))
        self.assertEqual(pyDSP_absorption.fft_imag.shape, encoded_seq_absorption.shape,
            "Expected FFT encoding with imaginary spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertEqual(pyDSP_absorption.fft_freqs.shape, encoded_seq_absorption.shape, "Expected FFT frequencies to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertTrue(pyDSP_absorption.spectrum_encoding.any() == pyDSP_absorption.fft_power.any(), 
            "Expected spectrum encoding attribute to be equal to chosen fft spectrum, power.")
        self.assertEqual(pyDSP_absorption.fft_power.dtype, "float64", "Expected data type of power spectrum to be float64, got {}.".format(pyDSP_absorption.fft_power.dtype))
        self.assertEqual(pyDSP_absorption.fft_real.dtype, "float64", "Expected data type of real spectrum to be float64, got {}.".format(pyDSP_absorption.fft_real.dtype))
        self.assertEqual(pyDSP_absorption.fft_imag.dtype, "float64", "Expected data type of imaginary spectrum to be float64, got {}.".format(pyDSP_absorption.fft_imag.dtype))
        self.assertEqual(pyDSP_absorption.fft_abs.dtype, "float64", "Expected data type of absolute spectrum to be float64, got {}.".format(pyDSP_absorption.fft_abs.dtype))
#4.) 
        pysar_localization = pySAR.PySAR(config_file=self.all_config_files[3])  #localization
        encoded_seq_localization = pysar_localization.get_aai_encoding(aa_indices4)            
        pyDSP_localization = pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=encoded_seq_localization) 
        
        self.assertEqual(pyDSP_localization.spectrum, "power", "Expected spectrum to be power, got {}.".format(pyDSP_localization.spectrum))
        self.assertEqual(pyDSP_localization.window_type, "hamming", "Expected window function to be hamming, got {}.".format(pyDSP_localization.window_type))
        self.assertIsInstance(pyDSP_localization.window, np.ndarray, "Expected window function to be a numpy array.")
        self.assertIsNone(pyDSP_localization.filter, "Expected filter function to be None on class initialisation.")
        self.assertIsNone(pyDSP_localization.filter_type, "Expected filter type to to be None, got {}.".format(pyDSP_localization.filter_type))
        self.assertEqual(pyDSP_localization.spectrum_encoding.shape, (pysar_localization.num_seqs, pysar_localization.sequence_length),
            "Expected spectrum encoding to be ({}, {}), got {}.".format(pysar_localization.num_seqs, pysar_localization.sequence_length, pyDSP_localization.spectrum_encoding.shape))
        self.assertEqual(pyDSP_localization.num_seqs, pysar_localization.num_seqs,
            "Expected num_seqs attribute in pyDSP_localization_thermostability class to be equal to that of pysar attribute: {}.".format(pysar_localization.num_seqs))
        self.assertEqual(pyDSP_localization.signal_len, pysar_localization.sequence_length,
            "Expectd signal_len attribute in pyDSP_localization_thermostability class to be equal that of pysar attribute: {}.".format(pysar_localization.sequence_length))
        self.assertEqual(pyDSP_localization.fft_power.dtype, 'float64', "Expected power spectrum to be of type float64, got {}.".format(pyDSP_localization.fft_power.dtype))
        self.assertEqual(pyDSP_localization.fft_power.shape, encoded_seq_localization.shape,
            "Expected FFT encoding with power spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_localization.shape))
        self.assertEqual(pyDSP_localization.fft_real.dtype, 'float64', "Expected real spectrum to be of type float64, got {}.".format(pyDSP_localization.fft_real.dtype))
        self.assertEqual(pyDSP_localization.fft_real.shape, encoded_seq_localization.shape,
            "Expected FFT encoding with real spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_localization.shape))
        self.assertEqual(pyDSP_localization.fft_abs.dtype, 'float64', "Expected absolute spectrum to be of type float64, got {}.".format(pyDSP_localization.fft_abs.dtype))
        self.assertEqual(pyDSP_localization.fft_abs.shape, encoded_seq_localization.shape,
            "Expected FFT encoding with absolute spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_localization.shape))
        self.assertEqual(pyDSP_localization.fft_imag.dtype, 'float64', "Expected imaginary spectrum to be of type float64, got {}.".format(pyDSP_localization.fft_imag.dtype))
        self.assertEqual(pyDSP_localization.fft_imag.shape, encoded_seq_localization.shape,
            "Expected FFT encoding with imaginary spectrum to be same shape as encoded sequences: {}.".format(encoded_seq_localization.shape))
        self.assertEqual(pyDSP_localization.fft_freqs.shape, encoded_seq_localization.shape, "Expected FFT frequencies to be same shape as encoded sequences: {}.".format(encoded_seq_absorption.shape))
        self.assertTrue(pyDSP_localization.spectrum_encoding.any() == pyDSP_localization.fft_power.any(), 
            "Expected spectrum encoding attribute to be equal to chosen fft spectrum, power.")
        self.assertEqual(pyDSP_localization.fft_power.dtype, "float64", "Expected data type of power spectrum to be float64, got {}.".format(pyDSP_localization.fft_power.dtype))
        self.assertEqual(pyDSP_localization.fft_real.dtype, "float64", "Expected data type of real spectrum to be float64, got {}.".format(pyDSP_localization.fft_real.dtype))
        self.assertEqual(pyDSP_localization.fft_imag.dtype, "float64", "Expected data type of imaginary spectrum to be float64, got {}.".format(pyDSP_localization.fft_imag.dtype))
        self.assertEqual(pyDSP_localization.fft_abs.dtype, "float64", "Expected data type of absolute spectrum to be float64, got {}.".format(pyDSP_localization.fft_abs.dtype))
#5.)
        with self.assertRaises(OSError, msg='OS Error raised, invalid config file path given.'):
            pyDSP_.PyDSP(config_file="blahblahblah")
            pyDSP_.PyDSP(config_file="test_data/notafile.json")
#6.)
        with self.assertRaises(TypeError, msg='Type Error raised, invalid config file path data type given.'):
            pyDSP_.PyDSP(config_file=4.21)
            pyDSP_.PyDSP(config_file=False)
#7.)
        with self.assertRaises(ValueError, msg='Value Error raised, protein sequences input parameter cannot be none or empty.'):
            pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=None)
            pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs="")
    
    def test_preprocessing(self):
        """ Testing preprocessing functionality of pyDSP class. """
        test_aaindices1 = "COHE430101"
        test_aaindices2 = "LEVM780105"
        test_aaindices3 = "QIAN880107"
        test_aaindices4 = "ROSG850102"
#1.)
        pysar_thermostability = pySAR.PySAR(config_file=self.all_config_files[0])  #thermostability
        encoded_seq_thermostability = pysar_thermostability.get_aai_encoding(test_aaindices1) #thermostability
        pyDSP_thermostability = pyDSP_.PyDSP(config_file=self.all_config_files[0], protein_seqs=encoded_seq_thermostability)
        pyDSP_thermostability.pre_processing()

        self.assertTrue(np.all((pyDSP_thermostability.fft_power==0)), "Expected power spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_thermostability.fft_real==0)), "Expected real spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_thermostability.fft_imag==0)), "Expected imaginary spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_thermostability.fft_abs==0)), "Expected absolute spectrum to be initialised into zeros array.")
        self.assertFalse(np.isnan(pyDSP_thermostability.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain null values.')
        self.assertFalse(np.isinf(pyDSP_thermostability.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain any +/- infinity values.')
#2.)
        pysar_enantioselectivity = pySAR.PySAR(config_file=self.all_config_files[1])  #enantioselectivity
        encoded_seq_enantioselectivity = pysar_enantioselectivity.get_aai_encoding(test_aaindices2) #enantioselectivity
        pyDSP_enantioselectivity = pyDSP_.PyDSP(config_file=self.all_config_files[1], protein_seqs=encoded_seq_enantioselectivity)
        pyDSP_enantioselectivity.pre_processing()

        self.assertTrue(np.all((pyDSP_enantioselectivity.fft_power==0)), "Expected power spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_enantioselectivity.fft_real==0)), "Expected real spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_enantioselectivity.fft_imag==0)), "Expected imaginary spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_enantioselectivity.fft_abs==0)), "Expected absolute spectrum to be initialised into zeros array.")
        self.assertFalse(np.isnan(pyDSP_enantioselectivity.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain null values.')
        self.assertFalse(np.isinf(pyDSP_enantioselectivity.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain any +/- infinity values.')
#3.)
        pysar_absorption = pySAR.PySAR(config_file=self.all_config_files[2])  #absorption
        encoded_seq_absorption = pysar_absorption.get_aai_encoding(test_aaindices3) #absorption
        pyDSP_absorption = pyDSP_.PyDSP(config_file=self.all_config_files[2], protein_seqs=encoded_seq_absorption)
        pyDSP_absorption.pre_processing()

        self.assertTrue(np.all((pyDSP_absorption.fft_power==0)), "Expected power spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_absorption.fft_real==0)), "Expected real spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_absorption.fft_imag==0)), "Expected imaginary spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_absorption.fft_abs==0)), "Expected absolute spectrum to be initialised into zeros array.")
        self.assertFalse(np.isnan(pyDSP_absorption.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain null values.')
        self.assertFalse(np.isinf(pyDSP_absorption.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain any +/- infinity values.')
#4.)
        pysar_localization = pySAR.PySAR(config_file=self.all_config_files[3])  #localization
        encoded_seq_localization = pysar_localization.get_aai_encoding(test_aaindices4) #localization
        pyDSP_localization = pyDSP_.PyDSP(config_file=self.all_config_files[3], protein_seqs=encoded_seq_localization)
        pyDSP_localization.pre_processing()

        self.assertTrue(np.all((pyDSP_localization.fft_power==0)), "Expected power spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_localization.fft_real==0)), "Expected real spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_localization.fft_imag==0)), "Expected imaginary spectrum to be initialised into zeros array.")
        self.assertTrue(np.all((pyDSP_localization.fft_abs==0)), "Expected absolute spectrum to be initialised into zeros array.")
        self.assertFalse(np.isnan(pyDSP_localization.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain null values.')
        self.assertFalse(np.isinf(pyDSP_localization.spectrum_encoding).any(), 
            'Expected sequences after pre-processing step to not contain any +/- infinity values.')

    def test_window(self):
        """ Testing window functions of pyDSP module. """
        all_aaindices = ["EISD860101", "GEIM800107", "NAKH900106", "QIAN880105"]
        all_windows = ['hamming', 'blackman', 'blackmanharris', 'gaussian', 'bartlett',
                'kaiser', 'barthann', 'bohman', 'chebwin', 'cosine', 'exponential',
                'flattop', 'hann', 'boxcar', 'nuttall', 'parzen', 'triang', 'tukey']
        all_shapes = [(466, ), (398, ), (298, ), (361, )]
#1.)    
        #iterate over all config files, all indices and all windows
        for config in range(0, len(self.all_config_files)):
            for index in all_aaindices:
                for window in all_windows:
                    #create instance of PySAR and PyDSP classes
                    pysar_ = pySAR.PySAR(config_file=self.all_config_files[config])
                    encoded_seq = pysar_.get_aai_encoding(index)
                    pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[config], protein_seqs=encoded_seq, window_type=window)
                    
                    self.assertEqual(pyDSP.window_type, window, 
                        "Expected window type to be {}, got {}.".format(pyDSP.window_type, window))
                    self.assertIsInstance(pyDSP.window, np.ndarray, 
                        "Expected window to be a numpy array, got {}.".format(pyDSP.window))
                    self.assertEqual(pyDSP.window.shape, all_shapes[config], 
                        "Expected shape of window to be {}, got {}.".format(all_shapes[config], pyDSP.window.shape))
                    self.assertFalse(np.isnan(pyDSP.window).any(), 
                        "Expected window output to contain no null/nan values.")
                    self.assertFalse(np.isinf(pyDSP.window).any(), 
                        'Expected sequences to not contain any +/- infinity values.')
                    
    def test_filter(self):
        """ Testing filter functions of pyDSP module. """   
        all_aaindices = ["EISD860101", "GEIM800107", "NAKH900106", "QIAN880105"]
        all_filters = ['savgol']
        all_shapes = [(466, ), (398, ), (298, ), (361, )]
#1.)    
        #iterate over all config files, all indices and all filters
        for config in range(0, len(self.all_config_files)):
            for index in all_aaindices:
                for filter in all_filters:
                    #create instance of PySAR and PyDSP classes
                    pysar_ = pySAR.PySAR(config_file=self.all_config_files[config])
                    encoded_seq = pysar_.get_aai_encoding(index)
                    pyDSP = pyDSP_.PyDSP(config_file=self.all_config_files[config], protein_seqs=encoded_seq, filter_type=filter)
                    
                    self.assertEqual(pyDSP.filter_type, filter, 
                        "Expected filter type to be {}, got {}.".format(pyDSP.filter_type, filter))
                    self.assertIsInstance(pyDSP.filter, np.ndarray, 
                        "Expected filter to be a numpy array, got {}.".format(pyDSP.filter))
                    self.assertEqual(pyDSP.filter.shape, all_shapes[config], 
                        "Expected shape of filter to be {}, got {}.".format(all_shapes[config], pyDSP.filter.shape))
                    self.assertFalse(np.isnan(pyDSP.filter).any(), 
                        "Expected filter output to contain no null/nan values.")
                    self.assertFalse(np.isinf(pyDSP.filter).any(), 
                        'Expected sequences to not contain any +/- infinity values.')

    def test_max_freq(self):
        """ Testing max frequency functionality. """
        all_aaindices = ["ISOY800101", "MEEJ800101", "PALJ810116", "QIAN880107"]
#1.)
        for config in self.all_config_files:
            for index in all_aaindices:
                #create instance of PySAR and PyDSP classes
                pysar_ = pySAR.PySAR(config_file=config)
                encoded_seq = pysar_.get_aai_encoding(index)
                pyDSP = pyDSP_.PyDSP(config_file=config, protein_seqs=encoded_seq)

                max_freq_, max_freq_index = pyDSP.max_freq(pyDSP.spectrum_encoding[0])
                self.assertIsInstance(max_freq_, float, 
                    "Expected max frequency attribute to be a float, got {}.".format(type(max_freq_)))
                self.assertIsInstance(max_freq_index, np.int64, 
                    "Expected max frequency index attribute to be a np.int64, got {}.".format(type(max_freq_index)))
#2.)
        with self.assertRaises(ValueError):
            pyDSP.max_freq(pyDSP.spectrum_encoding)

    def test_consensus_freq(self):
        """ Testing max frequency functionality. """
        all_aaindices = ["ISOY800101", "MEEJ800101", "PALJ810116", "QIAN880107"]
#1.)
        for config in self.all_config_files:
            for index in all_aaindices:
                #create instance of PySAR and PyDSP classes
                pysar_ = pySAR.PySAR(config_file=config)
                encoded_seq = pysar_.get_aai_encoding(index)
                pyDSP = pyDSP_.PyDSP(config_file=config, protein_seqs=encoded_seq)

                #calculate consensus freq
                consensus_freq = pyDSP.consensus_freq(pyDSP.spectrum_encoding[0])

                self.assertIsInstance(consensus_freq, float, 
                    "Expected consensus frequency attribute to be a float, got {}.".format(type(consensus_freq)))
#2.)
        with self.assertRaises(ValueError):
            pyDSP.consensus_freq(pyDSP.spectrum_encoding)