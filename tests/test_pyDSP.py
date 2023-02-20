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
        self.pysar = pySAR.PySAR(config_file=self.all_config_files[0])

    def test_pyDSP(self):
        """ Test class input parameters and attributes. """
        aa_indices1 = "EISD860101"
        aa_indices2 = "GEIM800107"
        aa_indices3 = "NAKH900106"
        aa_indices4 = "QIAN880105"
#1.)   
        encoded_seq1 = self.pysar.get_aai_encoding(aa_indices1)            

        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq1) #test_thermostability
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter)
        self.assertIsNone(pyDSP.filter_type, "Filter type expected to be X, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len)
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
        encoded_seq2 = self.pysar.get_aai_encoding(aa_indices2)            

        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[1], protein_seqs=encoded_seq2) #test_enantioselectivity
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter)
        self.assertIsNone(pyDSP.filter_type, "Filter type expected to be X, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs,self.pysar.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len)
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
        encoded_seq3 = self.pysar.get_aai_encoding(aa_indices3)            

        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[2], protein_seqs=encoded_seq3) #test_absorption
        #pre-encode sequences
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter)
        self.assertIsNone(pyDSP.filter_type, "Filter type expected to be X, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len)
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
        encoded_seq4 = self.pysar.get_aai_encoding(aa_indices4)            

        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[3], protein_seqs=encoded_seq4) #test_localization
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")
        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter)
        self.assertIsNone(pyDSP.filter_type, "Filter type expected to be X, got {}.".format(pyDSP.filter_type))
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len)
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
#5.)
        with self.assertRaises(OSError, msg='OS Error raised, invalid config file path given.'):
            pyDSP = pyDSP_.PyDSP(dsp_config="blahblahblah")
#6.)
        with self.assertRaises(TypeError, msg='Type Error raised, invalid config file path data type given.'):
            pyDSP = pyDSP_.PyDSP(dsp_config=4.21)
#7.)
        with self.assertRaises(ValueError, msg='Value Error raised, protein sequences input parameter cant be none.'):
            pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[3], protein_seqs=None)

    def test_preprocessing(self):
        """ Testing preprocessing functionality of pyDSP class. """
        test_aaindices1 = "COHE430101"
#1.)
        encoded_seq1 = self.pysar.get_aai_encoding(test_aaindices1)
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq1)
        pyDSP.pre_processing()
        self.assertTrue(np.all((pyDSP.fft_power==0)))
        self.assertTrue(np.all((pyDSP.fft_real==0)))
        self.assertTrue(np.all((pyDSP.fft_imag==0)))
        self.assertTrue(np.all((pyDSP.fft_abs==0)))
        self.assertFalse(np.isnan(pyDSP.spectrum_encoding).any(), 'Sequences should not contain null values.')
        
    def test_protein_spectra(self):
        """ Testing getting protein spectra from encoded protein sequences. """
        aa_indices1 = "COHE430101"
        aa_indices2 = "GRAR740101"
        aa_indices3 = "ISOY800106"
        aa_indices4 = "JOND920101"
#1.)
        encoded_seq1 = self.pysar.get_aai_encoding(aa_indices1)
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq1)
        self.assertTrue(pyDSP.fft_power.dtype, "complex128")
        self.assertTrue(pyDSP.fft_real.dtype, "complex128")
        self.assertTrue(pyDSP.fft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.fft_abs.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_power.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_real.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_abs.dtype, "complex128")        
#2.)
        encoded_seq2 = self.pysar.get_aai_encoding(aa_indices2)
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq2)
        self.assertTrue(pyDSP.fft_power.dtype, "complex128")
        self.assertTrue(pyDSP.fft_real.dtype, "complex128")
        self.assertTrue(pyDSP.fft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.fft_abs.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_power.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_real.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_abs.dtype, "complex128")  
#3.)
        encoded_seq3 = self.pysar.get_aai_encoding(aa_indices3)
        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq3)
        self.assertTrue(pyDSP.fft_power.dtype, "complex128")
        self.assertTrue(pyDSP.fft_real.dtype, "complex128")
        self.assertTrue(pyDSP.fft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.fft_abs.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_power.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_real.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_imag.dtype, "complex128")
        self.assertTrue(pyDSP.rfft_abs.dtype, "complex128")  

    def test_max_freq(self):
        """ Testing max frequency functionality. """
#1.)
        test_aaindices1 = "COHE430101"
        encoded_seq1 = self.pysar.get_aai_encoding(test_aaindices1)
        for config in self.all_config_files:
            pyDSP = pyDSP_.PyDSP(dsp_config=config, protein_seqs=encoded_seq1)
            max_freq_, max_freq_index = pyDSP.max_freq(pyDSP.spectrum_encoding[0])
            self.assertIsInstance(max_freq_, float, "")
            self.assertIsInstance(max_freq_index, np.int64, "")
#2.)
        with self.assertRaises(ValueError):
            max_freq_, max_freq_index = pyDSP.max_freq(pyDSP.spectrum_encoding)