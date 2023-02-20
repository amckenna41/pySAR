################################################################################
#################             pySAR Module Tests             #################
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
        self.all_protein_seqs = []
        self.pysar = pySAR.PySAR(config_file=self.all_config_files[0])

    def test_pyDSP(self):
        """ Test class input parameters and attributes. """
#1.)   
        aa_indices_1 = "NAKH920102"
        encoded_seq1 = self.pysar.get_aai_encoding(aa_indices_1)            

        pyDSP = pyDSP_.PyDSP(dsp_config=self.all_config_files[0], protein_seqs=encoded_seq1) #test_thermostability
        self.assertEqual(pyDSP.spectrum, "power")
        self.assertEqual(pyDSP.window_type, "hamming")

        self.assertIsInstance(pyDSP.window, np.ndarray)
        self.assertIsNone(pyDSP.filter)
        self.assertEqual(pyDSP.spectrum_encoding.shape, (self.pysar.num_seqs, self.pysar.seq_len))
        self.assertEqual(pyDSP.num_seqs, self.pysar.num_seqs)
        self.assertEqual(pyDSP.signal_len, self.pysar.seq_len)
        self.assertEqual(pyDSP.fft_power.shape, encoded_seq1.shape)
        self.assertTrue(pyDSP.fft_power.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_real.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_real.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_abs.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_abs.dtype, 'float64')
        # self.assertEqual(pyDSP.fft_imag.shape, encoded_seq1.shape)
        self.assertEqual(pyDSP.fft_imag.dtype, 'float64')
        self.assertTrue(pyDSP.spectrum_encoding.any() == pyDSP.fft_power.any())
        # self.assertEqual(pyDSP.fft_freqs.shape,encoded_seq1.shape)

        print(pyDSP.max_freq(pyDSP.spectrum_encoding[0]))
  
        #check column names using regex
        #check each column's values 
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)