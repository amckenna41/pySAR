#################################################################################
#################             Encoding Module Tests             #################
#################################################################################

import pandas as pd
import numpy as np
import os
import shutil
import re
import unittest
from aaindex import aaindex1
unittest.TestLoader.sortTestMethodsUsing = None

#stop sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pySAR.encoding as pysar_
import pySAR.globals_ as _globals

class PySARTests(unittest.TestCase):
    """
    Test suite for testing encoding module and functionality 
    in pySAR package. 

    Test Cases
    ----------
    test_pySAR_metadata:
        testing correct pysar software metadata.
    """
    def setUp(self):
        """ Import the 4 config files for each of the 4 datasets used for testing the pySAR methods. """
        #array of config files for each test dataset
        config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]

        #create instance of Encoding class 
        self.test_config1 = pysar_.Encoding(config_file=self.all_config_files[0])
        self.test_config2 = pysar_.Encoding(config_file=self.all_config_files[1])
        self.test_config3 = pysar_.Encoding(config_file=self.all_config_files[2])
        self.test_config4 = pysar_.Encoding(config_file=self.all_config_files[3])

        #list of canonical amino acids
        self.amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", 
            "Q", "R", "S", "T", "V", "W", "Y"]

        #expected dataframe columns for aai encoding
        self.expected_aai_encoding_output_columns = ["Index", "Category", "R2", "RMSE", "MSE", "MAE", 
            "RPD", "Explained Variance"]

        #expected dataframe columns for descriptors encoding
        self.expected_desc_encoding_output_columns = ["Descriptor", "Group", "R2", "RMSE", "MSE", "MAE", 
            "RPD", "Explained Variance"]

        #expected dataframe columns for aai + descriptor encoding
        self.expected_aai_desc_encoding_output_columns = ["Index", "Category", "Descriptor", "Group",
            "R2", "RMSE", "MSE", "MAE", "RPD", "Explained Variance"]

        #AAI record categories
        self.index_categories = ["sec_struct", "geometry", "polar", "charge", "composition", 
            "meta", "hydrophobic", "flexibility", "observable"]
        
        #descriptor groups/categories
        self.descriptor_groups = ["Composition", "Autocorrelation", "Sequence Order", "CTD", 
            "Conjoint Triad", "Pseudo Composition"]

        #list of available protein descriptors
        self.valid_descriptors = [
            'amino_acid_composition', 'dipeptide_composition', 'tripeptide_composition',
            'moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation',
            'ctd', 'ctd_composition', 'ctd_transition', 'ctd_distribution', 'conjoint_triad',
            'sequence_order_coupling_number','quasi_sequence_order',
            'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition'
        ]

        #set global vars to create temp test data folders
        _globals.OUTPUT_DIR = os.path.join('tests', _globals.OUTPUT_DIR)
        _globals.OUTPUT_FOLDER = os.path.join('tests', _globals.OUTPUT_FOLDER)

    # @unittest.skip("")
    def test_aai_encoding(self):
        """ Testing AAI encoding functionality in Encoding module. """
#1.)    
        test_aai1 = ["FAUJ880110", "GEIM800111"]
        test_encoding1 = self.test_config1.aai_encoding(aai_list=test_aai1, sort_by="R2")

        self.assertIsInstance(test_encoding1, pd.DataFrame, "")
        self.assertEqual(len(test_encoding1), 2, "") 
        self.assertEqual(set(list(test_encoding1["Index"])), set(test_aai1) , "") 
        self.assertEqual(test_encoding1["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["R2"].dtype, float, "")
        self.assertEqual(test_encoding1["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MAE"].dtype, float, "")
        self.assertEqual(test_encoding1["RPD"].dtype, float, "")
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding1["Category"]):
            self.assertIn(cat, self.index_categories)
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns)
#2.)
        test_aai2 = ["FAUJ880110", "GEIM800111", "JOND750102", "MAXF760102"]
        test_encoding2 = self.test_config2.aai_encoding(aai_list=test_aai2, sort_by="RMSE")

        self.assertIsInstance(test_encoding2, pd.DataFrame, "")
        self.assertEqual(len(test_encoding2), 4, "") 
        self.assertEqual(set(list(test_encoding2["Index"])), set(test_aai2) , "") 
        self.assertEqual(test_encoding2["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["R2"].dtype, float, "")
        self.assertEqual(test_encoding2["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MAE"].dtype, float, "")
        self.assertEqual(test_encoding2["RPD"].dtype, float, "")
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding2["Category"]):
            self.assertIn(cat, self.index_categories)
        for cols in test_encoding2.columns:
            self.assertIn(cols, self.expected_aai_encoding_output_columns)
#3.)
        test_aai3 = ["BIGC670101", "CHOP780211", "DESM900101", "FAUJ880113", "KANM800104"]
        test_encoding3 = self.test_config3.aai_encoding(aai_list=test_aai3, sort_by="MSE")

        self.assertIsInstance(test_encoding3, pd.DataFrame, "")
        self.assertEqual(len(test_encoding3), 5, "") 
        self.assertEqual(set(list(test_encoding3["Index"])), set(test_aai3) , "") 
        self.assertEqual(test_encoding3["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["R2"].dtype, float, "")
        self.assertEqual(test_encoding3["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MAE"].dtype, float, "")
        self.assertEqual(test_encoding3["RPD"].dtype, float, "")
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding3["Category"]):
            self.assertIn(cat, self.index_categories)
        for cols in test_encoding3.columns:
            self.assertIn(cols, self.expected_aai_encoding_output_columns)
#4.)
        test_aai4 = []
        test_encoding4 = self.test_config4.aai_encoding(aai_list=test_aai4, sort_by="MAE")

        self.assertIsInstance(test_encoding4, pd.DataFrame, "")
        self.assertEqual(len(test_encoding4), 566, "") 
        self.assertEqual(set(list(test_encoding4["Index"])), set(aaindex1.record_codes()), "") 
        self.assertEqual(test_encoding4["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding4["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding4["R2"].dtype, float, "")
        self.assertEqual(test_encoding4["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding4["MSE"].dtype, float, "")
        self.assertEqual(test_encoding4["MAE"].dtype, float, "")
        self.assertEqual(test_encoding4["RPD"].dtype, float, "")
        self.assertEqual(test_encoding4["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding4["Category"]):
            self.assertIn(cat, self.index_categories)
        for cols in test_encoding4.columns:
            self.assertIn(cols, self.expected_aai_encoding_output_columns)
#5.)
        test_aai5 = ["CHOP780211"]
        test_encoding5 = self.test_config3.aai_encoding(aai_list=test_aai5, sort_by="invalid_metric") #R2 will then be used as default metric

        self.assertIsInstance(test_encoding5, pd.DataFrame, "")
        self.assertEqual(len(test_encoding5), 1, "") 
        self.assertEqual(list(test_encoding5["Index"]), test_aai5, "") 
        self.assertEqual(test_encoding5["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding5["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding5["R2"].dtype, float, "")
        self.assertEqual(test_encoding5["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding5["MSE"].dtype, float, "")
        self.assertEqual(test_encoding5["MAE"].dtype, float, "")
        self.assertEqual(test_encoding5["RPD"].dtype, float, "")
        self.assertEqual(test_encoding5["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding5["Category"]):
            self.assertIn(cat, self.index_categories)
        for cols in test_encoding5.columns:
            self.assertIn(cols, self.expected_aai_encoding_output_columns)
#6.)
        test_aai6 = "blahblah" 
        with self.assertRaises(ValueError):
            test_encoding6 = self.test_config1.aai_encoding(aai_list=test_aai6, sort_by="RPD") 
#7.)
        test_aai7 = 1234 
        with self.assertRaises(TypeError):
            test_encoding7 = self.test_config2.aai_encoding(aai_list=test_aai7, sort_by="MSE")
#8.)
        test_aai8 = True 
        with self.assertRaises(TypeError):
            test_encoding8 = self.test_config3.aai_encoding(aai_list=test_aai8, sort_by="MAE")

    # @unittest.skip("Descriptor encoding functionality can take a lot of time, skipping.")
    def test_descriptor_encoding(self):
        """ Testing Descriptor encoding functionality in Encoding module. """
#1.)
        test_desc1 = ["amino_acid_composition"]
        test_encoding1 = self.test_config1.descriptor_encoding(desc_list=test_desc1, desc_combo=1, cutoff_index=1, sort_by="R2")

        self.assertIsInstance(test_encoding1, pd.DataFrame, "")
        self.assertEqual(len(test_encoding1), 1, "") 
        self.assertEqual(list(test_encoding1["Descriptor"]), test_desc1 , "") 
        self.assertEqual(test_encoding1["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["R2"].dtype, float, "")
        self.assertEqual(test_encoding1["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MAE"].dtype, float, "")
        self.assertEqual(test_encoding1["RPD"].dtype, float, "")
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding1["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns)
#2.)
        test_desc2 = ["ctd", "conjoint_triad", "dipeptide_composition"]
        test_encoding2 = self.test_config2.descriptor_encoding(desc_list=test_desc2, desc_combo=1, cutoff_index=1, sort_by="MSE")

        self.assertIsInstance(test_encoding2, pd.DataFrame, "")
        self.assertEqual(len(test_encoding2), 3, "") 
        self.assertEqual(set(list(test_encoding2["Descriptor"])), set(test_desc2) , "") 
        self.assertEqual(test_encoding2["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["R2"].dtype, float, "")
        self.assertEqual(test_encoding2["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MAE"].dtype, float, "")
        self.assertEqual(test_encoding2["RPD"].dtype, float, "")
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding2["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding2.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns)
#3.)
        test_desc3 = ["moran_auto"]
        test_encoding3 = self.test_config3.descriptor_encoding(desc_list=test_desc3, desc_combo=1, cutoff_index=1, sort_by="MAE")

        self.assertIsInstance(test_encoding3, pd.DataFrame, "")
        self.assertEqual(len(test_encoding3), 1, "") 
        self.assertEqual(list(test_encoding3["Descriptor"]), test_desc3 , "") 
        self.assertEqual(test_encoding3["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["R2"].dtype, float, "")
        self.assertEqual(test_encoding3["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MAE"].dtype, float, "")
        self.assertEqual(test_encoding3["RPD"].dtype, float, "")
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding3["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding3.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns)
#4.)
        test_desc4 = []
        test_encoding4 = self.test_config1.descriptor_encoding(desc_list=test_desc4, desc_combo=1, cutoff_index=1, sort_by="RPD")

        self.assertIsInstance(test_encoding4, pd.DataFrame, "")
        self.assertEqual(len(test_encoding4), 15, "") 
        self.assertEqual(test_encoding4["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding4["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding4["R2"].dtype, float, "")
        self.assertEqual(test_encoding4["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding4["MSE"].dtype, float, "")
        self.assertEqual(test_encoding4["MAE"].dtype, float, "")
        self.assertEqual(test_encoding4["RPD"].dtype, float, "")
        self.assertEqual(test_encoding4["Explained Variance"].dtype, float, "")
        for desc in list(test_encoding4["Descriptor"]):
            self.assertIn(desc, self.valid_descriptors)
        for cat in list(test_encoding4["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding4.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns)
#5.)
        invalid_test_desc5 = "invalid_descriptor_name" 
        with self.assertRaises(ValueError):
            test_encoding5 = self.test_config1.descriptor_encoding(desc_list=invalid_test_desc5, desc_combo=1, cutoff_index=1, sort_by="Explained Variance")
#6.)
        invalid_test_desc6 = 12345 
        with self.assertRaises(TypeError):
            test_encoding6 = self.test_config1.descriptor_encoding(desc_list=invalid_test_desc6, desc_combo=1, cutoff_index=1, sort_by="MAE")
#7.)
        invalid_test_desc7 = True 
        with self.assertRaises(TypeError):
            test_encoding7 = self.test_config1.descriptor_encoding(desc_list=invalid_test_desc7, desc_combo=1, cutoff_index=1, sort_by="RMSE")

    # @unittest.skip("AAI + Descriptor encoding functionality can take a lot of time, skipping.")
    def test_aai_descriptor_encoding(self):
        """ Testing AAI + Descriptor encoding functionality in Encoding module. """
#1.)    
        test_aai1 = ["FAUJ880110"]
        test_desc1 = ["ctd"]
        test_encoding1 = self.test_config1.aai_descriptor_encoding(aai_list=test_aai1, desc_list=test_desc1, 
            desc_combo=1, cutoff_index=1, sort_by="R2")

        self.assertIsInstance(test_encoding1, pd.DataFrame, "")
        self.assertEqual(len(test_encoding1), 1, "") 
        self.assertEqual(list(test_encoding1["Index"]), test_aai1, "") 
        self.assertEqual(list(test_encoding1["Descriptor"]), test_desc1, "") 
        self.assertEqual(test_encoding1["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding1["R2"].dtype, float, "")
        self.assertEqual(test_encoding1["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MSE"].dtype, float, "")
        self.assertEqual(test_encoding1["MAE"].dtype, float, "")
        self.assertEqual(test_encoding1["RPD"].dtype, float, "")
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding1["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns)
#2.)    
        test_aai2 = ["BIGC670101", "DAYM780201"]
        test_desc2 = ["tripeptide_composition", "quasi_sequence_order", "sequence_order_coupling_number"]
        test_encoding2 = self.test_config1.aai_descriptor_encoding(aai_list=test_aai2, desc_list=test_desc2, 
            desc_combo=1, cutoff_index=1, sort_by="MSE")

        self.assertIsInstance(test_encoding2, pd.DataFrame, "")
        self.assertEqual(len(test_encoding2), 6, "") 
        self.assertEqual(set(list(test_encoding2["Index"])), set(test_aai2), "") 
        self.assertEqual(set(list(test_encoding2["Descriptor"])), set(test_desc2), "") 
        self.assertEqual(test_encoding2["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding2["R2"].dtype, float, "")
        self.assertEqual(test_encoding2["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MSE"].dtype, float, "")
        self.assertEqual(test_encoding2["MAE"].dtype, float, "")
        self.assertEqual(test_encoding2["RPD"].dtype, float, "")
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding2["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding2.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns)
#3.)    
        test_aai3 = ["GEOR030107", "KARS160113", "COWR900101"]
        test_desc3 = ["amino_acid_composition", "ctd_distribution"]
        test_encoding3 = self.test_config2.aai_descriptor_encoding(aai_list=test_aai3, desc_list=test_desc3, 
            desc_combo=1, cutoff_index=1, sort_by="MSE")

        self.assertIsInstance(test_encoding3, pd.DataFrame, "")
        self.assertEqual(len(test_encoding3), 6, "") 
        self.assertEqual(set(list(test_encoding3["Index"])), set(test_aai3), "") 
        self.assertEqual(set(list(test_encoding3["Descriptor"])), set(test_desc3), "") 
        self.assertEqual(test_encoding3["Index"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["Category"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["Descriptor"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["Group"].dtype, "string[python]", "")
        self.assertEqual(test_encoding3["R2"].dtype, float, "")
        self.assertEqual(test_encoding3["RMSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MSE"].dtype, float, "")
        self.assertEqual(test_encoding3["MAE"].dtype, float, "")
        self.assertEqual(test_encoding3["RPD"].dtype, float, "")
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float, "")
        for cat in list(test_encoding3["Group"]):
            self.assertIn(cat, self.descriptor_groups)
        for col in test_encoding3.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns)
#4.)    
        test_aai4 = ["invalid_aai_index"]
        test_desc4 = ["invalid_descriptor_name"]
        
        with self.assertRaises(ValueError):
            test_encoding4 = self.test_config1.aai_descriptor_encoding(aai_list=test_aai4, desc_list=test_desc4, 
                desc_combo=1, cutoff_index=1, sort_by="MSE")
#6.)
        test_aai5 = 12345
        test_desc5 = 1000
        
        with self.assertRaises(TypeError):
            test_encoding4 = self.test_config1.aai_descriptor_encoding(aai_list=test_aai5, desc_list=test_desc5, 
                desc_combo=1, cutoff_index=1, sort_by="MAE")

#7.)    Below inputs result in all AAI Indices being encoded with all descriptors, commenting out due to time constraints
        # test_aai7 = []
        # test_desc7 = []
        # test_encoding7 = self.test_config1.aai_descriptor_encoding(aai_list=test_aai7, desc_list=test_desc7, 
        #     desc_combo=1, cutoff_index=1, sort_by="MAE")

        # self.assertIsInstance(test_encoding7, pd.DataFrame, "")
        # self.assertEqual(len(test_encoding7), 8490, "") 
        # self.assertEqual(set(list(test_encoding7["Index"])), set(test_aai3), "") 
        # self.assertEqual(set(list(test_encoding7["Descriptor"])), set(test_desc3), "") 
        # self.assertEqual(test_encoding7["Index"].dtype, "string[python]", "")
        # self.assertEqual(test_encoding7["Category"].dtype, "string[python]", "")
        # self.assertEqual(test_encoding7["Descriptor"].dtype, "string[python]", "")
        # self.assertEqual(test_encoding7["Group"].dtype, "string[python]", "")
        # self.assertEqual(test_encoding7["R2"].dtype, float, "")
        # self.assertEqual(test_encoding7["RMSE"].dtype, float, "")
        # self.assertEqual(test_encoding7["MSE"].dtype, float, "")
        # self.assertEqual(test_encoding7["MAE"].dtype, float, "")
        # self.assertEqual(test_encoding7["RPD"].dtype, float, "")
        # self.assertEqual(test_encoding7["Explained Variance"].dtype, float, "")
        # for cat in list(test_encoding7["Group"]):
        #     self.assertIn(cat, self.descriptor_groups)
        # for col in test_encoding7.columns:
        #     self.assertIn(col, self.expected_aai_desc_encoding_output_columns)

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        if (os.path.isdir(_globals.OUTPUT_DIR)):
            shutil.rmtree(_globals.OUTPUT_DIR, ignore_errors=False, onerror=None)
        if (os.path.isdir(_globals.OUTPUT_FOLDER)):
            shutil.rmtree(_globals.OUTPUT_FOLDER, ignore_errors=False, onerror=None)

        # del _globals.OUTPUT_DIR
        # del _globals.OUTPUT_FOLDER
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)