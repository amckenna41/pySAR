#################################################################################
#################             Encoding Module Tests             #################
#################################################################################

import pandas as pd
import os
import shutil
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

class EncodingTests(unittest.TestCase):
    """
    Test suite for testing encoding module and functionality 
    in pySAR package. 

    Test Cases
    ==========
    test_aai_encoding:
        testing correct aai encoding Encoding class functionality.
    test_descriptor_encoding:
        testing correct descriptor encoding Encoding class functionality.
    test_aai_descriptor_encoding:
        testing correct aai + descriptor encoding Encoding class functionality.
    """
    def setUp(self):
        """ Import the 4 config files for each of the 4 datasets used for testing the Encoding methods. """
        #array of config files for each test dataset
        config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]

        #create instance of Encoding class for each config file
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
            'moreaubroto_autocorrelation', 'moran_autocorrelation', 'geary_autocorrelation',
            'ctd', 'ctd_composition', 'ctd_transition', 'ctd_distribution', 'conjoint_triad',
            'sequence_order_coupling_number', 'quasi_sequence_order',
            'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition'
        ]

        #temporary unit test output folder
        self.test_output_folder = os.path.join("tests", "test_outputs")

    @unittest.skip("")
    def test_aai_encoding(self):
        """ Testing AAI encoding functionality in Encoding module. """
#1.)    
        test_aai1 = ["FAUJ880110", "GEIM800111"]
        test_encoding1 = self.test_config1.aai_encoding(aai_indices=test_aai1, sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding1, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding1)))
        self.assertEqual(len(test_encoding1), 2, 
            "Expected 2 rows in output dataframe, got {}.".format(len(test_encoding1))) 
        self.assertEqual(set(list(test_encoding1["Index"])), set(test_aai1), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding1["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding1["Index"].dtype))
        self.assertEqual(test_encoding1["Category"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding1["Category"].dtype))
        self.assertEqual(test_encoding1["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding1["R2"].dtype))
        self.assertEqual(test_encoding1["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding1["RMSE"].dtype))
        self.assertEqual(test_encoding1["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding1["MSE"].dtype))
        self.assertEqual(test_encoding1["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding1["MAE"].dtype))
        self.assertEqual(test_encoding1["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding1["RPD"].dtype))
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding1["Explained Variance"].dtype))
        for cat in list(test_encoding1["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_encoding_output_columns))
#2.)
        test_aai2 = ["FAUJ880110", "GEIM800111", "JOND750102", "MAXF760102"]
        test_encoding2 = self.test_config2.aai_encoding(aai_indices=test_aai2, sort_by="RMSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding2, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding2)))
        self.assertEqual(len(test_encoding2), 4, 
            "Expected 4 rows in output dataframe, got {}.".format(len(test_encoding2))) 
        self.assertEqual(set(list(test_encoding2["Index"])), set(test_aai2), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding2["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding2["Index"].dtype))
        self.assertEqual(test_encoding2["Category"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding2["Category"].dtype))
        self.assertEqual(test_encoding2["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding2["R2"].dtype))
        self.assertEqual(test_encoding2["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding2["RMSE"].dtype))
        self.assertEqual(test_encoding2["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding2["MSE"].dtype))
        self.assertEqual(test_encoding2["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding2["MAE"].dtype))
        self.assertEqual(test_encoding2["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding2["RPD"].dtype))
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding2["Explained Variance"].dtype))
        for cat in list(test_encoding2["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for col in test_encoding2.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_encoding_output_columns))
#3.)
        test_aai3 = ["BIGC670101", "CHOP780211", "DESM900101", "FAUJ880113", "KANM800104"]
        test_encoding3 = self.test_config3.aai_encoding(aai_indices=test_aai3, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding3, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding3)))
        self.assertEqual(len(test_encoding3), 5, 
            "Expected 5 rows in output dataframe, got {}.".format(len(test_encoding3))) 
        self.assertEqual(set(list(test_encoding3["Index"])), set(test_aai3), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding3["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding3["Index"].dtype))
        self.assertEqual(test_encoding3["Category"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding3["Category"].dtype))
        self.assertEqual(test_encoding3["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding3["R2"].dtype))
        self.assertEqual(test_encoding3["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding3["RMSE"].dtype))
        self.assertEqual(test_encoding3["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding3["MSE"].dtype))
        self.assertEqual(test_encoding3["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding3["MAE"].dtype))
        self.assertEqual(test_encoding3["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding3["RPD"].dtype))
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding3["Explained Variance"].dtype))
        for cat in list(test_encoding3["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for col in test_encoding3.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_encoding_output_columns))
#4.)
        test_aai4 = [] #passing in no indices into the function will calculate all 566+ indices
        test_encoding4 = self.test_config4.aai_encoding(aai_indices=test_aai4, sort_by="MAE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding4, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding4)))
        self.assertEqual(len(test_encoding4), 566, 
            "Expected 566 rows in output dataframe, got {}.".format(len(test_encoding4))) 
        self.assertEqual(set(list(test_encoding4["Index"])), set(aaindex1.record_codes()), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding4["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding4["Index"].dtype))
        self.assertEqual(test_encoding4["Category"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding4["Category"].dtype))
        self.assertEqual(test_encoding4["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding4["R2"].dtype))
        self.assertEqual(test_encoding4["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding4["RMSE"].dtype))
        self.assertEqual(test_encoding4["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding4["MSE"].dtype))
        self.assertEqual(test_encoding4["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding4["MAE"].dtype))
        self.assertEqual(test_encoding4["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding4["RPD"].dtype))
        self.assertEqual(test_encoding4["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding4["Explained Variance"].dtype))
        for cat in list(test_encoding4["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for col in test_encoding4.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_encoding_output_columns))
#5.)
        test_aai5 = ["CHOP780211"]
        test_encoding5 = self.test_config3.aai_encoding(aai_indices=test_aai5, sort_by="invalid_metric", output_folder=self.test_output_folder) #R2 will then be used as default metric

        self.assertIsInstance(test_encoding5, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding5)))
        self.assertEqual(len(test_encoding5), 1, 
            "Expected 1 rows in output dataframe, got {}.".format(len(test_encoding5))) 
        self.assertEqual(set(list(test_encoding5["Index"])), set(test_aai5), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding5["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding5["Index"].dtype))
        self.assertEqual(test_encoding5["Category"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding5["Category"].dtype))
        self.assertEqual(test_encoding5["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding5["R2"].dtype))
        self.assertEqual(test_encoding5["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding5["RMSE"].dtype))
        self.assertEqual(test_encoding5["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding5["MSE"].dtype))
        self.assertEqual(test_encoding5["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding5["MAE"].dtype))
        self.assertEqual(test_encoding5["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding5["RPD"].dtype))
        self.assertEqual(test_encoding5["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding5["Explained Variance"].dtype))
        for cat in list(test_encoding5["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for col in test_encoding5.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_encoding_output_columns))
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aaindex_results.csv")),
            "Output csv storing encoding results not found.")
#6.)
        test_aai6 = "blahblah" 
        test_aai7 = "DESM9001ZZ"
        with self.assertRaises(ValueError):
            self.test_config1.aai_encoding(aai_indices=test_aai6, sort_by="RPD", output_folder=self.test_output_folder)
            self.test_config1.aai_encoding(aai_indices=test_aai7, sort_by="RMSE", output_folder=self.test_output_folder)
#7.)
        test_aai8 = 1234 
        test_aai9 = True 
        with self.assertRaises(TypeError):
            self.test_config2.aai_encoding(aai_indices=test_aai8, sort_by="MSE", output_folder=self.test_output_folder)
            self.test_config3.aai_encoding(aai_indices=test_aai9, sort_by="MAE", output_folder=self.test_output_folder)

    @unittest.skip("Descriptor encoding functionality can take a lot of time, skipping.")
    def test_descriptor_encoding(self):
        """ Testing Descriptor encoding functionality in Encoding module. """
#1.)
        test_desc1 = "amino_acid_composition"
        test_encoding1 = self.test_config1.descriptor_encoding(descriptors=test_desc1, desc_combo=1, 
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding1, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding1)))
        self.assertEqual(len(test_encoding1), 1, 
            "Expected 1 rows in output dataframe, got {}.".format(len(test_encoding1))) 
        self.assertEqual(test_encoding1["Descriptor"].values[0], test_desc1, 
            "Output index values don't match expected, got {}.".format(test_encoding1["Descriptor"].values[0])) 
        self.assertEqual(test_encoding1["Group"].values[0], "Composition", 
            "Output group values don't match expected, got {}.".format(test_encoding1["Group"].values[0]))
        self.assertEqual(test_encoding1["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding1["Descriptor"].dtype))
        self.assertEqual(test_encoding1["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding1["Group"].dtype))
        self.assertEqual(test_encoding1["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding1["R2"].dtype))
        self.assertEqual(test_encoding1["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding1["RMSE"].dtype))
        self.assertEqual(test_encoding1["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding1["MSE"].dtype))
        self.assertEqual(test_encoding1["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding1["MAE"].dtype))
        self.assertEqual(test_encoding1["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding1["RPD"].dtype))
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding1["Explained Variance"].dtype))
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_desc_encoding_output_columns))
#2.)
        test_desc2 = "moran_auto"
        test_encoding2 = self.test_config2.descriptor_encoding(descriptors=test_desc2, desc_combo=1,
            sort_by="MAE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding2, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding2)))
        self.assertEqual(len(test_encoding2), 1, 
            "Expected 1 rows in output dataframe, got {}.".format(len(test_encoding2))) 
        self.assertEqual(test_encoding2["Descriptor"].values[0], test_desc2, 
            "Output index values don't match expected, got {}.".format(test_encoding2["Descriptor"].values[0])) 
        self.assertEqual(test_encoding2["Group"].values[0], "Autocorrelation", 
            "Output group values don't match expected, got {}.".format(test_encoding2["Group"].values[0]))   
        self.assertEqual(test_encoding2["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding2["Descriptor"].dtype))
        self.assertEqual(test_encoding2["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding2["Group"].dtype))
        self.assertEqual(test_encoding2["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding2["R2"].dtype))
        self.assertEqual(test_encoding2["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding2["RMSE"].dtype))
        self.assertEqual(test_encoding2["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding2["MSE"].dtype))
        self.assertEqual(test_encoding2["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding2["MAE"].dtype))
        self.assertEqual(test_encoding2["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding2["RPD"].dtype))
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding2["Explained Variance"].dtype))
        for col in test_encoding2.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_desc_encoding_output_columns))
#3.)
        test_desc3 = ["ctd", "conjoint_triad", "dipeptide_composition"]
        test_encoding3 = self.test_config2.descriptor_encoding(descriptors=test_desc3, desc_combo=1, 
            sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding3, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding3)))
        self.assertEqual(len(test_encoding3), 3, 
            "Expected 3 rows in output dataframe, got {}.".format(len(test_encoding3))) 
        self.assertEqual(set(list(test_encoding3["Descriptor"])), set(test_desc3), 
            "Output index values don't match expected, got {}.".format(list(test_encoding3["Descriptor"]))) 
        self.assertEqual(set(list(test_encoding3["Group"])), set(["Composition", "Conjoint Triad", "CTD"]), 
            "Output group values don't match expected, got {}.".format(list(test_encoding3["Group"])))
        self.assertEqual(test_encoding3["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding3["Descriptor"].dtype))
        self.assertEqual(test_encoding3["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding3["Group"].dtype))
        self.assertEqual(test_encoding3["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding3["R2"].dtype))
        self.assertEqual(test_encoding3["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding3["RMSE"].dtype))
        self.assertEqual(test_encoding3["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding3["MSE"].dtype))
        self.assertEqual(test_encoding3["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding3["MAE"].dtype))
        self.assertEqual(test_encoding3["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding3["RPD"].dtype))
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding3["Explained Variance"].dtype))
        for col in test_encoding3.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_desc_encoding_output_columns))
#4.)
        test_desc4 = [] #no descriptors passed into encoding function will calculate/import all descriptors for dataset
        test_encoding4 = self.test_config1.descriptor_encoding(descriptors=test_desc4, desc_combo=1, 
            sort_by="RPD", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding4, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding4)))
        self.assertEqual(test_encoding4["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding4["Descriptor"].dtype))
        self.assertEqual(test_encoding4["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding4["Group"].dtype))
        self.assertEqual(test_encoding4["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding4["R2"].dtype))
        self.assertEqual(test_encoding4["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding4["RMSE"].dtype))
        self.assertEqual(test_encoding4["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding4["MSE"].dtype))
        self.assertEqual(test_encoding4["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding4["MAE"].dtype))
        self.assertEqual(test_encoding4["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding4["RPD"].dtype))
        self.assertEqual(test_encoding4["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding4["Explained Variance"].dtype))
        for group in list(test_encoding4["Group"]):
            self.assertIn(group, self.descriptor_groups, 
                "Group {} not found in list of groups:\n{}".format(group, self.descriptor_groups))
        for col in test_encoding4.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_desc_encoding_output_columns))
        for desc in list(test_encoding4["Descriptor"]):
            self.assertIn(desc, self.valid_descriptors, 
                "Descriptor {} not found in list of available descriptors:\n{}".format(desc, self.valid_descriptors))
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "desc_results.csv")),
            "Output csv storing encoding results not found.")
#5.)
        invalid_test_desc5 = "invalid_descriptor_name" 
        with self.assertRaises(ValueError):
            self.test_config1.descriptor_encoding(descriptors=invalid_test_desc5, desc_combo=1, sort_by="Explained Variance")
#6.)
        invalid_test_desc6 = 12345 
        invalid_test_desc7 = True 
        with self.assertRaises(TypeError):
            self.test_config1.descriptor_encoding(descriptors=invalid_test_desc6, desc_combo=1, sort_by="MAE")
            self.test_config1.descriptor_encoding(descriptors=invalid_test_desc7, desc_combo=1, sort_by="RMSE")

    # @unittest.skip("AAI + Descriptor encoding functionality can take a lot of time, skipping.")
    def test_aai_descriptor_encoding(self):
        """ Testing AAI + Descriptor encoding functionality in Encoding module. """
#1.)    
        test_aai1 = "FAUJ880110"
        test_desc1 = "ctd"
        test_encoding1 = self.test_config1.aai_descriptor_encoding(aai_indices=test_aai1, descriptors=test_desc1, 
            desc_combo=1, sort_by="R2", output_folder=self.test_output_folder)
        
        print("test_encoding1")
        print(test_encoding1)
        self.assertIsInstance(test_encoding1, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding1)))
        self.assertEqual(len(test_encoding1), 1, 
            "Expected 1 rows in output dataframe, got {}.".format(len(test_encoding1))) 
        self.assertEqual(test_encoding1["Index"].values[0], test_aai1, 
            "Output index values don't match expected, got {}.".format(test_encoding1["Index"].values[0])) 
        self.assertEqual(test_encoding1["Category"].values[0], "geometry", 
            "Output group values don't match expected, got {}.".format(test_encoding1["Group"].values[0]))  
        self.assertEqual(test_encoding1["Descriptor"].values[0], test_desc1, 
            "Output index values don't match expected, got {}.".format(test_encoding1["Descriptor"].values[0])) 
        self.assertEqual(test_encoding1["Group"].values[0], "CTD", 
            "Output group values don't match expected, got {}.".format(test_encoding1["Group"].values[0]))  
        self.assertEqual(test_encoding1["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding1["Index"].dtype))
        self.assertEqual(test_encoding1["Category"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding1["Category"].dtype))    
        self.assertEqual(test_encoding1["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding1["Descriptor"].dtype))
        self.assertEqual(test_encoding1["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding1["Group"].dtype))
        self.assertEqual(test_encoding1["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding1["R2"].dtype))
        self.assertEqual(test_encoding1["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding1["RMSE"].dtype))
        self.assertEqual(test_encoding1["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding1["MSE"].dtype))
        self.assertEqual(test_encoding1["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding1["MAE"].dtype))
        self.assertEqual(test_encoding1["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding1["RPD"].dtype))
        self.assertEqual(test_encoding1["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding1["Explained Variance"].dtype))
        for col in test_encoding1.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_desc_encoding_output_columns))
#2.)    
        test_aai2 = "BIGC670101, DAYM780201" 
        test_desc2 = ["tripeptide_composition", "quasi_sequence_order", "sequence_order_coupling_number"]
        test_encoding2 = self.test_config2.aai_descriptor_encoding(aai_indices=test_aai2, descriptors=test_desc2, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding2, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding2)))
        self.assertEqual(len(test_encoding2), 6, 
            "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding2))) 
        self.assertEqual(set(list(test_encoding2["Index"])), set(test_aai2.replace(' ', '').split(',')), 
            "Expected index column to be type string, got {}.".format(test_encoding2["Index"].dtype))
        self.assertEqual(set(list(test_encoding2["Descriptor"])), set(test_desc2), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding2["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding2["Index"].dtype))
        self.assertEqual(test_encoding2["Category"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding2["Category"].dtype))    
        self.assertEqual(test_encoding2["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding2["Descriptor"].dtype))
        self.assertEqual(test_encoding2["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding2["Group"].dtype))
        self.assertEqual(test_encoding2["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding2["R2"].dtype))
        self.assertEqual(test_encoding2["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding2["RMSE"].dtype))
        self.assertEqual(test_encoding2["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding2["MSE"].dtype))
        self.assertEqual(test_encoding2["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding2["MAE"].dtype))
        self.assertEqual(test_encoding2["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding2["RPD"].dtype))
        self.assertEqual(test_encoding2["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding2["Explained Variance"].dtype))
        for cat in list(test_encoding2["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for group in list(test_encoding2["Group"]):
            self.assertIn(group, self.descriptor_groups, 
                "Group {} not found in list of groups:\n{}".format(group, self.descriptor_groups))
        for col in test_encoding2.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_desc_encoding_output_columns))
#3.)    
        test_aai3 = "GEOR030107, KARS160113, COWR900101"
        test_desc3 = ["amino_acid_composition", "ctd_distribution"]
        test_encoding3 = self.test_config3.aai_descriptor_encoding(aai_indices=test_aai3, descriptors=test_desc3, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder) #**
        
        self.assertIsInstance(test_encoding3, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding3)))
        self.assertEqual(len(test_encoding3), 6, 
            "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding3)))  #**
        self.assertEqual(set(list(test_encoding3["Index"])), set(test_aai3.replace(' ', '').split(',')), 
            "Expected index column to be type string, got {}.".format(list(test_encoding3["Index"])))
        self.assertEqual(set(list(test_encoding3["Descriptor"])), set(test_desc3), 
            "Output index values don't match expected.") 
        self.assertEqual(test_encoding3["Index"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding3["Index"].dtype))
        self.assertEqual(test_encoding3["Category"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding3["Category"].dtype))    
        self.assertEqual(test_encoding3["Descriptor"].dtype, "string[python]", 
            "Expected index column to be type string, got {}.".format(test_encoding3["Descriptor"].dtype))
        self.assertEqual(test_encoding3["Group"].dtype, "string[python]",
            "Expected category column to be type string, got {}.".format(test_encoding3["Group"].dtype))
        self.assertEqual(test_encoding3["R2"].dtype, float, 
            "Expected R2 column to be type float, got {}.".format(test_encoding3["R2"].dtype))
        self.assertEqual(test_encoding3["RMSE"].dtype, float, 
            "Expected RMSE column to be type float, got {}.".format(test_encoding3["RMSE"].dtype))
        self.assertEqual(test_encoding3["MSE"].dtype, float,
            "Expected MSE column to be type float, got {}.".format(test_encoding3["MSE"].dtype))
        self.assertEqual(test_encoding3["MAE"].dtype, float,
            "Expected MAE column to be type float, got {}.".format(test_encoding3["MAE"].dtype))
        self.assertEqual(test_encoding3["RPD"].dtype, float,
            "Expected RPD column to be type float, got {}.".format(test_encoding3["RPD"].dtype))
        self.assertEqual(test_encoding3["Explained Variance"].dtype, float,
            "Expected Explained Variance column to be type float, got {}.".format(test_encoding3["Explained Variance"].dtype))
        for cat in list(test_encoding3["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        for group in list(test_encoding3["Group"]):
            self.assertIn(group, self.descriptor_groups, 
                "Group {} not found in list of groups:\n{}".format(group, self.descriptor_groups))
        for col in test_encoding3.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
                "Column {} not found in list of column:\n{}".format(col, self.expected_aai_desc_encoding_output_columns))
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
            "Output csv storing encoding results not found.")
#4.)    
        test_aai4 = ["invalid_aai_index"]
        test_aai5 = ""
        test_desc4 = ["invalid_descriptor_name"]
        with self.assertRaises(ValueError):
            self.test_config1.aai_descriptor_encoding(aai_indices=test_aai4, descriptors=test_desc4, desc_combo=1, sort_by="MSE")
            self.test_config1.aai_descriptor_encoding(aai_indices=test_aai5, descriptors=test_desc4, desc_combo=1, sort_by="MSE")
#6.)
        test_aai6 = 12345
        test_desc5 = 1000
        with self.assertRaises(TypeError):
            self.test_config1.aai_descriptor_encoding(aai_indices=test_aai6, descriptors=test_desc5, desc_combo=1, sort_by="MAE")

#7.)    ** Below inputs result in all AAI Indices being encoded with all descriptors, commenting out due to time and resource constraints **
        # test_aai7 = []
        # test_desc6 = []
        # test_encoding7 = self.test_config1.aai_descriptor_encoding(aai_indices=test_aai7, descriptors=test_desc6, 
        #     desc_combo=1, sort_by="MAE", output_folder=self.test_output_folder)

        # self.assertIsInstance(test_encoding7, pd.DataFrame, 
        #     "Expected output to be a dataframe, got {}.".format(type(test_encoding7)))
        # self.assertEqual(len(test_encoding7), 8490, 
        #     "Expected 8490 rows in output dataframe, got {}.".format(len(test_encoding7))) 
        # self.assertEqual(list(test_encoding7["Index"]), test_aai3, 
        #     "Expected index column to be type string, got {}.".format(test_encoding7["Index"].dtype))
        # self.assertEqual(list(test_encoding7["Descriptor"]), test_desc1, 
        #     "Output index values don't match expected.") 
        # self.assertEqual(test_encoding7["Index"].dtype, "string[python]", 
        #     "Expected index column to be type string, got {}.".format(test_encoding7["Index"].dtype))
        # self.assertEqual(test_encoding7["Category"].dtype, "string[python]", 
        #     "Expected index column to be type string, got {}.".format(test_encoding7["Category"].dtype))    
        # self.assertEqual(test_encoding7["Descriptor"].dtype, "string[python]", 
        #     "Expected index column to be type string, got {}.".format(test_encoding7["Descriptor"].dtype))
        # self.assertEqual(test_encoding7["Group"].dtype, "string[python]",
        #     "Expected category column to be type string, got {}.".format(test_encoding7["Group"].dtype))
        # self.assertEqual(test_encoding7["R2"].dtype, float, 
        #     "Expected R2 column to be type float, got {}.".format(test_encoding7["R2"].dtype))
        # self.assertEqual(test_encoding7["RMSE"].dtype, float, 
        #     "Expected RMSE column to be type float, got {}.".format(test_encoding7["RMSE"].dtype))
        # self.assertEqual(test_encoding7["MSE"].dtype, float,
        #     "Expected MSE column to be type float, got {}.".format(test_encoding7["MSE"].dtype))
        # self.assertEqual(test_encoding7["MAE"].dtype, float,
        #     "Expected MAE column to be type float, got {}.".format(test_encoding7["MAE"].dtype))
        # self.assertEqual(test_encoding7["RPD"].dtype, float,
        #     "Expected RPD column to be type float, got {}.".format(test_encoding7["RPD"].dtype))
        # self.assertEqual(test_encoding7["Explained Variance"].dtype, float,
        #     "Expected Explained Variance column to be type float, got {}.".format(test_encoding5["Explained Variance"].dtype))
        # for cat in list(test_encoding7["Category"]):
        #     self.assertIn(cat, self.index_categories, 
        #         "Category {} not found in list of categories:\n{}".format(cat, self.index_categories))
        # for group in list(test_encoding7["Group"]):
        #     self.assertIn(group, self.descriptor_groups, 
        #         "Group {} not found in list of groups:\n{}".format(group, self.descriptor_groups))
        # for col in test_encoding7.columns:
        #     self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
        #         "Column {} not found in list of column:\n{}".format(col, self.expected_aai_desc_encoding_output_columns))

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        #removing any of the temp files created such as the results files, if
        #you want to verify the results files are actually being created then
        #comment out the below code block
        if (os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME)):
            shutil.rmtree(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, ignore_errors=False, onerror=None)
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)