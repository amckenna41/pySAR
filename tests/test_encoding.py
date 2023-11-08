#################################################################################
#################             Encoding Module Tests             #################
#################################################################################

import pandas as pd
import os
import shutil
import unittest
from aaindex import aaindex1
import numpy as np
unittest.TestLoader.sortTestMethodsUsing = None
#suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore")
                        
import pySAR.encoding as pysar_
import pySAR.globals_ as _globals

class EncodingTests(unittest.TestCase):
    """
    Test suite for testing encoding module and functionality in pySAR package. 

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
        self.test_config_thermostability = pysar_.Encoding(config_file=self.all_config_files[0])
        self.test_config_enantioselectivity = pysar_.Encoding(config_file=self.all_config_files[1])
        self.test_config_absorption = pysar_.Encoding(config_file=self.all_config_files[2])
        self.test_config_localization = pysar_.Encoding(config_file=self.all_config_files[3])

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

    # @unittest.skip("Skipping aai encoding tests.")
    def test_aai_encoding(self):
        """ Testing AAI encoding functionality in Encoding module. """
#1.)    
        test_aai1 = ["FAUJ880110", "GEIM800111"] #thermostability dataset and config
        test_encoding_thermostability = self.test_config_thermostability.aai_encoding(aai_indices=test_aai1, sort_by="R2", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_thermostability)))
        self.assertEqual(len(test_encoding_thermostability), 2, 
            "Expected 2 rows in output dataframe, got {}.".format(len(test_encoding_thermostability))) 
        self.assertEqual(set(list(test_encoding_thermostability["Index"])), set(test_aai1), 
            "Output index values don't match expected, got {}.".format(set(list(test_encoding_thermostability["Index"])))) 
        for cat in list(test_encoding_thermostability["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}.".format(cat, self.index_categories))
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_thermostability[col])))  
#2.)
        test_aai2 = ["FAUJ880110", "GEIM800111", "JOND750102", "MAXF760102"] #enantioselectivity dataset and config
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.aai_encoding(aai_indices=test_aai2, sort_by="RMSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_enantioselectivity)))
        self.assertEqual(len(test_encoding_enantioselectivity), 4, 
            "Expected 4 rows in output dataframe, got {}.".format(len(test_encoding_enantioselectivity))) 
        self.assertEqual(set(list(test_encoding_enantioselectivity["Index"])), set(test_aai2), 
            "Output index values don't match expected, got {}.".format(set(list(test_encoding_enantioselectivity["Index"])))) 
        for cat in list(test_encoding_enantioselectivity["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}.".format(cat, self.index_categories))
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_enantioselectivity[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_enantioselectivity[col])))  
#3.)
        test_aai3 = ["BIGC670101", "CHOP780211", "DESM900101", "FAUJ880113", "KANM800104"] #absorption dataset and config
        test_encoding_absorption = self.test_config_absorption.aai_encoding(aai_indices=test_aai3, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_absorption)))
        self.assertEqual(len(test_encoding_absorption), 5, 
            "Expected 5 rows in output dataframe, got {}.".format(len(test_encoding_absorption))) 
        self.assertEqual(set(list(test_encoding_absorption["Index"])), set(test_aai3), 
            "Output index values don't match expected, got {}.".format(set(list(test_encoding_absorption["Index"])))) 
        for cat in list(test_encoding_absorption["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}.".format(cat, self.index_categories))
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_absorption[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_absorption[col])))  
#4.)
        test_aai4 = [] #passing in no indices into the function will calculate all 566+ indices - localization dataset and config
        test_encoding_localization = self.test_config_localization.aai_encoding(aai_indices=test_aai4, sort_by="MAE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_localization, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_localization)))
        self.assertEqual(len(test_encoding_localization), 566,
            "Expected 566 rows in output dataframe, got {}.".format(len(test_encoding_localization))) 
        self.assertEqual(set(list(test_encoding_localization["Index"])), set(aaindex1.record_codes()),
            "Output index values don't match expected, got {}.".format(set(list(test_encoding_localization["Index"])))) 
        for cat in list(test_encoding_localization["Category"]):
            self.assertIn(cat, self.index_categories, 
                "Category {} not found in list of categories:\n{}.".format(cat, self.index_categories))
        for col in test_encoding_localization.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_localization[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_localization[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_localization[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_localization[col])))  
#5.)
        test_aai6 = "blahblah" 
        test_aai7 = "DESM9001ZZ"
        with self.assertRaises(ValueError):
            self.test_config_thermostability.aai_encoding(aai_indices=test_aai6, sort_by="RPD", output_folder=self.test_output_folder)
            self.test_config_enantioselectivity.aai_encoding(aai_indices=test_aai7, sort_by="RMSE", output_folder=self.test_output_folder)
#7.)
        test_aai8 = 1234 
        test_aai9 = True 
        with self.assertRaises(TypeError):
            self.test_config_absorption.aai_encoding(aai_indices=test_aai8, sort_by="MSE", output_folder=self.test_output_folder)
            self.test_config_localization.aai_encoding(aai_indices=test_aai9, sort_by="MAE", output_folder=self.test_output_folder)

    # @unittest.skip("Descriptor encoding functionality can take a lot of time, skipping.")
    def test_descriptor_encoding(self):
        """ Testing Descriptor encoding functionality in Encoding module. """ 
#1.)
        test_desc1 = "amino_acid_composition"
        test_encoding_thermostability = self.test_config_thermostability.descriptor_encoding(descriptors=test_desc1, desc_combo=1, 
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_thermostability)))
        self.assertEqual(len(test_encoding_thermostability), 1, 
            "Expected 1 row in output dataframe, got {}.".format(len(test_encoding_thermostability))) 
        self.assertEqual(test_encoding_thermostability["Descriptor"].values[0], test_desc1, 
            "Output index values don't match expected, got {}.".format(test_encoding_thermostability["Descriptor"].values[0])) 
        self.assertEqual(test_encoding_thermostability["Group"].values[0], "Composition", 
            "Output group values don't match expected, got {}.".format(test_encoding_thermostability["Group"].values[0]))
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_thermostability[col])))  
#2.)
        test_desc2 = "moran_auto"
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.descriptor_encoding(descriptors=test_desc2, desc_combo=1,
            sort_by="MAE", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_enantioselectivity)))
        self.assertEqual(len(test_encoding_enantioselectivity), 1, 
            "Expected 1 row in output dataframe, got {}.".format(len(test_encoding_enantioselectivity))) 
        self.assertEqual(test_encoding_enantioselectivity["Descriptor"].values[0], test_desc2, 
            "Output index values don't match expected, got {}.".format(test_encoding_enantioselectivity["Descriptor"].values[0])) 
        self.assertEqual(test_encoding_enantioselectivity["Group"].values[0], "Autocorrelation", 
            "Output group values don't match expected, got {}.".format(test_encoding_enantioselectivity["Group"].values[0]))   
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_enantioselectivity[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_enantioselectivity[col])))  
#3.)
        test_desc3 = ["ctd", "conjoint_triad", "dipeptide_composition"]
        test_encoding_absorption = self.test_config_absorption.descriptor_encoding(descriptors=test_desc3, desc_combo=1, 
            sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_absorption)))
        self.assertEqual(len(test_encoding_absorption), 3, 
            "Expected 3 rows in output dataframe, got {}.".format(len(test_encoding_absorption))) 
        self.assertEqual(set(list(test_encoding_absorption["Descriptor"])), set(test_desc3), 
            "Output index values don't match expected, got {}.".format(list(test_encoding_absorption["Descriptor"]))) 
        self.assertEqual(set(list(test_encoding_absorption["Group"])), set(["Composition", "Conjoint Triad", "CTD"]), 
            "Output group values don't match expected, got {}.".format(list(test_encoding_absorption["Group"])))
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_absorption[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_absorption[col]))) 
#4.)
        test_desc4 = [] #no descriptors passed into encoding function will calculate/import all descriptors for dataset
        test_encoding_thermostability = self.test_config_thermostability.descriptor_encoding(descriptors=test_desc4, desc_combo=1, 
            sort_by="RPD", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_thermostability)))
        self.assertEqual(len(test_encoding_thermostability), 15, 
            "Expected 15 rows in output dataframe, got {}.".format(len(test_encoding_thermostability))) 
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_encoding_output_columns))
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_thermostability[col]))) 
        for group in list(test_encoding_thermostability["Group"]):
            self.assertIn(group, self.descriptor_groups, 
                "Group {} not found in list of groups:\n{}.".format(group, self.descriptor_groups))
        for desc in list(test_encoding_thermostability["Descriptor"]):
            self.assertIn(desc, self.valid_descriptors, 
                "Descriptor {} not found in list of available descriptors:\n{}.".format(desc, self.valid_descriptors)) 
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "desc_results.csv")),
            "Output csv storing encoding results not found.")
#5.)
        invalid_test_desc5 = "invalid_descriptor_name" 
        invalid_test_desc6 = "blahblahblah" 
        with self.assertRaises(ValueError):
            self.test_config_thermostability.descriptor_encoding(descriptors=invalid_test_desc5, desc_combo=1, sort_by="MSE")
            self.test_config_enantioselectivity.descriptor_encoding(descriptors=invalid_test_desc6, desc_combo=1, sort_by="RMSE")
#6.)
        invalid_test_desc7 = 12345 
        invalid_test_desc8 = True 
        with self.assertRaises(TypeError):
            self.test_config_absorption.descriptor_encoding(descriptors=invalid_test_desc7, desc_combo=1, sort_by="MAE")
            self.test_config_localization.descriptor_encoding(descriptors=invalid_test_desc8, desc_combo=1, sort_by="RPD")

    # @unittest.skip("AAI + Descriptor encoding functionality can take a lot of time, skipping.")
    def test_aai_descriptor_encoding(self):
        """ Testing AAI + Descriptor encoding functionality in Encoding module. """ 
#1.)    
        test_aai1 = "FAUJ880110"  #thermostability
        test_desc1 = "tripeptide_composition"
        test_encoding_thermostability = self.test_config_thermostability.aai_descriptor_encoding(aai_indices=test_aai1, descriptors=test_desc1, 
            desc_combo=1, sort_by="R2", output_folder=self.test_output_folder) 
        
        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_thermostability)))
        self.assertEqual(len(test_encoding_thermostability), 1, 
            "Expected 1 row in output dataframe, got {}.".format(len(test_encoding_thermostability))) 
        self.assertEqual(test_encoding_thermostability["Index"].values[0], test_aai1, 
            "Output index values don't match expected, got {}.".format(test_encoding_thermostability["Index"].values[0])) 
        self.assertEqual(test_encoding_thermostability["Category"].values[0], "geometry", 
            "Output group values don't match expected, got {}.".format(test_encoding_thermostability["Group"].values[0]))  
        self.assertEqual(set(list(test_encoding_thermostability["Descriptor"].values)), {"tripeptide_composition"}, 
            "Output descriptor column values don't match expected, got\n{}.".format(test_encoding_thermostability["Descriptor"])) 
        self.assertEqual(test_encoding_thermostability["Group"].values[0], "Composition", 
            "Output group values don't match expected, got {}.".format(test_encoding_thermostability["Group"].values[0]))  
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_desc_encoding_output_columns))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_thermostability[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_thermostability[col]))) 
#2.)    
        test_aai2 = "BIGC670101, DAYM780201"  #enantioselectivity
        test_desc2 = ["ctd", "quasi_sequence_order", "sequence_order_coupling_number"]
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.aai_descriptor_encoding(aai_indices=test_aai2, descriptors=test_desc2, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_enantioselectivity)))
        self.assertEqual(len(test_encoding_enantioselectivity), 6, 
            "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding_enantioselectivity))) 
        self.assertEqual(set(list(test_encoding_enantioselectivity["Index"])), set(test_aai2.replace(' ', '').split(',')), 
            "Output Index column does not match expected, got\n{}.".format(test_encoding_enantioselectivity["Index"]))
        self.assertEqual(set(list(test_encoding_enantioselectivity["Category"].values)), {'composition', 'geometry'}, 
            "Output category values don't match expected, got {}.".format(test_encoding_enantioselectivity["Category"].values))  
        self.assertEqual(set(list(test_encoding_enantioselectivity["Descriptor"])), set(test_desc2), 
            "Output descriptor column values don't match expected, got\n{}.".format(test_encoding_enantioselectivity["Descriptor"])) 
        self.assertEqual(set(list(test_encoding_enantioselectivity["Group"].values)), {"Sequence Order", "CTD"}, 
            "Output group values don't match expected, got {}.".format(test_encoding_enantioselectivity["Group"].values))  
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_desc_encoding_output_columns))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_enantioselectivity[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_enantioselectivity[col]))) 
#3.)    
        test_aai3 = "GEOR030107, KARS160113, COWR900101"  #absorption 
        test_desc3 = ["amino_acid_composition", "ctd_distribution"]
        test_encoding_absorption = self.test_config_absorption.aai_descriptor_encoding(aai_indices=test_aai3, descriptors=test_desc3, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder) 
        
        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_absorption)))
        self.assertEqual(len(test_encoding_absorption), 6, 
            "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding_absorption))) 
        self.assertEqual(set(list(test_encoding_absorption["Index"])), set(test_aai3.replace(' ', '').split(',')), 
            "Output Index column does not match expected, got\n{}.".format(test_encoding_absorption["Index"]))
        self.assertEqual(set(list(test_encoding_absorption["Category"].values)), {'hydrophobic', 'meta', 'sec_struct'},  
            "Output category values don't match expected, got {}.".format(test_encoding_absorption["Category"].values)) 
        self.assertEqual(set(list(test_encoding_absorption["Descriptor"])), set(test_desc3), 
            "Output descriptor column values don't match expected, got\n{}.".format(test_encoding_absorption["Descriptor"])) 
        self.assertEqual(set(list(test_encoding_absorption["Group"].values)), {"Composition", "CTD"}, 
            "Output group values don't match expected, got {}.".format(test_encoding_absorption["Group"].values))  
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_desc_encoding_output_columns))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_absorption[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_absorption[col]))) 
#4.)
        test_aai4 = ["BEGF750103", "CIDH920103", "JOND920101"]  #localization 
        test_desc4 = ["dipeptide_composition", "ctd_transition"]
        test_encoding_localization = self.test_config_localization.aai_descriptor_encoding(aai_indices=test_aai4, descriptors=test_desc4, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_localization, pd.DataFrame, 
            "Expected output to be a dataframe, got {}.".format(type(test_encoding_localization)))
        self.assertEqual(len(test_encoding_localization), 6, 
            "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding_localization))) 
        self.assertEqual(set(list(test_encoding_localization["Index"])), set(test_aai4), 
            "Output Index column does not match expected, got\n{}.".format(test_encoding_localization["Index"]))
        self.assertEqual(set(list(test_encoding_localization["Category"].values)), {"sec_struct", "composition", "hydrophobic"}, 
            "Output category values don't match expected, got {}.".format(test_encoding_localization["Category"].values)) 
        self.assertEqual(set(list(test_encoding_localization["Descriptor"])), {"dipeptide_composition", "ctd_transition"}, 
            "Output descriptor column values don't match expected, got\n{}.".format(test_encoding_localization["Descriptor"])) 
        self.assertEqual(set(list(test_encoding_localization["Group"].values)), {"Composition", "CTD"}, 
            "Output group values don't match expected, got {}.".format(test_encoding_localization["Group"].values))  
        for col in test_encoding_localization.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_desc_encoding_output_columns))
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_localization[col].values)),
                    "Column {} expected to be of type string got {}.".format(col, type(test_encoding_localization[col])))
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_localization[col].values)),
                    "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_localization[col]))) 
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
            "Output csv storing encoding results not found.")
#5.)    
        test_aai5 = ["invalid_aai_index"]
        test_aai6 = ""
        test_desc5 = ["invalid_descriptor_name"]
        with self.assertRaises(ValueError):
            self.test_config_thermostability.aai_descriptor_encoding(aai_indices=test_aai5, descriptors=test_desc5, desc_combo=1, sort_by="MSE")
            self.test_config_enantioselectivity.aai_descriptor_encoding(aai_indices=test_aai6, descriptors=test_desc5, desc_combo=1, sort_by="MSE")
#6.)
        test_aai7 = 12345
        test_desc6 = 1000
        test_desc7 = False
        with self.assertRaises(TypeError):
            self.test_config_absorption.aai_descriptor_encoding(aai_indices=test_aai7, descriptors=test_desc6, desc_combo=1, sort_by="MAE")
            self.test_config_localization.aai_descriptor_encoding(aai_indices=test_aai7, descriptors=test_desc7, desc_combo=1, sort_by="MAE")

#7.)    ** Below inputs result in all AAI Indices being encoded with all descriptors, commenting out due to time and resource constraints **
        # test_aai8 = []
        # test_desc7 = []
        # test_encoding7 = self.test_encoding_thermostability.aai_descriptor_encoding(aai_indices=test_aai8, descriptors=test_desc7, 
        #     desc_combo=1, sort_by="MAE", output_folder=self.test_output_folder)

        # self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
        #     "Expected output to be a dataframe, got {}.".format(type(test_encoding_thermostability)))
        # self.assertEqual(len(test_encoding_thermostability), 6, 
        #     "Expected 6 rows in output dataframe, got {}.".format(len(test_encoding_thermostability))) 
        # self.assertEqual(set(list(test_encoding_thermostability["Index"])), set(test_aai3.replace(' ', '').split(',')), 
        #     "Expected index column to be type string, got {}.".format(test_encoding_thermostability["Index"].dtype))
        # self.assertEqual(test_encoding_thermostability["Category"].values[0], ["Composition", "CTD"], 
        #     "Output category values don't match expected, got {}.".format(test_encoding_thermostability["Category"].values[0]))   #**
        # self.assertEqual(set(list(test_encoding_thermostability["Descriptor"])), set(test_desc3), 
        #     "Output index values don't match expected.") 
        # self.assertEqual(test_encoding_thermostability["Group"].values[0], ["Composition", "CTD"], 
        #     "Output group values don't match expected, got {}.".format(test_encoding_thermostability["Group"].values[0]))  
        # for col in test_encoding_thermostability.columns:
        #     self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
        #         "Col {} not found in list of expected columns:\n{}.".format(col, self.expected_aai_desc_encoding_output_columns))
        #     if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
        #         self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
        #             "Column {} expected to be of type string got {}.".format(col, type(test_encoding_thermostability[col])))
        #     else:
        #         self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
        #             "Column {} expected to be of type np.float64 got {}.".format(col, type(test_encoding_thermostability[col]))) 
        # for group in list(test_encoding_thermostability["Group"]):
        #     self.assertIn(group, self.descriptor_groups, 
        #         "Group {} not found in list of groups:\n{}.".format(group, self.descriptor_groups))
        # for desc in list(test_encoding_thermostability["Descriptor"]):
        #     self.assertIn(desc, self.valid_descriptors, 
        #         "Descriptor {} not found in list of available descriptors:\n{}.".format(desc, self.valid_descriptors)) 
        # self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
        #     "Output dir storing encoding results not found.")
        # self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
        #     "Output csv storing encoding results not found.")

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