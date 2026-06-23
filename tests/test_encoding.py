#################################################################################
#################             Encoding Module Tests             #################
#################################################################################

import glob
import pandas as pd
import os
import shutil
import unittest
from aaindex import aaindex1
import numpy as np
unittest.TestLoader.sortTestMethodsUsing = None
                        
import pySAR.encoding as pysar_
import pySAR.globals_ as _globals
from pySAR.descriptors import Descriptors

# @unittest.skip("")
class EncodingTests(unittest.TestCase):
    """
    Test suite for testing encoding module and functionality in pySAR package. 

    Test Cases
    ==========
    test_encoding_init:
        testing Encoding class initialization and attribute assignment.
    test_validate_inputs:
        testing validate_inputs method for normalisation, deduplication and error handling.
    test_validate_desc_combo:
        testing validate_desc_combo for valid and invalid combination values.
    test_build_features:
        testing build_features method for AAI, descriptor and combined feature types.
    test_aai_encoding:
        testing correct aai encoding Encoding class functionality.
    test_aai_encoding_params:
        testing parameter-level behaviour of aai_encoding: sample_mode, max_models, n_jobs, random_state and sort_by fallback.
    test_aai_encoding_output_files:
        testing that aai_encoding creates the expected output directory and results CSV file.
    test_aai_encoding_comma_string_input:
        testing aai_encoding with a comma-separated string passed as aai_indices.
    test_descriptor_encoding:
        testing correct descriptor encoding Encoding class functionality.
    test_descriptor_encoding_combos:
        testing descriptor encoding with desc_combo=2 and desc_combo=3, and invalid combo values.
    test_aai_descriptor_encoding:
        testing correct aai + descriptor encoding Encoding class functionality.
    test_output_sorting:
        testing that encoding results are sorted correctly by each supported metric.
    test_metric_value_ranges:
        testing that all numeric metric columns contain valid non-negative float64 values.
    test_encoding_result_dataclass:
        testing EncodingResult dataclass construction, attributes, and from_dataframe() factory.
    test_export_best_model:
        testing that export_best_model=True saves a best_model.pkl in the output folder.
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
            'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition',
            # new protpy v1.4.1 descriptors
            'gravy', 'aromaticity', 'instability_index', 'isoelectric_point', 'molecular_weight',
            'charge_distribution', 'hydrophobic_polar_charged_composition', 'secondary_structure_propensity',
            'kmer_composition', 'reduced_alphabet_composition', 'motif_composition',
            'amino_acid_pair_composition', 'aliphatic_index', 'extinction_coefficient', 'boman_index',
            'aggregation_propensity', 'hydrophobic_moment', 'shannon_entropy'
        ]

        #temporary unit test output folder
        self.test_output_folder = os.path.join("tests", "test_outputs")

    def _assert_columns(self, df, expected_columns):
        for col in df.columns:
            self.assertIn(col, expected_columns,
                f"Col {col} not found in expected columns:\n{expected_columns}.")

    def _assert_string_columns(self, df, columns):
        for col in columns:
            self.assertTrue(all(isinstance(row, str) for row in list(df[col].values)),
                f"Column {col} expected to be string values got {df[col].dtype}.")

    def _assert_metric_columns_are_float(self, df):
        metric_cols = ["R2", "RMSE", "MSE", "MAE", "RPD", "Explained Variance"]
        for col in metric_cols:
            self.assertTrue(np.issubdtype(df[col].dtype, np.floating),
                f"Expected floating dtype for column {col}, got {df[col].dtype}.")

    def test_encoding_init(self):
        """ Testing Encoding class initialization and attribute assignment. """
#1.)    validate instantiation from each config file stores config_file and defaults verbose to True
        for config in self.all_config_files:
            enc = pysar_.Encoding(config_file=config)
            self.assertIsInstance(enc, pysar_.Encoding,
                f"Expected Encoding instance, got {type(enc)}.")
            self.assertEqual(enc.config_file, config,
                f"Expected config_file attribute to match input, got {enc.config_file}.")
            self.assertTrue(enc.verbose,
                "Expected verbose to be True by default.")
#2.)    verbose=False is stored correctly when explicitly set
        enc_quiet = pysar_.Encoding(config_file=self.all_config_files[0], verbose=False)
        self.assertFalse(enc_quiet.verbose,
            "Expected verbose to be False when explicitly set.")
#3.)    __str__ returns a non-empty string containing the config filename
        enc = self.test_config_thermostability
        str_val = str(enc)
        self.assertIsInstance(str_val, str,
            f"Expected __str__ to return str, got {type(str_val)}.")
        self.assertIn(os.path.basename(self.all_config_files[0]), str_val,
            f"Expected config filename in __str__ output, got:\n{str_val}.")
#4.)    __repr__ wraps __str__ and also contains the config filename
        repr_val = repr(enc)
        self.assertIsInstance(repr_val, str,
            f"Expected __repr__ to return str, got {type(repr_val)}.")
        self.assertIn(os.path.basename(self.all_config_files[0]), repr_val,
            f"Expected config filename in __repr__ output, got:\n{repr_val}.")

    def test_aai_encoding(self):
        """ Testing AAI encoding functionality in Encoding module. """
#1.)    
        test_aai1 = ["FAUJ880110", "GEIM800111"] #thermostability dataset and config
        test_encoding_thermostability = self.test_config_thermostability.aai_encoding(aai_indices=test_aai1, sort_by="R2", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_thermostability)}.")
        self.assertEqual(len(test_encoding_thermostability), 2, 
            f"Expected 2 rows in output dataframe, got {len(test_encoding_thermostability)}.") 
        self.assertEqual(set(list(test_encoding_thermostability["Index"])), set(test_aai1), 
            f'Output index values don\'t match expected, got {set(list(test_encoding_thermostability["Index"]))}.')
        for cat in list(test_encoding_thermostability["Category"]):
            self.assertIn(cat, self.index_categories, 
                f"Category {cat} not found in list of categories:\n{self.index_categories}.")
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_thermostability[col])}.")  
#2.)
        test_aai2 = ["FAUJ880110", "GEIM800111", "JOND750102", "MAXF760102"] #enantioselectivity dataset and config
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.aai_encoding(aai_indices=test_aai2, sort_by="RMSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_enantioselectivity)}.")
        self.assertEqual(len(test_encoding_enantioselectivity), 4, 
            f"Expected 4 rows in output dataframe, got {len(test_encoding_enantioselectivity)}.") 
        self.assertEqual(set(list(test_encoding_enantioselectivity["Index"])), set(test_aai2), 
            f'Output index values don\'t match expected, got {set(list(test_encoding_enantioselectivity["Index"]))}.')
        for cat in list(test_encoding_enantioselectivity["Category"]):
            self.assertIn(cat, self.index_categories, 
                f"Category {cat} not found in list of categories:\n{self.index_categories}.")
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_enantioselectivity[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_enantioselectivity[col])}.")  
#3.)
        test_aai3 = ["BIGC670101", "CHOP780211", "DESM900101", "FAUJ880113", "KANM800104"] #absorption dataset and config
        test_encoding_absorption = self.test_config_absorption.aai_encoding(aai_indices=test_aai3, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_absorption)}.")
        self.assertEqual(len(test_encoding_absorption), 5, 
            f"Expected 5 rows in output dataframe, got {len(test_encoding_absorption)}.") 
        self.assertEqual(set(list(test_encoding_absorption["Index"])), set(test_aai3), 
            f'Output index values don\'t match expected, got {set(list(test_encoding_absorption["Index"]))}.')
        for cat in list(test_encoding_absorption["Category"]):
            self.assertIn(cat, self.index_categories, 
                f"Category {cat} not found in list of categories:\n{self.index_categories}.")
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_absorption[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_absorption[col])}.")  
#4.)
        test_aai4 = ["BIGC670101", "CHOP780211"] #localization dataset and config
        test_encoding_localization = self.test_config_localization.aai_encoding(aai_indices=test_aai4, sort_by="MAE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_localization, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_localization)}.")
        self.assertEqual(len(test_encoding_localization), 2,
            f"Expected 2 rows in output dataframe, got {len(test_encoding_localization)}.")
        self.assertEqual(set(list(test_encoding_localization["Index"])), set(test_aai4),
            f'Output index values don\'t match expected, got {set(list(test_encoding_localization["Index"]))}.')
        for cat in list(test_encoding_localization["Category"]):
            self.assertIn(cat, self.index_categories,
                f"Category {cat} not found in list of categories:\n{self.index_categories}.")
        for col in test_encoding_localization.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Index" or col == "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_localization[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_localization[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_localization[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_localization[col])}.")
#5.)
        test_aai6 = "blahblah" 
        test_aai7 = "ZZZZZZZZXX"  # clearly invalid — no close AAI index match
        with self.assertRaises(ValueError):
            self.test_config_thermostability.aai_encoding(aai_indices=test_aai6, sort_by="RPD", output_folder=self.test_output_folder)
        with self.assertRaises(ValueError):
            self.test_config_enantioselectivity.aai_encoding(aai_indices=test_aai7, sort_by="RMSE", output_folder=self.test_output_folder)
#7.)
        test_aai8 = 1234 
        test_aai9 = True 
        with self.assertRaises(TypeError):
            self.test_config_absorption.aai_encoding(aai_indices=test_aai8, sort_by="MSE", output_folder=self.test_output_folder)
        with self.assertRaises(TypeError):
            self.test_config_localization.aai_encoding(aai_indices=test_aai9, sort_by="MAE", output_folder=self.test_output_folder)

    @unittest.skip(
        "Full AAI sweep encodes all ~566 indices and takes several minutes. "
        "Run manually with an extended timeout (e.g. pytest --timeout=600) when needed."
    )
    def test_aai_encoding_full_sweep(self):
        """
        Testing AAI encoding with all available AAI indices (no aai_indices filter).
        Skipped in CI: encodes all ~566 indices and takes several minutes.
        Run manually with an extended timeout when needed.
        """
        all_indices = aaindex1.record_codes()
        test_encoding_full = self.test_config_localization.aai_encoding(
            aai_indices=[], sort_by="MAE", output_folder=self.test_output_folder
        )

        self.assertIsInstance(test_encoding_full, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_full)}.")
        self.assertEqual(len(test_encoding_full), len(all_indices),
            f"Expected {len(all_indices)} rows in output dataframe, got {len(test_encoding_full)}.")
        self.assertEqual(set(list(test_encoding_full["Index"])), set(all_indices),
            f'Output index values don\'t match expected AAI record codes.')
        for cat in list(test_encoding_full["Category"]):
            self.assertIn(cat, self.index_categories,
                f"Category {cat} not found in list of categories:\n{self.index_categories}.")
        for col in test_encoding_full.columns:
            self.assertIn(col, self.expected_aai_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if col in ("Index", "Category"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_full[col].values)),
                    f"Column {col} expected to be string values, got {test_encoding_full[col].dtype}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_full[col].values)),
                    f"Column {col} expected to be np.float64 values, got {test_encoding_full[col].dtype}.")

    @unittest.skip("Descriptor encoding functionality can take a lot of time, skipping.")
    def test_descriptor_encoding(self):
        """ Testing Descriptor encoding functionality in Encoding module. """ 
#1.)
        test_desc1 = "amino_acid_composition"
        test_encoding_thermostability = self.test_config_thermostability.descriptor_encoding(descriptors=test_desc1, desc_combo=1, 
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_thermostability)}.")
        self.assertEqual(len(test_encoding_thermostability), 1, 
            f"Expected 1 row in output dataframe, got {len(test_encoding_thermostability)}.") 
        self.assertEqual(test_encoding_thermostability["Descriptor"].values[0], test_desc1, 
            f'Output index values don\'t match expected, got {test_encoding_thermostability["Descriptor"].values[0]}.')
        self.assertEqual(test_encoding_thermostability["Group"].values[0], "Composition", 
            f'Output group values don\'t match expected, got {test_encoding_thermostability["Group"].values[0]}.')
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_thermostability[col])}.")  
#2.)
        test_desc2 = "moran_autocorrelation"
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.descriptor_encoding(descriptors=test_desc2, desc_combo=1,
            sort_by="MAE", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_enantioselectivity)}.")
        self.assertEqual(len(test_encoding_enantioselectivity), 1, 
            f"Expected 1 row in output dataframe, got {len(test_encoding_enantioselectivity)}.") 
        self.assertEqual(test_encoding_enantioselectivity["Descriptor"].values[0], test_desc2, 
            f'Output index values don\'t match expected, got {test_encoding_enantioselectivity["Descriptor"].values[0]}.')
        self.assertEqual(test_encoding_enantioselectivity["Group"].values[0], "Autocorrelation", 
            f'Output group values don\'t match expected, got {test_encoding_enantioselectivity["Group"].values[0]}.')
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_enantioselectivity[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_enantioselectivity[col])}.")  
#3.)
        test_desc3 = ["ctd", "conjoint_triad", "dipeptide_composition"]
        test_encoding_absorption = self.test_config_absorption.descriptor_encoding(descriptors=test_desc3, desc_combo=1, 
            sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_absorption)}.")
        self.assertEqual(len(test_encoding_absorption), 3, 
            f"Expected 3 rows in output dataframe, got {len(test_encoding_absorption)}.") 
        self.assertEqual(set(list(test_encoding_absorption["Descriptor"])), set(test_desc3), 
            f'Output index values don\'t match expected, got {list(test_encoding_absorption["Descriptor"])}.')
        self.assertEqual(set(list(test_encoding_absorption["Group"])), {"Composition", "Conjoint Triad", "CTD"}, 
            f'Output group values don\'t match expected, got {list(test_encoding_absorption["Group"])}.')
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_absorption[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_absorption[col])}.") 
#4.) New composition descriptors from protpy>=1.4.1 - single-value outputs
        test_desc4a = "gravy"
        test_encoding_gravy = self.test_config_thermostability.descriptor_encoding(descriptors=test_desc4a, desc_combo=1,
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_gravy, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_gravy)}.")
        self.assertEqual(len(test_encoding_gravy), 1,
            f"Expected 1 row in output dataframe, got {len(test_encoding_gravy)}.")
        self.assertEqual(test_encoding_gravy["Descriptor"].values[0], test_desc4a,
            f'Output descriptor values don\'t match expected, got {test_encoding_gravy["Descriptor"].values[0]}.')
        self.assertEqual(test_encoding_gravy["Group"].values[0], "Composition",
            f'Output group values don\'t match expected, got {test_encoding_gravy["Group"].values[0]}.')
        for col in test_encoding_gravy.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_desc_encoding_output_columns}.")
#5.) New composition descriptors - multi-value outputs
        test_desc5a = "charge_distribution"
        test_encoding_charge = self.test_config_absorption.descriptor_encoding(descriptors=test_desc5a, desc_combo=1,
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_charge, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_charge)}.")
        self.assertEqual(len(test_encoding_charge), 1,
            f"Expected 1 row in output dataframe, got {len(test_encoding_charge)}.")
        self.assertEqual(test_encoding_charge["Descriptor"].values[0], test_desc5a,
            f'Output descriptor values don\'t match expected, got {test_encoding_charge["Descriptor"].values[0]}.')
        self.assertEqual(test_encoding_charge["Group"].values[0], "Composition",
            f'Output group values don\'t match expected, got {test_encoding_charge["Group"].values[0]}.')
        for col in test_encoding_charge.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_desc_encoding_output_columns}.")
#6.) Multiple new composition descriptors together
        test_desc6a = ["molecular_weight", "isoelectric_point", "shannon_entropy", "aliphatic_index", "boman_index"]
        test_encoding_multi_new = self.test_config_enantioselectivity.descriptor_encoding(descriptors=test_desc6a, desc_combo=1,
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_multi_new, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_multi_new)}.")
        self.assertEqual(len(test_encoding_multi_new), 5,
            f"Expected 5 rows in output dataframe, got {len(test_encoding_multi_new)}.")
        self.assertEqual(set(list(test_encoding_multi_new["Descriptor"])), set(test_desc6a),
            f'Output descriptor values don\'t match expected, got {list(test_encoding_multi_new["Descriptor"])}.')
        self.assertTrue(all(g == "Composition" for g in test_encoding_multi_new["Group"].values),
            f'All new descriptors should be in Composition group, got {list(test_encoding_multi_new["Group"])}.')
        for col in test_encoding_multi_new.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_desc_encoding_output_columns}.")
#7.) New descriptors mixed with old descriptors
        test_desc7a = ["hydrophobic_moment", "aggregation_propensity", "secondary_structure_propensity", "reduced_alphabet_composition"]
        test_encoding_mixed_new = self.test_config_localization.descriptor_encoding(descriptors=test_desc7a, desc_combo=1,
            sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_mixed_new, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_mixed_new)}.")
        self.assertEqual(len(test_encoding_mixed_new), 4,
            f"Expected 4 rows in output dataframe, got {len(test_encoding_mixed_new)}.")
        self.assertEqual(set(list(test_encoding_mixed_new["Descriptor"])), set(test_desc7a),
            f'Output descriptor values don\'t match expected, got {list(test_encoding_mixed_new["Descriptor"])}.')
        self.assertTrue(all(g == "Composition" for g in test_encoding_mixed_new["Group"].values),
            f'All new descriptors should be in Composition group, got {list(test_encoding_mixed_new["Group"])}.')
#8.)
        test_desc8 = [] #no descriptors passed into encoding function will calculate/import all descriptors for dataset
        test_encoding_thermostability = self.test_config_thermostability.descriptor_encoding(descriptors=test_desc8, desc_combo=1, 
            sort_by="RPD", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_thermostability)}.")
        self.assertEqual(len(test_encoding_thermostability), 33, 
            f"Expected 33 rows in output dataframe, got {len(test_encoding_thermostability)}.") 
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_encoding_output_columns}.")
            if (col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_thermostability[col])}.") 
        for group in list(test_encoding_thermostability["Group"]):
            self.assertIn(group, self.descriptor_groups, 
                f"Group {group} not found in list of groups:\n{self.descriptor_groups}.")
        for desc in list(test_encoding_thermostability["Descriptor"]):
            self.assertIn(desc, self.valid_descriptors, 
                f"Descriptor {desc} not found in list of available descriptors:\n{self.valid_descriptors}.") 
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "desc_results.csv")),
            "Output csv storing encoding results not found.")
#9.)
        invalid_test_desc5 = "invalid_descriptor_name" 
        invalid_test_desc6 = "blahblahblah" 
        with self.assertRaises(ValueError):
            self.test_config_thermostability.descriptor_encoding(descriptors=invalid_test_desc5, desc_combo=1, sort_by="MSE")
        with self.assertRaises(ValueError):
            self.test_config_enantioselectivity.descriptor_encoding(descriptors=invalid_test_desc6, desc_combo=1, sort_by="RMSE")
#10.)
        invalid_test_desc7 = 12345 
        invalid_test_desc8 = True 
        with self.assertRaises(TypeError):
            self.test_config_absorption.descriptor_encoding(descriptors=invalid_test_desc7, desc_combo=1, sort_by="MAE")
        with self.assertRaises(TypeError):
            self.test_config_localization.descriptor_encoding(descriptors=invalid_test_desc8, desc_combo=1, sort_by="RPD")

    @unittest.skip("AAI + Descriptor encoding functionality can take a lot of time, skipping.")
    def test_aai_descriptor_encoding(self):
        """ Testing AAI + Descriptor encoding functionality in Encoding module. """ 
#1.)    
        test_aai1 = "FAUJ880110"  #thermostability
        test_desc1 = "tripeptide_composition"
        test_encoding_thermostability = self.test_config_thermostability.aai_descriptor_encoding(aai_indices=test_aai1, descriptors=test_desc1, 
            desc_combo=1, sort_by="R2", output_folder=self.test_output_folder) 
        
        self.assertIsInstance(test_encoding_thermostability, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_thermostability)}.")
        self.assertEqual(len(test_encoding_thermostability), 1, 
            f"Expected 1 row in output dataframe, got {len(test_encoding_thermostability)}.") 
        self.assertEqual(test_encoding_thermostability["Index"].values[0], test_aai1, 
            f'Output index values don\'t match expected, got {test_encoding_thermostability["Index"].values[0]}.')
        self.assertEqual(test_encoding_thermostability["Category"].values[0], "geometry", 
            f'Output group values don\'t match expected, got {test_encoding_thermostability["Group"].values[0]}.')
        self.assertEqual(set(list(test_encoding_thermostability["Descriptor"].values)), {"tripeptide_composition"}, 
            f'Output descriptor column values don\'t match expected, got\n{test_encoding_thermostability["Descriptor"]}.')
        self.assertEqual(test_encoding_thermostability["Group"].values[0], "Composition", 
            f'Output group values don\'t match expected, got {test_encoding_thermostability["Group"].values[0]}.')
        for col in test_encoding_thermostability.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_thermostability[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_thermostability[col])}.") 
#2.)    
        test_aai2 = "BIGC670101, DAYM780201"  #enantioselectivity
        test_desc2 = ["ctd", "quasi_sequence_order", "sequence_order_coupling_number"]
        test_encoding_enantioselectivity = self.test_config_enantioselectivity.aai_descriptor_encoding(aai_indices=test_aai2, descriptors=test_desc2, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_enantioselectivity, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_enantioselectivity)}.")
        self.assertEqual(len(test_encoding_enantioselectivity), 6, 
            f"Expected 6 rows in output dataframe, got {len(test_encoding_enantioselectivity)}.") 
        self.assertEqual(set(list(test_encoding_enantioselectivity["Index"])), set(test_aai2.replace(' ', '').split(',')), 
            f'Output Index column does not match expected, got\n{test_encoding_enantioselectivity["Index"]}.')
        self.assertEqual(set(list(test_encoding_enantioselectivity["Category"].values)), {'composition', 'geometry'}, 
            f'Output category values don\'t match expected, got {test_encoding_enantioselectivity["Category"].values}.')
        self.assertEqual(set(list(test_encoding_enantioselectivity["Descriptor"])), set(test_desc2), 
            f'Output descriptor column values don\'t match expected, got\n{test_encoding_enantioselectivity["Descriptor"]}.')
        self.assertEqual(set(list(test_encoding_enantioselectivity["Group"].values)), {"Sequence Order", "CTD"}, 
            f'Output group values don\'t match expected, got {test_encoding_enantioselectivity["Group"].values}.')
        for col in test_encoding_enantioselectivity.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_enantioselectivity[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_enantioselectivity[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_enantioselectivity[col])}.") 
#3.)    
        test_aai3 = "GEOR030107, KARS160113, COWR900101"  #absorption 
        test_desc3 = ["amino_acid_composition", "ctd_distribution"]
        test_encoding_absorption = self.test_config_absorption.aai_descriptor_encoding(aai_indices=test_aai3, descriptors=test_desc3, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder) 
        
        self.assertIsInstance(test_encoding_absorption, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_absorption)}.")
        self.assertEqual(len(test_encoding_absorption), 6, 
            f"Expected 6 rows in output dataframe, got {len(test_encoding_absorption)}.") 
        self.assertEqual(set(list(test_encoding_absorption["Index"])), set(test_aai3.replace(' ', '').split(',')), 
            f'Output Index column does not match expected, got\n{test_encoding_absorption["Index"]}.')
        self.assertEqual(set(list(test_encoding_absorption["Category"].values)), {'hydrophobic', 'meta', 'sec_struct'},  
            f'Output category values don\'t match expected, got {test_encoding_absorption["Category"].values}.')
        self.assertEqual(set(list(test_encoding_absorption["Descriptor"])), set(test_desc3), 
            f'Output descriptor column values don\'t match expected, got\n{test_encoding_absorption["Descriptor"]}.')
        self.assertEqual(set(list(test_encoding_absorption["Group"].values)), {"Composition", "CTD"}, 
            f'Output group values don\'t match expected, got {test_encoding_absorption["Group"].values}.')
        for col in test_encoding_absorption.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_absorption[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_absorption[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_absorption[col])}.") 
#4.)
        test_aai4 = ["BEGF750103", "CIDH920103", "JOND920101"]  #localization 
        test_desc4 = ["dipeptide_composition", "ctd_transition"]
        test_encoding_localization = self.test_config_localization.aai_descriptor_encoding(aai_indices=test_aai4, descriptors=test_desc4, 
            desc_combo=1, sort_by="MSE", output_folder=self.test_output_folder) 

        self.assertIsInstance(test_encoding_localization, pd.DataFrame, 
            f"Expected output to be a dataframe, got {type(test_encoding_localization)}.")
        self.assertEqual(len(test_encoding_localization), 6, 
            f"Expected 6 rows in output dataframe, got {len(test_encoding_localization)}.") 
        self.assertEqual(set(list(test_encoding_localization["Index"])), set(test_aai4), 
            f'Output Index column does not match expected, got\n{test_encoding_localization["Index"]}.')
        self.assertEqual(set(list(test_encoding_localization["Category"].values)), {"sec_struct", "composition", "hydrophobic"}, 
            f'Output category values don\'t match expected, got {test_encoding_localization["Category"].values}.')
        self.assertEqual(set(list(test_encoding_localization["Descriptor"])), {"dipeptide_composition", "ctd_transition"}, 
            f'Output descriptor column values don\'t match expected, got\n{test_encoding_localization["Descriptor"]}.')
        self.assertEqual(set(list(test_encoding_localization["Group"].values)), {"Composition", "CTD"}, 
            f'Output group values don\'t match expected, got {test_encoding_localization["Group"].values}.')
        for col in test_encoding_localization.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
            if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
                self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_localization[col].values)),
                    f"Column {col} expected to be of type string got {type(test_encoding_localization[col])}.")
            else:
                self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_localization[col].values)),
                    f"Column {col} expected to be of type np.float64 got {type(test_encoding_localization[col])}.") 
        self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
            "Output dir storing encoding results not found.")
        self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
            "Output csv storing encoding results not found.")
#5.) New composition descriptors (protpy>=1.4.1) combined with AAI encoding
        test_aai5a = "FAUJ880110"
        test_desc5a = "gravy"
        test_encoding_gravy = self.test_config_thermostability.aai_descriptor_encoding(aai_indices=test_aai5a, descriptors=test_desc5a,
            desc_combo=1, sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_gravy, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_gravy)}.")
        self.assertEqual(len(test_encoding_gravy), 1,
            f"Expected 1 row in output dataframe, got {len(test_encoding_gravy)}.")
        self.assertEqual(test_encoding_gravy["Index"].values[0], test_aai5a,
            f'Output index values don\'t match expected, got {test_encoding_gravy["Index"].values[0]}.')
        self.assertEqual(test_encoding_gravy["Descriptor"].values[0], test_desc5a,
            f'Output descriptor values don\'t match expected, got {test_encoding_gravy["Descriptor"].values[0]}.')
        self.assertEqual(test_encoding_gravy["Group"].values[0], "Composition",
            f'Output group values don\'t match expected, got {test_encoding_gravy["Group"].values[0]}.')
        for col in test_encoding_gravy.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
#6.) Multiple new descriptors combined with multiple AAI indices
        test_aai6a = "FAUJ880110, BIGC670101"
        test_desc6a = ["molecular_weight", "shannon_entropy", "hydrophobic_polar_charged_composition"]
        test_encoding_multi_new = self.test_config_enantioselectivity.aai_descriptor_encoding(aai_indices=test_aai6a, descriptors=test_desc6a,
            desc_combo=1, sort_by="R2", output_folder=self.test_output_folder)

        self.assertIsInstance(test_encoding_multi_new, pd.DataFrame,
            f"Expected output to be a dataframe, got {type(test_encoding_multi_new)}.")
        self.assertEqual(len(test_encoding_multi_new), 6,
            f"Expected 6 rows in output dataframe, got {len(test_encoding_multi_new)}.")
        self.assertEqual(set(list(test_encoding_multi_new["Index"])), set(test_aai6a.replace(' ', '').split(',')),
            f'Output Index column does not match expected, got\n{test_encoding_multi_new["Index"]}.')
        self.assertEqual(set(list(test_encoding_multi_new["Descriptor"])), set(test_desc6a),
            f'Output descriptor column values don\'t match expected, got\n{test_encoding_multi_new["Descriptor"]}.')
        self.assertTrue(all(g == "Composition" for g in test_encoding_multi_new["Group"].values),
            f'All new descriptors should be in Composition group, got {list(test_encoding_multi_new["Group"])}.')
        for col in test_encoding_multi_new.columns:
            self.assertIn(col, self.expected_aai_desc_encoding_output_columns,
                f"Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.")
#7.)    
        test_aai7 = ["invalid_aai_index"]
        test_aai8 = ""
        test_desc7 = ["invalid_descriptor_name"]
        with self.assertRaises(ValueError):
            self.test_config_thermostability.aai_descriptor_encoding(aai_indices=test_aai7, descriptors=test_desc7, desc_combo=1, sort_by="MSE")
        with self.assertRaises(ValueError):
            self.test_config_enantioselectivity.aai_descriptor_encoding(aai_indices=test_aai8, descriptors=test_desc7, desc_combo=1, sort_by="MSE")
#8.)
        test_aai9 = 12345
        test_desc8 = 1000
        test_desc9 = False
        with self.assertRaises(TypeError):
            self.test_config_absorption.aai_descriptor_encoding(aai_indices=test_aai9, descriptors=test_desc8, desc_combo=1, sort_by="MAE")
        with self.assertRaises(TypeError):
            self.test_config_localization.aai_descriptor_encoding(aai_indices=test_aai9, descriptors=test_desc9, desc_combo=1, sort_by="MAE")

#9.)    ** Below inputs result in all AAI Indices being encoded with all descriptors, commenting out due to time and resource constraints **
        # test_aai8 = []
        # test_desc7 = []
        # test_encoding7 = self.test_encoding_thermostability.aai_descriptor_encoding(aai_indices=test_aai8, descriptors=test_desc7, 
        #     desc_combo=1, sort_by="MAE", output_folder=self.test_output_folder)

        # self.assertIsInstance(test_encoding_thermostability, pd.DataFrame,
        #     f'Expected output to be a dataframe, got {type(test_encoding_thermostability)}.')
        # self.assertEqual(len(test_encoding_thermostability), 6,
        #     f'Expected 6 rows in output dataframe, got {len(test_encoding_thermostability)}.')
        # self.assertEqual(set(list(test_encoding_thermostability["Index"])), set(test_aai3.replace(' ', '').split(',')), 
        #     f'Expected index column to be type string, got {test_encoding_thermostability["Index"].dtype}.')
        # self.assertEqual(test_encoding_thermostability["Category"].values[0], ["Composition", "CTD"], 
        #     f'Output category values don\'t match expected, got {test_encoding_thermostability["Category"].values[0]}.')   #**
        # self.assertEqual(set(list(test_encoding_thermostability["Descriptor"])), set(test_desc3), 
        #     "Output index values don't match expected.") 
        # self.assertEqual(test_encoding_thermostability["Group"].values[0], ["Composition", "CTD"], 
        #     f'Output group values don\'t match expected, got {test_encoding_thermostability["Group"].values[0]}.')
        # for col in test_encoding_thermostability.columns:
        #     self.assertIn(col, self.expected_aai_desc_encoding_output_columns, 
        #         f'Col {col} not found in list of expected columns:\n{self.expected_aai_desc_encoding_output_columns}.')
        #     if (col == "Index" or col == "Category" or col == "Descriptor" or col == "Group"):
        #         self.assertTrue(all(isinstance(row, str) for row in list(test_encoding_thermostability[col].values)),
        #             f'Column {col} expected to be of type string got {type(test_encoding_thermostability[col])}.')
        #     else:
        #         self.assertTrue(all(isinstance(row, np.float64) for row in list(test_encoding_thermostability[col].values)),
        #             f'Column {col} expected to be of type np.float64 got {type(test_encoding_thermostability[col])}.')
        # for group in list(test_encoding_thermostability["Group"]):
        #     self.assertIn(group, self.descriptor_groups, 
        #         f'Group {group} not found in list of groups:\n{self.descriptor_groups}.')
        # for desc in list(test_encoding_thermostability["Descriptor"]):
        #     self.assertIn(desc, self.valid_descriptors, 
        #         f'Descriptor {desc} not found in list of available descriptors:\n{self.valid_descriptors}.')
        # self.assertTrue(os.path.isdir(self.test_output_folder + "_" + _globals.CURRENT_DATETIME), 
        #     "Output dir storing encoding results not found.")
        # self.assertTrue(os.path.isfile(os.path.join(self.test_output_folder + "_" + _globals.CURRENT_DATETIME, "aai_desc_results.csv")),
        #     "Output csv storing encoding results not found.")

    def test_validate_inputs(self):
        """ Testing validate_inputs method for normalisation, deduplication and error handling. """
       
        valid_aai = aaindex1.record_codes()
#1.)    None, [] and '' each return the full sorted list of valid values
        for empty_input in [None, [], ""]:
            with self.subTest(empty_input=empty_input):
                result = self.test_config_thermostability.validate_inputs(empty_input, valid_aai, "AAI")
                self.assertEqual(sorted(result), sorted(list(valid_aai)),
                    f"Expected all valid values for empty input {empty_input!r}, got {len(result)} entries.")
#2.)    single valid string returns a single-element list
        result = self.test_config_thermostability.validate_inputs("FAUJ880110", valid_aai, "AAI")
        self.assertEqual(result, ["FAUJ880110"],
            f"Expected ['FAUJ880110'], got {result}.")
#3.)    comma-separated string is split and parsed into individual values
        result = self.test_config_thermostability.validate_inputs("FAUJ880110, GEIM800111", valid_aai, "AAI")
        self.assertEqual(set(result), {"FAUJ880110", "GEIM800111"},
            f"Expected parsed set from comma string, got {set(result)}.")
#4.)    list with duplicates is de-duplicated and sorted
        result = self.test_config_thermostability.validate_inputs(
            ["GEIM800111", "FAUJ880110", "FAUJ880110"], valid_aai, "AAI")
        self.assertEqual(result, sorted({"FAUJ880110", "GEIM800111"}),
            f"Expected de-duplicated sorted list, got {result}.")
#5.)    invalid string value raises ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.validate_inputs("INVALID_INDEX_XYZ", valid_aai, "AAI")
        with self.assertRaises(ValueError):
            self.test_config_thermostability.validate_inputs(["FAUJ880110", "INVALID_INDEX_XYZ"], valid_aai, "AAI")
#6.)    non-string/non-list types raise TypeError
        with self.assertRaises(TypeError):
            self.test_config_thermostability.validate_inputs(12345, valid_aai, "AAI")
        with self.assertRaises(TypeError):
            self.test_config_thermostability.validate_inputs(True, valid_aai, "AAI")
#7.)    list containing non-string elements raises TypeError
        with self.assertRaises(TypeError):
            self.test_config_thermostability.validate_inputs([123, 456], valid_aai, "AAI")

    # @unittest.skip("")
    def test_validate_desc_combo(self):
        """ Testing validate_desc_combo for valid and invalid combination values. """
#1.)    valid values 1, 2 and 3 should not raise any exception
        for valid_combo in [1, 2, 3]:
            with self.subTest(valid_combo=valid_combo):
                try:
                    self.test_config_thermostability.validate_desc_combo(valid_combo)
                except ValueError:
                    self.fail(f"validate_desc_combo raised ValueError unexpectedly for valid combo {valid_combo}.")
#2.)    values outside {1, 2, 3} raise ValueError
        for invalid_combo in [0, 4, -1, 5, 100]:
            with self.subTest(invalid_combo=invalid_combo):
                with self.assertRaises(ValueError,
                        msg=f"Expected ValueError for invalid desc_combo {invalid_combo}."):
                    self.test_config_thermostability.validate_desc_combo(invalid_combo)

    # @unittest.skip("")
    def test_build_features(self):
        """ Testing build_features method for AAI, descriptor and combined feature types. """

        desc_instance = Descriptors(config_file=self.all_config_files[0])
        index = "FAUJ880110"
        descriptor = "amino_acid_composition"
#1.)    feature_type="aai" returns a DataFrame with at least one column
        aai_features = self.test_config_thermostability.build_features(feature_type="aai", index=index)
        self.assertIsInstance(aai_features, pd.DataFrame,
            f"Expected DataFrame for AAI features, got {type(aai_features)}.")
        self.assertGreater(len(aai_features.columns), 0,
            "Expected at least one feature column for AAI encoding.")
#2.)    feature_type="descriptor" returns a DataFrame with at least one column
        desc_features = self.test_config_thermostability.build_features(
            feature_type="descriptor", descriptor_entry=descriptor, desc_instance=desc_instance)
        self.assertIsInstance(desc_features, pd.DataFrame,
            f"Expected DataFrame for descriptor features, got {type(desc_features)}.")
        self.assertGreater(len(desc_features.columns), 0,
            "Expected at least one feature column for descriptor encoding.")
#3.)    feature_type="aai_descriptor" returns a DataFrame with the combined column count
        combined_features = self.test_config_thermostability.build_features(
            feature_type="aai_descriptor", index=index, descriptor_entry=descriptor, desc_instance=desc_instance)
        self.assertIsInstance(combined_features, pd.DataFrame,
            f"Expected DataFrame for combined features, got {type(combined_features)}.")
        self.assertEqual(
            len(combined_features.columns), len(aai_features.columns) + len(desc_features.columns),
            f"Expected {len(aai_features.columns) + len(desc_features.columns)} combined columns, got {len(combined_features.columns)}.")
#4.)    unknown feature_type raises ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.build_features(feature_type="unknown_type", index=index)
#5.)    feature_type="aai" with index=None raises ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.build_features(feature_type="aai", index=None)
#6.)    feature_type="descriptor" with desc_instance=None raises ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.build_features(
                feature_type="descriptor", descriptor_entry=descriptor, desc_instance=None)

    # @unittest.skip("")
    def test_aai_encoding_params(self):
        """ Testing parameter-level behaviour of aai_encoding: sample_mode, max_models, n_jobs, random_state and sort_by fallback. """
        
        three_indices = ["FAUJ880110", "GEIM800111", "JOND750102"]
#1.)    sample_mode=True limits output to at most 10 rows
        result_sample = self.test_config_thermostability.aai_encoding(
            aai_indices=None, sample_mode=True, sort_by="R2", output_folder=self.test_output_folder)
        self.assertIsInstance(result_sample, pd.DataFrame,
            f"Expected DataFrame for sample_mode, got {type(result_sample)}.")
        self.assertLessEqual(len(result_sample), 10,
            f"Expected at most 10 rows for sample_mode=True, got {len(result_sample)}.")
#2.)    max_models=3 limits output to exactly 3 rows
        result_limited = self.test_config_thermostability.aai_encoding(
            aai_indices=None, max_models=3, sort_by="R2", output_folder=self.test_output_folder)
        self.assertEqual(len(result_limited), 3,
            f"Expected exactly 3 rows for max_models=3, got {len(result_limited)}.")
#3.)    max_models=0 raises ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.aai_encoding(
                aai_indices=three_indices, max_models=0, sort_by="R2", output_folder=self.test_output_folder)
#4.)    invalid sort_by falls back to R2 and still returns a valid DataFrame
        result_bad_sort = self.test_config_thermostability.aai_encoding(
            aai_indices=three_indices, sort_by="InvalidMetric", output_folder=self.test_output_folder)
        self.assertIsInstance(result_bad_sort, pd.DataFrame,
            "Expected DataFrame even with invalid sort_by (should silently fall back to R2).")
        self.assertEqual(len(result_bad_sort), 3,
            f"Expected 3 rows with invalid sort_by fallback, got {len(result_bad_sort)}.")
#5.)    n_jobs=2 produces the same index set and row count as n_jobs=1
        result_serial = self.test_config_thermostability.aai_encoding(
            aai_indices=three_indices, sort_by="R2", n_jobs=1, output_folder=self.test_output_folder)
        result_parallel = self.test_config_thermostability.aai_encoding(
            aai_indices=three_indices, sort_by="R2", n_jobs=2, output_folder=self.test_output_folder)
        self.assertEqual(len(result_serial), len(result_parallel),
            f"Expected same row count for n_jobs=1 and n_jobs=2, got {len(result_serial)} vs {len(result_parallel)}.")
        self.assertEqual(set(result_serial["Index"]), set(result_parallel["Index"]),
            "Expected same set of indices for n_jobs=1 and n_jobs=2.")
#6.)    same random_state produces identical results across two calls
        result_seed_a = self.test_config_thermostability.aai_encoding(
            aai_indices=three_indices, sort_by="R2", random_state=42, output_folder=self.test_output_folder)
        result_seed_b = self.test_config_thermostability.aai_encoding(
            aai_indices=three_indices, sort_by="R2", random_state=42, output_folder=self.test_output_folder)
        pd.testing.assert_frame_equal(
            result_seed_a.reset_index(drop=True), result_seed_b.reset_index(drop=True),
            check_dtype=False)

    # @unittest.skip("")
    def test_aai_encoding_output_files(self):
        """ Testing that aai_encoding creates the expected output directory and results CSV file. """
        
        test_aai = ["FAUJ880110", "GEIM800111"]
        self.test_config_thermostability.aai_encoding(
            aai_indices=test_aai, sort_by="R2", output_folder=self.test_output_folder)
        #check expected output directory and results CSV were both created
        expected_output_dir = self.test_output_folder + "_" + _globals.CURRENT_DATETIME
        self.assertTrue(os.path.isdir(expected_output_dir),
            f"Expected output directory to be created at {expected_output_dir}.")
        self.assertTrue(
            os.path.isfile(os.path.join(expected_output_dir, "aaindex_results.csv")),
            f"Expected aaindex_results.csv to exist in {expected_output_dir}.")

    # @unittest.skip("")
    def test_aai_encoding_comma_string_input(self):
        """ Testing aai_encoding with a comma-separated string passed as aai_indices. """
#1.)    comma-separated string is parsed and each index produces a separate output row
        test_aai_str = "FAUJ880110, GEIM800111, JOND750102"
        result = self.test_config_thermostability.aai_encoding(
            aai_indices=test_aai_str, sort_by="R2", output_folder=self.test_output_folder)
        expected_indices = {"FAUJ880110", "GEIM800111", "JOND750102"}
        self.assertIsInstance(result, pd.DataFrame,
            f"Expected DataFrame for comma-separated input, got {type(result)}.")
        self.assertEqual(len(result), 3,
            f"Expected 3 rows from comma-separated string input, got {len(result)}.")
        self.assertEqual(set(list(result["Index"])), expected_indices,
            f"Expected indices {expected_indices}, got {set(list(result['Index']))}.")
#2.)    single index as a plain string (no comma) also works correctly
        test_aai_single = "FAUJ880110"
        result_single = self.test_config_enantioselectivity.aai_encoding(
            aai_indices=test_aai_single, sort_by="R2", output_folder=self.test_output_folder)
        self.assertEqual(len(result_single), 1,
            f"Expected 1 row for single string index, got {len(result_single)}.")
        self.assertEqual(result_single["Index"].values[0], test_aai_single,
            f"Expected index {test_aai_single}, got {result_single['Index'].values[0]}.")

    # @unittest.skip("")
    def test_descriptor_encoding_combos(self):
        """ Testing descriptor encoding with desc_combo=2 and desc_combo=3, and invalid combo values. """
#1.)    desc_combo=2 with 2 descriptors produces one row with a '+' joined label
        test_descs2 = ["amino_acid_composition", "dipeptide_composition"]
        result_combo2 = self.test_config_thermostability.descriptor_encoding(
            descriptors=test_descs2, desc_combo=2, sort_by="R2", output_folder=self.test_output_folder)
        self.assertIsInstance(result_combo2, pd.DataFrame,
            f"Expected DataFrame for desc_combo=2, got {type(result_combo2)}.")
        self.assertEqual(len(result_combo2), 1,
            f"Expected 1 row for 2 descriptors with desc_combo=2, got {len(result_combo2)}.")
        descriptor_label = result_combo2["Descriptor"].values[0]
        self.assertIn("+", descriptor_label,
            f"Expected '+' separator in combined descriptor label, got '{descriptor_label}'.")
        self.assertEqual(set(descriptor_label.split("+")), set(test_descs2),
            f"Expected descriptor parts {set(test_descs2)}, got {set(descriptor_label.split('+'))}.")
#2.)    desc_combo=3 with 3 descriptors produces one row with two '+' separators
        test_descs3 = ["amino_acid_composition", "dipeptide_composition", "conjoint_triad"]
        result_combo3 = self.test_config_thermostability.descriptor_encoding(
            descriptors=test_descs3, desc_combo=3, sort_by="R2", output_folder=self.test_output_folder)
        self.assertIsInstance(result_combo3, pd.DataFrame,
            f"Expected DataFrame for desc_combo=3, got {type(result_combo3)}.")
        self.assertEqual(len(result_combo3), 1,
            f"Expected 1 row for 3 descriptors with desc_combo=3, got {len(result_combo3)}.")
        descriptor_label3 = result_combo3["Descriptor"].values[0]
        self.assertEqual(descriptor_label3.count("+"), 2,
            f"Expected 2 '+' separators in triple descriptor label, got '{descriptor_label3}'.")
#3.)    invalid desc_combo values raise ValueError
        with self.assertRaises(ValueError):
            self.test_config_thermostability.descriptor_encoding(
                descriptors=["amino_acid_composition"], desc_combo=4, sort_by="R2",
                output_folder=self.test_output_folder)
        with self.assertRaises(ValueError):
            self.test_config_thermostability.descriptor_encoding(
                descriptors=["amino_acid_composition"], desc_combo=0, sort_by="R2",
                output_folder=self.test_output_folder)

    # @unittest.skip("")
    def test_output_sorting(self):
        """ Testing that encoding results are sorted correctly by each supported metric. """
       
        test_aai = ["FAUJ880110", "GEIM800111", "JOND750102", "MAXF760102"]
#1.)    ascending sort: RMSE, MSE, MAE and RPD should each produce an ascending-ordered column
        for metric in ["RMSE", "MSE", "MAE", "RPD"]:
            with self.subTest(metric=metric, order="ascending"):
                result = self.test_config_thermostability.aai_encoding(
                    aai_indices=test_aai, sort_by=metric, output_folder=self.test_output_folder)
                values = list(result[metric].values)
                self.assertEqual(values, sorted(values),
                    f"Expected ascending sort for {metric}, got {values}.")
#2.)    descending sort: R2 and Explained Variance should each produce a descending-ordered column
        for metric in ["R2", "Explained Variance"]:
            with self.subTest(metric=metric, order="descending"):
                result = self.test_config_thermostability.aai_encoding(
                    aai_indices=test_aai, sort_by=metric, output_folder=self.test_output_folder)
                values = list(result[metric].values)
                self.assertEqual(values, sorted(values, reverse=True),
                    f"Expected descending sort for {metric}, got {values}.")

    # @unittest.skip("")
    def test_metric_value_ranges(self):
        """ Testing that all numeric metric columns contain valid float64 values and non-negative error metrics. """
        
        test_aai = ["FAUJ880110", "GEIM800111", "JOND750102"]
        result = self.test_config_thermostability.aai_encoding(
            aai_indices=test_aai, sort_by="R2", output_folder=self.test_output_folder)
#1.)    RMSE, MSE, MAE and RPD must all be >= 0
        for metric in ["RMSE", "MSE", "MAE", "RPD"]:
            with self.subTest(metric=metric):
                for val in result[metric].values:
                    self.assertGreaterEqual(float(val), 0.0,
                        f"Expected {metric} >= 0, got {val}.")
#2.)    all metric columns must contain np.float64 values
        self._assert_metric_columns_are_float(result)

    def test_encoding_result_dataclass(self):
        """ Testing EncodingResult dataclass construction and attributes. """
        from pySAR.encoding import EncodingResult

        test_aai = ["FAUJ880110", "GEIM800111", "JOND750102"]
        df = self.test_config_thermostability.aai_encoding(
            aai_indices=test_aai, sort_by="R2", output_folder=self.test_output_folder)
#1.)    from_dataframe returns an EncodingResult instance
        er = EncodingResult.from_dataframe(df, elapsed_time=1.5)
        self.assertIsInstance(er, EncodingResult,
            f"from_dataframe() should return an EncodingResult, got {type(er)}.")
#2.)    metrics attribute is the original DataFrame
        self.assertIsInstance(er.metrics, pd.DataFrame,
            "EncodingResult.metrics should be a DataFrame.")
        self.assertEqual(len(er.metrics), len(df),
            "EncodingResult.metrics should have same row count as input DataFrame.")
#3.)    best_index is a non-empty string corresponding to the first row
        self.assertIsInstance(er.best_index, str,
            "EncodingResult.best_index should be a string.")
        self.assertNotEqual(er.best_index, '',
            "EncodingResult.best_index should not be empty.")
        self.assertEqual(er.best_index, str(df.iloc[0, 0]),
            f"best_index should match first row first column value: {df.iloc[0, 0]}.")
#4.)    best_r2 is a float
        self.assertIsInstance(er.best_r2, float,
            "EncodingResult.best_r2 should be a float.")
#5.)    elapsed_time is preserved
        self.assertAlmostEqual(er.elapsed_time, 1.5, places=5,
            msg="EncodingResult.elapsed_time should match the value passed to from_dataframe().")
#6.)    best_model_path defaults to None
        self.assertIsNone(er.best_model_path,
            "EncodingResult.best_model_path should be None when not provided.")
#7.)    from_dataframe with an empty DataFrame returns sensible defaults
        empty_df = pd.DataFrame(columns=["Index", "Category", "R2"])
        er_empty = EncodingResult.from_dataframe(empty_df)
        self.assertEqual(er_empty.best_index, '',
            "best_index should be empty string for empty DataFrame.")
        self.assertTrue(np.isnan(er_empty.best_r2),
            "best_r2 should be NaN for empty DataFrame.")

    def test_get_aai_features_concurrent_cache(self):
        """ _get_aai_features must return identical results for all concurrent callers.

        Five threads request the same index simultaneously; the TOCTOU fix ensures
        only one thread computes the features while the others wait on the shared Future.
        All threads must receive equal DataFrames.
        """
        import threading
        import pandas as pd

        index = "ANDN920101"
        results: dict = {}
        errors: list = []

        def worker(i: int) -> None:
            try:
                results[i] = self.test_config_thermostability._get_aai_features(index)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0,
            f"Worker threads raised exceptions: {errors}")
        self.assertEqual(len(results), 5,
            "Expected results from all 5 threads.")
        ref = results[0]
        for i in range(1, 5):
            pd.testing.assert_frame_equal(results[i], ref,
                check_like=False,
                obj=f"results[{i}] vs results[0]")

    def test_export_best_model(self):
        """ Testing that export_best_model=True saves a best_model.pkl inside the output folder. """
        import tempfile
        import pickle
        from pySAR.model import Model

        test_aai = ["FAUJ880110", "GEIM800111"]
        tmp_dir = tempfile.mkdtemp(prefix="test_export_best_model_")
        try:
            df = self.test_config_thermostability.aai_encoding(
                aai_indices=test_aai,
                sort_by="R2",
                output_folder=tmp_dir,
                export_best_model=True,
            )
#1.)    return value is still a DataFrame
            self.assertIsInstance(df, pd.DataFrame,
                f"aai_encoding() with export_best_model=True should still return DataFrame, got {type(df)}.")
#2.)    best_model.pkl was created somewhere under tmp_dir
            pkl_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for f in files:
                    if f == 'best_model.pkl':
                        pkl_files.append(os.path.join(root, f))
            self.assertGreaterEqual(len(pkl_files), 1,
                "Expected at least one best_model.pkl to be written under the output folder.")
#3.)    the pickle is a valid dict with 'model' and 'scaler' keys
            with open(pkl_files[0], 'rb') as fh:
                payload = pickle.load(fh)
            self.assertIsInstance(payload, dict,
                "best_model.pkl should contain a dict.")
            self.assertIn('model', payload,
                "best_model.pkl dict should contain a 'model' key.")
            self.assertIn('scaler', payload,
                "best_model.pkl dict should contain a 'scaler' key.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """
        #removing any of the temp files created such as the results files, if
        #you want to verify the results files are actually being created then
        #comment out the below code block
        for _ts_dir in glob.glob(self.test_output_folder + "_*"):
            shutil.rmtree(_ts_dir, ignore_errors=True)
                
if __name__ == '__main__':
    #run all unit tests
    unittest.main(verbosity=2)
