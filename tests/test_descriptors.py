################################################################################
#################            Descriptors Module Tests          #################
################################################################################

import pandas as pd
import numpy as np
import os
import re
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import pySAR.descriptors as descr

# @unittest.skip("")
class DescriptorTests(unittest.TestCase):
    """
    Test suite for testing Descriptors module and functionality in pySAR package. 

    Test Cases
    ==========
    test_descriptor:
        testing correct overall Descriptor class and module functionality.
    test_descriptor_groups:
        testing correct list of descriptor groups.
    test_all_descriptors_list:
        testing correct list of valid descriptors and combinations of descriptors.
    test_valid_descriptors:
        testing correct list of valid descriptors.
    test_descriptor_import:
        testing correct import functionality for pre-calculated descriptor csv.
    test_amino_acid_composition:
        testing correct amino acid composition descriptor functionality.
    test_dipeptide_composition:
        testing correct dipeptide composition descriptor functionality.
    test_tripeptide_composition:
        testing correct tripeptide composition descriptor functionality.
    test_moreaubroto_autocorrelation:
        testing correct Moreaubroto autocorrellation descriptor functionality.
    test_moran_autocorrelation:
        testing correct Moran autocorrellation descriptor functionality.
    test_geary_autocorrelation:
        testing correct Geary autocorrellation descriptor functionality.
    test_ctd:
        testing correct CTD descriptor functionality.
    test_conjoint_triad:
        testing correct Conjoint Triad descriptor functionality.
    test_sequence_order_coupling_number:
        testing correct Sequence Order Coupling Number descriptor functionality.
    test_quasi_sequence_order:
        testing correct Quasi Sequence Order descriptor functionality.
    test_pseudo_amino_acid_composition:
        testing correct Pseudo Amino Acid Composition descriptor functionality.
    test_amphiphilic_pseudo_amino_acid_composition:
        testing correct Amphiphilic Pseudo Amino Acid Composition descriptor functionality.
    test_gravy:
        testing correct GRAVY (Grand Average of Hydropathicity) descriptor functionality.
    test_aromaticity:
        testing correct aromaticity descriptor functionality.
    test_instability_index:
        testing correct instability index descriptor functionality.
    test_isoelectric_point:
        testing correct isoelectric point descriptor functionality.
    test_molecular_weight:
        testing correct molecular weight descriptor functionality.
    test_charge_distribution:
        testing correct charge distribution descriptor functionality.
    test_hydrophobic_polar_charged_composition:
        testing correct hydrophobic/polar/charged composition descriptor functionality.
    test_secondary_structure_propensity:
        testing correct secondary structure propensity descriptor functionality.
    test_kmer_composition:
        testing correct k-mer composition descriptor functionality.
    test_reduced_alphabet_composition:
        testing correct reduced alphabet composition descriptor functionality.
    test_motif_composition:
        testing correct motif composition descriptor functionality.
    test_amino_acid_pair_composition:
        testing correct amino acid pair composition descriptor functionality.
    test_aliphatic_index:
        testing correct aliphatic index descriptor functionality.
    test_extinction_coefficient:
        testing correct extinction coefficient descriptor functionality.
    test_boman_index:
        testing correct Boman index descriptor functionality.
    test_aggregation_propensity:
        testing correct aggregation propensity descriptor functionality.
    test_hydrophobic_moment:
        testing correct hydrophobic moment descriptor functionality.
    test_shannon_entropy:
        testing correct Shannon entropy descriptor functionality.
    test_get_all_descriptors:
        testing correct functionality for calculating all descriptors for a dataset of sequences.
    test_n_jobs_parallel:
        testing parallel descriptor computation via the n_jobs parameter.
    test_get_descriptor_encoding:
        testing correct descriptor encoding functionality.
    """
    def setUp(self):
        """ Import the 4 config files for each of the 4 datasets used for testing the descriptor methods. """        
        #array of config files for each test dataset
        config_path = os.path.join('tests', 'test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]
        
        #path to pre-calculated protein descriptors for thermostability dataset
        self.test_descriptors_path = os.path.join('tests', 'test_data', 'test_thermostability_descriptors.csv')

        #array of the total number of protein seqs per dataset
        self.num_seqs = [261, 152, 81, 254]

        #list of canonical amino acids
        self.amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", 
            "Q", "R", "S", "T", "V", "W", "Y"]

    # @unittest.skip("")
    def test_descriptor(self):
        """ Test descriptor initialisation process. Verify the initial input parameters and descriptor attributes are correct. """
#1.)
        desc = descr.Descriptors(config_file=self.all_config_files[0], descriptors_csv=self.test_descriptors_path) #pre-calculated descriptors from thermostability dataset

        #verify num_seqs descriptors attribute is correct
        self.assertEqual(desc.num_seqs, self.num_seqs[0],
            f'Expected {self.num_seqs[0]} number of sequences, got {desc.num_seqs}.')

        print("desc.amino_acid_compositio")
        print(desc.amino_acid_composition)
        print(desc.amino_acid_composition.shape)
        #verify that all input sequences dont have any gaps/missing amino acids
        for seq in desc.protein_seqs:
            self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#2.)
        self.assertEqual(desc.amino_acid_composition.shape, (self.num_seqs[0], 20), 
            f'Attribute shape should be [{self.num_seqs[0]}, {20}], got {desc.amino_acid_composition.shape}.')
        self.assertEqual(desc.dipeptide_composition.shape, (self.num_seqs[0], 400), 
            f'Attribute shape should be [{self.num_seqs[0]}, {400}], got {desc.dipeptide_composition.shape}.')
        self.assertEqual(desc.tripeptide_composition.shape, (self.num_seqs[0], 8000), 
            f'Attribute shape should be [{self.num_seqs[0]}, {8000}], got {desc.tripeptide_composition.shape}.')
        self.assertEqual(desc.moreaubroto_autocorrelation.shape, (self.num_seqs[0], 240), 
            f'Attribute shape should be [{self.num_seqs[0]}, {240}], got {desc.moreaubroto_autocorrelation.shape}.')
        self.assertEqual(desc.moran_autocorrelation.shape, (self.num_seqs[0], 240), 
            f'Attribute shape should be [{self.num_seqs[0]}, {240}], got {desc.moran_autocorrelation.shape}.')
        self.assertEqual(desc.geary_autocorrelation.shape, (self.num_seqs[0], 240), 
            f'Attribute shape should be [{self.num_seqs[0]}, {240}], got {desc.geary_autocorrelation.shape}.')
        self.assertEqual(desc.ctd.shape, (self.num_seqs[0], 21), 
            f'Attribute shape should be [{self.num_seqs[0]}, {21}], got {desc.ctd.shape}.')
        self.assertEqual(desc.ctd_composition.shape, (self.num_seqs[0], 3), 
            f'Attribute shape should be [{self.num_seqs[0]}, {3}], got {desc.ctd_composition.shape}.')
        self.assertEqual(desc.ctd_transition.shape, (self.num_seqs[0], 3), 
            f'Attribute shape should be [{self.num_seqs[0]}, {3}], got {desc.ctd_transition.shape}.')
        self.assertEqual(desc.ctd_distribution.shape, (self.num_seqs[0], 15), 
            f'Attribute shape should be [{self.num_seqs[0]}, {15}], got {desc.ctd_distribution.shape}.')
        self.assertEqual(desc.conjoint_triad.shape, (self.num_seqs[0], 343), 
            f'Attribute shape should be [{self.num_seqs[0]}, {343}], got {desc.conjoint_triad.shape}.')
        self.assertEqual(desc.sequence_order_coupling_number.shape, (self.num_seqs[0], 30), 
            f'Attribute shape should be [{self.num_seqs[0]}, {30}], got {desc.sequence_order_coupling_number.shape}.')
        self.assertEqual(desc.quasi_sequence_order.shape, (self.num_seqs[0], 50), 
            f'Attribute shape should be [{self.num_seqs[0]}, {50}], got {desc.quasi_sequence_order.shape}.')
        self.assertEqual(desc.pseudo_amino_acid_composition.shape, (self.num_seqs[0], 50), 
            f'Attribute shape should be [{self.num_seqs[0]}, {50}], got {desc.pseudo_amino_acid_composition.shape}.')
        self.assertEqual(desc.amphiphilic_pseudo_amino_acid_composition.shape, (self.num_seqs[0], 80), 
            f'Attribute shape should be [{self.num_seqs[0]}, {80}], got {desc.amphiphilic_pseudo_amino_acid_composition.shape}.')
        self.assertEqual(desc.all_descriptors.shape, (self.num_seqs[0], 10552),
            f'Attribute shape should be [{self.num_seqs[0]}, {10552}], got {desc.all_descriptors.shape}.')
        # All descriptors are present in the updated pre-calculated CSV
        self.assertEqual(desc.gravy.shape, (self.num_seqs[0], 1), f'Got {desc.gravy.shape}.')
        self.assertEqual(desc.aromaticity.shape, (self.num_seqs[0], 1), f'Got {desc.aromaticity.shape}.')
        self.assertEqual(desc.instability_index.shape, (self.num_seqs[0], 1), f'Got {desc.instability_index.shape}.')
        self.assertEqual(desc.isoelectric_point.shape, (self.num_seqs[0], 1), f'Got {desc.isoelectric_point.shape}.')
        self.assertEqual(desc.molecular_weight.shape, (self.num_seqs[0], 1), f'Got {desc.molecular_weight.shape}.')
        self.assertEqual(desc.charge_distribution.shape, (self.num_seqs[0], 3), f'Got {desc.charge_distribution.shape}.')
        self.assertEqual(desc.hydrophobic_polar_charged_composition.shape, (self.num_seqs[0], 3), f'Got {desc.hydrophobic_polar_charged_composition.shape}.')
        self.assertEqual(desc.secondary_structure_propensity.shape, (self.num_seqs[0], 3), f'Got {desc.secondary_structure_propensity.shape}.')
        self.assertEqual(desc.kmer_composition.shape, (self.num_seqs[0], 400), f'Got {desc.kmer_composition.shape}.')
        self.assertEqual(desc.reduced_alphabet_composition.shape, (self.num_seqs[0], 6), f'Got {desc.reduced_alphabet_composition.shape}.')
        self.assertEqual(desc.motif_composition.shape, (self.num_seqs[0], 8), f'Got {desc.motif_composition.shape}.')
        self.assertEqual(desc.amino_acid_pair_composition.shape, (self.num_seqs[0], 400), f'Got {desc.amino_acid_pair_composition.shape}.')
        self.assertEqual(desc.aliphatic_index.shape, (self.num_seqs[0], 1), f'Got {desc.aliphatic_index.shape}.')
        self.assertEqual(desc.extinction_coefficient.shape, (self.num_seqs[0], 2), f'Got {desc.extinction_coefficient.shape}.')
        self.assertEqual(desc.boman_index.shape, (self.num_seqs[0], 1), f'Got {desc.boman_index.shape}.')
        self.assertEqual(desc.aggregation_propensity.shape, (self.num_seqs[0], 2), f'Got {desc.aggregation_propensity.shape}.')
        self.assertEqual(desc.hydrophobic_moment.shape, (self.num_seqs[0], 2), f'Got {desc.hydrophobic_moment.shape}.')
        self.assertEqual(desc.shannon_entropy.shape, (self.num_seqs[0], 1), f'Got {desc.shannon_entropy.shape}.')
#3.)
        #testing on remaining 3 datasets/config files that don't have a pre-calculated descriptors csv
        for config in range(1, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[config])

            #verify num_seqs descriptors attribute is correct
            self.assertEqual(desc.num_seqs, self.num_seqs[config], 
                f'Expected {self.num_seqs[config]} number of sequences, got {desc.num_seqs}.')

            #verify that all input sequences dont have any gaps/missing amino acids
            for seq in desc.protein_seqs:
                self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#4.)
            #verify all descriptor attributes are initialised to empty dataframes
            self.assertTrue(desc.amino_acid_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.dipeptide_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.tripeptide_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.moreaubroto_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.moran_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.geary_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.ctd.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.ctd_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.ctd_transition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.ctd_distribution.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.conjoint_triad.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.sequence_order_coupling_number.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.quasi_sequence_order.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.pseudo_amino_acid_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.amphiphilic_pseudo_amino_acid_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.gravy.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.aromaticity.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.instability_index.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.isoelectric_point.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.molecular_weight.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.charge_distribution.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.hydrophobic_polar_charged_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.secondary_structure_propensity.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.kmer_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.reduced_alphabet_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.motif_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.amino_acid_pair_composition.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.aliphatic_index.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.extinction_coefficient.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.boman_index.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.aggregation_propensity.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.hydrophobic_moment.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.shannon_entropy.empty, 'Attribute should be initialised to an empty dataframe.')
            self.assertTrue(desc.all_descriptors.empty, 'Attribute should be initialised to an empty dataframe.')
#5.)
            #test Type and OS error exceptions are thrown if invalid parameters input
            with self.assertRaises(TypeError, msg='Type Error raised, incorrect datatype input to class.'):
                descr.Descriptors(config_file=123)
                descr.Descriptors(config_file=None)
#6.)
            with self.assertRaises(OSError, msg='OS Error raised, filepath to config file not found.'):
                descr.Descriptors(config_file="incorrect_filepath.json")
                descr.Descriptors(config_file="")

    # @unittest.skip("")
    def test_descriptor_groups(self):
        """ Testing the descriptor groups dictionary which stores the specific group that a descriptor attribute is a member of. """
        #testing on all 4 datasets and config file
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[dataset])
#1.)
            self.assertEqual(list(desc.descriptor_groups.keys()), desc.all_descriptors_list(),
                f"Descriptor groups list is incorrect, got:\n{list(desc.descriptor_groups.keys())}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("Composition"), 21,
                f"Expected there to be 21 composition groups, got {list(desc.descriptor_groups.values()).count('Composition')}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("Autocorrelation"), 3,
                f"Expected there to be 3 autocorrelation groups, got {list(desc.descriptor_groups.values()).count('Autocorrelation')}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1,
                f"Expected there to be 1 conjoint triad groups, got {list(desc.descriptor_groups.values()).count('Conjoint Triad')}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("Sequence Order"), 2,
                f"Expected there to be 2 sequence order groups, got {list(desc.descriptor_groups.values()).count('Sequence Order')}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("CTD"), 4,
                f"Expected there to be 4 CTD groups, got {list(desc.descriptor_groups.values()).count('CTD')}.")
            self.assertEqual(list(desc.descriptor_groups.values()).count("Pseudo Composition"), 2,
                f"Expected there to be 2 pseudo composition groups, got {list(desc.descriptor_groups.values()).count('Pseudo Composition')}.")
            self.assertEqual(len(desc.descriptor_groups.keys()), len(desc.all_descriptors_list()),
                f"Expected {len(desc.all_descriptors_list())} total descriptor groups, got {len(desc.descriptor_groups.keys())}.")
#2.)
            #testing correct descriptor group is returned for each descriptor attribute
            self.assertEqual(desc.descriptor_groups['amino_acid_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['dipeptide_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['tripeptide_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['moran_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['geary_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['moreaubroto_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['ctd'], "CTD")
            self.assertEqual(desc.descriptor_groups['ctd_composition'], "CTD")
            self.assertEqual(desc.descriptor_groups['ctd_distribution'], "CTD")
            self.assertEqual(desc.descriptor_groups['ctd_transition'], "CTD")
            self.assertEqual(desc.descriptor_groups['conjoint_triad'], "Conjoint Triad")
            self.assertEqual(desc.descriptor_groups['pseudo_amino_acid_composition'], "Pseudo Composition")
            self.assertEqual(desc.descriptor_groups['quasi_sequence_order'], "Sequence Order")
            self.assertEqual(desc.descriptor_groups['sequence_order_coupling_number'], "Sequence Order")
            self.assertEqual(desc.descriptor_groups['amphiphilic_pseudo_amino_acid_composition'], "Pseudo Composition")
            # new protpy v1.4.1 descriptors all belong to the Composition group
            self.assertEqual(desc.descriptor_groups['gravy'], "Composition")
            self.assertEqual(desc.descriptor_groups['aromaticity'], "Composition")
            self.assertEqual(desc.descriptor_groups['instability_index'], "Composition")
            self.assertEqual(desc.descriptor_groups['isoelectric_point'], "Composition")
            self.assertEqual(desc.descriptor_groups['molecular_weight'], "Composition")
            self.assertEqual(desc.descriptor_groups['charge_distribution'], "Composition")
            self.assertEqual(desc.descriptor_groups['hydrophobic_polar_charged_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['secondary_structure_propensity'], "Composition")
            self.assertEqual(desc.descriptor_groups['kmer_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['reduced_alphabet_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['motif_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['amino_acid_pair_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['aliphatic_index'], "Composition")
            self.assertEqual(desc.descriptor_groups['extinction_coefficient'], "Composition")
            self.assertEqual(desc.descriptor_groups['boman_index'], "Composition")
            self.assertEqual(desc.descriptor_groups['aggregation_propensity'], "Composition")
            self.assertEqual(desc.descriptor_groups['hydrophobic_moment'], "Composition")
            self.assertEqual(desc.descriptor_groups['shannon_entropy'], "Composition")
#3.)
            self.assertIsInstance(desc.descriptor_groups, dict, f"Expected dict, got {type(desc.descriptor_groups)}.")

    # @unittest.skip("")
    def test_all_descriptors_list(self):
        """ Testing function that returns various combinations of available descriptors using built-in itertools library. """
        #testing on all 4 datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[dataset])
            desc_list_1 = desc.all_descriptors_list(desc_combo=1)
            desc_list_2 = desc.all_descriptors_list(desc_combo=2)
            desc_list_3 = desc.all_descriptors_list(desc_combo=3)
#1.)
            self.assertEqual(len(desc_list_1), 33, f"Expected 33 descriptor combinations, got {len(desc_list_1)}.")
            self.assertEqual(len(desc_list_2), 528, f"Expected 528 descriptor combinations, got {len(desc_list_2)}.")
            self.assertEqual(len(desc_list_3), 5456, f"Expected 5456 descriptor combinations, got {len(desc_list_3)}.")
#2.)
            self.assertIsInstance(desc_list_1, list, f"Expected list, got {type(desc_list_1)}.")
            self.assertIsInstance(desc_list_2, list, f"Expected list, got {type(desc_list_2)}.")
            self.assertIsInstance(desc_list_3, list, f"Expected list, got {type(desc_list_3)}.")

    # @unittest.skip("")
    def test_valid_descriptors(self):
        """ Testing list of valid descriptors available in descriptors module. """
#1.)
        for config in self.all_config_files:
            desc = descr.Descriptors(config_file=config)
            valid_desc = desc.valid_descriptors
    
            self.assertEqual(len(valid_desc), 33, f"Expected there to be 33 total descriptors, got {len(valid_desc)}.")
            self.assertIsInstance(valid_desc, list, f"Expected valid_desc to be a list, got {type(valid_desc)}.")
            self.assertIn('sequence_order_coupling_number', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('geary_autocorrelation', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('moran_autocorrelation', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('moreaubroto_autocorrelation', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('ctd', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('ctd_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('ctd_transition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('ctd_distribution', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('conjoint_triad', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('pseudo_amino_acid_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('amino_acid_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('tripeptide_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('dipeptide_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('quasi_sequence_order', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('amphiphilic_pseudo_amino_acid_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            # new protpy v1.4.1 descriptors
            self.assertIn('gravy', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('aromaticity', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('instability_index', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('isoelectric_point', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('molecular_weight', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('charge_distribution', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('hydrophobic_polar_charged_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('secondary_structure_propensity', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('kmer_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('reduced_alphabet_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('motif_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('amino_acid_pair_composition', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('aliphatic_index', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('extinction_coefficient', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('boman_index', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('aggregation_propensity', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('hydrophobic_moment', valid_desc, "Descriptor should be in list of valid descriptors.")
            self.assertIn('shannon_entropy', valid_desc, "Descriptor should be in list of valid descriptors.")

    # @unittest.skip("")
    def test_descriptor_import(self):
        """ Testing import function that allows for pre-calculated descriptors to be imported from a csv. """ 
#1.)
        desc = descr.Descriptors(self.all_config_files[0]) #pre-calculated thermostability descriptors
        desc.import_descriptors(self.test_descriptors_path)

        self.assertFalse(desc.amino_acid_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.dipeptide_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.tripeptide_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.moreaubroto_autocorrelation.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.moran_autocorrelation.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.geary_autocorrelation.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.ctd.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.ctd_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.ctd_transition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.ctd_distribution.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.conjoint_triad.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.sequence_order_coupling_number.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.quasi_sequence_order.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.pseudo_amino_acid_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.amphiphilic_pseudo_amino_acid_composition.empty, "Descriptor dataframe should not be empty.")
        self.assertFalse(desc.all_descriptors.empty, "Descriptor dataframe should not be empty.")
        # All protpy descriptors are now present in the updated pre-calculated CSV
        self.assertFalse(desc.gravy.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.aromaticity.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.instability_index.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.isoelectric_point.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.molecular_weight.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.charge_distribution.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.hydrophobic_polar_charged_composition.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.secondary_structure_propensity.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.kmer_composition.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.reduced_alphabet_composition.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.motif_composition.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.amino_acid_pair_composition.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.aliphatic_index.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.extinction_coefficient.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.boman_index.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.aggregation_propensity.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.hydrophobic_moment.empty, "Descriptor should be present in the updated pre-calculated CSV.")
        self.assertFalse(desc.shannon_entropy.empty, "Descriptor should be present in the updated pre-calculated CSV.")
#2.)
        with self.assertRaises(OSError):
            desc.import_descriptors("invalid_csv.csv")
            desc.import_descriptors("blahblahblah")
#3.)
        with self.assertRaises(TypeError):
            desc.import_descriptors(1234)
            desc.import_descriptors(False)

    # @unittest.skip("")
    def test_amino_acid_composition(self):
        """ Testing Amino Acid Composition protein descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            aa_comp = desc.get_amino_acid_composition()

            self.assertFalse(aa_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(desc.amino_acid_composition.equals(aa_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aa_comp.shape, (self.num_seqs[dataset], 20), f'Descriptor not correct shape, got {aa_comp.shape}.') 
            self.assertIsInstance(aa_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(aa_comp)}.')
            self.assertTrue(aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(aa_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(aa_comp.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(aa_comp.dtypes)}.")
            self.assertEqual(self.amino_acids, list(aa_comp.columns), 
                f'Incorrect column values found in output dataframe: {aa_comp.columns}.')

    # @unittest.skip("")
    def test_dipeptide_composition(self):
        """ Testing Dipeptide Composition protein descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            dipeptide_comp = desc.get_dipeptide_composition()

            self.assertFalse(dipeptide_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.dipeptide_composition.equals(dipeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(dipeptide_comp.shape, (self.num_seqs[dataset], 400), f'Descriptor not correct shape, got {dipeptide_comp.shape}.')
            self.assertIsInstance(dipeptide_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(dipeptide_comp)}.')
            self.assertTrue(dipeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(dipeptide_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(dipeptide_comp.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(dipeptide_comp.dtypes)}.")
            for col in list(dipeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, f"Column contains an invalid amino acid {col[0]}.")
                self.assertIn(col[1], self.amino_acids, f"Column contains an invalid amino acid {col[1]}.")

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_tripeptide_composition(self):
        """ Testing Tripeptide Composition protein descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            tripeptide_comp = desc.get_tripeptide_composition()

            self.assertFalse(tripeptide_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.tripeptide_composition.equals(tripeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(tripeptide_comp.shape, (self.num_seqs[dataset], 8000), f'Descriptor not correct shape, got {tripeptide_comp.shape}.')
            self.assertIsInstance(tripeptide_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(tripeptide_comp)}.')
            self.assertTrue(tripeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(tripeptide_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.int64 for col in list(tripeptide_comp.dtypes)), 
                f"Column datatypes should be np.int64, got:\n{list(tripeptide_comp.dtypes)}.")
            for col in list(tripeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{3}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, f"Column contains an invalid amino acid {col[0]}.")
                self.assertIn(col[1], self.amino_acids, f"Column contains an invalid amino acid {col[1]}.")
                self.assertIn(col[2], self.amino_acids, f"Column contains an invalid amino acid {col[2]}.")

    # @unittest.skip("")
    def test_moreaubroto_autocorrelation(self):
        """ Testing moreaubroto autocorrelation descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            moreau_broto = desc.get_moreaubroto_autocorrelation()

            self.assertFalse(moreau_broto.empty, 'Descriptor dataframe should not be empty.') 
            self.assertTrue(desc.moreaubroto_autocorrelation.equals(moreau_broto), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(moreau_broto.shape, (self.num_seqs[dataset], 240), f'Descriptor not correct shape, got {moreau_broto.shape}.')
            self.assertIsInstance(moreau_broto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(moreau_broto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(moreau_broto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(moreau_broto.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(moreau_broto.dtypes)}.")
            #check all columns follow pattern of MoreauBrotoAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(moreau_broto.columns):
                self.assertTrue(bool(re.match(r"MBAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    f"Column name doesn't match expected regex pattern: {col}.")  

    # @unittest.skip("")
    def test_moran_autocorrelation(self):
        """ Testing Moran autocorrelation descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)            
            #get descriptor values
            moran_auto = desc.get_moran_autocorrelation()

            self.assertFalse(moran_auto.empty, 'Descriptor dataframe should not be empty.') 
            self.assertTrue(desc.moran_autocorrelation.equals(moran_auto), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(moran_auto.shape, (self.num_seqs[dataset], 240), f'Descriptor not correct shape, got {moran_auto.shape}.')
            self.assertIsInstance(moran_auto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(moran_auto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(moran_auto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(moran_auto.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(moran_auto.dtypes)}.")
            #check all columns follow pattern of MoranAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(moran_auto.columns):
                self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    f"Column name doesn't match expected regex pattern: {col}.")

    # @unittest.skip("")
    def test_geary_autocorrelation(self):
        """ Testing Geary autocorrelation descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            geary_auto = desc.get_geary_autocorrelation()

            self.assertFalse(geary_auto.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(desc.geary_autocorrelation.equals(geary_auto), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(geary_auto.shape, (self.num_seqs[dataset], 240), f'Descriptor not correct shape, got {geary_auto.shape}.')
            self.assertIsInstance(geary_auto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(geary_auto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(geary_auto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(geary_auto.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(geary_auto.dtypes)}.")
            #check all columns follow pattern of GAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(geary_auto.columns):
                self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    f"Column name doesn't match expected regex pattern: {col}.")
    
    # @unittest.skip("")
    def test_ctd(self):
        """ Testing CTD descriptor attributes and methods. """
        ctd_properties = ["hydrophobicity", "normalized_vdwv", "polarity", "charge",
            "secondary_struct", "solvent_accessibility", "polarizability"]
        
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            ctd = desc.get_ctd()

            self.assertFalse(ctd.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(desc.ctd.equals(ctd), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd.shape, (self.num_seqs[dataset], 21), f'Descriptor not of correct, got {ctd.shape}.')
            self.assertIsInstance(ctd, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(ctd.dtypes)}.")
            #iterate over all columns and check its name follows expected format
            for col in list(ctd.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]_[0-9]{2}_[0-9]{3}_" + prop , col)))), 
                                f"Column name does not follow expected format: {col}.")
                self.assertTrue(matching_col, f"Column name's property name not found and doesn't match format: {col}.")
#2.)                
            #get descriptor values
            ctd_comp = desc.get_ctd_composition() 

            self.assertFalse(ctd_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_composition.equals(ctd_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_comp.shape, (self.num_seqs[dataset], 3), f'Descriptor not of correct, got {ctd_comp.shape}.')
            self.assertIsInstance(ctd_comp, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_comp.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(ctd_comp.dtypes)}.")
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_comp.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                                f"Column name does not follow expected format: {col}.")
                self.assertTrue(matching_col, f"Column name's property name not found and doesn't match format: {col}.")
#3.)
            #get descriptor values
            ctd_trans = desc.get_ctd_transition()

            self.assertFalse(ctd_trans.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_transition.equals(ctd_trans), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_trans.shape, (self.num_seqs[dataset], 3), f'Descriptor not of correct, got {ctd_trans.shape}.')
            self.assertIsInstance(ctd_trans, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_trans.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_trans).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_trans.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(ctd_trans.dtypes)}.")
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_trans.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                                f"Column name does not follow expected format: {col}.")
                self.assertTrue(matching_col, f"Column name's property name not found and doesn't match format: {col}.")
#4.)
            #get descriptor values
            ctd_distr = desc.get_ctd_distribution()

            self.assertFalse(ctd_distr.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_distribution.equals(ctd_distr), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_distr.shape, (self.num_seqs[dataset], 15), f'Descriptor not of correct, got {ctd_distr.shape}.')
            self.assertIsInstance(ctd_distr, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_distr.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_distr).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_distr.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(ctd_distr.dtypes)}.")
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_distr.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{3}_" + prop, col)))), 
                                f"Column name does not follow expected format: {col}.")
                self.assertTrue(matching_col, f"Column name's property name not found and doesn't match format: {col}.")

    # @unittest.skip("")
    def test_conjoint_triad(self):
        """ Testing Conjoint Triad descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.) 
            #get descriptor values
            conjoint_triad = desc.get_conjoint_triad()
   
            self.assertFalse(conjoint_triad.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.conjoint_triad.equals(conjoint_triad), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(conjoint_triad.shape, (self.num_seqs[dataset], 343), f'Descriptor not of correct shape, got {conjoint_triad.shape}.')
            self.assertIsInstance(conjoint_triad, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(conjoint_triad.any().isnull().sum()==0,'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(conjoint_triad).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.int64 for col in list(conjoint_triad.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(conjoint_triad.dtypes)}.")
            #iterate over all columns and check its name follows expected format
            for col in list(conjoint_triad.columns):
                self.assertTrue(bool(re.match(r"conj_triad_[0-9]{3}", col)), 
                    f"Column name doesn't match expected regex pattern: {col}.")   

    # @unittest.skip("")
    def test_sequence_order_coupling_number(self):
        """ Testing sequence order coupling number descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            sequence_order_coupling_number = desc.get_sequence_order_coupling_number()

            self.assertFalse(sequence_order_coupling_number.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.sequence_order_coupling_number.equals(sequence_order_coupling_number), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(sequence_order_coupling_number.shape, (self.num_seqs[dataset], 30), f'Descriptor not of correct shape, got {sequence_order_coupling_number.shape}.')
            self.assertIsInstance(sequence_order_coupling_number, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(sequence_order_coupling_number.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(sequence_order_coupling_number).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(sequence_order_coupling_number.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(sequence_order_coupling_number.dtypes)}.")
            #check all columns follow pattern of SOCNX or SOCNXY where x & y integers between 0 and 9
            for col in list(sequence_order_coupling_number.columns):
                self.assertTrue((bool(re.match(r'SOCN_SW[0-9]', col)) or bool(re.match(r'SOCN_SW[0-9][0-9]', col))), 
                    f"Column name doesn't match expected regex pattern: {col}.")   

    # @unittest.skip("")
    def test_quasi_sequence_order(self):
        """ Testing Quasi sequence order descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            quasi_sequence_order = desc.get_quasi_sequence_order()
#1.)
            self.assertFalse(quasi_sequence_order.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.quasi_sequence_order.equals(quasi_sequence_order), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(quasi_sequence_order.shape, (self.num_seqs[dataset], 50), f'Descriptor not of correct shape, got {quasi_sequence_order.shape}.')
            self.assertIsInstance(quasi_sequence_order, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(quasi_sequence_order.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(quasi_sequence_order).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(quasi_sequence_order.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(quasi_sequence_order.dtypes)}.")
            #check all columns follow pattern of QSO_X, where x is an integer between 0 and 9
            for col in list(quasi_sequence_order.columns):
                self.assertTrue((bool(re.match(r'QSO_SW[0-9]', col))), 
                    f"Column name doesn't match expected regex pattern: {col}.")

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping.")
    def test_pseudo_amino_acid_composition(self):
        """ Testing Pseudo Amino Acid Composition descriptor attributes and methods. """

        #running unit test on one of the datasets due to length of computation - thermostability
        desc = descr.Descriptors(self.all_config_files[0])
#1.)
        #get descriptor values
        pseudo_aa_comp = desc.get_pseudo_amino_acid_composition()

        self.assertFalse(pseudo_aa_comp.empty, 'Descriptor dataframe should not be empty.')
        self.assertTrue(desc.pseudo_amino_acid_composition.equals(pseudo_aa_comp), 'Output dataframe and class attribute dataframes must be the same.')
        self.assertEqual(pseudo_aa_comp.shape, (self.num_seqs[0], 50), f'Descriptor not of correct shape, got {pseudo_aa_comp.shape}.')
        self.assertIsInstance(pseudo_aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(pseudo_aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(pseudo_aa_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(pseudo_aa_comp.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(pseudo_aa_comp.dtypes)}.")
        #check all columns follow pattern of PAACX, where x is an integer between 0 and 9
        for col in list(pseudo_aa_comp.columns):
            self.assertTrue(bool(re.match(r"PAAC[0-9]", col)), 
                f"Column doesn't follow correct naming convention: {col}.")

        desc = descr.Descriptors(self.all_config_files[1])

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping.")
    def test_amphiphilic_pseudo_amino_acid_composition(self):
        """ Testing Amphiphilic Pseudo Amino Acid Composition descriptor attributes and methods. """
        #running unit test on one of the datasets due to length of computation - thermostability
        desc = descr.Descriptors(self.all_config_files[0])
#1.)
        #get descriptor values
        amphiphilic_pseudo_aac = desc.get_amphiphilic_pseudo_amino_acid_composition()

        self.assertFalse(amphiphilic_pseudo_aac.empty, 'Descriptor dataframe should not be empty.')
        self.assertTrue(desc.amphiphilic_pseudo_amino_acid_composition.equals(amphiphilic_pseudo_aac), 'Output dataframe and class attribute dataframes must be the same.')
        self.assertEqual(amphiphilic_pseudo_aac.shape, (self.num_seqs[1], 80), f'Descriptor not of correct shape, got {amphiphilic_pseudo_aac.shape}.')
        self.assertIsInstance(amphiphilic_pseudo_aac, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(amphiphilic_pseudo_aac.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(amphiphilic_pseudo_aac).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(amphiphilic_pseudo_aac.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(amphiphilic_pseudo_aac.dtypes)}.")
        #check all columns follow pattern of APAAC_X, where x is an integer between 0 and 9
        for col in list(amphiphilic_pseudo_aac.columns):
            self.assertTrue(bool(re.match(r"APAAC_[0-9]", col)), 
                f"Column doesn't follow correct naming convention: {col}.")
        
    # @unittest.skip("")
    def test_gravy(self):
        """ Testing GRAVY (Grand Average of Hydropathicity) descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            gravy = desc.get_gravy()

            self.assertFalse(gravy.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.gravy.equals(gravy), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(gravy.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {gravy.shape}.')
            self.assertIsInstance(gravy, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(gravy)}.')
            self.assertTrue(gravy.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(gravy).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(gravy.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(gravy.dtypes)}.")
            self.assertEqual(list(gravy.columns), ['GRAVY'],
                f"Expected column name 'GRAVY', got {list(gravy.columns)}.")

    # @unittest.skip("")
    def test_aromaticity(self):
        """ Testing aromaticity descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            aromaticity = desc.get_aromaticity()

            self.assertFalse(aromaticity.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.aromaticity.equals(aromaticity), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aromaticity.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {aromaticity.shape}.')
            self.assertIsInstance(aromaticity, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(aromaticity)}.')
            self.assertTrue(aromaticity.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(aromaticity).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(aromaticity.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(aromaticity.dtypes)}.")
            self.assertEqual(list(aromaticity.columns), ['Aromaticity'],
                f"Expected column name 'Aromaticity', got {list(aromaticity.columns)}.")

    # @unittest.skip("")
    def test_instability_index(self):
        """ Testing instability index descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            instability_index = desc.get_instability_index()

            self.assertFalse(instability_index.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.instability_index.equals(instability_index), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(instability_index.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {instability_index.shape}.')
            self.assertIsInstance(instability_index, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(instability_index)}.')
            self.assertTrue(instability_index.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(instability_index).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(instability_index.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(instability_index.dtypes)}.")
            self.assertEqual(list(instability_index.columns), ['InstabilityIndex'],
                f"Expected column name 'InstabilityIndex', got {list(instability_index.columns)}.")

    # @unittest.skip("")
    def test_isoelectric_point(self):
        """ Testing isoelectric point descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            isoelectric_point = desc.get_isoelectric_point()

            self.assertFalse(isoelectric_point.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.isoelectric_point.equals(isoelectric_point), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(isoelectric_point.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {isoelectric_point.shape}.')
            self.assertIsInstance(isoelectric_point, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(isoelectric_point)}.')
            self.assertTrue(isoelectric_point.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(isoelectric_point).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(isoelectric_point.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(isoelectric_point.dtypes)}.")
            self.assertEqual(list(isoelectric_point.columns), ['IsoelectricPoint'],
                f"Expected column name 'IsoelectricPoint', got {list(isoelectric_point.columns)}.")

    # @unittest.skip("")
    def test_molecular_weight(self):
        """ Testing molecular weight descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            molecular_weight = desc.get_molecular_weight()

            self.assertFalse(molecular_weight.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.molecular_weight.equals(molecular_weight), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(molecular_weight.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {molecular_weight.shape}.')
            self.assertIsInstance(molecular_weight, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(molecular_weight)}.')
            self.assertTrue(molecular_weight.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(molecular_weight).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(molecular_weight.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(molecular_weight.dtypes)}.")
            self.assertEqual(list(molecular_weight.columns), ['MolecularWeight'],
                f"Expected column name 'MolecularWeight', got {list(molecular_weight.columns)}.")

    # @unittest.skip("")
    def test_charge_distribution(self):
        """ Testing charge distribution descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            charge_distribution = desc.get_charge_distribution()

            self.assertFalse(charge_distribution.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.charge_distribution.equals(charge_distribution), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(charge_distribution.shape, (self.num_seqs[dataset], 3), f'Descriptor not correct shape, got {charge_distribution.shape}.')
            self.assertIsInstance(charge_distribution, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(charge_distribution)}.')
            self.assertTrue(charge_distribution.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(charge_distribution).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(charge_distribution.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(charge_distribution.dtypes)}.")
            self.assertEqual(list(charge_distribution.columns), ['PositiveCharge', 'NegativeCharge', 'NetCharge'],
                f"Expected columns ['PositiveCharge', 'NegativeCharge', 'NetCharge'], got {list(charge_distribution.columns)}.")

    # @unittest.skip("")
    def test_hydrophobic_polar_charged_composition(self):
        """ Testing hydrophobic/polar/charged composition descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            hpc_comp = desc.get_hydrophobic_polar_charged_composition()

            self.assertFalse(hpc_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.hydrophobic_polar_charged_composition.equals(hpc_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(hpc_comp.shape, (self.num_seqs[dataset], 3), f'Descriptor not correct shape, got {hpc_comp.shape}.')
            self.assertIsInstance(hpc_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(hpc_comp)}.')
            self.assertTrue(hpc_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(hpc_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(hpc_comp.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(hpc_comp.dtypes)}.")
            self.assertEqual(list(hpc_comp.columns), ['Hydrophobic', 'Polar', 'Charged'],
                f"Expected columns ['Hydrophobic', 'Polar', 'Charged'], got {list(hpc_comp.columns)}.")

    # @unittest.skip("")
    def test_secondary_structure_propensity(self):
        """ Testing secondary structure propensity descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            sec_struct = desc.get_secondary_structure_propensity()

            self.assertFalse(sec_struct.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.secondary_structure_propensity.equals(sec_struct), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(sec_struct.shape, (self.num_seqs[dataset], 3), f'Descriptor not correct shape, got {sec_struct.shape}.')
            self.assertIsInstance(sec_struct, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(sec_struct)}.')
            self.assertTrue(sec_struct.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(sec_struct).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(sec_struct.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(sec_struct.dtypes)}.")
            self.assertEqual(list(sec_struct.columns), ['Helix', 'Sheet', 'Coil'],
                f"Expected columns ['Helix', 'Sheet', 'Coil'], got {list(sec_struct.columns)}.")

    # @unittest.skip("")
    def test_kmer_composition(self):
        """ Testing k-mer composition descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            kmer_comp = desc.get_kmer_composition()

            self.assertFalse(kmer_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.kmer_composition.equals(kmer_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(kmer_comp.shape, (self.num_seqs[dataset], 400), f'Descriptor not correct shape, got {kmer_comp.shape}.')
            self.assertIsInstance(kmer_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(kmer_comp)}.')
            self.assertTrue(kmer_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(kmer_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(kmer_comp.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(kmer_comp.dtypes)}.")
            #check all columns follow pattern of XY where x & y are amino acids (k=2)
            for col in list(kmer_comp.columns):
                self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), f"Column name doesn't match expected k=2 pattern: {col}.")
                self.assertIn(col[0], self.amino_acids, f"Column contains an invalid amino acid {col[0]}.")
                self.assertIn(col[1], self.amino_acids, f"Column contains an invalid amino acid {col[1]}.")

    # @unittest.skip("")
    def test_reduced_alphabet_composition(self):
        """ Testing reduced alphabet composition descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            reduced_alpha = desc.get_reduced_alphabet_composition()

            self.assertFalse(reduced_alpha.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.reduced_alphabet_composition.equals(reduced_alpha), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(reduced_alpha.shape, (self.num_seqs[dataset], 6), f'Descriptor not correct shape, got {reduced_alpha.shape}.')
            self.assertIsInstance(reduced_alpha, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(reduced_alpha)}.')
            self.assertTrue(reduced_alpha.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(reduced_alpha).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(reduced_alpha.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(reduced_alpha.dtypes)}.")
            #check all columns follow ReducedAlphabet_N pattern
            for col in list(reduced_alpha.columns):
                self.assertTrue(bool(re.match(r'^ReducedAlphabet_[0-9]+$', col)),
                    f"Column name doesn't match expected pattern: {col}.")

    @unittest.skip("")
    def test_motif_composition(self):
        """ Testing motif composition descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            motif_comp = desc.get_motif_composition()

            self.assertFalse(motif_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.motif_composition.equals(motif_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(motif_comp.shape, (self.num_seqs[dataset], 8), f'Descriptor not correct shape, got {motif_comp.shape}.')
            self.assertIsInstance(motif_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(motif_comp)}.')
            self.assertTrue(motif_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.int64 for col in list(motif_comp.dtypes)),
                f"Column datatypes should be np.int64, got:\n{list(motif_comp.dtypes)}.")

    # @unittest.skip("")
    def test_amino_acid_pair_composition(self):
        """ Testing amino acid pair composition descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            aa_pair_comp = desc.get_amino_acid_pair_composition()

            self.assertFalse(aa_pair_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.amino_acid_pair_composition.equals(aa_pair_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aa_pair_comp.shape, (self.num_seqs[dataset], 400), f'Descriptor not correct shape, got {aa_pair_comp.shape}.')
            self.assertIsInstance(aa_pair_comp, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(aa_pair_comp)}.')
            self.assertTrue(aa_pair_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(aa_pair_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(aa_pair_comp.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(aa_pair_comp.dtypes)}.")

    # @unittest.skip("")
    def test_aliphatic_index(self):
        """ Testing aliphatic index descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            aliphatic_index = desc.get_aliphatic_index()

            self.assertFalse(aliphatic_index.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.aliphatic_index.equals(aliphatic_index), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aliphatic_index.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {aliphatic_index.shape}.')
            self.assertIsInstance(aliphatic_index, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(aliphatic_index)}.')
            self.assertTrue(aliphatic_index.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(aliphatic_index).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(aliphatic_index.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(aliphatic_index.dtypes)}.")
            self.assertEqual(list(aliphatic_index.columns), ['AliphaticIndex'],
                f"Expected column name 'AliphaticIndex', got {list(aliphatic_index.columns)}.")

    # @unittest.skip("")
    def test_extinction_coefficient(self):
        """ Testing extinction coefficient descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            ext_coeff = desc.get_extinction_coefficient()

            self.assertFalse(ext_coeff.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.extinction_coefficient.equals(ext_coeff), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ext_coeff.shape, (self.num_seqs[dataset], 2), f'Descriptor not correct shape, got {ext_coeff.shape}.')
            self.assertIsInstance(ext_coeff, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(ext_coeff)}.')
            self.assertTrue(ext_coeff.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ext_coeff).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.int64 for col in list(ext_coeff.dtypes)),
                f"Column datatypes should be np.int64, got:\n{list(ext_coeff.dtypes)}.")
            self.assertEqual(list(ext_coeff.columns), ['ExtCoeff_Reduced', 'ExtCoeff_Oxidized'],
                f"Expected columns ['ExtCoeff_Reduced', 'ExtCoeff_Oxidized'], got {list(ext_coeff.columns)}.")

    # @unittest.skip("")
    def test_boman_index(self):
        """ Testing Boman index descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            boman_index = desc.get_boman_index()

            self.assertFalse(boman_index.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.boman_index.equals(boman_index), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(boman_index.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {boman_index.shape}.')
            self.assertIsInstance(boman_index, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(boman_index)}.')
            self.assertTrue(boman_index.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(boman_index).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(boman_index.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(boman_index.dtypes)}.")
            self.assertEqual(list(boman_index.columns), ['BomanIndex'],
                f"Expected column name 'BomanIndex', got {list(boman_index.columns)}.")

    # @unittest.skip("")
    def test_aggregation_propensity(self):
        """ Testing aggregation propensity descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            agg_prop = desc.get_aggregation_propensity()

            self.assertFalse(agg_prop.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.aggregation_propensity.equals(agg_prop), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(agg_prop.shape, (self.num_seqs[dataset], 2), f'Descriptor not correct shape, got {agg_prop.shape}.')
            self.assertIsInstance(agg_prop, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(agg_prop)}.')
            self.assertTrue(agg_prop.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(agg_prop).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertEqual(agg_prop['AggregProneRegions'].dtype, np.int64,
                f"AggregProneRegions column should be np.int64, got {agg_prop['AggregProneRegions'].dtype}.")
            self.assertEqual(agg_prop['AggregProneFraction'].dtype, np.float64,
                f"AggregProneFraction column should be np.float64, got {agg_prop['AggregProneFraction'].dtype}.")
            self.assertEqual(list(agg_prop.columns), ['AggregProneRegions', 'AggregProneFraction'],
                f"Expected columns ['AggregProneRegions', 'AggregProneFraction'], got {list(agg_prop.columns)}.")

    # @unittest.skip("")
    def test_hydrophobic_moment(self):
        """ Testing hydrophobic moment descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            hydrophobic_moment = desc.get_hydrophobic_moment()

            self.assertFalse(hydrophobic_moment.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.hydrophobic_moment.equals(hydrophobic_moment), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(hydrophobic_moment.shape, (self.num_seqs[dataset], 2), f'Descriptor not correct shape, got {hydrophobic_moment.shape}.')
            self.assertIsInstance(hydrophobic_moment, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(hydrophobic_moment)}.')
            self.assertTrue(hydrophobic_moment.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(hydrophobic_moment).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(hydrophobic_moment.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(hydrophobic_moment.dtypes)}.")
            self.assertEqual(list(hydrophobic_moment.columns), ['HydrophobicMoment_Mean', 'HydrophobicMoment_Max'],
                f"Expected columns ['HydrophobicMoment_Mean', 'HydrophobicMoment_Max'], got {list(hydrophobic_moment.columns)}.")

    # @unittest.skip("")
    def test_shannon_entropy(self):
        """ Testing Shannon entropy descriptor attributes and methods. """
        #run tests on all 4 test datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
#1.)
            #get descriptor values
            shannon_entropy = desc.get_shannon_entropy()

            self.assertFalse(shannon_entropy.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.shannon_entropy.equals(shannon_entropy), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(shannon_entropy.shape, (self.num_seqs[dataset], 1), f'Descriptor not correct shape, got {shannon_entropy.shape}.')
            self.assertIsInstance(shannon_entropy, pd.DataFrame, f'Descriptor should be of type DataFrame, got {type(shannon_entropy)}.')
            self.assertTrue(shannon_entropy.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(shannon_entropy).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(shannon_entropy.dtypes)),
                f"Column datatypes should be np.float64, got:\n{list(shannon_entropy.dtypes)}.")
            self.assertEqual(list(shannon_entropy.columns), ['ShannonEntropy'],
                f"Expected column name 'ShannonEntropy', got {list(shannon_entropy.columns)}.")

    @unittest.skip("Skipping as calculating all descriptors takes a long time.")
    def test_get_all_descriptors(self):
        """ Testing functionality for calculating all protein descriptors for a datast of protein sequences.
            Only testing on the thermostability dataset/config as its protein descriptors have been
            pre-calcualted. Testing on the other datasets could take several hours each. """
#1.)
        #only testing on thermostability dataset to access pre-calculated descriptors
        desc = descr.Descriptors(self.all_config_files[0])
        all_descriptors = desc.get_all_descriptors()

        self.assertIsInstance(all_descriptors, pd.DataFrame, f'Expected function output to be of type DataFrame, got {type(all_descriptors)}.')
        self.assertEqual(all_descriptors.shape, (self.num_seqs[0], 10551), f"Expected shape of output to be {self.num_seqs[0]} x 10551, got {all_descriptors.shape}.")

#2.)
        #sequence_col prepends the named dataset column as the first column
        all_descriptors_named = desc.get_all_descriptors(sequence_col='name')

        self.assertIsInstance(all_descriptors_named, pd.DataFrame,
            f'Expected DataFrame with sequence_col, got {type(all_descriptors_named)}.')
        self.assertEqual(all_descriptors_named.shape, (self.num_seqs[0], 10552),
            f"Expected shape {self.num_seqs[0]} x 10552 with sequence_col, got {all_descriptors_named.shape}.")
        self.assertEqual(all_descriptors_named.columns[0], 'name',
            f"First column should be 'name', got '{all_descriptors_named.columns[0]}'.")

#3.)
        #invalid sequence_col raises ValueError
        with self.assertRaises(ValueError):
            desc.get_all_descriptors(sequence_col='nonexistent_column')

    # @unittest.skip("")
    def test_n_jobs_parallel(self):
        """ Testing parallel descriptor computation via the n_jobs parameter.
            Uses the smallest dataset (absorption, 81 seqs) for sequence-level comparison
            and the pre-calculated thermostability descriptors for the get_all_descriptors path. """
#1.)
        #n_jobs defaults to 1
        desc = descr.Descriptors(config_file=self.all_config_files[2])  # absorption - 81 seqs
        self.assertEqual(desc.n_jobs, 1, f'Default n_jobs should be 1, got {desc.n_jobs}.')

#2.)
        #invalid/zero/negative n_jobs values are clamped to 1
        desc_zero = descr.Descriptors(config_file=self.all_config_files[2], n_jobs=0)
        self.assertEqual(desc_zero.n_jobs, 1, 'n_jobs=0 should be clamped to 1.')
        desc_neg = descr.Descriptors(config_file=self.all_config_files[2], n_jobs=-4)
        self.assertEqual(desc_neg.n_jobs, 1, 'Negative n_jobs should be clamped to 1.')

#3.)
        #parallel sequence-level computation (n_jobs>1) must produce numerically identical results
        #to sequential computation (n_jobs=1) for the same descriptor
        desc_seq = descr.Descriptors(config_file=self.all_config_files[2], n_jobs=1)
        desc_par = descr.Descriptors(config_file=self.all_config_files[2], n_jobs=4)

        aa_comp_seq = desc_seq.get_amino_acid_composition()
        aa_comp_par = desc_par.get_amino_acid_composition()

        self.assertTrue(aa_comp_seq.equals(aa_comp_par),
            'Parallel and sequential amino acid composition results must be numerically identical.')
        self.assertEqual(aa_comp_seq.shape, aa_comp_par.shape,
            f'Shape mismatch: sequential {aa_comp_seq.shape} vs parallel {aa_comp_par.shape}.')
        self.assertFalse(aa_comp_par.empty, 'Parallel output dataframe should not be empty.')
        self.assertEqual(aa_comp_par.shape, (self.num_seqs[2], 20),
            f'Parallel output should be {self.num_seqs[2]} x 20, got {aa_comp_par.shape}.')
        self.assertTrue(aa_comp_par.any().isnull().sum() == 0,
            'Parallel output should not contain any null values.')
        self.assertTrue(np.isinf(aa_comp_par).values.sum() == 0,
            'Parallel output should not contain any +/- infinity values.')

#4.)
        #n_jobs=4 via get_all_descriptors: use thermostability config (pre-calculated descriptors
        #already loaded at init) so the parallel dispatch path runs without long recomputation
        desc_all_par = descr.Descriptors(config_file=self.all_config_files[0], n_jobs=4)
        all_desc_par = desc_all_par.get_boman_index()

        self.assertIsInstance(all_desc_par, pd.DataFrame,
            f'Expected DataFrame from parallel get_all_descriptors, got {type(all_desc_par)}.')
        self.assertEqual(all_desc_par.shape, (self.num_seqs[0], 1),
            f'Expected shape ({self.num_seqs[0]}, 10551) from parallel run, got {all_desc_par.shape}.')
        self.assertTrue(all_desc_par.any().isnull().sum() == 0,
            'Parallel get_all_descriptors output should not contain any null values.')

    # @unittest.skip("")
    def test_get_descriptor_encoding(self):
        """ Testing get_descriptor_encoding function by passing string of approximate descriptor names in to get encoding. """
        desc = descr.Descriptors(self.all_config_files[0]) #using thermostability config to access pre-calculated descriptors
#1.)    
        aa_comp_desc = desc.get_descriptor_encoding("amino_comp")

        self.assertIsInstance(aa_comp_desc, pd.DataFrame, f'Descriptor attribute should be a dataframe, got {type(aa_comp_desc)}.')
        self.assertEqual(aa_comp_desc.shape, (self.num_seqs[0], 20), 
            f"Attribute shape should be ({self.num_seqs[0]}, {20}), got {aa_comp_desc.shape}.")
        self.assertTrue(aa_comp_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(aa_comp_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(aa_comp_desc.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(aa_comp_desc.dtypes)}.")
        self.assertEqual(self.amino_acids, list(aa_comp_desc.columns), 
            f'Incorrect column values found in output dataframe: {aa_comp_desc.columns}.')
#2.)
        geary_auto_desc = desc.get_descriptor_encoding("geary_auto")

        self.assertIsInstance(geary_auto_desc, pd.DataFrame, f'Descriptor attribute should be a dataframe, got {type(geary_auto_desc)}.')
        self.assertEqual(geary_auto_desc.shape, (self.num_seqs[0], 240), 
            f"Attribute shape should be ({self.num_seqs[0]}, {240}), got {geary_auto_desc.shape}.")
        self.assertTrue(geary_auto_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(geary_auto_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(geary_auto_desc.dtypes)), 
            f"Column datatypes should be np.float64, got:\n{list(geary_auto_desc.dtypes)}.")
        #check all columns follow pattern of GAuto_X_Y where x is the asscession number of
        #the AAindex record and y is the count of the descriptor
        for col in list(geary_auto_desc.columns):
            self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)), 
                f"Column name doesn't match expected regex pattern: {col}.")
#3.)
        socn_desc = desc.get_descriptor_encoding("sequence_order_coupling")

        self.assertIsInstance(socn_desc, pd.DataFrame, f'Descriptor attribute should be a dataframe, got {socn_desc}.')
        self.assertEqual(socn_desc.shape, (self.num_seqs[0], 30), 
            f"Attribute shape should be ({self.num_seqs[0]}, {30}), got {socn_desc.shape}.")
        self.assertTrue(socn_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(socn_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(socn_desc.dtypes)), 
            f"Column datatypes should be np.float64, got:\n{list(socn_desc.dtypes)}.")
        #check all columns follow pattern of SOCNX or SOCNXY where x & y integers between 0 and 9
        for col in list(socn_desc.columns):
            self.assertTrue((bool(re.match(r'SOCN_SW[0-9]', col)) or bool(re.match(r'SOCN_SW[0-9][0-9]', col))), 
                f"Column name doesn't match expected regex pattern: {col}.")   
#4.)
        dipeptide_comp_desc = desc.get_descriptor_encoding("dipeptide")
        
        self.assertIsInstance(dipeptide_comp_desc, pd.DataFrame, f'Descriptor attribute should be a dataframe, got {dipeptide_comp_desc}.')
        self.assertEqual(dipeptide_comp_desc.shape, (self.num_seqs[0], 400), 
            f"Attribute shape should be ({self.num_seqs[0]}, {400}), got {dipeptide_comp_desc.shape}.")
        self.assertTrue(dipeptide_comp_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(dipeptide_comp_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(dipeptide_comp_desc.dtypes)),
            f"Column datatypes should be np.float64, got:\n{list(dipeptide_comp_desc.dtypes)}.")
        for col in list(dipeptide_comp_desc.columns):
            #check all columns follow pattern of XY where x & y are amino acids 
            self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "")      
            self.assertIn(col[0], self.amino_acids, f"Column contains invalid amino acid: {col[0]}.")
            self.assertIn(col[1], self.amino_acids, f"Column contains invalid amino acid: {col[1]}.")
#5.)
        ctd_transition_desc = desc.get_descriptor_encoding("ctd_transition")
        ctd_properties = ["hydrophobicity", "normalized_vdwv", "polarity", "charge",
            "secondary_struct", "solvent_accessibility", "polarizability"]
        
        self.assertIsInstance(ctd_transition_desc, pd.DataFrame, f'Descriptor attribute should be a dataframe, got {ctd_transition_desc}.')
        self.assertEqual(ctd_transition_desc.shape, (self.num_seqs[0], 3), 
            f"Attribute shape should be ({self.num_seqs[0]}, {3}), got {ctd_transition_desc.shape}.")
        self.assertTrue(ctd_transition_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(ctd_transition_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(ctd_transition_desc.dtypes)), 
                f"Column datatypes should be np.float64, got:\n{list(ctd_transition_desc.dtypes)}.")
        #check all columns follow correct format
        for col in list(ctd_transition_desc.columns):
            matching_col = False
            for prop in ctd_properties:
                if (col.endswith(prop)):
                    matching_col = True
                    self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                        (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                            f"Column name does not follow expected format: {col}.")
            self.assertTrue(matching_col, f"Column name's property name not found and doesn't match format: {col}.")
#6.)    
        with self.assertRaises(ValueError):
            desc.get_descriptor_encoding("invalid")
            desc.get_descriptor_encoding("blahblahblah")
            desc.get_descriptor_encoding("12345")
#7.)
        with self.assertRaises(TypeError):
            desc.get_descriptor_encoding(1234)
            desc.get_descriptor_encoding(5.5)
            desc.get_descriptor_encoding(False)

    def tearDown(self):
        """ Cleanup tests and delete datasets/config files. """
        del self.all_config_files