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
    test_get_all_descriptors:
        testing correct functionality for calculating all descriptors for a dataset of sequences.
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

    def test_descriptor(self):
        """ Test descriptor initialisation process. Verify the initial input parameters and descriptor attributes are correct. """
#1.)
        desc = descr.Descriptors(config_file=self.all_config_files[0]) #pre-calculated descriptors from thermostability dataset

        #verify num_seqs descriptors attribute is correct
        self.assertEqual(desc.num_seqs, self.num_seqs[0], 
            'Expected {} number of sequences, got {}.'.format(self.num_seqs[0], desc.num_seqs))

        #verify that all input sequences dont have any gaps/missing amino acids
        for seq in desc.protein_seqs:
            self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#2.)
        self.assertEqual(desc.amino_acid_composition.shape, (self.num_seqs[0], 20), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 20, desc.amino_acid_composition.shape))
        self.assertEqual(desc.dipeptide_composition.shape, (self.num_seqs[0], 400), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 400, desc.dipeptide_composition.shape))
        self.assertEqual(desc.tripeptide_composition.shape, (self.num_seqs[0], 8000), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 8000, desc.tripeptide_composition.shape))
        self.assertEqual(desc.moreaubroto_autocorrelation.shape, (self.num_seqs[0], 240), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 240, desc.moreaubroto_autocorrelation.shape))
        self.assertEqual(desc.moran_autocorrelation.shape, (self.num_seqs[0], 240), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 240, desc.moran_autocorrelation.shape))
        self.assertEqual(desc.geary_autocorrelation.shape, (self.num_seqs[0], 240), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 240, desc.geary_autocorrelation.shape))
        self.assertEqual(desc.ctd.shape, (self.num_seqs[0], 21), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 21, desc.ctd.shape))
        self.assertEqual(desc.ctd_composition.shape, (self.num_seqs[0], 3), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 3, desc.ctd_composition.shape))
        self.assertEqual(desc.ctd_transition.shape, (self.num_seqs[0], 3), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 3, desc.ctd_transition.shape))
        self.assertEqual(desc.ctd_distribution.shape, (self.num_seqs[0], 15), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 15, desc.ctd_distribution.shape))
        self.assertEqual(desc.conjoint_triad.shape, (self.num_seqs[0], 343), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 343, desc.conjoint_triad.shape))
        self.assertEqual(desc.sequence_order_coupling_number.shape, (self.num_seqs[0], 30), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 30, desc.sequence_order_coupling_number.shape))
        self.assertEqual(desc.quasi_sequence_order.shape, (self.num_seqs[0], 50), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 50, desc.quasi_sequence_order.shape))
        self.assertEqual(desc.pseudo_amino_acid_composition.shape, (self.num_seqs[0], 50), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 50, desc.pseudo_amino_acid_composition.shape))
        self.assertEqual(desc.amphiphilic_pseudo_amino_acid_composition.shape, (self.num_seqs[0], 80), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 80, desc.amphiphilic_pseudo_amino_acid_composition.shape))
        self.assertEqual(desc.all_descriptors.shape, (self.num_seqs[0], 9714), 
            'Attribute shape should be [{}, {}], got {}.'.format(self.num_seqs[0], 9714, desc.all_descriptors.shape))
#3.)
        #testing on remaining 3 datasets/config files that don't have a pre-calculated descriptors csv
        for config in range(1, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[config])

            #verify num_seqs descriptors attribute is correct
            self.assertEqual(desc.num_seqs, self.num_seqs[config], 
                'Expected {} number of sequences, got {}.'.format(self.num_seqs[config], desc.num_seqs))

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

    def test_descriptor_groups(self):
        """ Testing the descriptor groups dictionary which stores the specific group that a descriptor attribute is a member of. """
        #testing on all 4 datasets and config file
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[dataset])
#1.)
            self.assertEqual(list(desc.descriptor_groups.keys()), desc.all_descriptors_list(),
                "Descriptor groups list is incorrect, got:\n{}.".format(list(desc.descriptor_groups.keys())))
            self.assertEqual(list(desc.descriptor_groups.values()).count("Composition"), 3, 
                "Expected there to be 3 composition groups, got {}.".format(list(desc.descriptor_groups.values()).count("Composition")))
            self.assertEqual(list(desc.descriptor_groups.values()).count("Autocorrelation"), 3,
                "Expected there to be 3 autocorrelation groups, got {}.".format(list(desc.descriptor_groups.values()).count("Autocorrelation")))
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1,
                "Expected there to be 1 conjoint triad groups, got {}.".format(list(desc.descriptor_groups.values()).count("Conjoint Triad")))
            self.assertEqual(list(desc.descriptor_groups.values()).count("Sequence Order"), 2,
                "Expected there to be 2 sequence order groups, got {}.".format(list(desc.descriptor_groups.values()).count("Sequence Order")))
            self.assertEqual(list(desc.descriptor_groups.values()).count("CTD"), 4,
                "Expected there to be 4 CTD groups, got {}.".format(list(desc.descriptor_groups.values()).count("CTD")))
            self.assertEqual(list(desc.descriptor_groups.values()).count("Pseudo Composition"), 2,
                "Expected there to be 2 pseudo composition groups, got {}.".format(list(desc.descriptor_groups.values()).count("Pseudo Composition")))
            self.assertEqual(len(desc.descriptor_groups.keys()), len(desc.all_descriptors_list()),
                "Expected {} total descriptor groups, got {}.".format(len(desc.all_descriptors_list()), len(desc.descriptor_groups.keys())))
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
#3.)
            self.assertIsInstance(desc.descriptor_groups, dict, "Expected dict, got {}.".format(type(desc.descriptor_groups)))

    def test_all_descriptors_list(self):
        """ Testing function that returns various combinations of available descriptors using built-in itertools library. """
        #testing on all 4 datasets and config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(config_file=self.all_config_files[dataset])
            desc_list_1 = desc.all_descriptors_list(desc_combo=1)
            desc_list_2 = desc.all_descriptors_list(desc_combo=2)
            desc_list_3 = desc.all_descriptors_list(desc_combo=3)
#1.)
            self.assertEqual(len(desc_list_1), 15, "Expected 15 descriptor combinations, got {}.".format(len(desc_list_1)))
            self.assertEqual(len(desc_list_2), 105, "Expected 105 descriptor combinations, got {}.".format(len(desc_list_2)))
            self.assertEqual(len(desc_list_3), 455, "Expected 455 descriptor combinations, got {}.".format(len(desc_list_3)))
#2.)
            self.assertIsInstance(desc_list_1, list, "Expected list, got {}.".format(type(desc_list_1)))
            self.assertIsInstance(desc_list_2, list, "Expected list, got {}.".format(type(desc_list_2)))
            self.assertIsInstance(desc_list_3, list, "Expected list, got {}.".format(type(desc_list_3)))

    def test_valid_descriptors(self):
        """ Testing list of valid descriptors available in descriptors module. """
#1.)
        for config in self.all_config_files:
            desc = descr.Descriptors(config_file=config)
            valid_desc = desc.valid_descriptors
    
            self.assertEqual(len(valid_desc), 15, "Expected there to be 15 total descriptors, got {}.".format(len(valid_desc)))
            self.assertIsInstance(valid_desc, list, "Expected valid_desc to be a list, got {}.".format(type(valid_desc)))
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
#2.)
        with self.assertRaises(OSError):
            desc.import_descriptors("invalid_csv.csv")
            desc.import_descriptors("blahblahblah")
#3.)
        with self.assertRaises(TypeError):
            desc.import_descriptors(1234)
            desc.import_descriptors(False)

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
            self.assertEqual(aa_comp.shape, (self.num_seqs[dataset], 20), 'Descriptor not correct shape, got {}.'.format(aa_comp.shape)) 
            self.assertIsInstance(aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame, got {}.'.format(type(aa_comp)))
            self.assertTrue(aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(aa_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(aa_comp.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(aa_comp.dtypes)))
            self.assertEqual(self.amino_acids, list(aa_comp.columns), 
                'Incorrect column values found in output dataframe: {}.'.format(aa_comp.columns))

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
            self.assertEqual(dipeptide_comp.shape, (self.num_seqs[dataset], 400), 'Descriptor not correct shape, got {}.'.format(dipeptide_comp.shape))
            self.assertIsInstance(dipeptide_comp, pd.DataFrame, 'Descriptor should be of type DataFrame, got {}.'.format(type(dipeptide_comp)))
            self.assertTrue(dipeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(dipeptide_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(dipeptide_comp.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(dipeptide_comp.dtypes)))
            for col in list(dipeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, "Column contains an invalid amino acid {}.".format(col[0]))
                self.assertIn(col[1], self.amino_acids, "Column contains an invalid amino acid {}.".format(col[1]))

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
            self.assertEqual(tripeptide_comp.shape, (self.num_seqs[dataset], 8000), 'Descriptor not correct shape, got {}.'.format(tripeptide_comp.shape))
            self.assertIsInstance(tripeptide_comp, pd.DataFrame, 'Descriptor should be of type DataFrame, got {}.'.format(type(tripeptide_comp)))
            self.assertTrue(tripeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(tripeptide_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(tripeptide_comp.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(tripeptide_comp.dtypes)))
            for col in list(tripeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{3}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, "Column contains an invalid amino acid {}.".format(col[0]))
                self.assertIn(col[1], self.amino_acids, "Column contains an invalid amino acid {}.".format(col[1]))
                self.assertIn(col[2], self.amino_acids, "Column contains an invalid amino acid {}.".format(col[2]))

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
            self.assertEqual(moreau_broto.shape, (self.num_seqs[dataset], 240), 'Descriptor not correct shape, got {}.'.format(moreau_broto.shape))
            self.assertIsInstance(moreau_broto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(moreau_broto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(moreau_broto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(moreau_broto.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(moreau_broto.dtypes)))
            #check all columns follow pattern of MoreauBrotoAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(moreau_broto.columns):
                self.assertTrue(bool(re.match(r"MBAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))  

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
            self.assertEqual(moran_auto.shape, (self.num_seqs[dataset], 240), 'Descriptor not correct shape, got {}.'.format(moran_auto.shape))
            self.assertIsInstance(moran_auto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(moran_auto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(moran_auto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(moran_auto.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(moran_auto.dtypes)))
            #check all columns follow pattern of MoranAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(moran_auto.columns):
                self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))

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
            self.assertEqual(geary_auto.shape, (self.num_seqs[dataset], 240), 'Descriptor not correct shape, got {}.'.format(geary_auto.shape))
            self.assertIsInstance(geary_auto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(geary_auto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(geary_auto).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(geary_auto.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(geary_auto.dtypes)))
            #check all columns follow pattern of GAuto_X_Y where x is the asscession number of
            #the AAindex record and y is the count of the descriptor
            for col in list(geary_auto.columns):
                self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))
    
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
            self.assertEqual(ctd.shape, (self.num_seqs[dataset], 21), 'Descriptor not of correct, got {}.'.format(ctd.shape))
            self.assertIsInstance(ctd, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(ctd.dtypes)))
            #iterate over all columns and check its name follows expected format
            for col in list(ctd.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]_[0-9]{2}_[0-9]{3}_" + prop , col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
#2.)                
            #get descriptor values
            ctd_comp = desc.get_ctd_composition() 

            self.assertFalse(ctd_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_composition.equals(ctd_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_comp.shape, (self.num_seqs[dataset], 3), 'Descriptor not of correct, got {}.'.format(ctd_comp.shape))
            self.assertIsInstance(ctd_comp, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_comp.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(ctd_comp.dtypes)))
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_comp.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
#3.)
            #get descriptor values
            ctd_trans = desc.get_ctd_transition()

            self.assertFalse(ctd_trans.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_transition.equals(ctd_trans), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_trans.shape, (self.num_seqs[dataset], 3), 'Descriptor not of correct, got {}.'.format(ctd_trans.shape))
            self.assertIsInstance(ctd_trans, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_trans.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_trans).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_trans.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(ctd_trans.dtypes)))
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_trans.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
#4.)
            #get descriptor values
            ctd_distr = desc.get_ctd_distribution()

            self.assertFalse(ctd_distr.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(desc.ctd_distribution.equals(ctd_distr), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(ctd_distr.shape, (self.num_seqs[dataset], 15), 'Descriptor not of correct, got {}.'.format(ctd_distr.shape))
            self.assertIsInstance(ctd_distr, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_distr.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(ctd_distr).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_distr.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(ctd_distr.dtypes)))
            #iterate over all columns and check its name follows expected format
            for col in list(ctd_distr.columns):
                matching_col = False
                for prop in ctd_properties:
                    if (col.endswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                            (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{3}_" + prop, col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))

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
            self.assertEqual(conjoint_triad.shape, (self.num_seqs[dataset], 343), 'Descriptor not of correct shape, got {}.'.format(conjoint_triad.shape))
            self.assertIsInstance(conjoint_triad, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(conjoint_triad.any().isnull().sum()==0,'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(conjoint_triad).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.int64 for col in list(conjoint_triad.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(conjoint_triad.dtypes)))
            #iterate over all columns and check its name follows expected format
            for col in list(conjoint_triad.columns):
                self.assertTrue(bool(re.match(r"conj_triad_[0-9]{3}", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))   

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
            self.assertEqual(sequence_order_coupling_number.shape, (self.num_seqs[dataset], 30), 'Descriptor not of correct shape, got {}.'.format(sequence_order_coupling_number.shape))
            self.assertIsInstance(sequence_order_coupling_number, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(sequence_order_coupling_number.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(sequence_order_coupling_number).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(sequence_order_coupling_number.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(sequence_order_coupling_number.dtypes)))
            #check all columns follow pattern of SOCNX or SOCNXY where x & y integers between 0 and 9
            for col in list(sequence_order_coupling_number.columns):
                self.assertTrue((bool(re.match(r'SOCN_SW[0-9]', col)) or bool(re.match(r'SOCN_SW[0-9][0-9]', col))), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))   

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
            self.assertEqual(quasi_sequence_order.shape, (self.num_seqs[dataset], 50), 'Descriptor not of correct shape, got {}.'.format(quasi_sequence_order.shape))
            self.assertIsInstance(quasi_sequence_order, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(quasi_sequence_order.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(np.isinf(quasi_sequence_order).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
            self.assertTrue(all(col == np.float64 for col in list(quasi_sequence_order.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(quasi_sequence_order.dtypes)))
            #check all columns follow pattern of QSO_X, where x is an integer between 0 and 9
            for col in list(quasi_sequence_order.columns):
                self.assertTrue((bool(re.match(r'QSO_SW[0-9]', col))), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))

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
        self.assertEqual(pseudo_aa_comp.shape, (self.num_seqs[0], 50), 'Descriptor not of correct shape, got {}.'.format(pseudo_aa_comp.shape))
        self.assertIsInstance(pseudo_aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(pseudo_aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(pseudo_aa_comp).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(pseudo_aa_comp.dtypes)),
                "Column datatypes should be np.float64, got:\n{}.".format(list(pseudo_aa_comp.dtypes)))
        #check all columns follow pattern of PAACX, where x is an integer between 0 and 9
        for col in list(pseudo_aa_comp.columns):
            self.assertTrue(bool(re.match(r"PAAC[0-9]", col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))

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
        self.assertEqual(amphiphilic_pseudo_aac.shape, (self.num_seqs[1], 80), 'Descriptor not of correct shape, got {}.'.format(amphiphilic_pseudo_aac.shape))
        self.assertIsInstance(amphiphilic_pseudo_aac, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(amphiphilic_pseudo_aac.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(amphiphilic_pseudo_aac).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(amphiphilic_pseudo_aac.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(amphiphilic_pseudo_aac.dtypes)))
        #check all columns follow pattern of APAAC_X, where x is an integer between 0 and 9
        for col in list(amphiphilic_pseudo_aac.columns):
            self.assertTrue(bool(re.match(r"APAAC_[0-9]", col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))
        
    def test_get_all_descriptors(self):
        """ Testing functionality for calculating all protein descriptors for a datast of protein sequences. 
            Only testing on the thermostability dataset/config as its protein descriptors have been 
            pre-calcualted. Testing on the other datasets could take several hours each. """
#1.)        
        #only testing on thermostability dataset to access pre-calculated descriptors
        desc = descr.Descriptors(self.all_config_files[0])
        all_descriptors = desc.get_all_descriptors()

        self.assertIsInstance(all_descriptors, pd.DataFrame, 'Expected function output to be of type DataFrame, got {}.'.format(type(all_descriptors)))
        self.assertEqual(all_descriptors.shape, (261, 9714), "Expected shape of output to be 261 x 9714, got {}.".format(all_descriptors.shape))

    # @unittest.skip("Test case requires recalculating all descriptors which is redundant to the above tests") **
    def test_get_descriptor_encoding(self):
        """ Testing get_descriptor_encoding function by passing string of approximate descriptor names in to get encoding. """
        desc = descr.Descriptors(self.all_config_files[0]) #using thermostability config to access pre-calculated descriptors
#1.)    
        aa_comp_desc = desc.get_descriptor_encoding("amino_comp")

        self.assertIsInstance(aa_comp_desc, pd.DataFrame, 'Descriptor attribute should be a dataframe, got {}.'.format(type(aa_comp_desc)))
        self.assertEqual(aa_comp_desc.shape, (self.num_seqs[0], 20), 
            "Attribute shape should be ({}, {}), got {}.".format(self.num_seqs[0], 20, aa_comp_desc.shape))
        self.assertTrue(aa_comp_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(aa_comp_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(aa_comp_desc.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(aa_comp_desc.dtypes)))
        self.assertEqual(self.amino_acids, list(aa_comp_desc.columns), 
            'Incorrect column values found in output dataframe: {}.'.format(aa_comp_desc.columns))
#2.)
        geary_auto_desc = desc.get_descriptor_encoding("geary_auto")

        self.assertIsInstance(geary_auto_desc, pd.DataFrame, 'Descriptor attribute should be a dataframe, got {}.'.format(type(geary_auto_desc)))
        self.assertEqual(geary_auto_desc.shape, (self.num_seqs[0], 240), 
            "Attribute shape should be ({}, {}), got {}.".format(self.num_seqs[0], 240, geary_auto_desc.shape))
        self.assertTrue(geary_auto_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(geary_auto_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(geary_auto_desc.dtypes)), 
            "Column datatypes should be np.float64, got:\n{}.".format(list(geary_auto_desc.dtypes)))
        #check all columns follow pattern of GAuto_X_Y where x is the asscession number of
        #the AAindex record and y is the count of the descriptor
        for col in list(geary_auto_desc.columns):
            self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)), 
                "Column name doesn't match expected regex pattern: {}.".format(col))
#3.)
        socn_desc = desc.get_descriptor_encoding("sequence_order_coupling")

        self.assertIsInstance(socn_desc, pd.DataFrame, 'Descriptor attribute should be a dataframe, got {}.'.format(socn_desc))
        self.assertEqual(socn_desc.shape, (self.num_seqs[0], 30), 
            "Attribute shape should be ({}, {}), got {}.".format(self.num_seqs[0], 30, socn_desc.shape))
        self.assertTrue(socn_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(socn_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(socn_desc.dtypes)), 
            "Column datatypes should be np.float64, got:\n{}.".format(list(socn_desc.dtypes)))
        #check all columns follow pattern of SOCNX or SOCNXY where x & y integers between 0 and 9
        for col in list(socn_desc.columns):
            self.assertTrue((bool(re.match(r'SOCN_SW[0-9]', col)) or bool(re.match(r'SOCN_SW[0-9][0-9]', col))), 
                "Column name doesn't match expected regex pattern: {}.".format(col))   
#4.)
        dipeptide_comp_desc = desc.get_descriptor_encoding("dipeptide")
        
        self.assertIsInstance(dipeptide_comp_desc, pd.DataFrame, 'Descriptor attribute should be a dataframe, got {}.'.format(dipeptide_comp_desc))
        self.assertEqual(dipeptide_comp_desc.shape, (self.num_seqs[0], 400), 
            "Attribute shape should be ({}, {}), got {}.".format(self.num_seqs[0], 400, dipeptide_comp_desc.shape))
        self.assertTrue(dipeptide_comp_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(dipeptide_comp_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(dipeptide_comp_desc.dtypes)),
            "Column datatypes should be np.float64, got:\n{}.".format(list(dipeptide_comp_desc.dtypes)))
        for col in list(dipeptide_comp_desc.columns):
            #check all columns follow pattern of XY where x & y are amino acids 
            self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "")      
            self.assertIn(col[0], self.amino_acids, "Column contains invalid amino acid: {}.".format(col[0]))
            self.assertIn(col[1], self.amino_acids, "Column contains invalid amino acid: {}.".format(col[1]))
#5.)
        ctd_transition_desc = desc.get_descriptor_encoding("ctd_transition")
        ctd_properties = ["hydrophobicity", "normalized_vdwv", "polarity", "charge",
            "secondary_struct", "solvent_accessibility", "polarizability"]
        
        self.assertIsInstance(ctd_transition_desc, pd.DataFrame, 'Descriptor attribute should be a dataframe, got {}.'.format(ctd_transition_desc))
        self.assertEqual(ctd_transition_desc.shape, (self.num_seqs[0], 3), 
            "Attribute shape should be ({}, {}), got {}.".format(self.num_seqs[0], 3, ctd_transition_desc.shape))
        self.assertTrue(ctd_transition_desc.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(np.isinf(ctd_transition_desc).values.sum()==0, 'Descriptor should not contain any +/- infinity values.')
        self.assertTrue(all(col == np.float64 for col in list(ctd_transition_desc.dtypes)), 
                "Column datatypes should be np.float64, got:\n{}.".format(list(ctd_transition_desc.dtypes)))
        #check all columns follow correct format
        for col in list(ctd_transition_desc.columns):
            matching_col = False
            for prop in ctd_properties:
                if (col.endswith(prop)):
                    matching_col = True
                    self.assertTrue(((bool(re.search(r"CTD_[A-Z]_[0-9]{2}_" + prop, col))) or \
                        (bool(re.search(r"CTD_[A-Z]{1}_[0-9]{2}_[0-9]{2}_" + prop, col)))), 
                            "Column name does not follow expected format: {}.".format(col))
            self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
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