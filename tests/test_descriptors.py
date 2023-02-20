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
from pySAR.descriptors import *

class DescriptorTests(unittest.TestCase):
    """
    Test suite for testing Descriptors module and functionality 
    in pySAR package. 

    Test Cases
    ----------
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

        #array of the total number of protein seqs per dataset
        self.num_seqs = [261, 152, 81, 254]

        #list of canonical amino acids
        self.amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", 
            "Q", "R", "S", "T", "V", "W", "Y"]

    def test_descriptor(self):
        """ Test descriptor initialisation process. Verify the initial input parameters
            and descriptor attributes are correct. """
        #testing on all 4 datasets/config files
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(desc_config=self.all_config_files[dataset])
#1.)
            #verify num_seqs descriptors attribute is correct
            self.assertEqual(desc.num_seqs, self.num_seqs[dataset], 
                'Expected {} number of sequences, got {}.'.format(self.num_seqs[dataset], desc.num_seqs))

            #verify that all input sequences dont have any gaps/missing amino acids
            for seq in desc.protein_seqs:
                self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#2.)
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
#3.)
            #test Value and Type error exceptions are thrown if invalid parameters input
            with self.assertRaises(TypeError, msg='Type Error raised, incorrect datatype input to class.'):
                fail_desc = descr.Descriptors(desc_config=123)

            with self.assertRaises(TypeError, msg='Type Error raised, incorrect datatype input to class.'):
                fail_desc = descr.Descriptors(desc_config=None)

            with self.assertRaises(OSError, msg='OS Error raised, filepath to config file not found.'):
                fail_desc = descr.Descriptors(desc_config="incorrect_filepath.json")

            with self.assertRaises(OSError, msg='OS Error raised, filepath to config file not found.'):
                fail_desc = descr.Descriptors(desc_config="")

    def test_descriptor_groups(self):
        """ Testing the descriptor groups dictionary which stores the specific group
        that a descriptor attribute is a member of. """
        #testing on all 4 datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(desc_config=self.all_config_files[dataset])
#1.)
            self.assertEqual(list(desc.descriptor_groups.keys()), desc.all_descriptors_list())
            self.assertEqual(list(desc.descriptor_groups.values()).count("Composition"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Autocorrelation"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Sequence Order"), 2)
            self.assertEqual(list(desc.descriptor_groups.values()).count("CTD"), 4)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Pseudo Composition"), 2)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1)
            self.assertEqual(len(desc.descriptor_groups.keys()), len(desc.all_descriptors_list()))
#2.)
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
        """ Testing function that returns various combinations of available descriptors 
            using built-in itertools library. """
        #testing on all 4 datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(desc_config=self.all_config_files[dataset])
            desc_list_1 = desc.all_descriptors_list(desc_combo=1)
            desc_list_2 = desc.all_descriptors_list(desc_combo=2)
            desc_list_3 = desc.all_descriptors_list(desc_combo=3)
#1.)
            self.assertEqual(len(desc_list_1), 15, 
                "Expected 15 descriptor combinations, got {}.".format(len(desc_list_1)))
            self.assertEqual(len(desc_list_2), 105,
                "Expected 105 descriptor combinations, got {}.".format(len(desc_list_2)))
            self.assertEqual(len(desc_list_3), 455,
                "Expected 455 descriptor combinations, got {}.".format(len(desc_list_3)))
#2.)
            self.assertIsInstance(desc_list_1, list, "Expected list, got {}.".format(type(desc_list_1)))
            self.assertIsInstance(desc_list_2, list, "Expected list, got {}.".format(type(desc_list_2)))
            self.assertIsInstance(desc_list_3, list, "Expected list, got {}.".format(type(desc_list_3)))

    def test_valid_descriptors(self):
        """ Testing function that returns the list of valid descriptors available in descriptors module. """
        desc = descr.Descriptors(desc_config=self.all_config_files[0])
        valid_desc = desc.valid_descriptors()
#1.)
        self.assertEqual(len(valid_desc), 15, "Expected there to be 15 total descriptors, got {}.".format(len(valid_desc)))
        self.assertIsInstance(valid_desc, list)
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

    def test_amino_acid_composition(self):
        """ Testing Amino Acid Composition protein descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            aa_comp = desc.get_amino_acid_composition()
#1.)
            self.assertFalse(aa_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(desc.amino_acid_composition.equals(aa_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aa_comp.shape, (self.num_seqs[dataset], 20), 'Descriptor not of correct shape.') 
            self.assertIsInstance(aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertEqual(self.amino_acids, list(aa_comp.columns), 'Incorrect column values found in output dataframe: {}.'.format(aa_comp.columns))
            self.assertTrue(aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(aa_comp.dtypes)), "Column datatypes should be np.float64.")

    def test_dipeptide_composition(self):
        """ Testing Dipeptide Composition protein descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            dipeptide_comp = desc.get_dipeptide_composition()
#1.)
            self.assertTrue(desc.dipeptide_composition.equals(dipeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertFalse(dipeptide_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(dipeptide_comp.shape, (self.num_seqs[dataset], 400), 'Descriptor not of correct shape ({}, 400).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(dipeptide_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(dipeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(dipeptide_comp.dtypes)), "Column datatypes should be np.float64.")
            for col in list(dipeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{2}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, "")
                self.assertIn(col[1], self.amino_acids, "")

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_tripeptide_composition(self):
        """ Testing Tripeptide Composition protein descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            tripeptide_comp = desc.get_tripeptide_composition()
#1.)
            self.assertFalse(tripeptide_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(tripeptide_comp.shape, (self.num_seqs[dataset], 8000), 'Descriptor not of correct ({}, 8000).'.format(self.num_seqs[dataset]))
            self.assertTrue(desc.tripeptide_composition.equals(tripeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertIsInstance(tripeptide_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(tripeptide_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(tripeptide_comp.dtypes)), "Column datatypes should be np.float64.")
            for col in list(tripeptide_comp.columns):
                #check all columns follow pattern of XY where x & y are amino acids 
                self.assertTrue(bool(re.match(r'^[A-Z]{3}$', col)), "")      
                self.assertIn(col[0], self.amino_acids, "")
                self.assertIn(col[1], self.amino_acids, "")
                self.assertIn(col[2], self.amino_acids, "")

    def test_moreaubroto_autocorrelation(self):
        """ Testing moreaubroto autocorrelation descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            moreaubroto = desc.get_moreaubroto_autocorrelation()
#1.)
            self.assertFalse(moreaubroto.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(moreaubroto.shape, (self.num_seqs[dataset], 240), 'Descriptor not of correct ({}, 240)'.format(self.num_seqs[dataset]))
            self.assertIsInstance(moreaubroto, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(moreaubroto.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(moreaubroto.dtypes)), "Column datatypes should be np.float64.")

            #check all columns follow pattern of MoreauBrotoAuto_X_Y where x is the asscession number of
            #   the AAindex record and y is the count of the descriptor
            for col in list(moreaubroto.columns):
                self.assertTrue(bool(re.match(r"MBAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))  

    def test_moran_autocorrelation(self):
        """ Testing Moran autocorrelation descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
            
            #get descriptor values
            moran = desc.get_moran_autocorrelation()
#1.)
            self.assertEqual(moran.shape, (self.num_seqs[dataset], 240), 'Descriptor not of correct ({}, 240).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(moran, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertFalse(moran.empty, 'Descriptor dataframe should not be empty.')
            self.assertTrue(moran.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(moran.dtypes)), "Column datatypes should be np.float64.")

            #check all columns follow pattern of MoranAuto_X_Y where x is the asscession number of
            #   the AAindex record and y is the count of the descriptor
            for col in list(moran.columns):
                self.assertTrue(bool(re.match(r"MAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))

    def test_geary_autocorrelation(self):
        """ Testing Geary autocorrelation descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            geary = desc.get_geary_autocorrelation()
#1.)
            self.assertFalse(geary.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(geary.shape, (self.num_seqs[dataset], 240), 'Descriptor not of correct ({}, 240).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(geary, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(geary.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(geary.dtypes)), "Column datatypes should be np.float64.")

            #check all columns follow pattern of GearyAuto_X_Y where x is the asscession number of
            #   the AAindex record and y is the count of the descriptor
            for col in list(geary.columns):
                self.assertTrue(bool(re.match(r"GAuto_[A-Z0-9]{10}_[0-9]", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))
    
    def test_ctd(self):
        """ Testing CTD descriptor attributes and methods. """
        properties = ["hydrophobicity", "normalized_vdwv", "polarity", "charge",
            "secondary_struct", "solvent_accessibility", "polarizability"]
        
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            ctd = desc.get_ctd()
#1.)
            self.assertFalse(ctd.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(ctd.shape, (self.num_seqs[dataset], 147), 'Descriptor not of correct ({}, 147).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(ctd, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd.dtypes)), "Column datatypes should be np.float64.")

            #iterate over all columns and check its name follows expected format
            for col in list(ctd.columns):
                matching_col = False
                for prop in properties:
                    if (col.startswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(prop + r"_CTD_[A-Z]_[0-9][0-9]", col))) or \
                            (bool(re.search(prop + r"_CTD_[A-Z]{1}_[0-9][0-9]_[0-9][0-9]", col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
                
            #get descriptor values
            ctd_comp = desc.get_ctd_composition() 
#2.)
            self.assertFalse(ctd_comp.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(ctd_comp.shape, (self.num_seqs[dataset], 3), 'Descriptor not of correct ({}, 3).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(ctd_comp, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_comp.dtypes)), "Column datatypes should be np.float64.")

            #iterate over all columns and check its name follows expected format
            for col in list(ctd_comp.columns):
                matching_col = False
                for prop in properties:
                    if (col.startswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(prop + r"_CTD_[A-Z]_[0-9][0-9]", col))) or \
                            (bool(re.search(prop + r"_CTD_[A-Z]{1}_[0-9][0-9]_[0-9][0-9]", col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))

            #get descriptor values
            ctd_trans = desc.get_ctd_transition()
#3.)
            self.assertFalse(ctd_trans.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(ctd_trans.shape, (self.num_seqs[dataset], 3), 'Descriptor not of correct ({}, 3).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(ctd_trans, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_trans.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_trans.dtypes)), "Column datatypes should be np.float64.")

            #iterate over all columns and check its name follows expected format
            for col in list(ctd_trans.columns):
                matching_col = False
                for prop in properties:
                    if (col.startswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(prop + r"_CTD_[A-Z]_[0-9][0-9]", col))) or \
                            (bool(re.search(prop + r"_CTD_[A-Z]{1}_[0-9][0-9]_[0-9][0-9]", col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))
#4.)
            #get descriptor values
            ctd_distr = desc.get_ctd_distribution()

            self.assertFalse(ctd_distr.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(ctd_distr.shape, (self.num_seqs[dataset], 15), 'Descriptor not of correct ({}, 15).'.format(self.num_seqs[dataset]))
            self.assertIsInstance(ctd_distr, pd.DataFrame, "Descriptor should be of type DataFrame.")
            self.assertTrue(ctd_distr.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(ctd_distr.dtypes)), "Column datatypes should be np.float64.")

            #iterate over all columns and check its name follows expected format
            for col in list(ctd_distr.columns):
                matching_col = False
                for prop in properties:
                    if (col.startswith(prop)):
                        matching_col = True
                        self.assertTrue(((bool(re.search(prop + r"_CTD_[A-Z]_[0-9][0-9]", col))) or \
                            (bool(re.search(prop + r"_CTD_[A-Z]{1}_[0-9][0-9]_[0-9][0-9]", col)))), 
                                "Column name does not follow expected format: {}.".format(col))
                self.assertTrue(matching_col, "Column name's property name not found and doesn't match format: {}.".format(col))

    def test_conjoint_triad(self):
        """ Testing Conjoint Triad descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            conjoint_triad = desc.get_conjoint_triad()
#1.)
            self.assertFalse(conjoint_triad.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(conjoint_triad.shape, (self.num_seqs[dataset], 343), 'Descriptor not of correct shape (1, 343).')
            self.assertIsInstance(conjoint_triad, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(conjoint_triad.any().isnull().sum()==0,'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.int64 for col in list(conjoint_triad.dtypes)), "Column datatypes should be np.float64.")

            #iterate over all columns and check its name follows expected format
            for col in list(conjoint_triad.columns):
                self.assertTrue(bool(re.match(r"[0-9]{3}", col)), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))   

    def test_sequence_order_coupling_number(self):
        """ Testing sequence order coupling number descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            sequence_order_coupling_number = desc.get_sequence_order_coupling_number()
#1.)
            self.assertFalse(sequence_order_coupling_number.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(sequence_order_coupling_number.shape, (self.num_seqs[dataset], 30), 'Descriptor not of correct shape (1, 30).')
            self.assertIsInstance(sequence_order_coupling_number, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(sequence_order_coupling_number.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(sequence_order_coupling_number.dtypes)), "Column datatypes should be np.float64.")

            #check all columns follow pattern of SOCNX or SOCNXY where x & y integers between 0 and 9
            for col in list(sequence_order_coupling_number.columns):
                self.assertTrue((bool(re.match(r'SOCN_SW[0-9]', col)) or bool(re.match(r'SOCN_SW[0-9][0-9]', col))), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))   

    def test_quasi_sequence_order(self):
        """ Testing Quasi sequence order descriptor attributes and methods. """
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get descriptor values
            quasi_sequence_order = desc.get_quasi_sequence_order()
#1.)
            self.assertFalse(quasi_sequence_order.empty, 'Descriptor dataframe should not be empty.')
            self.assertEqual(quasi_sequence_order.shape, (self.num_seqs[dataset], 50), 'Descriptor not of correct shape (1, 100).')
            self.assertIsInstance(quasi_sequence_order, pd.DataFrame, 'Descriptor should be of type DataFrame.')
            self.assertTrue(quasi_sequence_order.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
            self.assertTrue(all(col == np.float64 for col in list(quasi_sequence_order.dtypes)), "Column datatypes should be np.float64.")

            #check all columns follow pattern of QSO_X, where x is an integer between 0 and 9
            for col in list(quasi_sequence_order.columns):
                self.assertTrue((bool(re.match(r'QSO_SW[0-9]', col))), 
                    "Column name doesn't match expected regex pattern: {}.".format(col))

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping.")
    def test_pseudo_amino_acid_composition(self):
        """ Testing Pseudo Amino Acid Composition descriptor attributes and methods. """
#1.)
        #running unit test on one of the datasets due to length of computation
        desc = descr.Descriptors(self.all_config_files[0])

        #get descriptor values
        pseudo_aa_comp = desc.get_pseudo_amino_acid_composition()

        self.assertFalse(pseudo_aa_comp.empty, 'Descriptor dataframe should not be empty.')
        self.assertEqual(pseudo_aa_comp.shape, (self.num_seqs[0], 50), 'Descriptor not of correct shape (1,50).')
        self.assertIsInstance(pseudo_aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(pseudo_aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(all(col == np.float64 for col in list(pseudo_aa_comp.dtypes)), "Column datatypes should be np.float64.")

        #check all columns follow pattern of PAACX, where x is an integer between 0 and 9
        for col in list(pseudo_aa_comp.columns):
            self.assertTrue(bool(re.match(r"PAAC[0-9]", col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))
#2.)
        desc = descr.Descriptors(self.all_config_files[1])

        #get descriptor values
        pseudo_aa_comp = desc.get_pseudo_amino_acid_composition()

        self.assertFalse(pseudo_aa_comp.empty, 'Descriptor dataframe should not be empty.')
        self.assertEqual(pseudo_aa_comp.shape, (self.num_seqs[0], 50), 'Descriptor not of correct shape (1,50).')
        self.assertIsInstance(pseudo_aa_comp, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(pseudo_aa_comp.any().isnull().sum()==0, 'Descriptor should not contain any null values.')
        self.assertTrue(all(col == np.float64 for col in list(pseudo_aa_comp.dtypes)), "Column datatypes should be np.float64.")

        #check all columns follow pattern of PAACX, where x is an integer between 0 and 9
        for col in list(pseudo_aa_comp.columns):
            self.assertTrue(bool(re.match(r"PAAC[0-9]", col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping.")
    def test_amphiphilic_pseudo_amino_acid_composition(self):
        """ Testing Amphiphilic Pseudo Amino Acid Composition descriptor attributes and methods. """
        #running unit test on one of the datasets due to length of computation
        desc = descr.Descriptors(self.all_config_files[1])

        #get descriptor values
        amphiphilic_pseudo_aac = desc.get_amphiphilic_pseudo_amino_acid_composition()
#1.)
        self.assertFalse(amphiphilic_pseudo_aac.empty, 'Descriptor dataframe should not be empty.')
        self.assertEqual(amphiphilic_pseudo_aac.shape, (self.num_seqs[1], 80), 'Descriptor not of correct shape (1, 80).')
        self.assertIsInstance(amphiphilic_pseudo_aac, pd.DataFrame, 'Descriptor should be of type DataFrame.')
        self.assertTrue(amphiphilic_pseudo_aac.any().isnull().sum()==0,'Descriptor should not contain any null values.')
        self.assertTrue(all(col == np.float64 for col in list(amphiphilic_pseudo_aac.dtypes)), "Column datatypes should be np.float64.")

        #check all columns follow pattern of APAAC_X, where x is an integer between 0 and 9
        for col in list(amphiphilic_pseudo_aac.columns):
            self.assertTrue(bool(re.match(r"APAAC_[0-9]", col)), 
                "Column doesn't follow correct naming convention: {}.".format(col))
                
    @unittest.skip("Test case requires recalculating all descriptors which is redundant to the above tests")
    def test_get_descriptor_encoding(self):
        """ Testing get_descriptor_encoding function. """

        valid_desc = [
            'amino_acid_composition', 'dipeptide_composition', \
            'moreaubroto_autocorrelation', 'moran_autocorrelation', 'geary_autocorrelation', \
            'ctd', 'ctd_composition', 'ctd_transition', 'ctd_distribution', 'conjoint_triad', \
            'sequence_order_coupling_number', 'tripeptide_composition', 'sequence_order_coupling_number', \
            'quasi_sequence_order', 'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition'
        ]
#1.)
        #run tests on all test datasets
        for dataset in range(0, len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #test each descriptor is calculated when string of its name passed to func
            for val_desc in range(0, len(valid_desc)):

                encoded_desc = desc.get_descriptor_encoding(valid_desc[val_desc])
                self.assertIsNotNone(encoded_desc, 'Descriptor attribute should not be none.')
                self.assertFalse(encoded_desc.empty, 'Descriptor attribute should not be empty.')
                self.assertTrue(encoded_desc.shape[0], len(desc.protein_seqs), 
                    '1st dimension of descriptor attribute should equal num of protein seqs.')

    def tearDown(self):
        """ Cleanup tests and delete datasets/config files. """
        del self.all_config_files
