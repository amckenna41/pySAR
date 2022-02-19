################################################################################
#################            Descriptors Module Tests          #################
################################################################################

import pandas as pd
import numpy as np
import os
import sys
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import random
import json
from json import JSONDecodeError

import pySAR.descriptors_ as descr
import pySAR.descriptors.composition as descriptor_comp
from pySAR.descriptors import *
import pySAR.utils as utils

class DescriptorTests(unittest.TestCase):

    def setUp(self):
        """ 
        Import the 4 config files for each of the 4 datasets used for testing the descriptor methods. 
        """        
        #array of config files for each test dataset
        config_path = os.path.join('tests','test_config')
        self.all_config_files = [
            os.path.join(config_path, "test_thermostability.json"), 
            os.path.join(config_path, "test_enantioselectivity.json"),
            os.path.join(config_path, "test_absorption.json"), 
            os.path.join(config_path, "test_localization.json")
        ]

        #array of the total number of protein seqs per dataset
        self.num_seqs = [261, 152, 81, 254]

    def test_descriptor(self):
        """ 
        Test descriptor initialisation process. Verify the initial input parameters
        and descriptor attributes are correct. 
        """
        #testing on all 4 datasets
        # for dataset in range(0,len(self.all_test_datasets)):
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(desc_config=self.all_config_files[dataset])
#1.)
            #verify num_seqs descriptors attribute is correct
            # self.assertEqual(desc.num_seqs, len(self.test_dataset1['sequence']),
            self.assertEqual(desc.num_seqs, self.num_seqs[dataset],
                'num_seqs attribute not equal to the number of sequences found.')

            #verify that all input sequences dont have any gaps/missing amino acids
            for seq in desc.protein_seqs:
                self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#2.)
            #verify all descriptor attributes are initialised to empty dataframes
            self.assertTrue(desc.aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.dipeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.tripeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.normalized_moreaubroto_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.moran_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.geary_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.ctd.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.comp.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.transition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.distribution.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.conjoint_triad.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.seq_order_coupling_number.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.quasi_seq_order.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.pseudo_aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.amp_pseudo_aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.all_descriptors.empty, 'Attribute should be initialised to an empty dataframe')
#3.)
            #test Value and Type error exceptions are thrown if invalid parameters input
            with self.assertRaises(TypeError):
                fail_desc = descr.Descriptors(desc_config=123)

            with self.assertRaises(TypeError):
                fail_desc = descr.Descriptors(desc_config=None)

            with self.assertRaises(OSError):
                fail_desc = descr.Descriptors(desc_config="incorrect_filepath.json")

            with self.assertRaises(OSError):
                fail_desc = descr.Descriptors(desc_config="")

    def test_descriptor_groups(self):
        """ Testing the descriptor groups dictionary which stores the specific group
        that a descriptor attribute is a member of. """

        #testing on all 4 datasets
        # for dataset in range(0,len(self.all_test_datasets)):
        for dataset in range(0,len(self.all_config_files)):
            # desc = descr.Descriptors(self.all_configs[dataset])
            desc = descr.Descriptors(desc_config=self.all_config_files[dataset])
#1.)
            self.assertEqual(list(desc.descriptor_groups.keys()), desc.all_descriptors_list())
            self.assertEqual(list(desc.descriptor_groups.values()).count("Composition"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Autocorrelation"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Quasi-Sequence-Order"), 2)
            self.assertEqual(list(desc.descriptor_groups.values()).count("CTD"), 4)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Pseudo Composition"), 2)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1)
            self.assertEqual(len(desc.descriptor_groups.keys()), len(desc.all_descriptors_list()))
#2.)
            self.assertEqual(desc.descriptor_groups['aa_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['dipeptide_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['moran_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['geary_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['comp'], "CTD")
            self.assertEqual(desc.descriptor_groups['distribution'], "CTD")
            self.assertEqual(desc.descriptor_groups['transition'], "CTD")
            self.assertEqual(desc.descriptor_groups['conjoint_triad'], "Conjoint Triad")
            self.assertEqual(desc.descriptor_groups['pseudo_aa_composition'], "Pseudo Composition")
            self.assertEqual(desc.descriptor_groups['quasi_seq_order'], "Quasi-Sequence-Order")
            self.assertEqual(desc.descriptor_groups['amp_pseudo_aa_composition'], "Pseudo Composition")
#3.)
            self.assertIsInstance(desc.descriptor_groups, dict)

    def test_aa_composition(self):
        """ Testing Amino Acid Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            aa_comp = desc.get_aa_composition()

            self.assertFalse(aa_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(desc.aa_composition.equals(aa_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertEqual(aa_comp.shape, (self.num_seqs[dataset],20), 'Descriptor not of correct shape.') 
            self.assertIsInstance(aa_comp, pd.DataFrame, 'Descriptor not of type DataFrame.')
            self.assertEqual(descriptor_comp.aminoAcids, list(aa_comp.columns), 'Incorrect column values found.')
            self.assertTrue(aa_comp.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    def test_dipeptide_composition(self):
        """ Testing Dipeptide Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            dipeptide_comp = desc.get_dipeptide_composition()

            self.assertTrue(desc.dipeptide_composition.equals(dipeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertFalse(dipeptide_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(dipeptide_comp.shape, (self.num_seqs[dataset],400),'Descriptor not of correct shape.')
            self.assertIsInstance(dipeptide_comp, pd.DataFrame,'Descriptor not of type DataFrame.')
            self.assertTrue(dipeptide_comp.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_tripeptide_composition(self):
        """ Testing Tripeptide Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            tripeptide_comp = desc.get_tripeptide_composition()

            self.assertFalse(tripeptide_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(tripeptide_comp.shape, (self.num_seqs[dataset],8000),'Descriptor not of correct shape.')
            self.assertTrue(desc.tripeptide_composition.equals(tripeptide_comp), 'Output dataframe and class attribute dataframes must be the same.')
            self.assertIsInstance(tripeptide_comp, pd.DataFrame,'Descriptor not of type DataFrame.')
            self.assertTrue(tripeptide_comp.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    def test_normalized_moreaubroto_autocorrelation(self):
        """ Testing normalized moreaubroto autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            norm_moreaubroto = desc.get_norm_moreaubroto_autocorrelation()

            self.assertFalse(norm_moreaubroto.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(norm_moreaubroto.shape, (self.num_seqs[dataset],240))
            self.assertIsInstance(norm_moreaubroto, pd.DataFrame)
            self.assertTrue(norm_moreaubroto.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    def test_moran_autocorrelation(self):
        """ Testing Moran autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
            
            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            moran = desc.get_moran_autocorrelation()

            self.assertEqual(moran.shape, (self.num_seqs[dataset],240))
            self.assertIsInstance(moran, pd.DataFrame)
            self.assertFalse(moran.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(moran.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    def test_geary_autocorrelation(self):
        """ Testing Geary autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            geary = desc.get_geary_autocorrelation()

            self.assertFalse(geary.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(geary.shape, (self.num_seqs[dataset],240))
            self.assertIsInstance(geary, pd.DataFrame)
            self.assertTrue(geary.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    def test_ctd(self):
        """ Testing CTD descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            ctd = desc.get_ctd()

            self.assertFalse(ctd.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(ctd.shape, (self.num_seqs[dataset],147))
            self.assertIsInstance(ctd, pd.DataFrame)
            self.assertTrue(ctd.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    def test_conjoint_triad(self):
        """ Testing Conjoint Triad descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            # desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            conjoint_triad = desc.get_conjoint_triad()

            self.assertFalse(conjoint_triad.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(conjoint_triad.shape, (self.num_seqs[dataset],343), 'Descriptor not of correct shape (1,343)')
            self.assertIsInstance(conjoint_triad, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertTrue(conjoint_triad.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    def test_seq_order_coupling_number(self):
        """ Testing sequence order coupling number descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            seq_order_coupling_number = desc.get_seq_order_coupling_number()

            self.assertFalse(seq_order_coupling_number.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(seq_order_coupling_number.shape, (self.num_seqs[dataset],30), 'Descriptor not of correct shape (1,60)')
            self.assertIsInstance(seq_order_coupling_number, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertTrue(seq_order_coupling_number.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    def test_quasi_seq_order(self):
        """ Testing Quasi sequence order descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            quasi_seq_order = desc.get_quasi_seq_order()

            self.assertFalse(quasi_seq_order.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(quasi_seq_order.shape, (self.num_seqs[dataset],30), 'Descriptor not of correct shape (1,100)')
            self.assertIsInstance(quasi_seq_order, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertTrue(quasi_seq_order.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_pseudo_AAC(self):
        """ Testing Pseudo Amino Acid Composition descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            pseudo_AAC = desc.get_pseudo_aa_composition()

            self.assertFalse(pseudo_AAC.empty, 'Descriptor dataframe should not be empty')
            self.assertEqual(pseudo_AAC.shape, (self.num_seqs[dataset],50), 'Descriptor not of correct shape (1,50)')
            self.assertIsInstance(pseudo_AAC, pd.DataFrame, 'Descriptor not of type DataFrame')
            self.assertTrue(pseudo_AAC.any().isnull().sum()== 0, 'Descriptor should not contain any null values.')

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_amp_pseudo_AAC(self):
        """ Testing Amphiphilic Pseudo Amino Acid Composition descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):
            desc = descr.Descriptors(self.all_config_files[dataset])
            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            amp_pseudo_AAC = desc.get_amp_pseudo_aa_composition()

            self.assertFalse(amp_pseudo_AAC.empty)
            self.assertEqual(amp_pseudo_AAC.shape, (self.num_seqs[dataset],80), 'Descriptor not of correct shape (1,80)')
            self.assertIsInstance(amp_pseudo_AAC, pd.DataFrame)
            self.assertTrue(amp_pseudo_AAC.any().isnull().sum()== 0,'Descriptor should not contain any null values.')

    @unittest.skip("Test case requires recalculating all descriptors which is redundant to the above tests")
    def test_get_descriptor_encoding(self):
        """ Testing get_descriptor_encoding function.  """

        valid_desc = [
            'aa_composition', 'dipeptide_composition', \
            'normalized_moreaubroto_autocorrelation', 'moran_autocorrelation', 'geary_autocorrelation', \
            'ctd', 'comp', 'transition', 'distribution', 'conjoint_triad', \
            'seq_order_coupling_number'
            # 'tripeptide_composition', 'seq_order_coupling_number', 'quasi_seq_order', 'pseudo_aa_composition'
        ]
#1.)
        #run tests on all test datasets
        for dataset in range(0,len(self.all_config_files)):

            #get random sequence in dataset to calculate descriptor values for
            # random_seq = random.randint(0,len(desc.protein_seqs)-1)
            desc = descr.Descriptors(self.all_config_files[dataset])

            #test each descriptor is calculated when string of its name passed to func
            for val_desc in range(0,len(valid_desc)):

                encoded_desc = desc.get_descriptor_encoding(valid_desc[val_desc])
                self.assertIsNotNone(encoded_desc)
                self.assertFalse(encoded_desc.empty)
                self.assertTrue(encoded_desc.shape[0],1)

    def tearDown(self):
        """ Cleanup tests and delete datasets/config files. """
        del self.all_config_files
