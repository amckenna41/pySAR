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

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
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
            self.assertEqual(desc.num_seqs, len(self.all_config_files[dataset]['sequence']),
                'num_seqs attribute not equal to the number of sequences found.')

            #verify that all input sequences dont have any gaps/missing amino acids
            for seq in desc.protein_seqs:
                self.assertNotIn('-', seq, 'There should be no gaps (-) in the sequences.')
#2.)
            print(desc.dipeptide_composition)
            #verify all descriptor attributes are initialised to empty dataframes
            self.assertTrue(desc.aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.dipeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.tripeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.normalized_moreaubroto_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.moran_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.geary_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.ctd.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.transition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.distribution.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.conjoint_triad.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.seq_order_coupling_number.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.quasi_seq_order.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.pseudo_aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.amp_pseudo_aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
            self.assertTrue(desc.all_descriptors.empty, 'Attribute should be initialised to an empty dataframe')
#3.)
            invalid_testdata1_copy = self.test_dataset1

            #test Value and Type error exceptions are thrown if invalid parameters input
            with self.assertRaises(TypeError):
                fail_desc = descr.Descriptors(desc_config=123)

            with self.assertRaises(TypeError):
                fail_desc = descr.Descriptors(desc_config=None)

            with self.assertRaises(OSError):
                fail_desc = descr.Descriptors(desc_config="incorrect_filepath.json")

            with self.assertRaises(ValueError):
                fail_desc = descr.Descriptors(desc_config="")

            with self.assertRaises(ValueError):
                fail_desc = descr.Descriptors(protein_seqs=invalid_testdata1_copy)

            #J,O,U and Z are all invalid amino acid values
            invalid_testdata1_copy['sequence'][0] = 'JOUZ'

            #descriptor module should raise ValueError if invalid sequence found in dataset
            with self.assertRaises(ValueError):
                fail_desc = descr.Descriptors(invalid_testdata1_copy['sequence'])


    # @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
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
