################################################################################
#################            Descriptors Module Tests          #################
################################################################################

import pandas as pd
import numpy as np
import os
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import random
import json
from json import JSONDecodeError

import pySAR.descriptors_ as descr
from pySAR.descriptors import *
import pySAR.utils as utils

class DescriptorTests(unittest.TestCase):

    def setUp(self):
        """ 
        Import the 4 test datasets and 4 config files used for testing the descriptor methods. 
        """
        #import datasets
        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests','test_data','test_thermostability.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1.')
        try:
            self.test_dataset2 = pd.read_csv(os.path.join('tests','test_data','test_enantioselectivity.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset2.')
        try:
            self.test_dataset3 = pd.read_csv(os.path.join('tests','test_data','test_localization.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset3.')
        try:
            self.test_dataset4 = pd.read_csv(os.path.join('tests','test_data','test_absorption.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset4.')

        #append all datasets to a list
        self.all_test_datasets = [self.test_dataset1, self.test_dataset2, self.test_dataset3,
                self.test_dataset4]

        #import config files
        config_filepath = os.path.join('tests','test_config')
        try:
            with open(os.path.join(config_filepath,'test_thermostability.json')) as f:
                self.test_params1 = json.load(f)
        except JSONDecodeError as e:
            print('Error getting config JSON file: {}.'.format(config_filepath))
            sys.exit()
        try:
            with open(os.path.join(config_filepath,'test_enantioselectivity.json')) as f:
                self.test_params2 = json.load(f)
        except JSONDecodeError as e:
            print('Error getting config JSON file: {}.'.format(config_filepath))
            sys.exit()
        try:
            with open(os.path.join(config_filepath,'test_absorption.json')) as f:
                self.test_params3 = json.load(f)
        except JSONDecodeError as e:
            print('Error getting config JSON file: {}.'.format(config_filepath))
            sys.exit()
        try:
            with open(os.path.join(config_filepath,'test_localization.json')) as f:
                self.test_params4 = json.load(f)
        except JSONDecodeError as e:
            print('Error getting config JSON file: {}.'.format(config_filepath))
            sys.exit()

        #append all configs to a list
        self.all_configs = [self.test_params1, self.test_params2, self.test_params3,
                self.test_params4]
        self.all_config_files = ["test_thermostability.json", "test_enatioselectivity.json",
            "test_absorption.json", "test_localization.json"]

    def test_descriptor(self):
        """ 
        Test descriptor initialisation process. Verify the initial input parameters
        and descriptor attributes are correct. 
        """

        #testing on all 4 datasets
        # for dataset in range(0,len(self.all_test_datasets)):
        for dataset in range(0,len(self.all_configs)):
            desc = descr.Descriptors(self.all_configs[dataset])
#1.)
            #verify num_seqs descriptors attribute is correct
            # self.assertEqual(desc.num_seqs, len(self.test_dataset1['sequence']),
            self.assertEqual(desc.num_seqs, len(self.all_test_datasets[dataset]['sequence']),
                'num_seqs attribute not equal to the number of sequences found.')

            #verify that all input sequences dont have any gaps/missing amino acids
            for seq in desc.protein_seqs:
                self.assertNotIn('-',seq, 'There should be no gaps (-) in the sequences.')
#2.)
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

            #descriptor module should raise ValueError when the full dataset DF is
            #   passed in instead of just the sequences.
            with self.assertRaises(ValueError):
                fail_desc = descr.Descriptors(invalid_testdata1_copy)

            #J,O,U and Z are all invalid amino acid values
            invalid_testdata1_copy['sequence'][0] = 'JOUZ'

            #descriptor module should raise ValueError if invalid sequence found in dataset
            with self.assertRaises(ValueError):
                fail_desc = descr.Descriptors(invalid_testdata1_copy['sequence'])

    def test_descriptor_groups(self):
        """ Testing the descriptor groups dictionary which stores the specific group
        that a descriptor attribute is a member of. """

        #testing on all 4 datasets
        for dataset in range(0,len(self.all_test_datasets)):
            desc = descr.Descriptors(self.all_test_datasets[dataset]['sequence'], desc_dataset="")
#1.)
            self.assertEqual(list(desc.descriptor_groups.keys()), desc.all_descriptors_list())
            self.assertEqual(list(desc.descriptor_groups.values()).count("Composition"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Autocorrelation"), 3)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Conjoint Triad"), 1)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Quasi-Sequence-Order"), 2)
            self.assertEqual(list(desc.descriptor_groups.values()).count("CTD"), 4)
            self.assertEqual(list(desc.descriptor_groups.values()).count("Pseudo Composition"), 2)
            self.assertEqual(len(desc.descriptor_groups.keys()), len(desc.all_descriptors_list()))
#2.)
            self.assertEqual(desc.descriptor_groups['aa_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['dipeptide_composition'], "Composition")
            self.assertEqual(desc.descriptor_groups['moran_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['geary_autocorrelation'], "Autocorrelation")
            self.assertEqual(desc.descriptor_groups['composition'], "CTD")
            self.assertEqual(desc.descriptor_groups['distribution'], "CTD")
            self.assertEqual(desc.descriptor_groups['transition'], "CTD")
            self.assertEqual(desc.descriptor_groups['pseudo_aa_composition'], "Pseudo Composition")
            self.assertEqual(desc.descriptor_groups['quasi_seq_order'], "Quasi-Sequence-Order")
            self.assertEqual(desc.descriptor_groups['amp_pseudo_aa_composition'], "Pseudo Composition")
#3.)
            self.assertIsInstance(desc.descriptor_groups, dict)

    def test_aa_composition(self):
        """ Testing Amino Acid Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            aa_comp = desc.get_aa_composition()

            self.assertEqual(aa_comp.shape, (1,20), 'Descriptor not of correct shape.')
            self.assertIsInstance(aa_comp, pd.DataFrame,'Descriptor not of type DataFrame.')
            self.assertEqual(AAComposition.AALetter, list(aa_comp.columns), 'Incorrect column values found.')
            self.assertFalse(aa_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(aa_comp.any().isnull().sum()== 0,'Descriptor contains null values.')

    def test_dipeptide_composition(self):
        """ Testing Dipeptide Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            dipeptide_comp = desc.get_dipeptide_composition()

            self.assertEqual(dipeptide_comp.shape, (1,400),'Descriptor not of correct shape.')
            self.assertIsInstance(dipeptide_comp, pd.DataFrame,'Descriptor not of type DataFrame.')
            self.assertFalse(dipeptide_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(dipeptide_comp.any().isnull().sum()== 0,'Descriptor contains null values.')

    # @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_tripeptide_composition(self):
        """ Testing Tripeptide Composition protein descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            tripeptide_comp = desc.get_tripeptide_composition()

            self.assertEqual(tripeptide_comp.shape, (1,8000),'Descriptor not of correct shape.')
            self.assertIsInstance(tripeptide_comp, pd.DataFrame,'Descriptor not of type DataFrame.')
            self.assertFalse(tripeptide_comp.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(tripeptide_comp.any().isnull().sum()== 0,'Descriptor contains null values.')

    def test_normalized_moreaubroto_autocorrelation(self):
        """ Testing normalized moreaubroto autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            norm_moreaubroto = desc.get_norm_moreaubroto_autocorrelation()

            self.assertEqual(norm_moreaubroto.shape, (1,240))
            self.assertIsInstance(norm_moreaubroto, pd.DataFrame)
            self.assertFalse(norm_moreaubroto.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(norm_moreaubroto.any().isnull().sum()== 0)

    def test_moran_autocorrelation(self):
        """ Testing Moran autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            moran = desc.get_moran_autocorrelation()

            self.assertEqual(moran.shape, (1,240))
            self.assertIsInstance(moran, pd.DataFrame)
            self.assertFalse(moran.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(moran.any().isnull().sum()== 0)

    def test_geary_autocorrelation(self):
        """ Testing Geary autocorrelation descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            geary = desc.get_geary_autocorrelation()

            self.assertEqual(geary.shape, (1,240))
            self.assertIsInstance(geary, pd.DataFrame)
            self.assertFalse(geary.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(geary.any().isnull().sum()== 0)

    def test_ctd(self):
        """ Testing CTD descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            ctd = desc.get_ctd()

            self.assertEqual(ctd.shape, (1,147))
            self.assertIsInstance(ctd, pd.DataFrame)
            self.assertFalse(ctd.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(ctd.any().isnull().sum()== 0)

    def test_conjoint_triad(self):
        """ Testing Conjoint Triad descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            conjoint_triad = desc.get_conjoint_triad()

            self.assertEqual(conjoint_triad.shape, (1,343), 'Descriptor not of correct shape (1,343)')
            self.assertIsInstance(conjoint_triad, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertFalse(conjoint_triad.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(conjoint_triad.any().isnull().sum()== 0,'Descriptor contains null values')

    def test_seq_order_coupling_number(self):
        """ Testing sequence order coupling number descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            seq_order_coupling_number = desc.get_seq_order_coupling_number()

            self.assertEqual(seq_order_coupling_number.shape, (1,60), 'Descriptor not of correct shape (1,60)')
            self.assertIsInstance(seq_order_coupling_number, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertFalse(seq_order_coupling_number.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(seq_order_coupling_number.any().isnull().sum()== 0,'Descriptor contains null values')

    def test_quasi_seq_order(self):
        """ Testing Quasi sequence order descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            quasi_seq_order = desc.get_quasi_seq_order()

            self.assertEqual(quasi_seq_order.shape, (1,100), 'Descriptor not of correct shape (1,100)')
            self.assertIsInstance(quasi_seq_order, pd.DataFrame,'Descriptor not of type DataFrame')
            self.assertFalse(quasi_seq_order.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(quasi_seq_order.any().isnull().sum()== 0,'Descriptor contains null values')

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_pseudo_AAC(self):
        """ Testing Pseudo Amino Acid Composition descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            pseudo_AAC = desc.get_pseudo_aa_composition()

            self.assertEqual(pseudo_AAC.shape, (1,50), 'Descriptor not of correct shape (1,50)')
            self.assertIsInstance(pseudo_AAC, pd.DataFrame, 'Descriptor not of type DataFrame')
            self.assertFalse(pseudo_AAC.empty, 'Descriptor dataframe should not be empty')
            self.assertTrue(pseudo_AAC.any().isnull().sum()== 0, 'Descriptor contains null values')

    @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_amp_pseudo_AAC(self):
        """ Testing Amphiphilic Pseudo Amino Acid Composition descriptor attributes and methods. """
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
            amp_pseudo_AAC = desc.get_amp_pseudo_aa_composition()

            self.assertEqual(amp_pseudo_AAC.shape, (1,80), 'Descriptor not of correct shape (1,80)')
            self.assertIsInstance(amp_pseudo_AAC, pd.DataFrame)
            self.assertFalse(amp_pseudo_AAC.empty)
            self.assertTrue(amp_pseudo_AAC.any().isnull().sum()== 0,'Descriptor contains null values')

    def test_get_descriptor_encoding(self):
        """ Testing get_descriptor_encoding function.  """

        valid_desc = [

        'aa_composition', 'dipeptide_composition', 'tripeptide_composition', \
        'normalized_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation', \
        'ctd', 'composition', 'transition', 'distribution', 'conjoint_triad', \
        'seq_order_coupling_number','quasi_seq_order'

        ]
#1.)
        #run tests on all test datasets
        for data in range(0,len(self.all_test_datasets)):

            #get random sequence in dataset to calculate descriptor values for
            random_seq = random.randint(0,len(self.all_test_datasets[data])-1)
            desc = descr.Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #test each descriptor is calculated when string of its name passed to func
            for val_desc in range(0,len(valid_desc)):

                encoded_desc = desc.get_descriptor_encoding(valid_desc[val_desc])
                self.assertIsNotNone(encoded_desc)
                self.assertFalse(encoded_desc.empty)
                self.assertTrue(encoded_desc.shape[0],1)

    def tearDown(self):
        """ Cleanup tests and delete datasets. """
        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4
