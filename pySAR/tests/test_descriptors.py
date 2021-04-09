################################################################################
#################            Descriptors Module Tests          #################
################################################################################

import pandas as pd
import numpy as np
import os
import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import random

from descriptors import Descriptors
from pySAR import pySAR
from PyBioMed.PyBioMed.PyProtein import AAComposition, Autocorrelation, CTD, ConjointTriad, QuasiSequenceOrder, PseudoAAC
import utils as utils

class DescriptorTests(unittest.TestCase):

    def setUp(self):

        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests','test_data','test_thermostability.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1')
        try:
            self.test_dataset2 = pd.read_csv(os.path.join('tests','test_data','test_enantioselectivity.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset2')
        try:
            self.test_dataset3 = pd.read_csv(os.path.join('tests','test_data','test_localization.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset3')
        try:
            self.test_dataset4 = pd.read_csv(os.path.join('tests','test_data','test_absorption.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset4')

        self.all_test_datasets = [self.test_dataset1, self.test_dataset2, self.test_dataset3,
                self.test_dataset4]

    def test_descriptor(self):

        desc = Descriptors(self.test_dataset1['sequence'], desc_dataset="")

        #verify num_seqs descriptors attribute is correct
        self.assertEqual(desc.num_seqs, len(self.test_dataset1['sequence']), 'num_seqs attribute not equal to the number of sequences found')
        #verify all_desc attribute is at its default of False
        self.assertFalse(desc.all_desc, 'all_desc parameter should initially be False')

        #verify that all input sequences dont have any gaps/missing amino acids
        for seq in desc.protein_seqs:
            self.assertNotIn('-',seq, 'There should be no gaps (-) in the sequences')

        #verify all descriptor attributes are initialised to empty dataframes
        self.assertTrue(desc.aa_composition.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.dipeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.tripeptide_composition.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.normalized_moreaubroto_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.moran_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.geary_autocorrelation.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.CTD.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.composition.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.transition.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.distribution.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.conjoint_triad.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.seq_order_coupling_number.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.quasi_seq_order.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.pseudo_AAC.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.amp_pseudo_AAC.empty, 'Attribute should be initialised to an empty dataframe')
        self.assertTrue(desc.all_descriptors.empty, 'Attribute should be initialised to an empty dataframe')


        invalid_testdata1_copy = self.test_dataset1
        invalid_testdata1_copy['sequence'][0] = 'JOUZ'

    #test importing descriptors.


        # with self.assertRaises(ValueError):
        #     fail_desc = Descriptors(invalid_testdata1_copy)

        #


    def test_aa_composition(self):

        print('Testing AA Composition Descriptor....')


        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            aa_comp = desc.get_aa_composition()

            self.assertEqual(aa_comp.shape, (1,20))
            self.assertIsInstance(aa_comp, pd.DataFrame)
            self.assertEqual(AAComposition.AALetter, list(aa_comp.columns))
            self.assertTrue(aa_comp.any().isnull().sum()== 0,'Descriptor contains null values')

    #         encoded_desc = desc.get_descriptor_encoding("aa_composition")
    #
    #         self.assertEqual(encoded_desc.shape, (1,20))
    #         self.assertIsInstance(encoded_desc, pd.DataFrame)
    #         self.assertEqual(AAComposition.AALetter, list(encoded_desc.columns))
    #         self.assertTrue(encoded_desc.any().isnull().sum()== 0)
    # #
    # #     #assert concatenation of 2 descriptors works
    # #     # self.assertEqual(aa_comp.shape == (self.test_dataset1.shape[0], 20))
    # #
    def test_dipeptide_composition(self):

        print('Testing Dipeptide Composition Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            dipeptide_comp = desc.get_dipeptide_composition()

            self.assertEqual(dipeptide_comp.shape, (1,400))
            self.assertIsInstance(dipeptide_comp, pd.DataFrame)
            self.assertTrue(dipeptide_comp.any().isnull().sum()== 0,'Descriptor contains null values')



    def test_tripeptide_composition(self):

        print('Testing Tripeptide Composition Descriptor....')



        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            tripeptide_comp = desc.get_tripeptide_composition()

            self.assertEqual(tripeptide_comp.shape, (1,8000))
            self.assertIsInstance(tripeptide_comp, pd.DataFrame)
            self.assertTrue(tripeptide_comp.any().isnull().sum()== 0,'Descriptor contains null values')


    #
    # def test_normalized_moreaubroto_autocorrelation(self):
    #     print('Testing Normalised MoreauBroto Autocorrelation Descriptor....')
    #
    #
    #     for data in range(0,len(self.all_test_datasets)):
    #         # print('\nRunning tests on dataset {} '.format(data))
    #
    #         random_seq = random.randint(0,len(self.all_test_datasets[data]))
    #
    #         desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")
    #
    #         #assert aa_comp dataframe is initially empty
    #         tripeptide_comp = desc.get_tripeptide_composition()
    #
    #         self.assertEqual(tripeptide_comp.shape, (1,8000))
    #         self.assertIsInstance(tripeptide_comp, pd.DataFrame)
    #         self.assertTrue(tripeptide_comp.any().isnull().sum()== 0)
    #
    #     pass
    #
    # def test_ctd(self):
    #     print('Testing CTD Descriptor....')

    #     pass
    #
    def test_conjoint_triad(self):
        print('Testing Conjoint Triad Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            conjoint_triad = desc.get_conjoint_triad()

            self.assertEqual(conjoint_triad.shape, (1,512), 'Descriptor not of correct shape (1,512)')
            self.assertIsInstance(conjoint_triad, pd.DataFrame)
            self.assertTrue(conjoint_triad.any().isnull().sum()== 0,'Descriptor contains null values')



    def test_seq_order_coupling_number(self):

        print('Testing Sequence Order Coupling Number Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            seq_order_coupling_number = desc.get_seq_order_coupling_number()

            self.assertEqual(seq_order_coupling_number.shape, (1,1), 'Descriptor not of correct shape (1,1)')
            self.assertIsInstance(seq_order_coupling_number, pd.DataFrame)
            self.assertTrue(seq_order_coupling_number.any().isnull().sum()== 0,'Descriptor contains null values')


    def test_quasi_seq_order(self):

        print('Testing Quasi Sequence Order Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            quasi_seq_order = desc.get_quasi_seq_order()

            self.assertEqual(quasi_seq_order.shape, (1,100), 'Descriptor not of correct shape (1,100)')
            self.assertIsInstance(quasi_seq_order, pd.DataFrame,'Descriptor not of type DataFrame'))
            self.assertTrue(quasi_seq_order.any().isnull().sum()== 0,'Descriptor contains null values')



    # @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_pseudo_AAC(self):
        print('Testing Pseudo Amino Acid Composition Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            #assert aa_comp dataframe is initially empty
            pseudo_AAC = desc.get_pseudo_AAC()

            self.assertEqual(pseudo_AAC.shape, (1,50), 'Descriptor not of correct shape (1,50)')
            self.assertIsInstance(pseudo_AAC, pd.DataFrame, 'Descriptor not of type DataFrame')
            self.assertTrue(pseudo_AAC.any().isnull().sum()== 0, 'Descriptor contains null values')


    # @unittest.skip("Descriptor can take quite a bit of time to calculate therefore skipping")
    def test_amp_pseudo_AAC(self):

        print('Testing Amphiphilic Amino Acid Composition Descriptor....')

        for data in range(0,len(self.all_test_datasets)):
            # print('\nRunning tests on dataset {} '.format(data))

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            amp_pseudo_AAC = desc.get_amp_pseudo_AAC()

            self.assertEqual(amp_pseudo_AAC.shape, (1,80), 'Descriptor not of correct shape (1,80)')
            self.assertIsInstance(amp_pseudo_AAC, pd.DataFrame)
            self.assertTrue(amp_pseudo_AAC.any().isnull().sum()== 0,'Descriptor contains null values')


    #
    def test_get_descriptor_encoding(self):

        valid_desc = ['aa_composition', 'dipeptide_composition', 'tripeptide_composition', \
            'norm_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation', \
             'ctd', 'composition', 'transition', 'distribution', 'conjoint_triad', \
             'seq_order_coupling_number','quasi_seq_order_descriptors',\
             'pseudo_aa_comp', 'amphipilic_pseudo_aa_comp']


        for data in range(0,len(self.all_test_datasets)):

            random_seq = random.randint(0,len(self.all_test_datasets[data]))

            desc = Descriptors(self.all_test_datasets[data]['sequence'][random_seq], desc_dataset="")

            for val_desc in range(0,len(valid_desc)):

                encoded_desc = desc.get_descriptor_encoding(valid_desc[val_desc])
                self.assertIsNotNone(encoded_desc)

      # type(encoded_desc) == aa_composition()

    # def all_descriptors_list(self, desc_combo=1):

    #
    def test_all_descriptors(self):
        print('Testing All Descriptors Functionality....')

        all_desc = ['aa_composition', 'dipeptide_composition', 'tripeptide_composition', \
                'norm_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation', \
                 'ctd', 'composition', 'transition', 'distribution', 'conjoint_triad', \
                 'seq_order_coupling_number','quasi_seq_order_descriptors',\
                 'pseudo_aa_comp', 'amphipilic_pseudo_aa_comp']

        desc = Descriptors(self.all_test_datasets[0])

        val_desc = desc.all_descriptors_list()

        self.assertEqual(len(val_desc), 15)
        self.assertIsInstance(val_desc, list)
        self.assertNotIn("all_descriptors", val_desc,
            'all_descriptors attribute should not be be returned by the all_descriptors \
            list function')


        for d in range(0, len(val_desc)):

            self.assertIn('_'+val_desc[d], list(desc.__dict__.keys(),
                'Descriptor {} not found in available descriptor attributes: {}'
                .format(val_desc[d],list(desc.__dict__.keys()))
            self.assertIn(val_desc[d], all_desc,
                'Descriptor ({}) not found in all descriptors attribute: {}'.format(val_desc[d], all_desc))

        val_desc = desc.all_descriptors_list(desc_combo=2)

        self.assertEqual(len(val_desc), 105,
            'There should be 105 total descriptor combinations, got {}'.format(len(val_desc)))

        val_desc = desc.all_descriptors_list(desc_combo=3)

        self.assertEqual(len(val_desc), 455,
            'There should be 455 total descriptor combinations, got {}'.format(len(val_desc)))


    #     # def all_descriptors_list(self, descCombo=1):
    #
    # def test_valid_descriptors(self):
    #     print('Testing Valid Descriptor....')
    #     pass


    def tearDown(self):
        """
        Cleanup tests and delete datasets.
        """
        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4







# https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names

#assert desc.all_descriptors.shape == []
