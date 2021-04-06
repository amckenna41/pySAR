################################################################################
#################            Descriptors Module Tests          #################
################################################################################

import pandas as pd
import numpy as np
import os
import unittest
import random

from descriptors import Descriptors
from ProtSAR import ProtSAR
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

    def test_descriptor(self):

        desc = Descriptors(self.test_dataset1['sequence'])

        self.assertEqual(desc.num_seqs, len(self.test_dataset1['sequence']))
        self.assertFalse(desc.all_desc)

        for seq in desc.protein_seqs:
            self.assertNotIn('-',seq)

        self.assertTrue(desc.aa_composition.empty)
        self.assertTrue(desc.dipeptide_composition.empty)
        self.assertTrue(desc.tripeptide_composition.empty)
        self.assertTrue(desc.normalized_moreaubroto_autocorrelation.empty)
        self.assertTrue(desc.moran_autocorrelation.empty)
        self.assertTrue(desc.geary_autocorrelation.empty)
        self.assertTrue(desc.CTD.empty)
        self.assertTrue(desc.conjoint_triad.empty)
        self.assertTrue(desc.seq_order_coupling_number.empty)
        self.assertTrue(desc.quasi_seq_order.empty)
        self.assertTrue(desc.pseudo_AAC.empty)
        self.assertTrue(desc.amp_pseudo_AAC.empty)
        self.assertTrue(desc.all_descriptors.empty)

        invalid_testdata1_copy = self.test_dataset1
        invalid_testdata1_copy['sequence'][0] = 'JOUZ'

        # with self.assertRaises(ValueError):
        #     fail_desc = Descriptors(invalid_testdata1_copy)

        #


    def test_aacomposition(self):

        print('Testing AA Composition Descriptor....')

        random_seq = random.randint(0,len(self.test_dataset1))

        desc = Descriptors(self.test_dataset1['sequence'][random_seq])


        #assert aa_comp dataframe is initially empty
        aa_comp = desc.get_aa_composition()

        #assert aa_comp contains no NAN or +/- infinity
        self.assertEqual(aa_comp.shape, (1,20))
        self.assertNotIn('-' ,desc.protein_seqs)
        self.assertIsInstance(aa_comp, pd.DataFrame)
        self.assertEqual(AAComposition.AALetter, list(aa_comp.columns))
        self.assertFalse(np.isnan(aa_comp).any())

        random_seq = random.randint(0,len(self.test_dataset2))

        desc = Descriptors(self.test_dataset2['sequence'][random_seq])

        aa_comp = desc.get_aa_composition()

        self.assertEqual(aa_comp.shape, (1,20))
        self.assertNotIn('-' ,desc.protein_seqs)
        self.assertIsInstance(aa_comp, pd.DataFrame)
        self.assertEqual(AAComposition.AALetter, list(aa_comp.columns))
        self.assertFalse(np.isnan(aa_comp).any())

        random_seq = random.randint(0,len(self.test_dataset3))

        desc = Descriptors(self.test_dataset3['sequence'][random_seq])

        aa_comp = desc.get_aa_composition()

        self.assertTrue(self.aa_composition.empty)

        self.assertEqual(aa_comp.shape, (1,20))
        self.assertNotIn('-' ,desc.protein_seqs)
        self.assertIsInstance(aa_comp, pd.DataFrame)
        self.assertEqual(AAComposition.AALetter, list(aa_comp.columns))
        self.assertFalse(np.isnan(aa_comp).any())

    #
    # #     #assert concatenation of 2 descriptors works
    # #     # self.assertEqual(aa_comp.shape == (self.test_dataset1.shape[0], 20))
    # #
    # def test_dipeptidecomposition(self):
    #
    #     print('Testing Dipeptide Composition Descriptor....')
    #
    #     random_seq = random.randint(0,len(self.test_dataset1))
    #
    #     desc = Descriptors(self.test_dataset1['sequence'][random_seq])
    #
    #     dipeptide_comp = desc.get_dipeptide_composition()
    #
    #     self.assertEqual(aa_comp.shape, (1,20))
    #     self.assertNotIn('-' ,desc.protein_seqs)
    #     self.assertIsInstance(aa_comp, pd.DataFrame)
    #     self.assertEqual(AAComposition.AALetter, list(aa_comp.columns))



    #     # desc = Descriptors(self.test_dataset1['sequence'])
    #     #
    #     # dipeptide_comp = desc.get_dipeptide_composition()
    #     #
    #     # self.assertIsInstance(dipeptide_comp, pd.DataFrame)
    #     # self.assertTrue('-' not in desc.protein_seqs)
    #     #
    #     # self.ass
    #     # self.assertEqual(dipeptide_comp.shape == (self.test_dataset1.shape[0], X))
    #
    #     pass
    #
    # def test_tripeptidecomposition(self):

    #     print('Testing Tripeptide Composition Descriptor....')

    #     pass
    #
    # def test_normalized_moreaubroto_autocorrelation(self):
    #     print('Testing Normalised MoreauBroto Autocorrelation Descriptor....')

    #     pass
    #
    # def test_ctd(self):
    #     print('Testing CTD Descriptor....')

    #     pass
    #
    # def test_conjoint_triad(self):
    #     print('Testing Conjoint Triad Descriptor....')

    #     pass
    #
    # def test_seq_order_coupling_number(self):
    #     print('Testing Sequence Order Coupling Number Descriptor....')

    #     pass
    #
    # def test_quasi_seq_order(self):
    #     print('Testing Quasi Sequence Order Descriptor....')
    #     pass
    #

    # def test_pseudo_AAC(self):
    #     print('Testing Pseudo Amino Acid Composition Descriptor....')
            # random_seq = random.randint(0,len(self.test_dataset1))

    #     pass
    #
    # def test_amp_pseudo_AAC(self):
    #     print('Testing Amphiphilic Amino Acid Composition Descriptor....')

    #     pass
    #
    # def test_get_descriptor_encoding(self):
    #
    #     # def get_descriptor_encoding(self,descriptor):
    #
    #     pass
    #
    # def test_all_descriptors(self):
    #     print('Testing All Descriptors Functionality....')

    #     pass
    #     # def all_descriptors_list(self, descCombo=1):
    #
    # def test_valid_descriptors(self):
    #     print('Testing Valid Descriptor....')
    #     pass


    def tearDown(self):

        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4







# https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names

#assert desc.all_descriptors.shape == []
