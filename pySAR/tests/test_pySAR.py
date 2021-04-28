
################################################################################
#################             pySAR Module Tests             #################
################################################################################

import pandas as pd
import numpy as np
import os
import shutil
import unittest
unittest.TestLoader.sortTestMethodsUsing = None

from pySAR import PySAR
import globals

class PySARTests(unittest.TestCase):

    def setUp(self):
        """ Import the 4 test datasets used for testing the pySAR methods. """
        try:
            self.test_dataset1 = pd.read_csv(os.path.join('tests','test_data',
                'test_thermostability.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset1')
        try:
            self.test_dataset2 = pd.read_csv(os.path.join('tests','test_data',
                'test_enantioselectivity.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset2')
        try:
            self.test_dataset3 = pd.read_csv(os.path.join('tests','test_data',
                'test_localization.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset3')
        try:
            self.test_dataset4 = pd.read_csv(os.path.join('tests','test_data',
                'test_absorption.txt'),sep=",", header=0)
        except:
            raise IOError('Error reading in test_dataset4')

        #set global vars to create temp test data folders
        globals.OUTPUT_DIR = os.path.join('tests',globals.OUTPUT_DIR)
        globals.OUTPUT_FOLDER = os.path.join('tests',globals.OUTPUT_FOLDER)

    def test_pySAR(self):
        """ Testing pySAR intialisation process and associated methods & attributes. """

        #test pysar.data contains no NAN or gaps.

        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_thermostability.txt')),
            activity="T50",algorithm="PLSre")
#1.)
        #testing attribute values
        self.assertEqual(test_pySAR.dataset,(os.path.join('tests','test_data','test_thermostability.txt')),
            'Dataset attribute does not equal what was input, got {}'.format(test_pySAR.dataset))
        self.assertEqual(test_pySAR.seq_col, "sequence",
            'Sequence column attribute is not correct, got {}, expected {}'.format(test_pySAR.seq_col,"sequence"))
        self.assertEqual(test_pySAR.activity, "T50", "Activity attribute name not correct, \
            expected {}, got {}. ".format("T50", test_pySAR.activity))
        self.assertEqual(test_pySAR.algorithm,"PLSre", 'Algorithm attribute not correct, \
            expected {}, got {}.'.format("PLSRegression",test_pySAR.algorithm))
        self.assertEqual(test_pySAR.test_split, 0.2)
        self.assertEqual(test_pySAR.aa_indices, "")
        self.assertEqual(test_pySAR.descriptors, "")
        self.assertEqual(test_pySAR._parameters, {}, 'Parameters attribute should be empty, got {}.'.format(test_pySAR.parameters))
        self.assertEqual(str(type(test_pySAR.aaindex)), "<class 'aaindex.AAIndex'>", 'AAIndex should be an instance of the AAIndex class, got {}.'.format(type(test_pySAR.aaindex)))
        self.assertIsInstance(test_pySAR.data,pd.DataFrame, 'Data should be a DataFrame, got {}'.format(type(test_pySAR.data)))
        self.assertEqual(test_pySAR.data.shape, (self.test_dataset1.shape))
        self.assertEqual(test_pySAR.num_seqs,261, 'Number of sequences should be 261, got {}'.format(test_pySAR.num_seqs))
        self.assertEqual(test_pySAR.seq_len, 466, 'Sequence length should be 466, got {}'.format(test_pySAR.seq_len))
        self.assertEqual((str(type(test_pySAR.model))), "<class 'model.Model'>")
        self.assertEqual(((type(test_pySAR.model.model).__name__)), "PLSRegression",'Model type should be of type PLSRegression, got {}'.format((type(test_pySAR.model.model).__name__)))
        self.assertTrue(os.path.isdir(globals.OUTPUT_DIR), 'Output directory directory has not been created properly.')  #validate output dirs have been created
        self.assertTrue(os.path.isdir(globals.OUTPUT_FOLDER),'Output Folder directory has not been created properly.')

#2.)
        #validate that if errorneous input parameters are input, that errors are raised
        with self.assertRaises(IOError,msg='IO Error raised, input parameters erroneous.'):
            test_pySAR1 = PySAR(dataset=(os.path.join('tests','test_data','blah_blah.txt')),
                activity="T50",algorithm="PLSre")

        with self.assertRaises(ValueError,msg='Value Error raised, input parameters erroneous.'):
            test_pySAR1 = PySAR(dataset=(os.path.join('tests','test_data','test_absorption.txt')),
                activity="peak",algorithm="PLSre",seq_col = "blahblahblah")

        with self.assertRaises(ValueError,msg='Value Error raised, input parameters erroneous.'):
            test_pySAR1 = PySAR(dataset=(os.path.join('tests','test_data','test_absorption.txt')),
                activity="blahblahblah",algorithm="PLSre")

    def test_get_seqs(self):
        """ Testing getting the protein sequences from the dataset. """

#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_localization.txt')),
            activity="log_GFP",algorithm="PLSre")

        test_seqs = test_pySAR.get_seqs()

        self.assertEqual(test_seqs.shape,(test_pySAR._num_seqs, ),'Shape of the sequences \
            not correct, expected {}, got {}.'.format(test_seqs.shape,(test_pySAR._num_seqs, )))
        self.assertIsInstance(test_seqs, pd.Series, 'Sequences not correct type, \
            expected {}, got {}'.format(pd.Series, type(test_seqs)))
        self.assertTrue(test_seqs[0].startswith("MSRLVAASW"), 'Error in first seqeuence \
            expected it to start with MSRLVAASW.')

    def test_get_activity(self):
        """ Testing getting activity from dataset function. """

#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_thermostability.txt')),
            activity="T50",algorithm="PLSre")
        activity = test_pySAR.get_activity()

        self.assertEqual(activity.shape[0], test_pySAR.num_seqs,'Shape of 1st dimension of Activity expected to be {} , got{} '.format(test_pySAR.num_seqs,activity.shape[1]))
        self.assertEqual(activity.shape[1], 1, 'Shape of 2nd dimension of Activity expected to be 1, got{} '.format(activity.shape[1]))
        self.assertIsInstance(activity, np.ndarray, 'Activity attribute should of type numpy array, got {}'.format(type(activity)))

        self.assertTrue((activity[:10] == np.array(([55. ],[43. ],[49. ],[39.8],\
            [52.9],[48.8],[45. ],[48.3],[61.5],[54.6]))).all())

#2.)
        test_pySAR_2 = PySAR(dataset=(os.path.join('tests','test_data','test_localization.txt')),
            activity="log_GFP",algorithm="PLSreg")
        activity_2 = test_pySAR_2.get_activity()

        self.assertEqual(activity_2.shape[0], test_pySAR_2.num_seqs,'Shape of 1st dimension of Activity should be {} , got{} '.format(test_pySAR_2.num_seqs,activity_2.shape[1]))
        self.assertEqual(activity_2.shape[1], 1, 'Shape of 2nd dimension of Activity should be 1, got{} '.format(activity.shape[1]))
        self.assertIsInstance(activity_2, np.ndarray, 'Activity attribute should of type numpy array, got {}'.format(type(activity_2)))

#3.)
        test_pySAR_3 = PySAR(dataset=(os.path.join('tests','test_data','test_absorption.txt')),
            activity="peak",algorithm="PLSreg")
        activity_3 = test_pySAR_3.get_activity()

        self.assertEqual(activity_3.shape[0], test_pySAR_2.num_seqs,'Shape of 1st dimension of Activity should be {} , got{} '.format(test_pySAR_3.num_seqs,activity_3.shape[1]))
        self.assertEqual(activity_3.shape[1], 1, 'Shape of 2nd dimension of Activity should be 1, got{} '.format(activity_3.shape[1]))
        self.assertIsInstance(activity_3, np.ndarray, 'Activity attribute should of type numpy array, got {}'.format(type(activity_3)))


    def test_get_aai_encoding(self):
        """ Testing getting the AAI encoding from the database for specific indices."""

        aa_indices = ["CHAM810101","ISOY800103"]
        aa_indices1 = "NAKH920102"
        error_aaindices = ["ABCD1234","ABCD12345"]
        error_aaindices1 = "XYZ4567"

#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_thermostability.txt')),
            activity="T50",algorithm="PLSre")
        aai_encoding = test_pySAR.get_aai_enoding(aa_indices)

        self.assertIsInstance(aai_encoding,np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}'.format(type(aai_encoding)))
        self.assertEqual(aai_encoding.shape[0],test_pySAR.num_seqs, 'The number of sequences \
            in the dataset expected to be {}, got {}'.format(test_pySAR.num_seqs,aai_encoding.shape[0]))
        self.assertEqual(aai_encoding.shape[1],test_pySAR.seq_len*len(aa_indices), 'The length \
            of the sequences expected to be {}, got {}'.format(test_pySAR.seq_len, str(aai_encoding.shape[1])))
        self.assertEqual(aai_encoding.dtype,np.float64, 'Datatype of elements \
            in numpy array expected be np.float64, got {}'.format(aai_encoding.dtype))
        self.assertTrue((np.array([0.78, 0.5, 1.02, 0.68, 0.68, 0.78, 0.36, 0.68, \
            0.36, 0.68],dtype=np.float64)==aai_encoding[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what is expected.')
#2.)
        test_pySAR_1 = PySAR(dataset=(os.path.join('tests','test_data','test_enantioselectivity.txt')),
            activity="e-value",algorithm="PLSre")
        aai_encoding_1 = test_pySAR_1.get_aai_enoding(aa_indices1)

        self.assertIsInstance(aai_encoding_1,np.ndarray,
            'AAI Encoding output expected to be a numpy array, got datatype {}.'.format(type(aai_encoding_1)))
        self.assertEqual(aai_encoding_1.shape[0],test_pySAR_1.num_seqs,'The number of sequences \
            in the dataset expected to be {}, got {}.'.format(test_pySAR_1.num_seqs,aai_encoding_1.shape[0]))
        self.assertEqual(aai_encoding_1.shape[1],test_pySAR_1.seq_len,'The length of the \
            sequences expected to be {}, got {}.'.format(test_pySAR_1.seq_len, str(aai_encoding_1.shape[1])))
        self.assertEqual(aai_encoding_1.dtype,np.float64,'Datatype of elements \
            in numpy array should be np.float64, got {}.'.format(aai_encoding_1.dtype))
        self.assertTrue((np.array([3.79, 3.51, 1.8, 6.11, 9.34, 3.79, 7.21, 4.68, 7.21,  \
            6.11],dtype=np.float64)==aai_encoding_1[0][:10]).all(),
                'The first 10 elements of sequence 0 do not match what is expected')

#3.)    #testing errenous indices
        with self.assertRaises(ValueError,msg='ValueError: Errorneous indices have been input'):
            aai_encoding = test_pySAR_1.get_aai_enoding(error_aaindices)
#4.)
        with self.assertRaises(ValueError,msg='ValueError: Errorneous indices have been input'):
            aai_encoding = test_pySAR_1.get_aai_enoding(error_aaindices1)
#5.)
        with self.assertRaises(TypeError,msg='TypeError: Errorneous indices datatypes have been input'):
            aai_encoding = test_pySAR_1.get_aai_enoding(1235)
            aai_encoding = test_pySAR_1.get_aai_enoding(40.89)
            aai_encoding = test_pySAR_1.get_aai_enoding(False)

    def test_aai_encoding(self):
        """ Testing AAI encoding functionality. """

        aa_indices = ["CHAM810101","ISOY800103"]
        aa_indices_1 = "NAKH920102"
        aa_indices_2 = "LIFS790103"
        aa_indices_3 = ["PTIO830101", "QIAN880136", "RACS820110"]
        all_indices = [aa_indices, aa_indices_1, aa_indices_2, aa_indices_3]
        #test with erroneous indices
#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_thermostability.txt')),
            activity="T50",algorithm="PLSre")

        with self.assertRaises(ValueError,msg='ValueError: Indices parameter cannot be None.'):
            test_aai_ = test_pySAR.aai_encoding(indices=None)
#2.)
        with self.assertRaises(ValueError,msg='ValueError: Spectrum parameter cannot be None.'):
            test_aai_ = test_pySAR.aai_encoding(spectrum=None)

#3.)
        for index in range(0, len(all_indices)):

            test_aai_ = test_pySAR.aai_encoding(spectrum='power', indices=all_indices[index])
            self.assertIsInstance(test_aai_,pd.Series, 'Output should be a DataFrame, got {}'.format(type(test_aai_)))
            self.assertEqual(len(test_aai_),8)
            self.assertEqual(test_aai_.dtype, object)
            self.assertIn('Index', test_aai_.index)
            # self.assertEqual(test_aai_['Index'], "CHAM810101, ISOY800103")
            self.assertIn('Category', test_aai_.index)
            # self.assertEqual(test_aai_['Category'], "CHAM810101, ISOY800103")
            self.assertIn('R2', test_aai_.index)
            self.assertIn('RMSE', test_aai_.index)
            self.assertIn('MSE', test_aai_.index)
            self.assertIn('MAE', test_aai_.index)
            self.assertIn('RPD', test_aai_.index)
            self.assertIn('Explained Var', test_aai_.index)

#4.)        testing that regression plot and results file saved to output folder
            self.assertTrue(os.path.isdir(os.path.join(globals.OUTPUT_FOLDER, 'model_regPlot.png')))
            self.assertTrue(os.path.isdir(os.path.join(globals.OUTPUT_FOLDER, 'aai_encoding.csv')))

    def test_desc_encoding(self):
        """ Testing Descriptor encoding functionality. """

        desc_1 = "aa_comp"
        desc_2 = "distribution"
        desc_3 = "conjoint_triad"
        desc_4 = ["moranauto", "quasi_seq_order"]
        all_desc = [desc_1, desc_2, desc_3, desc_4]

#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_enantioselectivity.txt')),
            activity="e-value",algorithm="PLSre")

        with self.assertRaises(ValueError,msg='ValueError: Descriptor parameter cannot be None.'):
            test_desc = test_pySAR.desc_encoding(descriptor=None)

#2.)
        for de in range(0,len(all_desc)):

            test_desc = test_pySAR.desc_encoding(descriptor="aa_comp")
            self.assertIsInstance(test_desc,pd.Series, 'Output should be a DataFrame, got {}'.format(type(test_desc)))
            self.assertEqual(len(test_desc),8)
            self.assertEqual(test_desc.dtype, object)
            self.assertIn('Descriptor', test_desc.index)
            # self.assertEqual(test_desc['Descriptor'], "Descriptor")
            self.assertIn('Group', test_desc.index)
            # self.assertEqual(test_desc['Group'], "ffasf")
            self.assertIn('R2', test_desc.index)
            self.assertIn('RMSE', test_desc.index)
            self.assertIn('MSE', test_desc.index)
            self.assertIn('MAE', test_desc.index)
            self.assertIn('RPD', test_desc.index)
            self.assertIn('Explained Var', test_desc.index)

#3.)        #testing that regression plot and results file saved to output folder
            self.assertTrue(os.path.isdir(os.path.join(globals.OUTPUT_FOLDER, 'model_regPlot.png')))
            self.assertTrue(os.path.isdir(os.path.join(globals.OUTPUT_FOLDER, 'desc_encoding.csv')))


    def test_aai_desc_encoding(self):
        """ Testing AAI + Descriptor encoding functionality. """

        aa_indices_1 = "CHAM810101"
        aa_indices_2 = "NAKH920102"
        aa_indices_3 = "LIFS790103"
        aa_indices_4 = ["PTIO830101", "QIAN880136", "RACS820110"]
        desc_1 = "aa_comp"
        desc_2 = "distribution"
        desc_3 = "conjoint_triad"
        desc_4 = ["moranauto", "quasi_seq_order"]

#1.)
        test_pySAR = PySAR(dataset=(os.path.join('tests','test_data','test_localization.txt')),
            activity="log_GFP",algorithm="PLSre")

        with self.assertRaises(ValueError,msg='ValueError: Descriptor and indices parameter cannot both be None.'):
            test_desc = test_pySAR.desc_encoding(descriptor=None)
            test_desc = test_pySAR.aai_desc_encoding(indices=None)
            test_desc = test_pySAR.aai_desc_encoding(descriptor="aa_comp")
            test_desc = test_pySAR.aai_desc_encoding(indices="LIFS790103")

    def tearDown(self):
        """ Delete any temp files or folders created during testing process. """

        del self.test_dataset1
        del self.test_dataset2
        del self.test_dataset3
        del self.test_dataset4

        #delete any temporary output directory or folders created
        if (os.path.isdir(globals.OUTPUT_DIR)):
            shutil.rmtree(globals.OUTPUT_DIR, ignore_errors=False, onerror=None)
        if (os.path.isdir(globals.OUTPUT_FOLDER)):
            shutil.rmtree(globals.OUTPUT_FOLDER, ignore_errors=False, onerror=None)
