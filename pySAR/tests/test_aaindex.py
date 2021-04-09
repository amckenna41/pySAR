################################################################################
#################             AAIndex Module Tests             #################
################################################################################

import os
import sys
from aaindex import AAIndex
from globals import DATA_DIR, OUTPUT_FOLDER, OUTPUT_DIR
import unittest
import requests
import urllib.request

class AAIndexTests(unittest.TestCase):

    def setUp(self):
        self.aaindex = AAIndex()
        self.TEST_DIR = "tests/test_data"

    def tearDown(self):
        pass

    @unittest.skip("Don't want to overload the FTP server each time tests are run")
    def test_download(self):
        """
        Test Case to check that the download functionality works for the required AAI1 file.
        The file will firstly be removed from the test directory and then redownloaded, and
        its presence in the directory will pass the test.

        """
        print('Testing AAIndex download...')

        #if AAI1 present in test dir then remove
        if (os.path.isfile(os.path.join(self.TEST_DIR, 'test_aaindex1'))):
            os.remove(os.path.join(self.TEST_DIR, 'test_aaindex1'))

        #Assert that OSError exception raised when erroneous directory input to download func
        with self.assertRaises(OSError):
            self.aaindex.download_aaindex(save_dir="Non-existent-directory")

        #download AAI1 to local test directory
        self.aaindex.download_aaindex(save_dir=self.TEST_DIR)

        #verify download functionality has worked as desired
        self.assertTrue(os.path.isfile(os.path.join(self.TEST_DIR, 'aaindex1')))

    @unittest.skip("Similarly, don't want to overload the FTP server each time tests are run")
    def test_url(self):
        """
        Test Case to check that the URL endpoints used for downloading the AAI databases
        return a 200 status code.

        """
        print('Testing AAIndex URLs...')

        AA_INDEX1_URL = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"
        AA_INDEX2_URL = "https://www.genome.jp/ftp/db/community/aaindex/aaindex2"
        AA_INDEX3_URL = "https://www.genome.jp/ftp/db/community/aaindex/aaindex3"
        wrong_AA_INDEX_URL = "https://www.genome.jp/ftp/BLAH/BLAH/BLAH/BLAH"

        #test URL endpoints for AAINDEX are active and give a 200 status code
        r = requests.get(AA_INDEX1_URL, allow_redirects = True)
        self.assertEqual(r.status_code, 200, 'URL not returning Status Code 200')

        r = requests.get(AA_INDEX2_URL, allow_redirects = True)
        self.assertEqual(r.status_code, 200, 'URL not returning Status Code 200')

        r = requests.get(AA_INDEX3_URL, allow_redirects = True)
        self.assertEqual(r.status_code, 200, 'URL not returning Status Code 200')

        r = requests.get(wrong_AA_INDEX_URL, allow_redirects = True)
        self.assertEqual(r.status_code, 404, 'URL not returning Status Code 404')
        #get correct status code here - prbably 404?

        #maybe try requests.mock

    def test_get_amino_acids(self):
        """
        Test Case to check that only valid Amino Acids are used within the AAIndex
        class.

        """
        print('Testing Valid Amino Acids...')

        valid_amino_acids = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        amino_acids = self.aaindex.get_amino_acids()

        for aa in amino_acids:
            self.assertIn(aa, valid_amino_acids)

    def test_num_features(self):
        """
        Test Case to check that the correct number of indices/features are present
        in the AAIndex object. To date, 566 indices are present in the database,
        features may be added to the AAI in time so the test takes this into account.

        """
        print('Testing correct number of indices...')

        self.assertTrue(self.aaindex.get_num_features() >= 566)

    def test_feature_size(self):
        """
        Test Case to check that the AAI has the correct dimensionality. Check that
        the number of keys in parsed JSON from the AAI is correct. The test takes
        into account if more features are added to the database in the future.

        """
        print('Testing correct dimensionality of AAIndex...')

        self.assertTrue(len(self.aaindex.get_feature_codes())>=566)

    def test_get_feature_from_code(self):
        """
        Test Case to check that the correct amino acid values are returned for
        the feature/index codes using the get_feature_from_code function.

        """
        print('Testing getting AAi features from their indices...')

        #initialise test feature codes and their correct amino acid values
        feature1 = 'AURR980103'
        feature1_vals = {'A': 1.05, 'L': 0.96, 'R': 0.81, 'K': 0.97, 'N': 0.91, 'M': 0.99, 'D': 1.39, 'F': 0.95, 'C': 0.6, 'P': 1.05, 'Q': 0.87, 'S': 0.96, 'E': 1.11, 'T': 1.03, 'G': 1.26, 'W': 1.06, 'H': 1.43, 'Y': 0.94, 'I': 0.95, 'V': 0.62, '-': 0}
        feature2 = 'FAUJ880113'
        feature2_vals = {'A': 4.76, 'L': 4.79, 'R': 4.3, 'K': 4.27, 'N': 3.64, 'M': 4.25, 'D': 5.69, 'F': 4.31, 'C': 3.67, 'P': 0.0, 'Q': 4.54, 'S': 3.83, 'E': 5.48, 'T': 3.87, 'G': 3.77, 'W': 4.75, 'H': 2.84, 'Y': 4.3, 'I': 4.81, 'V': 4.86, '-': 0}
        feature3 = 'NAGK730101'
        feature3_vals = {'A': 1.29, 'L': 1.23, 'R': 0.83, 'K': 1.23, 'N': 0.77, 'M': 1.23, 'D': 1.0, 'F': 1.23, 'C': 0.94, 'P': 0.7, 'Q': 1.1, 'S': 0.78, 'E': 1.54, 'T': 0.87, 'G': 0.72, 'W': 1.06, 'H': 1.29, 'Y': 0.63, 'I': 0.94, 'V': 0.97, '-': 0}
        feature4 = 'ROBB760105'
        feature4_vals = {'A': -2.3, 'L': 2.3, 'R': 0.4, 'K': -3.3, 'N': -4.1, 'M': 2.3, 'D': -4.4, 'F': 2.6, 'C': 4.4, 'P': -1.8, 'Q': 1.2, 'S': -1.7, 'E': -5.0, 'T': 1.3, 'G': -4.2, 'W': -1.0, 'H': -2.5, 'Y': 4.0, 'I': 6.7, 'V': 6.8, '-': 0}

        #get amino acid values for inputted feature/index codes
        feature_vals = self.aaindex.get_feature_from_code(feature1)['values']
        self.assertEqual(feature_vals == feature1_vals)

        feature_vals = self.aaindex.get_feature_from_code(feature2)['values']
        self.assertEqual(feature_vals == feature2_vals)

        feature_vals = self.aaindex.get_feature_from_code(feature3)['values']
        self.assertEqual(feature_vals == feature3_vals)

        feature_vals = self.aaindex.get_feature_from_code(feature4)['values']
        self.assertEqual(feature_vals == feature4_vals)


    def test_feature_codes(self):
        """
        Test Case to check that correct feature/index codes are in the parsed JSON
        of all AAI records. Also testing that random erroneous feature codes are
        not present.

        """
        print('Testing AAi feature indices...')

        #testing actual index codes
        feature1 = 'VHEG790101'
        feature2 = 'PONP800107'
        feature3 = 'OOBM770102'
        feature4 = 'NADH010101'

        #testing index codes are in the AAI1
        self.assertIn(feature1, self.aaindex.get_feature_codes())
        self.assertIn(feature2, self.aaindex.get_feature_codes())
        self.assertIn(feature3, self.aaindex.get_feature_codes())
        self.assertIn(feature4, self.aaindex.get_feature_codes())

        #testing bogus index codes
        feature5 = 'ABC1234'
        feature6 = 'ABC12345'
        feature7 = 'ABC123456'
        feature8 = 'ABC1234567'

        #testing errenous index codes are not in the AAI1
        self.assertNotIn(feature5, self.aaindex.get_feature_codes())
        self.assertNotIn(feature6, self.aaindex.get_feature_codes())
        self.assertNotIn(feature7, self.aaindex.get_feature_codes())
        self.assertNotIn(feature8, self.aaindex.get_feature_codes())

    def test_category(self):
        """
        Test Case to check that the correct category value is returned for each
        index.

        """
        print('Testing AAi categories for indices...')

        feature1 = 'TANS770109'
        feature2 = 'RADA880107'
        feature3 = 'ZIMJ680103'
        feature4 = 'WOLS870102'

        cat1 = self.aaindex.get_category(feature1)
        cat2 = self.aaindex.get_category(feature2)
        cat3 = self.aaindex.get_category(feature3)
        cat4 = self.aaindex.get_category(feature4)

        self.assertEqual(cat1, 'sec_struct')
        self.assertEqual(cat2, 'hydrophobic')
        self.assertEqual(cat3, 'polar')
        self.assertEqual(cat4, 'meta')

    def test_aaindex_names(self):
        """
        Test Case to check that the correct names/description is returned for each
        index.

        """
        print('Testing AAi names from indices...')

        names = self.aaindex.get_feature_names()

        name1 = names[50]
        name2 = names[140]
        name3 = names[368]
        name4 = names[560]

        self.assertEqual(name1, 'Frequency of the 3rd residue in turn (Chou-Fasman, 1978b)')
        self.assertTrue(name2.startswith('Average relative probability'))
        self.assertIn('chain reversal', name3)
        self.assertTrue(name4.endswith('(Karkbara-Knisley, 2016)'))

    def test_aaindex_refs(self):
        """
        Test Case to check that the correct references are returned for each
        index.

        """
        print('Testing AAI references from indices...')

        feature1 = 'VELV850101'
        feature2 = 'QIAN880139'
        feature3 = 'NAKH900112'
        feature4 = 'LEVM760103'

        ref1 = self.aaindex.get_ref_from_code(feature1)
        ref2 = self.aaindex.get_ref_from_code(feature2)
        ref3 = self.aaindex.get_ref_from_code(feature3)
        ref4 = self.aaindex.get_ref_from_code(feature4)

        self.assertTrue(ref1.startswith('Veljkovic'))
        self.assertIn("globular proteins using neural network models", ref2)
        self.assertIn("hydrophobicity of amino acid composition of mitochondrial proteins", ref3)
        self.assertTrue(ref4.endswith('Nucleic Acids Res. 28, 374 (2000).'))


if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests.test_aaindex -v (-v to give more verbose output)
# python -m unittest test_module1 test_module2
# python -m unittest test_module.TestClass
# python -m unittest test_module.TestClass.test_method
# python -m unittest tests/test_something.py
# python -m unittest -v test_module
# python -m unittest discover -s project_directory -p "*_test.py"
    # @unittest.expectedFailure
# assertIs(a, b)
# assertIsNot(a, b)

# assertIsNone(x)
# assertIsNotNone(x)

# assertIn(a, b)

# assertNotIn(a, b)
# assertIsInstance(a, b)
#
# assertNotIsInstance(a, b)

# assertGreaterer()
# assertLess()
