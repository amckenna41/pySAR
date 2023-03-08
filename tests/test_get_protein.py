################################################################################
###############             Get_protein Module Tests            ################
################################################################################

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import unittest
import shutil

from pySAR.get_protein import *

class GetProteinTests(unittest.TestCase):
    """
    Test suite for testing get_protein module and functionality 
    in pySAR package. 

    Test Cases
    ----------
    test_fasta:
        testing correct fasta functionality.
    test_protein_uniprot:
        testing correct protein uniprot functionality.
    test_protein_ncbi:
        testing correct protein ncbi functionality.
    test_protein_pdb:
        testing correct protein pdb functionality.
    """
    def setUp(self):
        """ Initialise test suite vars. """
        self.test_dir = os.path.join("tests", "test_data")
        self.temp_test_dir = os.path.join("tests", "temp_test_data")

        #create temp dir to store test outputs
        if not (os.path.isdir(self.temp_test_dir)):
            os.mkdir(self.temp_test_dir)

    def test_fasta(self):
        """ Test fasta function which returns protein seq from fasta file. """
        test_seq = fasta(os.path.join(self.test_dir, "test_fasta.fasta"))
#1.)
        with self.assertRaises(OSError):
            test_seq = fasta("incorrect_filepath.fasta")
        with self.assertRaises(ValueError):
            test_seq.index("Z")
        with self.assertRaises(ValueError):
            test_seq.index("B")
#2.)
        self.assertEqual(len(test_seq), 1255, 
            "Expected sequence to be of length 1255, got {}.".format(len(test_seq)))
        self.assertTrue(test_seq.startswith('MFIFLL'), 
            "Expected sequence to start with MFIFLL, got {}.".format(test_seq.startswith('MFIFLL')))
#3.)
        self.assertEqual(test_seq.find("X"), -1, "Expected X to not be found in sequence.")
        self.assertEqual(test_seq.find("B"), -1, "Expected B to not be found in sequence.")
        self.assertEqual(test_seq.find("Z"), -1, "Expected Z to not be found in sequence.")

    @unittest.skip("Don't want to overload the Uniprot server each time tests are run.")
    def test_protein_uniprot(self):
        """ Testing getting protein sequence from Uniprot database. """
        protid_1 = "P94485"
        protid_2 = "Q7CPU9"
        protid_3 = "Q802Y8"
        protid_4 = "Q04731"

        download_protein_from_uniprot(protid_1, save_dir=self.temp_test_dir)
        download_protein_from_uniprot(protid_2, save_dir=self.temp_test_dir)
        download_protein_from_uniprot(protid_3, save_dir=self.temp_test_dir)
        download_protein_from_uniprot(protid_4, save_dir=self.temp_test_dir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, protid_1 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, protid_2 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, protid_3 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, protid_4 + '.fasta')),
            "Protein file not found in download folder.")

    @unittest.skip("Don't want to overload the NCBI server each time tests are run.")
    def test_protein_ncbi(self):
        """ Testing getting protein sequence from NCBI database. """
        ncbi_1 = "P94485"
        ncbi_2 = "Q7CPU9"
        ncbi_3 = "Q802Y8"
        ncbi_4 = "Q04731"

        download_protein_from_ncbi(ncbi_1, save_dir=self.temp_test_dir)
        download_protein_from_ncbi(ncbi_2, save_dir=self.temp_test_dir)
        download_protein_from_ncbi(ncbi_3, save_dir=self.temp_test_dir)
        download_protein_from_ncbi(ncbi_4, save_dir=self.temp_test_dir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, ncbi_1 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, ncbi_2 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, ncbi_3 + '.fasta')),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, ncbi_4 + '.fasta')),
            "Protein file not found in download folder.")

    @unittest.skip("Don't want to overload the PDB server each time tests are run.")
    def test_protein_pdb(self):
        """ Testing getting protein sequence from PDB database. """
        pdb_1 = "7MLP"
        pdb_2 = "2GPZ"
        pdb_3 = "7JPH"
        pdb_4 = "7JPI"

        download_protein_from_pdb(pdb_1, save_dir=self.temp_test_dir)
        download_protein_from_pdb(pdb_2, save_dir=self.temp_test_dir)
        download_protein_from_pdb(pdb_3, save_dir=self.temp_test_dir)
        download_protein_from_pdb(pdb_4, save_dir=self.temp_test_dir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, pdb_1 + ".pdb")),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, pdb_2 + ".pdb")),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, pdb_3 + ".pdb")),
            "Protein file not found in download folder.")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_test_dir, pdb_4 + ".pdb")),
            "Protein file not found in download folder.")
        
    def tearDown(self):
        """ Cleanup tests and delete any temp dirs. """
        shutil.rmtree(self.temp_test_dir)