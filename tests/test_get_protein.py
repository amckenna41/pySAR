import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import unittest

from pySAR.get_protein import *

class GetProteinTests(unittest.TestCase):
    """
    Unit tests for testing the get_protein module of pySAR.
    """
    def test_fasta(self):
        """ Test get_protein_from_fasta function which returns protein seq from fasta file. """
        test_seq = get_protein_from_fasta(os.path.join("tests", "test_data", "test_fasta.fasta"))
#1.)
        with self.assertRaises(OSError):
            test_seq = get_protein_from_fasta("incorrect_filepath.fasta")
        with self.assertRaises(ValueError):
            test_seq.index("Z")
        with self.assertRaises(ValueError):
            test_seq.index("B")
#2.)
        self.assertEqual(len(test_seq), 91)
        self.assertTrue(test_seq.startswith('MNVKKA'))
#3.)
        self.assertEqual(test_seq.find("X"), -1)
        self.assertEqual(test_seq.find("B"), -1)
        self.assertEqual(test_seq.find("Z"), -1)

    @unittest.skip("Don't want to overload the Uniprot server each time tests are run.")
    def test_protein_uniprot(self):
        """ Testing getting protein sequence from Uniprot database. """
        protid_1 = "P94485"
        protid_2 = "Q7CPU9"
        protid_3 = "Q802Y8"
        protid_4 = "Q04731"
        invalid_protid_1 = "abcde"
        invalid_protid_2 = "fghi"
        protid_savedir = os.path.join("tests", "test_data")

        download_protein_from_uniprot(protid_1, save_dir=protid_savedir)
        download_protein_from_uniprot(protid_2, save_dir=protid_savedir)
        download_protein_from_uniprot(protid_3, save_dir=protid_savedir)
        download_protein_from_uniprot(protid_4, save_dir=protid_savedir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join(protid_savedir, protid_1)))
        self.assertTrue(os.path.isfile(os.path.join(protid_savedir, protid_2)))
        self.assertTrue(os.path.isfile(os.path.join(protid_savedir, protid_3)))
        self.assertTrue(os.path.isfile(os.path.join(protid_savedir, protid_4)))
#2.)
        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_uniprot(invalid_protid_1, save_dir=protid_savedir)

        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_uniprot(invalid_protid_2, save_dir=protid_savedir)

    @unittest.skip("Don't want to overload the NCBI server each time tests are run.")
    def test_protein_ncbi(self):
        """ Testing getting protein sequence from NCBI database. """
        ncbi_1 = "P94485"
        ncbi_2 = "Q7CPU9"
        ncbi_3 = "Q802Y8"
        ncbi_4 = "Q04731"
        invalid_protid_1 = "abcde"
        invalid_protid_2 = "fghi"
        ncib_savedir = os.path.join("tests", "test_data")

        download_protein_from_ncbi(ncbi_1, save_dir=ncib_savedir)
        download_protein_from_ncbi(ncbi_2, save_dir=ncib_savedir)
        download_protein_from_ncbi(ncbi_3, save_dir=ncib_savedir)
        download_protein_from_ncbi(ncbi_4, save_dir=ncib_savedir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join('pySAR', ncib_savedir, ncbi_1)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', ncib_savedir, ncbi_2)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', ncib_savedir, ncbi_3)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', ncib_savedir, ncbi_4)))
#2.)
        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_ncbi(invalid_protid_1, save_dir=ncib_savedir)

        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_ncbi(invalid_protid_2, save_dir=ncib_savedir)

    @unittest.skip("Don't want to overload the PDB server each time tests are run.")
    def test_protein_pdb(self):
        """ Testing getting protein sequence from PDB database. """
        pdb_1 = "7MLP"
        pdb_2 = "2GPZ"
        pdb_3 = "7JPH"
        pdb_4 = "7JPI"
        invalid_protid_1 = "abcde"
        invalid_protid_2 = "fghi"
        pdb_savedir = os.path.join("tests", "test_data")

        download_protein_from_pdb(pdb_1, save_dir=pdb_savedir)
        download_protein_from_pdb(pdb_2, save_dir=pdb_savedir)
        download_protein_from_pdb(pdb_3, save_dir=pdb_savedir)
        download_protein_from_pdb(pdb_4, save_dir=pdb_savedir)
#1.)
        self.assertTrue(os.path.isfile(os.path.join('pySAR', pdb_savedir, pdb_1)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', pdb_savedir, pdb_2)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', pdb_savedir, pdb_3)))
        self.assertTrue(os.path.isfile(os.path.join('pySAR', pdb_savedir, pdb_4)))
#2.)
        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_ncbi(invalid_protid_1, save_dir=pdb_savedir)

        with self.assertRaises(requests.exceptions.RequestException, msg='Requests Error raised, invalid url given.'):
            download_protein_from_ncbi(invalid_protid_2, save_dir=pdb_savedir)