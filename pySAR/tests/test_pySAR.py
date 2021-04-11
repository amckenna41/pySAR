
################################################################################
#################             pySAR Module Tests             #################
################################################################################

import unittest

from pySAR import PySAR


class ProDSPTests(unittest.TestCase):

    def setUp(self):
        """
        Import the 4 test datasets used for testing the ProDSP methods
        """
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


    def test_pySAR(self):
        pass
    #
    # def __init__(self,data_json=None,dataset="",seq_col="sequence", activity="",\
    #     aa_indices="", window="hamming", filter="",spectrum="power", descriptors="", \
    #     algorithm="",parameters={}, test_split=0.2):
    #
    #     self.data_json = data_json
    #     self.dataset = dataset
    #     self.seq_col = seq_col
    #     self.activity = activity
    #     self.window = window
    #     self.filter = filter
    #     self.spectrum = spectrum
    #     self.aa_indices = aa_indices
    #     self.descriptors = descriptors
    #     self.algorithm = algorithm
    #     self.parameters = parameters
    #     self.test_split = test_split


        pass

    #create dummy dataset with errorneious seqs and actvity cols

    def test_read_dataset():
        pass


    def test_preprocessing():
        pass


    def test_encoded_aai():
        pass

    def test_desc_encoding():
        pass


    def test_aai_desc_encoding():
        pass


    def test_get_seqs(self):
        pass

    def test_get_activity(self):
        pass

    def test_to_JSON(self):
        pass


#         sklearn.utils.assert_all_finite(X, *, allow_nan=False)  Throw an error if array contains NaNs or Infs.
#
#
#         assert data[activity] contains no NA
#
#         assert proAct.data.ndim == 2
#         assert X and Y in proAct.data have the same length
#
#         if __name__ == '__main__':
#             unittest.main(verbosity=2)
# #test_fft

#test_power

#test_real

#test_imag

#test_freqs

#test np.isinf == False

#test np.isnan == False
