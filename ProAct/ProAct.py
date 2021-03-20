
import pandas as pd
import numpy as np
from aaindex import *
from utils import *
from tests.test_aaindex import *
from model import *
from proteinSpectra import *
from evaluate import *
from plots import *
from descriptors import *
import argparse
import itertools
from difflib import get_close_matches
#1.) Dataset path and name passed as parameter to class - __init__ func
#2.) Import dataset and seqeunces into dataframe
#3.) Pre-processing all sequences
#4.) Import AAIndex Class
#5.) Exit class initialisation process
#6.) Class func to calcualte AAIndex def calculate_aai(index):
#7.) Class func to calculate descriptors def calcualte_descriptor(desc)
#8.) If AAI and Descriptors applied, concat function to concatenate features
#9.) Class func to build predictve model from accumulated features
#10.) Class func to plot and output results
# - Create config file that contains all the parameters etc , pass this into function
class ProAct():

    def __init__(self,dataset,activity,seq_col='sequence'):

        self.dataset = dataset
        self.seq_col = seq_col
        self.activity = activity

        using_aa_features = False
        using_desc_features = False

        if (os.path.isfile(dataset)):
            try:
                self.data = pd.read_csv(self.dataset, sep=",", header=0)

    #   self.data['activity'].fillna(0,inplace=True)    #fill with 0 any missing acitivty values

            except IOError as e:
                print('Error opening dataset file: ',dataset)
        else:
            print('Dataset file not in directory')
            return

        #pre-processing
        invalid_seq = valid_sequence(self.get_seqs())
        assert invalid_seq == {}, ('Invalid AA found in dataset '+ str(invalid_seq))
        assert activity in self.data
        assert seq_col in self.data

        # assertions....
        # self.pre_processing()  - check all sequences in dataset to see if they are correct and valid

    def get_seqs(self):

        return self.data[self.seq_col]

    def get_names(self):

        return self.data['name']

    def get_num_seqs(self):

        return len(self.data['name'])

    def get_seq_len(self):

        return len(self.data[self.seq_col][0])

    def get_activity(self):

        return self.data[self.activity].values.reshape((-1,1))

    def __len__(self):

        return len(self.data['name'])

    def using_aa(self):

        return using_aa_features

    def using_desc(self):

        return using_desc_features

    def head(self):

        return self.data.head()

    def aaindex_encoding(self, aaindex, indices):

        encoded_indices = []

        if not isinstance(indices, list):
            encoded_aai = aaindex.get_feature_from_code(indices)['values']
            # encoded_vals = list((aa_index.get_feature_from_code(indices)['values']).values())

            temp_seq_vals = []
            temp_all_seqs = []

            for protein in range(0, len(self.data['sequence'])):
                for aa in self.data['sequence'][protein]:
                    temp_seq_vals.append(encoded_aai[aa])

                temp_all_seqs.append(temp_seq_vals)
                temp_seq_vals = []


            encoded_ai_reshaped = np.reshape(temp_all_seqs, (self.get_num_seqs(), self.get_seq_len()))
            # encoded_ai_reshaped = np.reshape(temp_all_seqs, (152, 398))

            using_aa_features = True
            return encoded_ai_reshaped

        else:

            encoded_ai_reshaped = np.zeros((self.get_num_seqs(), self.get_seq_len()))

            #if multiple indices used then calcualte FFT encoding for each and then concatenate after each calculation
            for ind in range(0,len(indices)):
                encoded_aai = aaindex.get_feature_from_code(indices[ind])['values']

                temp_seq_vals = []
                temp_all_seqs = []

                for protein in range(0, len(self.data['sequence'])):
                    for aa in self.data['sequence'][protein]:
                        temp_seq_vals.append(encoded_aai[aa])

                    temp_all_seqs.append(temp_seq_vals)
                    temp_seq_vals = []

                encoded_ai_ = np.reshape(temp_all_seqs, (self.get_num_seqs(), self.get_seq_len()))

                if ind == 0:
                    encoded_ai_reshaped = encoded_ai_
                else:
                    encoded_ai_reshaped = np.concatenate((encoded_ai_reshaped,encoded_ai_), axis=1)

            using_aa_features = True

            self.encoded_ai_reshaped = encoded_ai_reshaped
            return self.encoded_ai_reshaped
            # encoded_df = pd.DataFrame(encoded_indices, columns=encoded_aai.keys())

    def desc_encoding(self, desc, descriptors):

        encoded_desc = []
        encoded_desc_vals = []

        if not isinstance(descriptors, list):
              encoded_desc = desc.get_descriptor_encoding(descriptors)

        #     encoded_desc = desc.get_descriptor_encoding(descriptors)
        #     for d in range(0,len(encoded_desc)):
        #         encoded_desc_vals.append(list(encoded_desc[d].values()))
        #
        # encoded_desc_vals = (list(itertools.chain.from_iterable(encoded_desc_vals)))
        # encoded_desc_vals = np.reshape(encoded_desc_vals, (self.get_num_seqs(), len(encoded_desc[0])))
        using_desc_features = True

        self.encoded_desc = encoded_desc

        return self.encoded_desc

    def __doc__(self):

        '''Documentaion of class:
           Description
           Parameters
           Attributes
           Example
           e.g model = PlsRegression()
           print(model.__doc__)
        '''
        pass

    #
    # def get_aaindex(indices):
    #
    #     loop through indices and get all index values
    #     self.aaindex[]
    #
    #     using_aaindex_feautures = True
    #
    #     if using_desc_features ==1:
    #         concat with aa_index_features
    #     return self.concatenated_aaindex_features
    #
    # def get_descriptors(desc):
    #
    #     loop through all dwscriptors:
    #     if self.descriptors doesnt exist then run import_descriptors
    #
    #     using_desc_feautures = True
    #
    #     if aa_index_features ==1:
    #         concat with using_desc_feautures
    #     return self.concartenatred_descriptor_feautes

#main for testing
def main2():

    aaindex2 = AAIndex2()

def main(args):

    dataset = str(args.dataset)
    activity = str(args.activity)
    aaindices = args.aaindices
    aa_spectrum = str(args.aa_spectrum)
    window = str(args.window)
    filter = str(args.filter)
    descriptors = args.descriptors
    model_ = str(args.model)
    model_params = args.model_params

    # from difflib import get_close_matches - validation for when using puts in model close to existing model:
    #e.g plsregresion instead of plsregression
    #matches = get_close_matches(model_, all_models))
    #chosen_model = matches[0]

    proAct = ProAct(dataset,activity)

    # indices = ['ZASB820101', 'ZHOH040101', 'ZHOH040102','ZHOH040102']
    indices = ['ZASB820101']
    desc = 'aa_comp'

    aa_index = AAIndex()
    #
    encoded_seqs = proAct.aaindex_encoding(aa_index,indices)
    proteinDSP = ProteinDSP(encoded_seqs)

    proteinDSP.encode_aaindex()

    encoded_aa = proteinDSP.fft
    encoded_aa_real = proteinDSP.fft.real
    encoded_aa_power = proteinDSP.power

    model = Model(model_)
    desc = Descriptors(proAct.get_seqs())

    encoded_desc = proAct.desc_encoding(desc, 'aa_comp')

    print(encoded_desc[0])

    #Concatenate AAIndex + Desc before building model

    model.build(encoded_aa_power, proAct.get_activity())

    Y_pred = model.model_fit.predict(model.X_test)

    r2 = r2_score(model.Y_test,Y_pred)
    metrics = get_all_metrics(model.model_fit, model.Y_test, Y_pred)

    output_results(args,metrics)

    plot_reg(model.Y_test, Y_pred, metrics['R2'])
    # #output results to CSV

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Protein Sequence Activity Relationship (name)')

    parser.add_argument('-dataset', '--dataset', type=str, default=(os.path.join('data','enantioselectivity.txt')),
                        help='', required=False)

    parser.add_argument('-activity', '--activity', type=str, default='e-value',
                        help='', required=False)

    parser.add_argument('-aaindices', '--aaindices', default=[],
                        help='',required=False)

    parser.add_argument('-aa_spectrum', '--aa_spectrum', default='power',
                        help='', required=False)

    parser.add_argument('-window', '--window', default='hamming',
                        help='', required=False)

    parser.add_argument('-filter', '--filter', default='',
                        help='', required=False)

    parser.add_argument('-descriptors', '--descriptors', default=[],
                        help='', required=False)

    parser.add_argument('-model', '--model', default='plsregression',
                        help='',required=False)

    parser.add_argument('-model_params', '--model_params', default={},
                        help='',required=False)

    args = parser.parse_args()

    main2()
