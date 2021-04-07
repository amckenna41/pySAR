#maybe move main up a dir
import pandas as pd
import numpy as np
# import  aaindex as aaindex
from aaindex import AAIndex
from utils import *
from model import *
from proDSP import *
from evaluate import *
from plots import *
from descriptors import *
from ProtSAR import *
from encoding import *
import datetime, time
import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Protein Sequence Activity Relationship (name)')

    parser.add_argument('-input_data', '--input_data', type=str, default="",
                        help='', required=False)

    args = parser.parse_args()

    input_data = args.input_data

    aaindex = AAIndex()
    # encoding = Encoding(data_json=input_data)
    # encoding = Encoding(data_json=input_data)

    #aa_df = encoding.aai_encoding(verbose=True)
    # desc_results = encoding.descriptor_encoding()

    #Filter encoding strategy input and then create class instances accordingly
    # proAct = ProtSAR(dataset="T50.txt" ,activity="T50",aa_indices=["LEVM780106"],algorithm="randomforest")
    proAct = ProtSAR(dataset="T50.txt" ,activity="T50",aa_indices="PONJ960101",algorithm="bagging")


    # encoded_seqs = proAct.aaindex_encoding(proAct.aa_indices)
    # proDSP = ProDSP(encoded_seqs, spectrum=proAct.spectrum, window=proAct.window, filter=proAct.filter)
    # proDSP.encode_seqs()
    aa_df = proAct.encode_aaindices()
    #

    # print(input_data)
    # aa_df = encoding.aai_descriptor_encoding()
    #
    # print(aa_df)
    # model = Model('randomforest')
    # def descriptor_encoding(self, descCombo = 1, verbose=True):

    # model.train_test_split(proAct.)
    # encoding.aai_encoding(combo2=2)
    # desc_results = encoding.descriptor_encoding(2)
    # print(desc_results)

    # aaindex = AAIndex()
    # encoded_seqs = proAct.aaindex_encoding(aaindex)
    # proDSP = ProDSP(encoded_seqs, spectrum=proAct.spectrum, window=proAct.window, filter=proAct.filter)
    # proDSP.encode_seqs()
    #
    # aa_df = proAct.encode_aaindices()
    # X = pd.DataFrame(proDSP.spectrum_encoding)
    # Y = proAct.get_activity()
    #
    # print(aa_df)
    # model.train_test_split(X, Y)
    # params = {'n_estimators':[100,200],'criterion':['mse']}
    #
    # model.hyperparameter_tuning(parameters=params)

    # X = pd.DataFrame(proDSP.spectrum_encoding)
    # print(X.shape)
    # Y = proAct.get_activity()
    # print(Y.shape)
    #
    # params = {'n_estimators':[100,200],'criterion':['mse']}
    #
    # hp = HyperparameterTuning(model,params, X, Y)
    # hp.hyperparameter_tuning()
    # aa_df = proAct.encode_aai_descriptor()
    # aa_df = proAct.encoded_desc()
    # print(aa_df)

#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Protein Sequence Activity Relationship (name)')
#
#     parser.add_argument('-input_data', '--input_data', type=str, default="",
#                         help='', required=False)

    # parser.add_argument('-dataset', '--dataset', type=str, default=(os.path.join('data','enantioselectivity.txt')),
    #                     help='', required=False)
    #
    # parser.add_argument('-activity', '--activity', type=str, default='e-value',
    #                     help='', required=False)
    #
    # parser.add_argument('-aaindices', '--aaindices', default=[],
    #                     help='',required=False)
    #
    # parser.add_argument('-aa_spectrum', '--aa_spectrum', default='power',
    #                     help='', required=False)
    #
    # parser.add_argument('-window', '--window', default='hamming',
    #                     help='', required=False)
    #
    # parser.add_argument('-filter', '--filter', default='',
    #                     help='', required=False)
    #
    # parser.add_argument('-descriptors', '--descriptors', default=[],
    #                     help='', required=False)
    #
    # parser.add_argument('-model', '--model', default='plsregression',
    #                     help='',required=False)
    #
    # parser.add_argument('-model_params', '--model_params', default={},
    #                     help='',required=False)
    #
    #
    # args = parser.parse_args()
    #
    # proAct = ProAct(args.input_data)


    # main2(args)
