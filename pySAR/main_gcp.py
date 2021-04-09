
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0,'/ProAct')
#
# from ProAct.aaindex import *
# from ProAct.utils import *
# from ProAct.model import *
# from ProAct.proDSP import *
# from ProAct.evaluate import *
# from ProAct.plots import *
# from ProAct.descriptors import *
# from ProAct.ProAct import *
# from ProAct.encoding import *

# cwd = os.getcwd()
# print('cwd= ',cwd)
# print('lisitng dir')
# print(os.listdir(cwd)) # returns list
# print(os.path.basename(cwd))
# # if os.path.basename(cwd) != 'ProAct':
# #     os.chdir('ProAct')
# print('cd into user/dir')
# os.chdir('/user_dir')
#
# cwd = os.getcwd()
# print(os.listdir(cwd)) # returns list
#
# print('cwd= ',cwd)

# from . globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR #think this works??
# from .globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR #think this also works??

from . globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR, CURRENT_DATETIME#think this also works??

from . aaindex import  AAIndex
from . model import Model
from . proDSP import ProDSP
from . evaluate import Evaluate
from . ProtSAR import ProtSAR
from . utils import *
from . descriptors import Descriptors
from . plots import plot_reg

import datetime, time
import argparse
import itertools
import pickle
import yaml
import io
from difflib import get_close_matches
import json
import sklearn
import scipy
from google.cloud import storage

BUCKET_NAME = "keras-python-models-2"

storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)


def upload_file(blob_path, filepath):

    """
    Description:
        Upload blob to bucket
    Args:
        blob_path (str): path of blob object in bucket
        filepath (str): local filepath of object

    Returns:
        None
    """
    print('Uploading blob to GCP Storage')
    blob = bucket.blob(blob_path)

    try:
        blob.upload_from_filename(filepath)
    except Exception as e:
        print("Error uploading blob {} to storage bucket {} ".format(blob_path, e.message))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Protein Sequence Activity Relationship (name)')

    parser.add_argument('-input_data', '--input_data', type=str, default="",
                        help='', required=False)

    parser.add_argument('-job_name', '--job_name', type=str, default="",
                        help='', required=False)

    parser.add_argument('-dataset', '--dataset', type=str, default='T50.txt',
                        help='', required=False)

    parser.add_argument('-activity', '--activity', type=str, default='T50',
                        help='', required=False)

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

    input_data = args.input_data

    aaindex = AAIndex()
    encoding = Encoding(data_json=input_data)
    aa_df = encoding.aai_encoding(aaindex, combo2 = True, cutoff=1, verbose=True)

    utils.save_results(aa_df,results.csv)


    #upload history blob
    blob_path = os.path.join(args.job_name, 'results_' + CURRENT_DATETIME +'.csv')
    blob = bucket.blob(blob_path)

    #upload history to bucket
    upload_file(blob_path,'results.csv')
