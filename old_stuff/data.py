import os
import pandas as pd

def load_data(dataset):

    assert dataset in ['t50', 'enantioselectivity', 'absorption','localization', 'solubility']

    dataset_path = os.path.join('data',dataset +'.txt')
    data_df = pd.read_csv(dataset_path, sep=",", header=0)

    return data_df

@property
def get_sequence(dataset, seq):

    return dataset['sequence'][seq]

@property
def get_num_mutants(dataset, seq):

    return data['m'][seq]

@property
def get_activity(dataset, seq, activity):

    return data[activity][seq]

@property
def get_total_seq(dataset):

    #count number of sequences in dataset - dataset size
    pass
