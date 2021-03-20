
#prepare dataset
import pandas as pd
import numpy as np
import os
from globals import *
import shutil
import json
import csv
from pathlib import Path

def valid_sequence(sequences):

    valid_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    invalid_indices = {}
    for seq in range(0,len(sequences)-1):
        for aa in range(0,len(sequences[seq])):
            if (sequences[seq][aa] not in valid_amino_acids):
                invalid_indices['Sequence: '+str(seq)] = aa
    return invalid_indices

def remove_gaps(dataset):

    data_df = pd.read_csv(dataset, sep=",", header=0)

    for row in range(0, len(data_df['sequence'])):
        data_df['sequence'][row] = data_df['sequence'][row].replace("-","")

    data_df.to_csv(dataset, index=False)


def output_results(args,metrics):

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_path = 'model_output_' + CURRENT_DATETIME
    # if os.path.isdir(os.path.join(OUTPUT_DIR,model_path)):
    if os.path.isdir(OUTPUT_FOLDER):
        # os.rmdir(os.path.join(OUTPUT_DIR,model_path))
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=False, onerror=None)

    # os.makedirs(os.path.join(OUTPUT_DIR,model_path))
    os.makedirs(OUTPUT_FOLDER)

    dataset = str(args.dataset)
    dataset = os.path.splitext(os.path.basename(dataset))[0]
    activity = str(args.activity)
    aaindices = args.aaindices
    aa_spectrum = str(args.aa_spectrum)
    window = str(args.window)
    filter = str(args.filter)
    descriptors = args.descriptors
    model_ = str(args.model)
    model_params = args.model_params

    df_values = [dataset, activity, aaindices, aa_spectrum, window, filter,
                descriptors, model_, model_params] + list(metrics.values())
    df_cols = ['Dataset','Activity','AA Indices','AA Spectrum','Window','Filter',
                'Descriptors','Model','Model Parameters'] + list(metrics.keys())

    print('len_vals', (df_values))
    print('len_colls', (df_cols))

    # output_df = pd.DataFrame([list(metrics.values())], columns = list(metrics.keys()))
    output_df = pd.DataFrame([df_values], columns = df_cols)

    # output_df.to_csv((os.path.join(OUTPUT_DIR,model_path, 'results.csv')), index=False)
    output_df.to_csv(os.path.join(OUTPUT_FOLDER,'results.csv'), index=False)

    # output_to_json((os.path.join(OUTPUT_DIR,model_path, 'results.csv')), df_cols)
    output_to_json((os.path.join(OUTPUT_FOLDER,'results.csv')), df_cols)

def output_to_json(csvfile, fields):

    #get csv file name - change .csv to .json
    csv_path = Path(csvfile).resolve()
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    json_out_name = os.path.join(os.path.dirname(csvfile),csv_basename+'.json')

    print('csvfile',csvfile)

    csv_out = open(csvfile, 'r')
    json_out = open(json_out_name, 'w')

    reader = csv.DictReader(csv_out, fields)

    print('json_out_name',json_out_name)

    for row in reader:
        json.dump(row, json_out)
        json_out.write('\n')


# Function to convert a CSV to JSON
# Takes the file paths as arguments
# def make_json(csvFilePath, jsonFilePath):
#
#     # create a dictionary
#     data = {}
#
#     # Open a csv reader called DictReader
#     with open(csvFilePath, encoding='utf-8') as csvf:
#         csvReader = csv.DictReader(csvf)
#
#         # Convert each row into a dictionary
#         # and add it to data
#         for rows in csvReader:
#
#             # Assuming a column named 'No' to
#             # be the primary key
#             key = rows['No']
#             data[key] = rows
#
#     # Open a json writer, and use the json.dumps()
#     # function to dump data
#     with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
#         jsonf.write(json.dumps(data, indent=4))
#
# # Driver Code
#
# # Decide the two file paths according to your
# # computer system
# csvFilePath = r'Names.csv'
# jsonFilePath = r'Names.json'
#
# # Call the make_json function
# make_json(csvFilePath, jsonFilePath)


def get_AAIndex_X(data_df_features):

    X = data_df_features
    X = np.array(X)
    X_concat = np.concatenate(X, axis = 0)
    X_len = X.shape[0]
    protein_len = len(X[0])
    X = X_concat.reshape([X_len, protein_len])

    return X

def get_Y(dataset):

    if dataset.lower() == 'T50':
        Y = 't50'
    elif dataset.lower() == 'enantioselectivity':
        Y = 'e-value'

    return Y

def standardise(data):

    data = data - np.mean(data)
    data = data/np.std(data)

    return data

def normalise():
    pass
