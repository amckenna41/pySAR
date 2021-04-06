
################################################################################
#################             Utilities Modules                #################
################################################################################

import pandas as pd
pd.options.mode.chained_assignment = None  #stop pandas warnings, default='warn'
import numpy as np
import os
import shutil
import json
import yaml
import csv
from pathlib import Path

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR


def valid_sequence(sequences):

    """
    Function that iterates through all protein sequences and validates that
    each sequence is made up of valid canonical amino acid letters. If no
    invalid values are found then None will be returned. If invalid letters
    are found in the sequence, the sequence index and the index of the value
    in the sequence will be appened to a dict.

    Parameters
    ----------
    sequences: np.ndarray
        array of protein sequences.

    Returns
    -------
    None or invalid_indices : None/dict
        if no invalid values found in the protein sequences, None returned. if
        invalid values found, dict returned in the form {sequence index: invalid
        value in sequence index}.

    """
    #valid canonical amino acid letters
    valid_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    invalid_indices = {}

    #iterate through all sequences, validating that there are no invalid values
    #   present in the sequences, if there are then append to dict
    for seq in range(0,len(sequences)-1):
        for aa in range(0,len(sequences[seq])):
            if (sequences[seq][aa] not in valid_amino_acids):
                invalid_indices['Sequence: '+str(seq)] = aa

    #if no invalid values found in sequences return None, else return dict of
    #   invalid index values
    if invalid_indices == {}:
        return None
    else:
        return invalid_indices

def remove_gaps(protein_seqs):

    """
    Function that removes any gaps ('-') from the protein sequences in the input.
    The descriptors cannot be calculated if a '-' value is passsed into their
    respective funtions so gaps need to be removed. Removing the gaps has the same
    effect as setting the value at the index of the sequence to 0 and has no effect
    on the descriptors calculation.

    Parameters
    ----------
    protein_seqs: np.ndarray
        array of protein sequences.

    Returns
    -------
    protein_seqs : np.ndarray
        returns the same inputted protein sequences but with any gaps ('-') removed.
    """

    for row in range(0, len(protein_seqs)):
        try:
            # print(row)
            protein_seqs[row] = protein_seqs[row].replace("-","")
        except ValueError:
            print('Error removing gaps from sequences')
            return None
    return protein_seqs

#check the output type for this func
def flatten(array):

    """
    Lambda function for flattening list of lists or array of lists into one
    1 Dimensional array/list. Input must contain an array of arrays of the same
    length. Input will be flattened into a 1-dimensional array of size M * N
    where M = len(array) and N = len(array[0]). The flattened output can then be
    reshaped into the requried shape and format.

    Parameters
    ----------
    array: np.ndarray / list
        array of arrays or list of lists to be flattened

    Returns
    -------
    flatten(array) : list/array
        flattened 1 dimensional output as calculated from Lambda function.

    """
    flatten = lambda array: [item for sublist in array for item in sublist]
    try:
        return flatten(array)
    except:
        raise ValueError('Error flattening array of type: {} and size {}'.format(type(array),len(array)))

def zero_padding(seqs):
    """
    Pad seqences in seqs with 0's such that every sequence is of the same length
    of max(len(seq)).

    Parameters
    ----------
    seqs: np.ndarray / list
        array or list of encoded protein sequences

    Returns
    -------
    seqs: np.ndarray / list
        input sequences but with every sequence in the object now zero paddded
        to be the same length.

    """
    max_len = len(max(seqs,key=len))

    for s in range(0,len(seqs)):
        if len(seqs[s])<max_len:
            seqs[s]+= [0] * (max_len - len(seqs[s]))

    return seqs

def parse_json(data_json):

    #To DO: Check if input file is of type json or yaml
        assert os.path.isfile(data_json), 'Input data file not correct filepath'
        _, file_extension = os.path.splitext(data_json)
        print(file_extension)

        if (file_extension == '.yml') or (file_extension == '.yaml'):
            print('hello')
            try:
                with open(input_path) as f:
                    # The FullLoader parameter handles the conversion from YAML
                    # scalar values to Python the dictionary format
                    data = yaml.load(f, Loader=yaml.FullLoader)
            except OSError('Error opening Yaml File {}'.format(data_json)):
                return None
        # elif (file_extension == ".json"):

        # elif (file_extension == '.json'):
        #     with open(input_path) as f:
        #         data = json.load(f)
        #     dataset = data['dataset']
        #     activity = data['activity']

        # data_json = json.loads(args.input_data)
        try:
            with open(data_json, 'r') as js:
                try:
                    data_json = json.loads(js.read())
                except ValueError('Error decoding JSON: {}'.format(data_json)):
                    return None
        except OSError('Error opening JSON file {} '.format(data_json)):
            return None

        try:
            dataset = data_json['dataset']
        except:
            raise KeyError('')

        seq_col = data_json['sequence_col']
        activity = data_json['activity']
        aa_indices =  data_json['aa_indices']

        if aa_indices!="":
            using_aa_features = True
        else:
            using_aa_features = False

        window =  data_json['window']
        filter =  data_json['filter']
        spectrum =  data_json['spectrum']
        descriptors =  data_json['descriptors']

        if descriptors!="":
            using_desc_features = True
        else:
            using_desc_features = False

        algorithm =  data_json['algorithm']
        parameters =  data_json['parameters']
        test_split =  data_json['test_split']

        return dataset, seq_col, activity, aa_indices, window, filter, \
        spectrum, descriptors, algorithm, parameters, test_split


def output_encoding(encoding_strat, model, output):

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = encoding_strat +'_' + model +'_' + CURRENT_DATETIME +'.csv'

    output.to_csv(os.path.join(OUTPUT_DIR,output_path), index=False)

    output_to_json((os.path.join(OUTPUT_DIR,output_path)), output.columns)

def create_output_dir():

    """
    Create output directory pointed to by global OUTPUT_DIR folder and create a
    folder according to the OUTPUT_FOLDER global variable within this directory,
    used for storing the outputs/results from current job. Each output folder will
    have a unique name as the current DateTime will be used in its naming.

    Parameters
    ----------

    Returns
    -------

    """
    #if directory doesnt exist then create it
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    #if output folder already exists then delete it
    if os.path.isdir(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=False, onerror=None)

    #create output folder in directory
    os.makedirs(OUTPUT_FOLDER)

def save_results(results, name):

    """
    Create output directory pointed to by global OUTPUT_DIR folder and create a
    folder according to the OUTPUT_FOLDER global variable within this directory,
    used for storing the outputs/results from current job. Each output folder will
    have a unique name as the current DateTime will be used in its naming.

    Parameters
    ----------

    Returns
    -------

    """
    if isinstance(results,dict):

        with open(os.path.join(OUTPUT_FOLDER, name+'.csv'), 'w') as f:
            w = csv.DictWriter(f, results.keys())
            w.writeheader()
            w.writerow(results)

    elif isinstance(results, pd.DataFrame):

        results.to_csv(os.path.join(OUTPUT_FOLDER, name+'.csv'), index=False)

def output_results(args,metrics):


    create_output_dir()

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


def output_to_yml(output):

    pass

# def get_top_k():
#
#     #output top K results from inputted results file
#     pass


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
