################################################################################
#################             Utilities Modules                #################
################################################################################

import pandas as pd
pd.options.mode.chained_assignment = None  #stop pandas warnings, default='warn'
import numpy as np
import os
import shutil
import csv

from .globals_ import DATA_DIR, OUTPUT_DIR, OUTPUT_FOLDER

def valid_sequence(sequences):
    """
    Function that iterates through all protein sequences and validates that
    each sequence is made up of valid canonical amino acid letters. If no
    invalid values are found then None will be returned. If invalid letters
    are found in the sequence, the sequence index and the index of the value
    in the sequence will be appened to a dict.

    Parameters
    ----------
    :sequences : list/np.ndarray
        list or array of protein sequences.

    Returns
    -------
    :None or invalid_indices : None/list
        if no invalid values found in the protein sequences, None returned. if
        invalid values found, list of dicts returned in the form
        {sequence index: invalid value in sequence index}.
    """
    #if input is string, cast to a list so it is iterable
    if isinstance(sequences,str):
        sequences = [sequences]

    #valid canonical amino acid letters
    valid_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',\
        'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','-']
    invalid_indices = []

    #iterate through all sequences, validating that there are no invalid values
    #present in the sequences, if there are then append to list of invalid indices
    for seq in range(0,len(sequences)):
        for aa in range(0,len(sequences[seq])):
            if (sequences[seq][aa] not in valid_amino_acids):
                invalid_indices.append({'Sequence #'+str(seq) : 'Index #'+str(aa)})

    #if no invalid values found in sequences return None, else return list of
    #dicts containing invalid index and invalid values
    if invalid_indices == []:
        return None
    else:
        return invalid_indices

def remove_gaps(sequences):
    """
    Function that removes any gaps ('-') from the protein sequences in the input.
    The descriptors cannot be calculated if a '-' value is passsed into their
    respective funtions so gaps need to be removed. Removing the gaps has the same
    effect as setting the value at the index of the sequence to 0 and has no effect
    on the descriptors calculation. Input can be a string, list of array of sequences.

    Parameters
    ----------
    :sequences : str/list/np.ndarray
        string of 1 protein sequence or array/list of protein sequences.

    Returns
    -------
    :protein_seqs : np.ndarray
        returns the same inputted protein sequences but with any gaps ('-') removed.
    """
    #bool needed to ensure correct output format if input is str
    is_string=False   

    #convert single string into 1 element list
    if isinstance(sequences, str):
      is_string = True
      sequences = [sequences]     

    #concatenate multiple sequences into 1 iterable list
    if isinstance(sequences, list) and \
      len(sequences)>1:
      # for i in range(0,len(protein_seqs)):
      #   protein_seqs[i] = ''.join(protein_seqs[i])
      sequences = [''.join(sequences)]

    #iterate through sequences, removing any gaps ('-')
    for row in range(0, len(sequences)):
        try:
            sequences[row] = sequences[row].replace("-","")
        except:
            raise ValueError('Error removing gaps from sequences at index {} '.format(row))

    #if input was str then join list of sequences into one str
    if is_string:
       sequences = ''.join(sequences)

    return sequences

def flatten(array):
    """
    Lambda function for flattening list of lists or array of lists into one
    1 Dimensional array/list. Input must contain an array of arrays of the same
    length. Input will be flattened into a 1-dimensional array of size M * N
    where M = len(array) and N = len(array[0]). The flattened output can then be
    reshaped into the required shape and format.

    Parameters
    ----------
    :array : np.ndarray / list
        array of arrays or list of lists to be flattened.

    Returns
    -------
    :flatten(array/list) : np.ndarray/list
        flattened 1 dimensional list or array.
    """
    #if input is a string then return input as cannot be flattened
    if isinstance(array, str):
        return array

    #create flatten lambda function
    flatten = lambda array: [item for sublist in array for item in sublist]

    #flatten array/list
    try:
        flattened_array = flatten(array)
    except:
        raise ValueError('Error flattening array of type: {} and size {} \
            '.format(type(array),len(array)))

    #if input is a numpy array then reshape to 1D numpy array else return list
    if isinstance(array,np.ndarray):
        return (np.array(flattened_array).reshape([-1,1]))
    else:
        return flattened_array

def zero_padding(sequences):
    """
    Pad sequences in input array with 0's such that every sequence is of the same length
    of max(len(sequences)).

    Parameters
    ----------
    :sequences : np.ndarray / list
        array or list of encoded protein sequences.

    Returns
    -------
    :sequences: np.ndarray / list
        input sequences but with every sequence in the object now zero paddded
        to be the same length.
    """
    #no need to zero-pad if only one sequence passed in
    if len(sequences) == 1:
        return sequences

    #get maximum length of all sequences
    max_len = len(max(sequences,key=len))

    #iterate through all sequences, padding with 0's to max_len
    for s in range(0,len(sequences)):
        if len(sequences[s])<max_len:
            sequences[s]+= str(0) * (max_len - len(sequences[s]))

    return sequences

def create_output_dir():
    """
    Create output directory pointed to by global OUTPUT_DIR folder and create a
    folder according to the OUTPUT_FOLDER global var within this directory,
    used for storing the outputs/results from current job. Each output folder will
    have a unique name as the current date & time (DateTime) will be used in its naming.

    """
    #if directory doesn't exist then create it
    if not os.path.isdir(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except:
            raise OSError('Error creating directory {} '.format(OUTPUT_DIR))

    #if output folder already exists then delete it
    if os.path.isdir(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=False, onerror=None)

    #create output folder in directory
    try:
        os.makedirs(OUTPUT_FOLDER)
    except:
        raise OSError('Error creating directory {} '.format(OUTPUT_FOLDER))

def save_results(results, name):
    """
    Save object DataFrame/Series containing metric names and their values captured from
    the encoding process. Save the results in this object to a CSV file named according
    to name input parameter. Function can also accept a dict of results.

    Parameters
    ----------
    :results : dict/pd.DataFrame/pd.Series
        object of the metrics and results from the encoding process. Ideally should
        be a dataframe/series but function also accepts a dict of results.
    :name : str
        name to call results file.
    """
    #output results to csv if results variable is a dictionary
    if isinstance(results,dict):
        with open(os.path.join(OUTPUT_FOLDER, name + '.csv'), 'w') as f:
            w = csv.DictWriter(f, results.keys())
            w.writeheader()
            w.writerow(results)

    #output results to csv if results variable is a dataframe or Series
    elif isinstance(results, pd.DataFrame) or isinstance(results,pd.Series):
        results.reset_index(drop=True, inplace=True)
        results.to_csv(os.path.join(OUTPUT_FOLDER, name+'.csv'))

    else:
        raise TypeError('Results Object must be of type: dict, pd.Series or pd.DataFrame, got object of type {}'
            .format(type(results)))
