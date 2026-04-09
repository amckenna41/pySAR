################################################################################
#################             Utilities Module                ##################
################################################################################

import pandas as pd
import numpy as np
import os
import csv

from .globals_ import OUTPUT_FOLDER, CURRENT_DATETIME

class Map(dict):
    """
    Instantiating this class will convert a dict such that it can be accessed using 
    dot notation which makes it easier for accessing the individual elements and 
    parameters of the config files. It also works for nested dicts.

    Parameters 
    ==========
    :dict: dict 
        input dictionary to be mapped into dot notation.

    Usage
    =====
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    # Add new key
    m.new_key = 'Hello world!'
    # Or
    m['new_key'] = 'Hello world!'
    print m.new_key
    print m['new_key']
    # Update values
    m.new_key = 'Yay!'
    # Or
    m['new_key'] = 'Yay!'
    # Delete key
    del m.new_key
    # Or
    del m['new_key']
   
    References
    ----------
    [1] https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if (isinstance(arg, dict)):
                for k, v in arg.items():
                    self[k] = v

        if (kwargs):
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]

def valid_sequence(sequences):
    """
    Function that iterates through all protein sequences and validates that
    each sequence is made up of valid canonical amino acid letters. If no
    invalid values are found then None will be returned. If invalid letters
    are found in the sequence, the sequence index and the index of the value
    within the sequence will be appened to a dict and returned. In the output
    dict, the sequence reference is not zero indexed so the index to the first 
    sequence will be 1 not 0. 

    Parameters
    ==========
    :sequences: list/np.ndarray
        list or array of protein sequences.

    Returns
    =======
    :None or invalid_indices: None/list
        if no invalid values found in the protein sequences, None returned. If
        invalid values found, list of dicts returned in the form
        {sequence index: invalid value in sequence index}.
    
    Usage
    -----
    seq = ["ACDEF", "GHIKLM", "ABCDE"]
    seq_check = valid_sequence(seq)
    #{'Sequence #3': '(B at index #2)'}
    """
    #if input is string, cast to a list so it is iterable
    if (isinstance(sequences, str)):
        sequences = [sequences]

    #valid canonical amino acid letters
    valid_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',\
        'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    invalid_indices = []

    #iterate through all sequences, validating that there are no invalid values
    #present in the sequences, if there are then append to list of invalid indices
    for seq in range(0, len(sequences)):
        for aa in range(0, len(sequences[seq])):
            if (sequences[seq][aa] not in valid_amino_acids):
                invalid_indices.append(
                    {f'Sequence #{seq+1}': f'({sequences[seq][aa]} at index #{aa})'})

    #if no invalid values found in sequences return None, else return list of
    #dicts containing invalid index and invalid values
    return invalid_indices or None

def remove_gaps(sequences):
    """
    Function that removes any gaps ('-') from the protein sequences in the input.
    The descriptors cannot be calculated if a '-' value is passsed into their
    respective funtions so gaps need to be removed. Removing the gaps has the same
    effect as setting the value at the index of the sequence to 0 and has no effect
    on the descriptor calculation. Input can be a string or list/array of sequences.

    Parameters
    ==========
    :sequences: str/list/np.ndarray
        string of 1 protein sequence or array/list of protein sequences.

    Returns
    =======
    :protein_seqs: np.ndarray
        returns the same inputted protein sequence(s) but with any gaps ('-') removed.
    """
    #string input: remove gaps and return as string
    if isinstance(sequences, str):
        return sequences.replace("-", "")

    #pd.Series input: process each element independently using vectorised str.replace
    if isinstance(sequences, pd.Series):
        return sequences.str.replace("-", "", regex=False).reset_index(drop=True)

    #list/array input: treat as single sequence of chars — join after removing gap chars
    cleaned = ''.join(str(c) for c in sequences if str(c) != '-')
    return [cleaned]

def flatten(array):
    """
    Lambda function for flattening list of lists or array of lists into one
    1-dimensional array/list. Input must contain an array of arrays of the same
    length. Input will be flattened into a 1-dimensional array of size (M * N, 1)
    where M = len(array) and N = len(array[0]). The flattened output can then be
    reshaped into the required shape and format.

    Parameters
    ==========
    :array: np.ndarray/list
        array of arrays or list of lists to be flattened.

    Returns
    =======
    :flatten(array/list): np.ndarray/list
        flattened 1-dimensional list or array.
    """
    #if input is a string then return input as cannot be flattened
    if (isinstance(array, str)):
        return array

    #create flatten lambda function
    _flatten = lambda array: [item for sublist in array for item in sublist]

    #flatten array/list
    try:
        flattened_array = _flatten(array)
    except (TypeError, ValueError):
        raise TypeError(f'Error flattening array of type: {type(array)} and size {len(array)}.')

    #if input is a numpy array then reshape to 1D numpy array else return list
    if (isinstance(array,np.ndarray)):
        return (np.array(flattened_array).reshape([-1, 1]))
    else:
        return flattened_array

def zero_padding(sequences): 
    """ 
    Pad sequences in input array with 0's such that every sequence is of the same length
    of max(len(sequences)).

    Parameters
    ==========
    :sequences: np.ndarray/list
        array or list of encoded protein sequences.

    Returns
    =======
    :sequences: np.ndarray/list
        input sequences but with every sequence in the object now zero paddded
        to be the same length.
    """
    #no need to zero-pad if only one sequence passed in
    if (len(sequences) == 1):
        return sequences

    is_series = isinstance(sequences, pd.Series)
    is_ndarray = isinstance(sequences, np.ndarray)

    #get maximum length of all sequences
    max_len = len(max(sequences, key=len))

    #check if any sequence is shorter than max_len
    seq_at = lambda i: sequences.iloc[i] if is_series else sequences[i]
    if not any(len(seq_at(s)) < max_len for s in range(len(sequences))):
        return sequences

    #determine element type to choose padding strategy
    first_elem = seq_at(0)
    if isinstance(first_elem, str):
        #string sequences: pad with '0' character
        if is_series:
            return sequences.str.ljust(max_len, '0')
        seqs_list = list(sequences)
        for s in range(len(seqs_list)):
            if len(seqs_list[s]) < max_len:
                seqs_list[s] = seqs_list[s].ljust(max_len, '0')
        return np.array(seqs_list, dtype=sequences.dtype) if is_ndarray else seqs_list
    else:
        #list/array sequences: extend shorter sequences with zeros
        seqs_list = [list(s) for s in sequences]
        for s in range(len(seqs_list)):
            diff = max_len - len(seqs_list[s])
            if diff > 0:
                seqs_list[s] = seqs_list[s] + [0] * diff
        return np.array(seqs_list, dtype=object) if is_ndarray else seqs_list

def save_results(results, file_name, output_folder=""):
    """
    Save object DataFrame/Series containing metric names and their values captured from
    the encoding process. Save the results in this object to a CSV file named according
    to name input parameter. Function can also accept a dict of results.

    Parameters
    ==========
    :results: dict/pd.DataFrame/pd.Series
        object of the metrics and results from the encoding process. Ideally should
        be a dataframe/series but function also accepts a dict of results.
    :file_name: str
        file name to call results file.
    
    Returns
    =======
    None
    """
    #append extension if not in file name
    if (os.path.splitext(file_name)[1] == ""):
        file_name = file_name + '.csv'

    #set output folder to default (already timestamped) or append timestamp to custom folder
    if not output_folder:
        output_folder = OUTPUT_FOLDER
    else:
        output_folder = output_folder + '_' + CURRENT_DATETIME

    #create output folder if it doesn't exist
    if not (os.path.isdir(output_folder)):
        os.makedirs(output_folder)

    #output results to csv if results variable is a dictionary
    if (isinstance(results, dict)):
        with open(os.path.join(output_folder, file_name), 'w') as f:
            w = csv.DictWriter(f, results.keys())
            w.writeheader()
            w.writerow(results)
    #output results to csv if results variable is a dataframe or Series
    elif (isinstance(results, pd.DataFrame) or isinstance(results, pd.Series)):
        results.reset_index(drop=True, inplace=True)
        results.to_csv(os.path.join(output_folder, file_name))
    else:
        raise TypeError(f'Results object must be of type: dict, pd.Series or pd.DataFrame, got object of type {type(results)}.')