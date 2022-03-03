################################################################################
###############              Quasi Sequence Order                ###############
################################################################################

import os
import json
import pandas as pd
import math
from json import JSONDecodeError
import sys

from pySAR.globals_ import DATA_DIR

from . import composition

#list of amino acids
aminoAcids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

def seq_order_coupling_number(sequence, lag=30,
    distance_matrix="schneider-wrede-physiochemical-distance-matrix.json"):
    """
    Calculate Sequence Order Coupling Number (SOCNum) features for the protein sequences.
    Sequence Order Coupling Number computes the dissimilarity between amino acid
    pairs. The distance between amino acid pairs is determined by d which varies
    between 1 to nlag. For each d, it computes the sum of the dissimilarities
    of all amino acid pairs. The number of output features can be calculated as N * 2,
    where N = lag, by default this value is 30 so 60 features are output.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.
    :lag : int (default=30)
        lag is the maximum lag; the length of the protein should be larger
        than lag. Default set to 30.
    :distance_matrix_path : str (default="schneider-wrede-physicochemical-distance-matrix")
        path to physiochemical distance matrix for calculating quasi sequence order.

    Returns
    -------
    :seq_order_df : pd.Dataframe
        Dataframe of SOCNum descriptor values for all protein sequences. Output
        will be of the shape N x 1, where N is the number of features calculated 
        from the descriptor (calculated as M * 2 where M = lag).

    References
    ----------
    [1]: Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
        Quasi-Sequence-Order Effect. Biochemical and Biophysical Research Communications,
        2000, 278, 477-483.
    [2]: Kuo-Chen Chou and Yu-Dong Cai. Prediction of Protein Sucellular Locations by
        GO-FunD-PseAA Predictor. Biochemical and Biophysical Research Communications,
        2004, 320, 1236-1239.
    [3]: Gisbert Schneider and Paul Wrede. The Rational Design of Amino Acid Sequences
        by Artifical Neural Networks and Simulated Molecular Evolution: Do Novo Design
        of an Idealized Leader Cleavge Site. Biophys Journal, 1994, 66, 335-344.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    #get filepath to distance matrix json
    if not (os.path.isfile(distance_matrix)):
        if not (os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(sys.modules['pySAR'].__file__)), DATA_DIR, distance_matrix))):
            raise OSError('Distance Matrix json ({}) not found.'.format(os.path.join(os.path.dirname(os.path.abspath(sys.modules['pySAR'].__file__)), DATA_DIR, distance_matrix)))
        else:
            distance_matrix_path = os.path.join(os.path.dirname(os.path.abspath(sys.modules['pySAR'].__file__)), DATA_DIR, distance_matrix)
    else:
        distance_matrix_path = distance_matrix
    
    #open distance matrix json if present
    try:
        with open(distance_matrix_path, "r") as f:
            distance_matrix = json.load(f)
    except:
        raise JSONDecodeError('Error getting config JSON file: {}.'.format(distance_matrix_path))

    #set default lag if invalid value input
    if (lag>=len(sequence) or (lag<0) or not (isinstance(lag, int))):
        lag=30

    seq_order = {}

    #iterate through sequence, calculating the SOCNum using the selected distance matrix
    for i in range(lag):
        tau = 0.0
        for j in range(len(sequence) - (i+1) ):
            current_aa = sequence[j]
            next_aa = sequence[j + (i+1)]
            tau = round(tau + math.pow(distance_matrix[current_aa + next_aa], 2),3)

        #append SOCNum of current lag to seq_order dict
        seq_order["SOCNum" + str(i + 1)] = tau

    #transform descriptor data into pandas dataframe
    seq_order_df = pd.DataFrame([list(seq_order.values())], columns=list(seq_order.keys()))

    return seq_order_df

def quasi_sequence_order(sequence, max_lag=30, weight=0.1,
    distance_matrix_path="schneider-wrede-physicochemical-distance-matrix.json"):
    """
    Calculate Quasi Sequence Order features for the protein sequences.
    The quasi-sequence-order descriptors were proposed by K.C. Chou, et.al. [1].
    They are derived from the distance matrix between the 20 amino acids. By default,
    the Scheider-Wrede physicochemical distance matrix was used. Also utilised in
    the descriptor calculation is the Grantham chemical distance matrix. Both of
    these matrices are used by Grantham et. al. in the calculation
    of the descriptor [4]. 100 values are calculated per sequence, thus generating
    an output of 100 x 1 per sequence.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.
    :max_lag : int (default=30)
        A value for a lag, the max value is equal to the length of shortest peptide minus one.
    :weight: float (default = 0.1)
        weighting factor
    :distance_matrix_path : str (default="schneider-wrede-physicochemical-distance-matrix")
        path to physiochemical distance matrix for calculating quasi sequence order.

    Returns
    -------
    :quasi_seq_order_df : pd.Dataframe
        dataframe of quasi-sequence-order descriptor values for the
        protein sequences, with output shape 100 x 1 where 100 is the 
        number of calculated features per sequence.

    References
    ----------
    [1]: Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
        Quasi-Sequence-Order Effect. Biochemical and Biophysical Research Communications
        2000, 278, 477-483.
    [2]: Kuo-Chen Chou and Yu-Dong Cai. Prediction of Protein sucellular locations by
        GO-FunD-PseAA predictor, Biochemical and Biophysical Research Communications,
        2004, 320, 1236-1239.
    [3]: Gisbert Schneider and Paul wrede. The Rational Design of Amino Acid
        Sequences by Artifical Neural Networks and Simulated Molecular Evolution: Do
        Novo Design of an Idealized Leader Cleavge Site. Biophys Journal, 1994, 66, 335-344.
    [4]: Grantham, R. (1974-09-06). "Amino acid difference formula to help explain protein
        evolution". Science. 185 (4154): 862–864. Bibcode:1974Sci...185..862G.
        doi:10.1126/science.185.4154.862. ISSN 0036-8075. PMID 4843792. S2CID 35388307.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    quasi_seq_order = {}
    right_part = 0.0

    #calculate quasi sequence order using sequence order coupling number for
    #proteins using lag and specificed physiochemical distance matrix
    for i in range(max_lag):
        right_part = right_part + seq_order_coupling_number(
            sequence, i + 1, distance_matrix_path
        )

    aa_comp = composition.AAComposition(sequence)
    temp = 1 + weight * right_part
    
    for index, i in enumerate(aminoAcids):
        quasi_seq_order["Quasi_seq_order1_" + str(index + 1)] = round(aa_comp[i] / temp, 6)

    right_part = []
    for i in range(max_lag):
        right_part.append(
            seq_order_coupling_number(sequence, i+1, distance_matrix_path)
            )

    temp = 1 + weight * sum(right_part)
    for index in range(20, 20 + max_lag):
        quasi_seq_order["Quasi_seq_order2_" + str(index + 1)] = round(
            weight * right_part[index - 20] / temp, 6
        )

    #transform descriptor data into pandas dataframe
    quasi_seq_order_df = pd.DataFrame([list(quasi_seq_order.values())], columns=list(quasi_seq_order.keys()))

    return quasi_seq_order_df
