################################################################################
###############              Quasi Sequence Order                ###############
################################################################################

import os
import json
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

def seq_order_coupling_number(sequence, maxlag=30,
    distance_matrix_path="..data/schneider-wrede-physicochemical-distance-matrix.json"):
    """
    Calculate Sequence Order Coupling Number (SOCNum) features for the protein sequences.
    Sequence Order Coupling Number computes the dissimilarity between amino acid
    pairs. The distance between amino acid pairs is determined by d which varies
    between 1 to nlag. For each d, it computes the sum of the dissimilarities
    of all amino pairs. The number of output features can be calculated as N * 2,
    where N = maxlag, by default this value is 30 so 60 features are output.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.
    :maxlag : int (default = 30)
        maxlag is the maximum lag and the length of the protein should be larger
        than maxlag. Default set to 30.
    :distance_matrix_path : str (default = "schneider-wrede-physicochemical-distance-matrix")
        path to physiochemical distance matrix for calculating quasi sequence order.

    Returns
    -------
    :seq_order_df : pd.Series
        Series of SOCNum descriptor values for all protein sequences. Output
        will be of the shape N x M, where N is the number of protein sequences and
        M is the number of features calculated from the descriptor (calculated as
        N * 2 where N = maxlag).

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
    #open distance matrix json if present
    try:
        with open(distance_matrix_path, "r") as f:
            distance_matrix = json.load(f)
    except:
        raise OSError('Distance Matrix json ({}) not found.'.format(distance_matrix_path))

    #calculate sequence order coupling number for proteins using maxlag and specificed
    #physiochemical distance matrix
    seq_order = {}
    for i in range(maxlag):

        tau = 0.0
        for j in range(len(sequence - i+1)):

            aa_1 = sequence[j]
            aa_2 = sequence[j+(i+1)]
            tau = tau + math.pow(distance_matrix[aa_1 + aa_2], 2)

        tau = round(tau, 3)
        seq_order["seq_order" + str(i+1)] = tau

    #transform descriptor data into pandas Series
    seq_order_df = pd.Series(data=(list(seq_order.values())), index=list(seq_order.keys()))

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
    an output of N x 100 where N is the numnber of sequences.

    Parameters
    ----------
    sequence : str
        protein sequence in str form.
    max_lag : int (default = 30)
        A value for a lag, the max value is equal to the length of shortest peptide minus one.
    weight: float (default = 0.1)
        weighting factor
    distance_matrix_path : str (default = "schneider-wrede-physicochemical-distance-matrix")
        path to physiochemical distance matrix for calculating quasi sequence order.

    Returns
    -------
    quasi_seq_order_df : pd.Series
        dataframe of quasi-sequence-order descriptor values for the
        protein sequences, with output shape N x 100 where N is the number
        of sequences and 100 the number of calculated features.

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
    result = {}

    right_part = 0.0

    #calculate quasi sequence order using sequence order coupling number for
    #proteins using maxlag and specificed physiochemical distance matrix
    for i in range(max_lag):
        rightpart = rightpart + seq_order_coupling_number(
            sequence, i + 1, distance_matrix_path
        )

    aa_comp = AAComposition(sequence)
    temp = 1 + weight * rightpart
    for index, i in enumerate(aminoAcids):
        result["Quasi_seq_order1_" + str(index + 1)] = round(aa_comp[i] / temp, 6)

    right_part = []
    for i in range(max_lag):
        right_part.append(
            seq_order_coupling_number(sequence, i+1, distance_matrix_path)
            )

    temp = 1 + weight * sum(right_part)
    for index in range(20, 20 + max_lag):
        result["Quasi_seq_order2_" + str(index + 1)] = round(
            weight * right_part[index - 20] / temp, 6
        )

    return result
