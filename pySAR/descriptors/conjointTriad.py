################################################################################
############                    Conjoint Triad                     #############
################################################################################

import pandas as pd

#amino acid triads
aa_triads = {
    1: ["A", "G", "V"],
    2: ["I", "L", "F", "P"],
    3: ["Y", "M", "T", "S"],
    4: ["H", "N", "Q", "W"],
    5: ["R", "K"],
    6: ["D", "E"],
    7: ["C"],
}

def conjoint_triad(sequence):
    """
    Calculate Conjoint Triad features (CTriad) for a protein sequence. CTF
    mainly considers neighbor relationships in protein sequences by encoding
    each protein sequence using the triad (continuous three amino acids)
    frequency distribution extracted from a 7-letter reduced alphabet [7]. This
    descriptor calculates 343 different features (7x7x7), with the output
    being of shape 343 x 1 for a sequence.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.

    Returns
    -------
    :ct_df : pd.Dataframe
        pandas Dataframe of CTriad descriptor values for all protein sequences. Dataframe
        will be of the shape 343 x 1, where 343 is the number of features calculated 
        from the descriptor for a sequence.

    References
    ----------
    [1]: J. Shen et al., “Predicting protein-protein interactions based only on sequences
        information,” Proc. Natl. Acad. Sci. U. S. A., vol. 104, no. 11, pp. 4337–4341, 2007.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    con_triad = {}
    _aa_triads = {}

    #expand aa_triads into form {AminoAcid: triad}
    for i in aa_triads:
        for j in aa_triads[i]:
            _aa_triads[j] = i

    #get protein number for each triad
    protein_num = sequence
    for i in _aa_triads:
        protein_num = protein_num.replace(i, str(_aa_triads[i]))

    #calculate 7 triads for amino acids
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                temp = str(i) + str(j) + str(k)
                con_triad[temp] = protein_num.count(temp)

    #transform descriptor values to dataframe
    con_triad_df = pd.DataFrame([list(con_triad.values())], columns=list(con_triad.keys()))

    return con_triad_df
