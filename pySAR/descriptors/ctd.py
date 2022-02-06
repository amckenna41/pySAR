################################################################################
#################                    CTD                       #################
################################################################################

import pandas as pd
import math

"""
References
----------
[1] Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim.
    Prediction of protein folding class using global description of amino
    acid sequence. Proc.Natl. Acad.Sci.USA, 1995, 92, 8700-8704.
[2] Inna Dubchak, Ilya Muchink, Christopher Mayor, Igor Dralyuk and Sung-Hou
    Kim. Recognition of a Protein Fold in the Context of the SCOP
    classification. Proteins: Structure, Function and
    Genetics, 1999, 35, 401-407.
"""

#list of physiochemical properties to use for calculating CTD descriptors, value
# of properties taken from referenced papers
_hydrophobicity = {"1": "RKEDQN", "2": "GASTPHY", "3": "CLVIMFW"}
# '1' -> Polar; '2' -> Neutral, '3' -> Hydrophobicity

_normalized_VDWV = {"1": "GASTPD", "2": "NVEQIL", "3": "MHKFRYW"}
# '1' -> (0-2.78); '2' -> (2.95-4.0), '3' -> (4.03-8.08)

_polarity = {"1": "LIFWCMVY", "2": "CPNVEQIL", "3": "KMHFRYW"}
# '1' -> (4.9-6.2); '2' -> (8.0-9.2), '3' -> (10.4-13.0)

_charge = {"1": "KR", "2": "ANCQGHILMFPSTWYV", "3": "DE"}
# '1' -> Positive; '2' -> Neutral, '3' -> Negative

_sec_struct = {"1": "EALMQKRH", "2": "VIYCWFT", "3": "GNPSD"}
# '1' -> Helix; '2' -> Strand, '3' -> coil

_solvent_accessibility = {"1": "ALFCGIVW", "2": "RKQEND", "3": "MPSTHY"}
# '1' -> Buried; '2' -> Exposed, '3' -> Intermediate

_polarizability = {"1": "GASDT", "2": "CPNVEQIL", "3": "KMHFRYW"}
# '1' -> (0-0.108); '2' -> (0.128-0.186), '3' -> (0.219-0.409)

all_properties = {}

def str_to_num(sequence, property):
    """
    Convert sequences str to number from input physiochemical property.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : dict
        dictionary of values fof specified physiochemical property.

    Returns
    -------
    :aaindex_metrics_df : pd.DataFrame
        dataframe of calculated metric values from generated predictive models
        encoded using indices in the AAI for the AAI encoding strategy.
    """
    sequence_converted = copy.deepcopy(sequence)
    for key, value in list(property.items()):
        for index in value:
            sequence_converted = sequence_converted.replace(index, key)

    return sequence_converted

def composition(sequence, property=_hydrophobicity):
    """
    Calculate composition physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : dict (default = _hydrophocity)
        dictionary of values of specified physiochemical property.

    Returns
    -------
    :aaindex_metrics_df : pd.DataFrame
        dataframe of calculated metric values from generated predictive models
        encoded using indices in the AAI for the AAI encoding strategy.
    """
    seq = str_to_num(sequence, property)
    property_name = [ k for k,v in locals().iteritems() if v == property][0]
    result = {}

    result[property_name + '_C_1'] = round(float(seq.count("1"))/len(sequence), 3)
    result[property_name + '_C_2'] = round(float(seq.count("2"))/len(sequence), 3)
    result[property_name + '_C_3'] = round(float(seq.count("3"))/len(sequence), 3)
    result_df = pd.Series(data=(list(result.values())), index=list(result.keys()))
    
    return result_df

def transition(sequence, property=_hydrophobicity):
    """
    Calculate transition physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : dict (default = _hydrophocity)
        dictionary of values of specified physiochemical property.

    Returns
    -------
    :aaindex_metrics_df : pd.DataFrame
        dataframe of calculated metric values from generated predictive models
        encoded using indices in the AAI for the AAI encoding strategy.
    """

    seq = str_to_num(sequence, property)
    property_name = [ k for k,v in locals().iteritems() if v == property][0]
    result = {}

    result[property_name + "_T_12"] = round(
        float(seq.count("12") + seq.count("21")) / (len(sequence-1),3)
    )
    result[property_name + "_T_13"] = round(
        float(seq.count("13") + seq.count("31")) / (len(sequence-1),3)
    )
    result[property_name + "_T_23"] = round(
        float(seq.count("23") + seq.count("32")) / (len(sequence-1),3)
    )

    return result

def distribution(sequence, property=_hydrophobicity):
    """
    Calculate distribution physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : dict (default = _hydrophocity)
        dictionary of values of specified physiochemical property.

    Returns
    -------
    :aaindex_metrics_df : pd.DataFrame
        dataframe of calculated metric values from generated predictive models
        encoded using indices in the AAI for the AAI encoding strategy.
    """
    result = {}
    seq = str_to_num(sequence, property)
    property_name = [ k for k,v in locals().iteritems() if v == property][0]

    for key, value in property.items():
       num = seq.count(key)
       ink = 1
       indexk = 0
       cds = []
       while ink <= num:
           indexk = seq.find(i,indexk) + 1
           cds.append(indexk)
           ink = ink + 1

       if cds == []:
           result[property_name + "_D_" + i + "_001"] = 0
           result[property_name + "_D_" + i + "_025"] = 0
           result[property_name + "_D_" + i + "_050"] = 0
           result[property_name + "_D_" + i + "_075"] = 0
           result[property_name + "_D_" + i + "_100"] = 0
       else:
           result[property_name + "_D_" + i + "_001"] = round(
            float(cds[0]) / len(seq) * 100, 3
           )
           result[property_name + "_D_" + i + "_025"] = round(
            float(cds[int(math.floor(len(num) * 0.25)) - 1]) / len(seq) * 100, 3
           )
           result[property_name + "_D_" + i + "_050"] = round(
            float(cds[int(math.floor(num * 0.5)) - 1]) / len(seq) * 100, 3
           )
           result[property_name + "_D_" + i + "_075"] = round(
            float(cds[int(math.floor(num * 0.75)) - 1]) / len(seq) * 100, 3
           )
           result[property_name + "_D_" + i + "_100"] = round(
            float(cds[-1]) / len(seq) * 100, 3
           )

       result_df = pd.Series(data=(list(result.values())), index=list(result.keys()))

    return result_df


def ctd_(sequence, property=_hydrophobicity):
    """
    Calculate Composition, transition and distribution (CTD) features of protein sequences.
    Composition is the number of amino acids of a particular property (e.g., hydrophobicity)
    divided by the total number of amino acids in a protein sequence. Transition
    characterizes the percent frequency with which amino acids of a particular
    property is followed by amino acids of a different property. Distribution
    measures the chain length within which the first, 25%, 50%, 75%, and 100% of
    the amino acids of a particular property are located, respectively [6].
    CTD functionality in the PyBioMed package uses the properties:
    Polarizability, Solvent Accessibility, Secondary Structure, Charge,
    Polarity, Normalized VDWV, Hydrophobicity. The output will be of shape
    N x 147 where N is the number of protein sequences. 21/147 will be
    composition, 21/147 will be transition and the remaining 105 are distribution.

    Returns
    -------
    :ctd_df : pd.DataFrame
        dataframe of CTD descriptor values for all protein sequences. DataFrame will
        be of the shape N x 147, where N is the number of protein sequences and
        147 is the number of features calculated from the descriptors.
    """

    comp = composition()
    trans = transition()
    distr = distribution()

    ctd = pd.concat([comp, trans, distr], axis=1)

    return ctd