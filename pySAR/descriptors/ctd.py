################################################################################
#################                    CTD                       #################
################################################################################

import pandas as pd
import math
import copy
from varname import nameof

"""
References
----------
    [1]: Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim.
        Prediction of protein folding class using global description of amino
        acid sequence. Proc.Natl. Acad.Sci.USA, 1995, 92, 8700-8704.
    [2]: Inna Dubchak, Ilya Muchink, Christopher Mayor, Igor Dralyuk and Sung-Hou
        Kim. Recognition of a Protein Fold in the Context of the SCOP
        classification. Proteins: Structure, Function and
        Genetics, 1999, 35, 401-407.
"""

hydrophobicity = {"name": "hydrophobicity", "1": "RKEDQN", "2": "GASTPHY", "3": "CLVIMFW"}
# '1' -> Polar; '2' -> Neutral, '3' -> Hydrophobicity

normalized_VDWV = {"name": "normalized_VDWV", "1": "GASTPD", "2": "NVEQIL", "3": "MHKFRYW"}
# '1' -> (0-2.78); '2' -> (2.95-4.0), '3' -> (4.03-8.08)

polarity = {"name": "polarity", "1": "LIFWCMVY", "2": "CPNVEQIL", "3": "KMHFRYW"}
# '1' -> (4.9-6.2); '2' -> (8.0-9.2), '3' -> (10.4-13.0)

charge = {"name": "charge", "1": "KR", "2": "ANCQGHILMFPSTWYV", "3": "DE"}
# '1' -> Positive; '2' -> Neutral, '3' -> Negative

sec_struct = {"name": "secondary_struct", "1": "EALMQKRH", "2": "VIYCWFT", "3": "GNPSD"}
# '1' -> Helix; '2' -> Strand, '3' -> coil

solvent_accessibility = {"name": "solvent_accessibility", "1": "ALFCGIVW", "2": "RKQEND", "3": "MPSTHY"}
# '1' -> Buried; '2' -> Exposed, '3' -> Intermediate

polarizability = {"name": "polarizability", "1": "GASDT", "2": "CPNVEQIL", "3": "KMHFRYW"}
# '1' -> (0-0.108); '2' -> (0.128-0.186), '3' -> (0.219-0.409)

#object of physiochemical properties to use for calculating CTD descriptors 
ctd_properties = {
    nameof(hydrophobicity): hydrophobicity,
    nameof(normalized_VDWV): normalized_VDWV,
    nameof(polarity): polarity,
    nameof(charge): charge,
    nameof(sec_struct): sec_struct,
    nameof(solvent_accessibility): solvent_accessibility,
    nameof(polarizability): polarizability
}

def str_to_num(sequence, property):
    """
    Convert sequences str to number from input physiochemical property.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : dict
        dictionary of values for specified physiochemical property.

    Returns
    -------
    :sequence_converted : str
        converted protein sequence into numerical format. 
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    sequence_converted = copy.deepcopy(sequence)

    for key, value in list(property.items()):
        if (key == "name"):
            continue
        for index in value:
            sequence_converted = sequence_converted.replace(index, key)

    return sequence_converted

def ctd_composition(sequence, property="hydrophobicity"):
    """
    Calculate composition physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : str (default = "hydrophocity")
        key for dictionary of values of specified physiochemical property.

    Returns
    -------
    :result_df : pd.DataFrame
        dataframe of calculated composition values for sequence using
        selected physiochemical property.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    #get CTD property values from dict, if property not in dict set to default of hydrophobicity
    if (property not in list(ctd_properties.keys())):
        prop = ctd_properties["hydrophobicity"]
    else:
        prop = ctd_properties[property]

    seq = str_to_num(sequence, prop)
    result = {}
    
    #calculate descriptor values, append to result dict
    result[prop["name"] + '_CTD_C_1'] = round(float(seq.count("1"))/len(sequence), 3) 
    result[prop["name"] + '_CTD_C_2'] = round(float(seq.count("2"))/len(sequence), 3)
    result[prop["name"] + '_CTD_C_3'] = round(float(seq.count("3"))/len(sequence), 3)

    #transform values and columns to DataFrame
    result_df = pd.DataFrame([list(result.values())], columns=list(result.keys()))

    return result_df

def ctd_transition(sequence, property="hydrophobicity"):
    """
    Calculate transition physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : str (default = "hydrophocity")
        key for dictionary of values of specified physiochemical property.

    Returns
    -------
    :result_df : pd.DataFrame
        dataframe of calculated transition values for sequence using
        selected physiochemical property.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    #get CTD property values from dict, if property not in dict set to default of hydrophobicity
    if (property not in list(ctd_properties.keys())):
        prop = ctd_properties["hydrophobicity"]
    else:
        prop = ctd_properties[property]

    seq = str_to_num(sequence, prop)
    result = {}

    #calculate descriptor values, append to result dict
    result[prop["name"] + "_CTD_T_12"] = round(
        float(seq.count("12") + seq.count("21")) / (len(sequence)-1), 3)
    result[prop["name"] + "_CTD_T_13"] = round(
        float(seq.count("13") + seq.count("31")) / (len(sequence)-1), 3)
    result[prop["name"] + "_CTD_T_23"] = round(
        float(seq.count("23") + seq.count("32")) / (len(sequence)-1), 3)

    #transform values and columns to DataFrame
    result_df = pd.DataFrame([list(result.values())], columns=list(result.keys()))

    return result_df

def ctd_distribution(sequence, property="hydrophobicity"):
    """
    Calculate distribution physiochemical/structural descriptor.

    Parameters
    ----------
    :sequence : str
        input protein sequence in str form.
    :property : str (default = "hydrophocity")
        key for dictionary of values of specified physiochemical property.

    Returns
    -------
    :result_df : pd.DataFrame
        dataframe of calculated distribution values for sequence using
        selected physiochemical property.
    """
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    #get CTD property values from dict, if property not in dict set to default of hydrophobicity
    if (property not in list(ctd_properties.keys())):
        prop = ctd_properties["hydrophobicity"]
    else:
        prop = ctd_properties[property]

    result = {}
    seq = str_to_num(sequence, prop)
    
    #iterate through sequence, calculating distribution descriptor values using property
    for key, value in prop.items():
       if (key=="name"):
        continue
       num = seq.count(key)
       ink = 1
       indexk = 0
       cds = []
       while ink <= num:
           indexk = seq.find(key,indexk) + 1
           cds.append(indexk)
           ink = ink + 1

       if cds == []:
           result[prop["name"] + "_CTD_D_" + key + "_001"] = 0
           result[prop["name"] + "_CTD_D_" + key + "_025"] = 0
           result[prop["name"] + "_CTD_D_" + key + "_050"] = 0
           result[prop["name"] + "_CTD_D_" + key + "_075"] = 0
           result[prop["name"] + "_CTD_D_" + key + "_100"] = 0
       else:
           result[prop["name"] + "_CTD_D_" + key + "_001"] = round(
            float(cds[0]) / len(seq) * 100, 3
           )
           result[prop["name"] + "_CTD_D_" + key + "_025"] = round(
            float(cds[int(math.floor(num * 0.25)) - 1]) / len(seq) * 100, 3
           )
           result[prop["name"] + "_CTD_D_" + key + "_050"] = round(
            float(cds[int(math.floor(num * 0.5)) - 1]) / len(seq) * 100, 3
           )
           result[prop["name"] + "_CTD_D_" + key + "_075"] = round(
            float(cds[int(math.floor(num * 0.75)) - 1]) / len(seq) * 100, 3
           )
           result[prop["name"] + "_CTD_D_" + key + "_100"] = round(
            float(cds[-1]) / len(seq) * 100, 3
           )

    #transform values and columns to DataFrame
    result_df = pd.DataFrame([list(result.values())], columns=list(result.keys()))

    return result_df

def ctd_(sequence, property=ctd_properties[nameof(hydrophobicity)], all_ctd=True):
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
    #check input sequence is a string, if not raise type error
    if not isinstance(sequence, str):
        raise TypeError('Input sequence must be a string, got input of type {}'.format(type(sequence)))

    #initialise ctd dataframes
    comp_df = pd.DataFrame()
    trans_df = pd.DataFrame()
    distr_df = pd.DataFrame()

    #if using single property, calculate each of the CTD descriptors individually
    if not all_ctd:
        comp_df = ctd_composition(sequence, property=property)
        trans_df = ctd_transition(sequence, property=property)
        distr_df = ctd_distribution(sequence, property=property)
    else:
        #if using all calculable properties, calculate CTD descriptors for each property
        for prop in ctd_properties:
            comp = ctd_composition(sequence, property=ctd_properties[prop])
            comp_df = pd.concat([comp_df, comp], axis=1)
            trans = ctd_transition(sequence, property=ctd_properties[prop])
            trans_df = pd.concat([trans_df, trans], axis=1)
            distr = ctd_distribution(sequence, property=ctd_properties[prop])
            distr_df = pd.concat([distr_df, distr], axis=1)

    #concatenate all descriptors
    ctd = pd.concat([comp_df, trans_df, distr_df], axis=1)
    
    return ctd