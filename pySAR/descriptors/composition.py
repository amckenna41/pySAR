################################################################################
#############                   Composition                       ##############
################################################################################

import re
import pandas as pd
import math
from ..aaindex import *

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

def AAComposition(sequence):
    """
    Calculate Amino Acid Composition (AAComp) of protein sequence. AAComp
    describes the fraction of each amino acid type within a protein sequence,
    and is calculated as:

    AA_Comp(s) = AA(t)/N(s)

    where AA_Comp(s) is the AAComp of protein sequence s, AA(t) is the number
    of amino acid types t (where t = 1,2,..,20) and N(s) is the length of the
    sequence s.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.

    Returns
    -------
    :composition_df : pd.DataFrame
        pandas dataframe of AAComp for protein sequence. Dataframe will
        be of the shape 20 x 1, where 20 is the number of features 
        calculated from the descriptor (for the 20 amino acids).
    """
    composition = {}
    for aa in aminoAcids:
        composition[aa] = round(float(sequence.count(aa)) / len(sequence) * 100, 3)

    #transform values and columns to DataFrame
    composition_df = pd.DataFrame([list(composition.values())], columns=list(composition.keys()))

    return composition_df


def DipeptideComposition(sequence):
    """
    Calculate Dipeptide Composition (DPComp) for protein sequence.
    Dipeptide composition is the fraction of each dipeptide type within a
    protein sequence. With dipeptides being of length 2 and there being 20
    canonical amino acids this creates 20^2 different combinations, thus a
    400-Dimensional vector will be produced such that:

    DPComp(s,t) = AA(s,t) / N -1

    where DPComp(s,t) is the dipeptide composition of the protein sequence
    for amino acid type s and t (where s and t = 1,2,..,20), AA(s,t) is the number
    of dipeptides represented by amino acid type s and t and N is the total number
    of dipeptides.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.

    Returns
    -------
    :dipepComposition_df : pd.DataFrame
        pandas dataframe of dipeptide composition for protein sequence. Dataframe will
        be of the shape 400 x 1, where 400 is the number of features calculated 
        from the descriptor (20^2 for the 20 canonical amino acids).
    """
    dipepComposition = {}
    for i in aminoAcids:
        for j in aminoAcids:
            dipep = i + j
            dipepComposition[dipep] = round(
                float(sequence.count(dipep)) / (len(sequence)-1) * 100, 2
            )
        # dipepComposition[i] = round(float(sequence.count(i)) / len(sequence) *100, 3)

    #transform values and columns to DataFrame
    dipepComposition_df = pd.DataFrame([list(dipepComposition.values())], columns=list(dipepComposition.keys()))

    return dipepComposition_df


def TripeptideComposition(sequence):
    """
    Calculate Tripeptide Composition (TPComp) of protein sequence.
    Tripeptide composition is the fraction of each tripeptide type within a
    protein sequence. With tripeptides being of length 3 and there being 20
    canonical amino acids this creates 20^3 different combinations, thus a
    8000-Dimensional vector will be produced such that:

    TPComp(s,t,u) = AA(s,t,u) / N -1

    where TPComp(s,t,u) is the tripeptide composition of the protein sequence
    for amino acid type s, t and u (where s, t and u = 1,2,..,20), AA(s,t,u) is
    the number of tripeptides represented by amino acid type s and t, and N is
    the total number of tripeptides.

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.

    Returns
    -------
    :tripepComposition_df : pd.DataFrame
        pandas DataFrame of tripeptide composition for protein sequence. Dataframe will
        be of the shape 8000 x 1, where 8000 is the number of features calculated 
        from the descriptor (20^3 for the 20 canonical amino acids).
    """
    tripepComposition = {}
    tripeptides = []    #tripeptides = list()

    #get list of tripeptides
    for i in aminoAcids:
        for j in aminoAcids:
            for k in aminoAcids:
                tripeptides.append(i + j + k)

    #get frequency of each tripeptide in the sequence
    for i in tripeptides:
        tripepComposition[i] = len(re.findall(i, sequence))

    #transform values and columns to DataFrame
    tripepComposition_df = pd.DataFrame([list(tripepComposition.values())], columns=list(tripepComposition.keys()))

    return tripepComposition_df

def pseudoAAC(sequence, lamda=30, weight=0.05, properties=["ARGP820101", "KUHL950101"]):
    """
    Calculate Pseudo Amino Acid Composition features for the protein sequence.
    Similar to the quasi-sequence order descriptor, the pseudo amino acid descriptor is
    made up of a 50-dimensional vector in which the first 20 components are a weighted
    sum of the amino acid composition and 30 are physiochemical square correlations as
    dictated by the lamda and properties parameters. This generates an output of 
    [(20 + lamda), 1] - 50 x 1. By default, the physiochemical properties used are 
    hydrophobicity (ARGP820101) and hydrophillicity (KUHL950101) indices, with a 
    lamda of 30 and weight of 0.05.

    Parameters 
    ----------
    :sequence : str
        protein sequence in str form.
    :lamda : int (default = 30)
        lamda parameter that reflects the rank correlation and should be a non-negative
        integer and not larger than the length of the protein sequence.
    :weight : float (default = 0.05)
        weighting factor allowing for weights to be added to the additioanl descriptor
        components with respect to the conventional amino acid components. 
    :properties : str/array (default =["ARGP820101", "KUHL950101"])
        single or multiple amino acid index properties from the AAI database used for 
        calculating the sequence-order 

    Returns
    -------
    :pseudoAAComp_df : pd.Dataframe
        pandas Dataframe of pseudo amino acid composition for protein sequence. Dataframe will
        be of the shape [(20 + lamda),1] - 50 x 1, where 50 is the number of features calculated 
        from the descriptor. 

    References
    ----------
    [1]: Chou, K. C. (2001). Prediction of protein cellular attributes using pseudo-amino acid 
         composition. Proteins, 43(3), 246–255. https://doi.org/10.1002/prot.1035
    """

    #set lamda to its default value if <0, > sequence len or not an int
    if ((lamda < 0) or (lamda > len(sequence)) or not isinstance(lamda, int)):
        lamda = 30

    #ensure at least 1 property input to function and or properties is a list so it can be iterated over
    if (properties == "" or properties == []):
        raise ValueError('At least one property value must be input to function.')
    if (isinstance(properties, str)):   #cast properties to list if str
        properties = [properties]

    #initialise dicts to store AAI properties and values
    aai_properties = {}
    aaindex = AAIndex()
    for prop in properties:
        aai_properties[prop] = {}

    #iterate through list of properties, getting property values from AAIndex
    for prop in properties:

        #get property values from AAIndex & reshape
        aaindex[prop]['values'].pop('-', None)
        prop_aminoacid_values = aaindex[prop]['values']
     
        #normalise property values, calculate mean and std dev
        aai_property_vals = {}
        for i, j in prop_aminoacid_values.items():
            aai_property_vals[i] = (j - (sum(prop_aminoacid_values.values()) / len(prop_aminoacid_values.values()))) / _std(prop_aminoacid_values.values(), ddof=0)

        aa_counter = 0
        #assign property and associated amino acid values to aai_property_vals array
        for i, j in aai_property_vals.items():
            aai_property_vals[aminoAcids[aa_counter]] = aai_property_vals[i]
            aa_counter+=1
        aai_properties[prop] = aai_property_vals

    print(aai_properties)
    #### Pseudo AAC 1 ####

    #calculate pseudo AAC for sequence
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + sequenceOrderCorrelationFactor(
            sequence, i + 1, aai_properties
        )

    #get amino acid composition
    aaComp = AAComposition(sequence)    
    result = {}
    #applying weighting factor to components
    temp = 1 + weight * rightpart

    #append each descriptor feature value to results dict
    for index, i in enumerate(aminoAcids):
        result["PseudoAAC1_" + str(index + 1)] = round(aaComp.iloc[0][i] / temp, 3) 

    ##### Pseudo AAC 2 ####
    #calculate pseudo AAC for sequence 
    rightpart = []
    for i in range(lamda):
        rightpart.append(sequenceOrderCorrelationFactor(sequence, i + 1, aai_properties))

    #applying weighting factor to components
    temp = 1 + weight * sum(rightpart)

    #append each descriptor feature value to results dict
    for index in range(20, 20 + lamda):
        result["PseudoAAC2_" + str(index + 1)] = round(
            weight * rightpart[index - 20] / temp * 100, 3
        )

    #convert resultant descriptor values into a dataframe
    pseudoAAComp_df = pd.DataFrame([list(result.values())], columns=list(result.keys()))

    return pseudoAAComp_df

def sequenceOrderCorrelationFactor(sequence, k=1, properties=[]):
    """
    Calculate the sequence order correlation factor with gap = k for the inputted
    physiochemical properties. 

    Parameters
    ----------
    :k : int (default = 1)
        gap in sequence for calculating factor.
    :properties : list (default = [])
        list of physiochemical properties.

    Returns
    -------
    :seqOrderCorrelationFactor : float
        sequence order correlation factor with gap = k.
    
    References
    ----------
    [1]: Manish C. Saraf, Gregory L. Moore, Costas D. Maranas, Using multiple 
        sequence correlation analysis to characterize functionally important 
        protein regions, Protein Engineering, Design and Selection, Volume 16, 
        Issue 6, June 2003, Pages 397–406, https://doi.org/10.1093/protein/gzg053
    
    """
    #ensure at least 1 property input to function and or properties is a list so it can be iterated over
    if (properties == "" or properties == []):
        raise ValueError('At least one property value must be input to function.')

    #cast properties to list if str
    if (isinstance(properties, str)):   
        properties = [properties]
    
    LengthSequence = len(sequence)
    res = []
    for i in range(LengthSequence - k):
        AA1 = sequence[i]
        AA2 = sequence[i + k]
        res.append(correlation_function(AA1, AA2, properties))
    result = round(sum(res) / (LengthSequence - k), 3)
    return result

def correlation_function(aa1, aa2, property):

    NumAAP = len(property)
    theta = 0.0
    for prop in property:
        temp = NormalizeEachAAP(property[prop]) #[{"ABCD": "A":ddsd, ...}]
        theta = theta + math.pow(temp[aa1] - temp[aa2], 2)
    result = round(theta / NumAAP, 3)
    return result


def NormalizeEachAAP(AAP):
    """
    ########################################################################################
    All of the amino acid indices are centralized and
    standardized before the calculation.
    Usage:
    result=NormalizeEachAAP(AAP)
    Input: AAP is a dict form containing the properties of 20 amino acids.
    Output: result is the a dict form containing the normalized properties
    of 20 amino acids.
    ########################################################################################
    """
    if len(AAP.values()) != 20:
        print("You can not input the correct number of properities of Amino acids!")
    else:
        Result = {}
        for i, j in AAP.items():
            Result[i] = (j - _mean(AAP.values())) / _std(AAP.values(), ddof=0)

    return Result

# def amphiphilicPseudoAAC(sequence, lamda=30, weight=0.5, properties=["ARGP820101", "KUHL950101"]):
#     """

#     """

#     #set lamda to its default value if <0, > sequence len or not an int
#     if ((lamda < 0) or (lamda > len(sequence)) or not isinstance(lamda, int)):
#         lamda = 30

#     #keys of dicts should be AA not properties
#     aai_properties = {}
#     aai_property_vals = {}
#     aaindex = AAIndex()

#     #ensure properties is a list so it can be iterated over
#     if (isinstance(properties, list) or len(properties) == 1):
#         properties = [properties]

#     #get amino acid values from AAI for property 
#     for prop in properties: 
#         aai_properties[prop] = aaindex.get_values_from_record(prop)



#     rightpart = 0.0
#     for i in range(lamda):
#         rightpart = rightpart + sum(
#             GetSequenceOrderCorrelationFactorForAPAAC(sequence, k=i + 1)
#         )



#     AAC = GetAAComposition(ProteinSequence)

#     result = {}
#     temp = 1 + weight * rightpart
#     for index, i in enumerate(AALetter):
#         result["APAAC" + str(index + 1)] = round(AAC[i] / temp, 3)

#     return result

def _mean(listvalue):
    """
    ########################################################################################
    The mean value of the list data.
    Usage:
    result=_mean(listvalue)
    ########################################################################################
    """
    return sum(listvalue) / len(listvalue)

def _std(array, ddof=1):
    """
    Calculate the standard deviation of the array data.

    Parameters
    ----------
    :array : np.array
        numpy array of floats.

    Returns
    -------
    :res : np.array
        input array after standard deviation transformation.
    """
    return math.sqrt(sum([math.pow(i - sum(array) / len(array), 2) for i in array]) 
        / (len(array) - ddof))

    #     def sequenceOrderCorrelationFactor(sequence, k=1, properties=[]):
    # """
    # Calculate the sequence order correlation factor with gap = k for the inputted
    # physiochemical properties. 

    # Parameters
    # ----------
    # :k : int (default = 1)
    #     gap in sequence for calculating factor.
    # :properties : list (default = [])
    #     list of physiochemical properties.

    # Returns
    # -------
    # :seqOrderCorrelationFactor : float
    #     sequence order correlation factor with gap = k.
    
    # References
    # ----------
    # [1]: Manish C. Saraf, Gregory L. Moore, Costas D. Maranas, Using multiple 
    #     sequence correlation analysis to characterize functionally important 
    #     protein regions, Protein Engineering, Design and Selection, Volume 16, 
    #     Issue 6, June 2003, Pages 397–406, https://doi.org/10.1093/protein/gzg053
    
    # """
    # #ensure at least 1 property input to function and or properties is a list so it can be iterated over
    # if (properties == "" or properties == []):
    #     raise ValueError('At least one property value must be input to function.')

    # #cast properties to list if str
    # if (isinstance(properties, str)):   
    #     properties = [properties]
    
    # res = []
    # for i in range(len(sequence) - k):
    #     AA1 = sequence[i]
    #     AA2 = sequence[i + k]

    #     theta = 0.0
    #     for prop in properties:
    #         print(prop)
    #         temp_prop = np.array(list(properties[prop].values()))
    #         print(temp_prop)
    #         # temp_prop = np.array(list(properties[j].values()))
    #         temp_prop = temp_prop.reshape(-1,1)

    #         #normalise property values
    #         # norm_prop = preprocessing.normalize(temp_prop)
    #         norm_prop = ((sum(temp_prop) / len(properties[prop].values()))) / _std(temp_prop, ddof=0)

    #         theta = theta + math.pow(norm_prop[AA1] - norm_prop[AA2], 2)

    #     result = round(theta / len(properties), 3)
    #     res.append(round(theta / len(properties), 3))

    # result = round(sum(res) / (len(sequence) - k), 3)
    # return result