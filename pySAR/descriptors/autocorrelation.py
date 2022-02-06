################################################################################
#############                   Autocorrelation                   ##############
################################################################################

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

from ..aaindex import *

#list of amino acids
aminoAcids = [
    "-",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

def norm_moreaubroto_autocorrelation(
    sequence, lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
    "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
    ):
    """
    Calculate Normalized MoreauBrotoAuto Autocorrelation (NMBAuto) descriptor.
    Autocorrelation descriptors are a class of topological descriptors,
    also known as molecular connectivity indices, that describe the level of
    correlation between two objects (protein or peptide sequences) in terms of
    their specific structural or physicochemical properties, which are
    defined based on the distribution of amino acid properties along the sequence.
    By default, 8 amino acid properties are used for deriving the descriptors. The derivations
    and detailed explanations of this type of descriptor is outlind in [2].
    The NMBAuto descriptor is a type of Autocorrelation descriptor that uses
    the property values as the basis for measurement. Each autocorrelation will
    generate the number of features depending on the lag value and number of
    properties input with total features = lag * number of properties. Using 
    the default 8 properties with default lag value of 30, 240 features are 
    generated, the default 8 properties are:

    AccNo. CIDH920105 - Normalized Average Hydrophobicity Scales
    AccNo. BHAR880101 - Average Flexibility Indices
    AccNo. CHAM820101 - Polarizability Parameter
    AccNo. CHAM820102 - Free Energy of Solution in Water, kcal/mole
    AccNo. CHOC760101 - Residue Accessible Surface Area in Tripeptide
    AccNo. BIGC670101 - Residue Volume
    AccNo. CHAM810101 - Steric Parameter
    AccNo. DAYM780201 - Relative Mutability

    Parameters
    ----------
    :sequence : str
        protein sequence in str form.
    :lag : int (default = 30)
        A value for a lag, the max value is equal to the length of shortest peptide minus one.

    Returns
    -------
    :norm_moreaubroto_autocorr_df : pd.Series
        pandas Series of NMBAuto values for protein sequence. Output will
        be of the shape N x 1, where N is the number of features calculated from
        the descriptor. By default, the shape will be 240 x 1 (30 features per 
        property - using 8 properties).

    References
    ----------
    [1]: D. Raimondi, G. Orlando, W. F. Vranken, and Y. Moreau,
        “Exploring the limitations of biophysical propensity scales coupled with
        machine learning for protein sequence analysis,” Sci. Rep., vol. 9, no. 1, p. 16932, 2019.
    [2]: S. A. K. Ong, H. H. Lin, Y. Z. Chen, Z. R. Li, and Z. Cao, “Efficacy
        of different protein descriptors in predicting protein functional families,”
        BMC Bioinformatics, vol. 8, p. 300, 2007.
    """

    #set default lag if invalid value input
    if (lag>=len(sequence) or (lag<0) or not (isinstance(lag, int))):
        lag=30

    #validate at least 1 property input to function
    if (properties == "" or properties == []):
        raise ValueError('At least one property must be passed into function.')

    #if 1 property passed into function, wrap it to an iterable list
    if (isinstance(properties, str)):
        properties = [properties]
    
    #initialise dicts to store AAI properties and values
    aai_properties = {}
    aai_property_vals = {}
    aaindex = AAIndex()

    #iterate through list of properties, getting property values from AAIndex
    for prop in properties:

        #get property values from AAIndex & reshape
        temp_prop = np.array(list(aaindex.get_values_from_record(prop).values()))
        temp_prop = temp_prop.reshape(-1,1)

        #normalise property values
        norm_prop = preprocessing.normalize(temp_prop)

        aa_counter = 0
        for i in range(0,len(norm_prop)):
            aai_property_vals[aminoAcids[aa_counter]] = norm_prop[i]
            aa_counter+=1

        aai_properties[prop] = aai_property_vals

    result = {}

    #iterate through list of properties, calculating autocorrelation values for the sequence, append to results dict
    for key in aai_properties:
        temp = 0
        for i in range(1,lag+1):
            for j in range(len(sequence)-i):
                temp = temp + aai_properties[key][sequence[j]] * aai_properties[key][sequence[j+1]]

            if len(sequence) - i == 0:
                result["MoreauBrotoAuto_" + key + "_"+str(i)] = round(
                    temp[0] / (len(sequence)), 3
                )
            else:
                result["MoreauBrotoAuto_" + key + "_"+str(i)] = round(
                    temp[0] / (len(sequence) - i), 3
                )

    #result_df = pd.DataFrame(data=list(result.values()), columns=list(result.keys()))
    #convert resultant descriptor values into a Series
    result_df = pd.Series(data=(list(result.values())), index=list(result.keys()))

    return result_df


def moran_autocorrelation(
    sequence, lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
    "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
    ):
    """
    **refer to NMBAuto docstring for autocorrelation description.
    Moran autocorrelation (MAuto) utilizes property deviations from the
    average values.

    Parameters
    ----------
    **refer to NMBAuto docstring for autocorrelation parameters.

    Returns
    -------
    :moran_autocorr_df : pd.Series
        pandas Series of MAuto values for protein sequence. Output will
        be of the shape N x 1, where N is the number of features calculated from
        the descriptor. By default, the shape will be 240 x 1 (30 features per 
        property - using 8 properties).

    References
    ----------
    [1]: B. Hollas, “An analysis of the autocorrelation descriptor for molecules,”
        J. Math. Chem., vol. 33, no. 2, pp. 91–101, 2003.
    [2]: S. A. K. Ong, H. H. Lin, Y. Z. Chen, Z. R. Li, and Z. Cao, “Efficacy
        of different protein descriptors in predicting protein functional families,”
        BMC Bioinformatics, vol. 8, p. 300, 2007.    
    """

    #set default lag if invalid value input
    if (lag>=len(sequence) or (lag<0) or not (isinstance(lag, int))):
        lag=30

    #validate at least 1 property input to function
    if (properties == "" or properties == []):
        raise ValueError('At least one property must be passed into function.')

    #if 1 property passed into function, wrap it to an iterable list
    if (isinstance(properties, str)):
        properties = [properties]

    #initialise dicts to store AAI properties and values
    aai_properties = {}
    aai_property_vals = {}
    aaindex = AAIndex()

    #iterate through list of properties, getting property values from AAIndex
    for prop in properties:

        #get property values from AAIndex & reshape
        temp_prop = np.array(list(aaindex.get_values_from_record(prop).values()))
        temp_prop = temp_prop.reshape(-1,1)

        #normalise property values
        norm_prop = preprocessing.normalize(temp_prop)
        aa_counter = 0

        for i in range(0,len(norm_prop)):
            aai_property_vals[aminoAcids[aa_counter]] = norm_prop[i]
            aa_counter+=1
        aai_properties[prop] = aai_property_vals

    result = {}

    #iterate through list of properties, calculating autocorrelation values for the sequence, append to results dict
    for key in aai_properties:
        cds = 0
        for aa in aminoAcids:
            cds = cds + sequence.count(aa) * aai_properties[key][aa]
        prop_mean = cds / len(sequence)

        cc = []
        for aa in sequence:
            cc.append(aai_properties[key][aa])

        k = (np.std(cc, ddof=0)) **2

        for i in range(1,lag+1):
            temp = 0

            for j in range(len(sequence)-i):
                temp = temp + aai_properties[key][sequence[j]] - prop_mean * (
                    aai_properties[key][sequence[j+i]] - prop_mean
                )

            if len(sequence) - i == 0:
                result["MoranAuto_" + key + "_"+str(i)] = round(
                    temp[0] / ((len(sequence)) / k, 5)
                )
            else:
                result["MoranAuto_" + key + "_"+str(i)] = round(
                    temp[0] / ((len(sequence) - i)/k), 5
                )

    #convert resultant descriptor values into a Series
    result_df = pd.Series(data=(list(result.values())), index=list(result.keys()))

    return result_df

def geary_autocorrelation(
    sequence, lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
    "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
    ):
    """
    *refer to NMBAuto docstring for autocorrelation description.
    Geary Autocorrelation (GAuto) utilizes the square-difference of property
    values instead of vector-products (of property values or deviations).
    
    Parameters
    ----------
    **refer to NMBAuto docstring for autocorrelation parameters.

    Returns
    -------
    :geary_autocorr_df : pd.Series
        pandas Series of GAuto values for protein sequence. Output will
        be of the shape N x 1, where N is the number of features calculated from
        the descriptor. By default, the shape will be 240 x 1 (30 features per 
        property - using 8 properties).

    References
    ----------
    [1]: B. Hollas, “An analysis of the autocorrelation descriptor for molecules,”
        J. Math. Chem., vol. 33, no. 2, pp. 91–101, 2003.
    [2]: S. A. K. Ong, H. H. Lin, Y. Z. Chen, Z. R. Li, and Z. Cao, “Efficacy
        of different protein descriptors in predicting protein functional families,”
        BMC Bioinformatics, vol. 8, p. 300, 2007. 
    """

    #set default lag if invalid value input
    if (lag>=len(sequence) or (lag<0) or not (isinstance(lag, int))):
        lag=30

    #validate at least 1 property input to function
    if (properties == "" or properties == []):
        raise ValueError('At least one property must be passed into function.')

    #if 1 property passed into function, wrap it to an iterable list
    if (isinstance(properties, str)):
        properties = [properties]
        
    #initialise dicts to store AAI properties and values
    aai_properties = {}
    aai_property_vals = {}
    aaindex = AAIndex()

    #iterate through list of properties, getting property values from AAIndex
    for prop in properties:

        #get property values from AAIndex & reshape
        temp_prop = np.array(list(aaindex.get_values_from_record(prop).values()))
        temp_prop = temp_prop.reshape(-1,1)

        #normalise property values
        norm_prop = preprocessing.normalize(temp_prop)
        aa_counter = 0

        for i in range(0,len(norm_prop)):
            aai_property_vals[aminoAcids[aa_counter]] = norm_prop[i]
            aa_counter+=1

        aai_properties[prop] = aai_property_vals

    result = {}

    #iterate through list of properties, calculating autocorrelation values for the sequence, append to results dict
    for key in aai_properties:

        cc = []
        for aa in sequence:
            cc.append(aai_properties[key][aa])

        k = ((np.std(cc,ddof=0)) ** 2) * len(sequence) / (len(sequence) - 1)

        for i in range(1,lag+1):
            temp = 0

        for j in range(len(sequence)-i):
            temp = (temp + (
                aai_properties[key][sequence[j]] - aai_properties[key][sequence[j+i]]) **2
            )

        if len(sequence) - i == 0:
            result["GAuto_" + key + "_"+str(i)] = round(
                temp[0] / (2* (len(sequence))) / k,3
            )
        else:
            result["GAuto_" + key + "_"+str(i)] = round(
                temp[0] / (2* (len(sequence) -i))/k,3
            )

    #convert resultant descriptor values into a Series
    result_df = pd.Series(data=(list(result.values())), index=list(result.keys()))

    return result_df
