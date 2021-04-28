
################################################################################
#################                  Descriptors                 #################
################################################################################

import pandas as pd
import numpy as np
import datetime, time
import itertools
import pickle
import yaml
import os
import time
from difflib import get_close_matches
import json
import sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import  AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
# from pySAR import PySAR
import pySAR
import utils as utils
from PyBioMed.PyBioMed.PyProtein import AAComposition, Autocorrelation, CTD, \
    ConjointTriad, QuasiSequenceOrder, PseudoAAC

#double check seqorder coupling number maxlag parameter
###big change made to conjojt triad, replace range(8) to range(7)?
class Descriptors():

    """
    Class for calculating a wide variety of protein descriptors. These descriptors
    have been used in a wide variety of Bioinformaitcs applications including
    protein strucutral and function class prediction, protein-protein interactions,
    subcellular location, secondary structure prediction etc. They represent the different
    structural, functional & interaction profiles of proteins by exploring the
    features in the groups of composition, correlation, distribtuion of the constituent
    residues and their biochemical and physiochemical properties.

    This class allows calculation of the following descriptors: amino-acid compostion (AAComp),
    dipeptide composition (DPComp), tripeptide composition (TPComp), Normalized Moreau-Broto
    autocorrelation (NMBAuto), Moran Autocorrelation (MAuto), Geary Autocorrelation (GAuto),
    Composition (C), Transition (T), Distribution (D), CTD, Conjoint Triad (CTriad), sequence
    order coupling number (SOCNum), Quasi-sequence-order (QSOrder), Pseudo amino-acid
    composition-type 1(PAAcomp) and Amphiphilic pseudo amino-acid composition -type 2 (APAAComp).

    Descriptors are calculated using the PyProtein library of the PyBioMed Python
    package:
    https://pybiomed.readthedocs.io/en/latest/reference/PyProtein.html
    "Copyright (c) 2016-2017, Zhijiang Yao, Jie Dong and Dongsheng Cao
    All rights reserved"
    The output of the descriptor functions in PyBioMed are incompatible with some
    modules and functions in pySAR therefore, the output from PyBioMed is
    restrucured and reformatted into dataframe to make it useable with pySAR.

    Attributes
    ----------
    protein_seqs : np.array
        array of protein sequences that descriptors will be calculated for.
    desc_dataset : str (default = "descriptors.csv")
        csv file storing pre-calculated descriptor values for a particular
        dataset of protein sequences. By default the class will look for a file
        named 'descriptors.csv' and import the pre-calculated descriptor values. If
        file not found then descriptor values will have to be calculated. It is
        reccomended to pre-calculate the descriptor values to save time and resources
        calculating each time.
    all_desc : bool (default = False)
        if set to true, upon instantiation of a Descriptor class object, all
        available descriptors will be calculated and saved into an output csv.
        This should only have to be done once to calculate all descriptors in 1 go.

    Methods
    -------

    References
    ----------
    [1]: Dong, J., Yao, ZJ., Zhang, L. et al. PyBioMed: a python library for
         various molecular representations of chemicals, proteins and DNAs and
         their interactions. J Cheminform 10, 16 (2018).
         https://doi.org/10.1186/s13321-018-0270-2
    [2]: Broto P, Moreau G, Vandicke C: Molecular structures: perception,
         autocorrelation descriptor and SAR studies. Eur J Med Chem 1984, 19: 71–78.
    [3]: Ong, S.A., Lin, H.H., Chen, Y.Z. et al. Efficacy of different protein
         descriptors in predicting protein functional families. BMC Bioinformatics
         8, 300 (2007). https://doi.org/10.1186/1471-2105-8-300
    [4]: Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim. Prediction
         of protein folding class using global description of amino acid sequence.
         Proc.Natl. Acad.Sci.USA, 1995, 92, 8700-8704.
    [5]: Juwen Shen, Jian Zhang, Xiaomin Luo, Weiliang Zhu, Kunqian Yu, Kaixian Chen,
         Yixue Li, Huanliang Jiang. Predicting proten-protein interactions based only
         on sequences inforamtion. PNAS. 2007 (104) 4337-4341.
    [6]: Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
         Quasi-Sequence-Order Effect. Biochemical and Biophysical Research
         Communications 2000, 278, 477-483.<- quasi-seq-order refernece
    [7]: Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using
         Pseudo-Amino Acid Composition. PROTEINS: Structure, Function, and
         Genetics, 2001, 43: 246-255.
    """


    def __init__(self, protein_seqs, desc_dataset="descriptors.csv", all_desc=False):

        self.protein_seqs = protein_seqs
        self.desc_dataset = desc_dataset
        self.all_desc = all_desc

        #if 1 protein sequence (1 string) input then convert to pandas Series object
        if isinstance(self.protein_seqs, str):
          self.protein_seqs = pd.Series(self.protein_seqs)

        #only the sequences should be passed in, not all columns in a dataset etc.
        if isinstance(self.protein_seqs, pd.DataFrame) and \
            len(self.protein_seqs.columns) > 1:
            raise ValueError('The full dataset must not be passed in, only the \
                columns containing the protein sequences.')

        #remove any gaps from protein sequences
        self.protein_seqs = utils.remove_gaps(self.protein_seqs)

        #validate that all inputtted protein sequences are valid and only contain
        #   valid amino acids, if not then raise ValueError
        invalid_seqs = utils.valid_sequence(self.protein_seqs)
        if invalid_seqs!=None:
            raise ValueError('Invalid Amino Acids found in protein sequence \
                dataset: {}'.format(invalid_seqs))

        #get the total number of inputted protein sequences
        self.num_seqs = len(self.protein_seqs)

        #initialise all descriptor attributes to empty dataframes
        self.aa_composition = pd.DataFrame()
        self.dipeptide_composition = pd.DataFrame()
        self.tripeptide_composition = pd.DataFrame()
        self.normalized_moreaubroto_autocorrelation = pd.DataFrame()
        self.moran_autocorrelation = pd.DataFrame()
        self.geary_autocorrelation = pd.DataFrame()
        self.CTD = pd.DataFrame()
        self.composition = pd.DataFrame()
        self.transition = pd.DataFrame()
        self.distribution = pd.DataFrame()
        self.conjoint_triad = pd.DataFrame()
        self.seq_order_coupling_number = pd.DataFrame()
        self.quasi_seq_order = pd.DataFrame()
        self.pseudo_AAC = pd.DataFrame()
        self.amp_pseudo_AAC = pd.DataFrame()
        self.all_descriptors = pd.DataFrame()

        #try importing descriptors csv with pre-calculated descriptor values,
        #   if not found then calculate all descriptors if all_desc is true
        self.desc_dataset = os.path.join(DATA_DIR, desc_dataset)
        if os.path.isfile(self.desc_dataset):
            self.import_descriptors(self.desc_dataset)
        else:
            #if all_desc parameter true then calculate all descriptor value and store
            #   in their respective attributes
            if all_desc:
                self.all_descriptors = self.get_all_descriptors()
                self.all_descriptors.to_csv(os.path.join(DATA_DIR, 'descriptors.csv'))
                #^save all descriptor values for next time^

        #create dictionary of descriptors and their associated groups
        keys = self.all_descriptors_list()
        values = ["Composition"]*3 + ["Autocorrelation"]*3 + ["CTD"]*4 + ["Conjoint Triad"] + \
            ["Quasi-Sequence-Order"]*2 + ["Pseudo Composition"]*2

        self.descriptor_groups = dict(zip(keys,values))

    def import_descriptors(self, descriptor_file):
        """
        By default, the class will search for a file in the DATA_DIR called
        desciptors_(desc_dataset).csv with ALL of the pre-calculated descriptor values.
        This function parses this file and sets the descriptor instance variables
        to the correct features. The descriptors file should be of size N x 9920,
        where N is the number of protein sequences in the dataset and 9920 is the
        total number of features that can be calculated from the 15 descriptors.
        The method does not work if the file is not of this size.
        To create a descriptors file, create an instance of the desc class,
        with the parameter all_desc=True; this will calculate all descriptors
        and store them to the DATA_DIR.

        Parameters
        ----------
        descriptor_file : str
            file name containing pre-calculated descriptor values. Ideally, the
            file should be stored in the DATA_DIR
        """
        print('Importing Descriptors File....')

        try:
            descriptor_df = pd.read_csv(descriptor_file)
        except IOError:
            print('Error opening descriptor file: {}'.format(descriptor_file))

        if (descriptor_df.shape)!= (self.num_seqs, 9920):
            raise ValueError('Descriptors file must be of the shape ({}x9920) such \
            that all descriptor values can be parsed from the file'.format(self.num_seqs))

        #replacing +/- infinity or NAN values with 0
        descriptor_df.replace([np.inf, -np.inf], np.nan)
        descriptor_df = descriptor_df.fillna(0)

        #get the feature columns for all descriptors
        self.aa_composition = descriptor_df.iloc[:,: 20]
        self.dipeptide_composition = descriptor_df.iloc[:,20:420]
        self.tripeptide_composition = descriptor_df.iloc[:,420:8420]
        self.normalized_moreaubroto_autocorrelation = descriptor_df.iloc[:,8420:8660]
        self.moran_autocorrelation = descriptor_df.iloc[:,8660: 8900]
        self.geary_autocorrelation = descriptor_df.iloc[:,8900:9140]
        self.CTD =  descriptor_df.iloc[:,9140:9287]
        self.composition = descriptor_df.iloc[:,9140:9161]
        self.transition = descriptor_df.iloc[:,9161:9182]
        self.distribution = descriptor_df.iloc[:,9182:9287]


        # self.conjoint_triad = descriptor_df.iloc[:,9287:9799]       <---not correct
        self.seq_order_coupling_number = descriptor_df.iloc[:,9799:9800]
        self.quasi_seq_order = descriptor_df.iloc[:,9800:9900]
        self.pseudo_AAC = descriptor_df.iloc[:,9900:9950]
        self.amp_pseudo_AAC = descriptor_df.iloc[:,9950:10030]
        self.all_descriptors = descriptor_df.iloc[:,:]

    def get_aa_composition(self):
        """
        Calculate Amino Acid Composition (AAComp) of protein sequences. AA_comp
        describes the fraction of each amino acid type within a protein sequence,
        and is calculated as:
        AA_Comp(s) = AA(t)/N(s)

        where AA_Comp(s) is the AA_comp of protein sequence s, AA(t) is the number
        of amino acid types t (where t = 1,2,..,20) and N(s) is the length of the
        sequence s.

        Returns
        -------
        aa_comp_df : pd.DataFrame
            dataframe of AA composition for all protein sequences. DataFrame will
            be of the shape N x 20, where N is the number of protein sequences and
            20 is the number of features calculated from the descriptor (for the
            20 amino acids).
        """
        print('\nGetting AA Composition Descriptors...')
        print('#####################################\n')

        aa_comp = []
        #get feature names for AAComp descriptor
        aa = list((AAComposition.CalculateAAComposition(self.protein_seqs[0])).keys())

        #iterate through all sequences, calculate descriptor values and append to aa_comp list
        time.sleep(1)
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Amid Composition", file=sys.stdout):
            AAComp=AAComposition.CalculateAAComposition(seq)
            aa_comp.append(list(AAComp.values()))

        #convert calculated AAComp values into dataframe
        aa_comp_df = pd.DataFrame(data=aa_comp, columns=aa)

        self.aa_composition = aa_comp_df     #set descriptor attribute

        return aa_comp_df

    def get_dipeptide_composition(self):
        """
        Calculate Dipeptide Composition (DPComp) of protein sequences.
        Dipeptide composition is the fraction of each dipeptide type within a
        protein sequence. With dipeptides being of length 2 and there being 20
        canonical amino acids this creates 20^2 different combinations, thus a
        400-Dimensional vector will be produced such that:

        DPComp(s,t) = AA(s,t) / N -1

        where DPComp(s,t) is the dipeptide composition of the protein sequence
        for amino acid type s and t (where s and t = 1,2,..,20), AA(s,t) is the number
        of dipeptides represented by amino acid type s and t and N is the total number
        of dipeptides.

        Returns
        -------
        dipeptide_comp_df : pd.DataFrame
            dataframe of dipeptide composition for all protein sequences. DataFrame will
            be of the shape N x 400, where N is the number of protein sequences and
            400 is the number of features calculated from the descriptor (20^2 for
            the 20 canonical amino acids).
        """
        print('\nGetting Dipeptide Composition Descriptors...')
        print('############################################\n')

        dipeptide_comp = []

        #get feature names for Dipeptide_comp descriptor
        dipeptides = list((AAComposition.CalculateDipeptideComposition(self.protein_seqs[0])).keys())

        #iterate through all sequences, calculate descriptor values and append to dipeptide_comp list
        time.sleep(1)
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Dipeptide Composition", file=sys.stdout):
            AADipeptide=AAComposition.CalculateDipeptideComposition(seq)
            dipeptide_comp.append(list(AADipeptide.values()))

        #convert calculated DPComp values into dataframe
        dipeptide_comp_df = pd.DataFrame(data=dipeptide_comp, columns=dipeptides)

        self.dipeptide_composition = dipeptide_comp_df  #set descriptor attribute

        return dipeptide_comp_df

    def get_tripeptide_composition(self):
        """
        Calculate Tripeptide Composition (TPComp) of protein sequences.
        Tripeptide composition is the fraction of each tripeptide type wihin a
        protein sequence. With tripeptides being of length 3 and there being 20
        canonical amino acids this creates 20^3 different combinations, thus a
        8000-Dimensional vector will be produced such that:

        TPComp(s,t,u) = AA(s,t,u) / N -1

        where TPComp(s,t,u) is the tripeptide composition of the protein sequence
        for amino acid type s, t and u (where s, t and u = 1,2,..,20), AA(s,t,u) is
        the number of tripeptides represented by amino acid type s and t and N is
        the total number of tripeptides.

        Returns
        -------
        tripeptide_df : pd.DataFrame
            dataframe of tripeptide composition for all protein sequences. DataFrame will
            be of the shape N x 8000, where N is the number of protein sequences and
            8000 is the number of features calcualted from the descriptor (20^3 for
            the 20 canonical amino acids).
        """
        print('\nGetting Tripeptide Composition Descriptors...')
        print('#############################################\n')

        tripeptide_comp = []

        #get feature names for TPComp descriptor
        tripeptides = list((AAComposition.GetSpectrumDict(self.protein_seqs[0])).keys())

        #iterate through all sequences, calculate descriptor values and append to tripeptide_comp list
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Tripeptide Comp", file=sys.stdout):
            AATripeptide=AAComposition.GetSpectrumDict(seq)
            tripeptide_comp.append(list(AATripeptide.values()))

        #convert calculated TPComp values into dataframe
        tripeptide_comp_df = pd.DataFrame(data=tripeptide_comp, columns=tripeptides)

        self.tripeptide_composition = tripeptide_comp_df    #set descriptor attribute

        return tripeptide_comp_df

    def get_norm_moreaubroto_autocorrelation(self,aa_encoding=Autocorrelation._AAProperty,
        aa_name=Autocorrelation._AAPropertyName):
        """
        Calculate Normalized MoreauBrotoAuto Autocorrelation (NMBAuto) descriptor.
        Autocorrelation descriptors are a class of topological descriptors,
        also known as molecular connectivity indices, that describe the level of
        correlation between two objects (protein or peptide sequences) in terms of
        their specific structural or physicochemical property [2], which are
        defined based on the distribution of amino acid properties along the sequence.
        8 amino acid properties are used for deriving the descriptors. The derivations
        and detailed explanations of this type of descriptor is outlind in [3].
        Each autocorrelation will generate 240 features, using the default 8 properties:

        AccNo. CIDH920105 - Normalized Average Hydrophobicity Scales
        AccNo. BHAR880101 - Average Flexibility Indices
        AccNo. CHAM820101 - Polarizability Parameter
        AccNo. CHAM820102 - Free Energy of Solution in Water, kcal/mole
        AccNo. CHOC760101 - Residue Accessible Surface Area in Tripeptide
        AccNo. BIGC670101 - Residue Volume
        AccNo. CHAM810101 - Steric Parameter
        AccNo. DAYM780201 - Relative Mutability

        The NMBAuto descriptor is a type of Autocorrelation descriptor that uses
        the property values as the basis for measurement [3].

        Parameters
        ----------
        aa_encoding : list of dicts (default = Autocorrelation._AAProperty )
            list of dictionaries of amino acid values for particular properties,
            in the form "A": 0.2, "B":0.4 etc. By default 8 properties are used,
            _AAProperty = (_Hydrophobicity,_AvFlexibility,_Polarizability,
            _FreeEnergy,_ResidueASA,_ResidueVol,_Steric,_Mutability).

        aa_name : list of str (default = Autocorrelation._AAPropertyName)
            list of strings containing the name of each property. Used for the
            column/feature names in the output dataframe. Default is just a list
            of string representations of each of the mentioned 8 properties.

        Returns
        -------
        norm_moreaubroto_autocorr_df : pd.DataFrame
            dataframe of NMBAuto values for all protein sequences. DataFrame will
            be of the shape N x 240, where N is the number of protein sequences and
            240 is the number of features calcualted from the descriptor (30 features
            per property - using 8 properties.)
        """
        print('\nGetting Normalized Moreaubroto Autocorrelation Descriptors...')
        print('######################################\n')

        #initialise lists used to store the autocorrelation values and keys
        norm_moreaubroto_autocorr = []
        norm_moreaubroto_autocorr_keys = []
        norm_moreaubroto_autocorr_values = []

        #iterate through protein sequences and calculate NMBAuto values, append to list
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Autocorrelation", file=sys.stdout):
            norm_moreau_broto = Autocorrelation.CalculateNormalizedMoreauBrotoAuto(seq, aa_encoding,aa_name)
            norm_moreaubroto_autocorr.append(norm_moreau_broto)

        #get keys/column names generated from the descriptor calculation
        for name in range(0,len(aa_name)):
          norm_moreaubroto_autocorr_keys.append(list(norm_moreaubroto_autocorr[0][aa_name[name]].keys()))

        #flatten column names/keys into 1 1-D list
        norm_moreaubroto_autocorr_keys = utils.flatten(norm_moreaubroto_autocorr_keys)

        '''
        Reformatting the output of the descriptors from the PyBioMed package.
        The PyBioMed descriptor function returns a dict of dicts of the resultant
        values. For uniformity with the rest of the descriptor calculations,
        this output is transformed such that it can be output into a dataframe.
        '''
        for x in range(0,len(list(norm_moreaubroto_autocorr))):
            for y in range(0,len(list(norm_moreaubroto_autocorr[x].values()))):
                norm_moreaubroto_autocorr_values.append(
                list((list((list(norm_moreaubroto_autocorr)[x]).values()))[y].values())
                )

        #flatten list of lists of the autocorrelation values
        norm_moreaubroto_autocorr_values = utils.flatten(norm_moreaubroto_autocorr_values)

        #reshape into N x M array where N is the number of protein sequences and M
        #   is the number of AA properties used in the correlation
        norm_moreaubroto_autocorr_values_ = np.reshape(norm_moreaubroto_autocorr_values,
            (len(self.protein_seqs),len(norm_moreaubroto_autocorr_keys)))

        #convert NMBAutp array values into a dataframe
        norm_moreaubroto_autocorr_df = pd.DataFrame(data=norm_moreaubroto_autocorr_values_,
            columns=norm_moreaubroto_autocorr_keys)

        self.normalized_moreaubroto_autocorrelation = norm_moreaubroto_autocorr_df  #set descriptor attribute

        return norm_moreaubroto_autocorr_df

    def get_moran_autocorrelation(self,aa_encoding=Autocorrelation._AAProperty,
        aa_name=Autocorrelation._AAPropertyName):
        """
        **refer to NMBAuto for autocorrelation description.
        Moran autocorrelation (MAuto) utilizes property deviations from the
        average values [3].

        Parameters
        ----------
        aa_encoding : list of dicts (default = Autocorrelation._AAProperty )
            **refer to NMBAuto doctring.
        aa_name : list of str (default = Autocorrelation._AAPropertyName)
            **refer to NMBAuto doctring.

        Returns
        -------
        moran_autocorr_df : pd.DataFrame
            dataframe of MAuto values for all protein sequences. DataFrame will
            be of the shape N x 240, where N is the number of protein sequences and
            240 is the number of features calcualted from the descriptor (30 features
            per property - using 8 properties.)
        """
        print('\nGetting Moran Autocorrelation Descriptors...')
        print('######################################\n')

        #initialise lists used to store the MAuto values and keys
        moran_autocorr = []
        moran_autocorr_keys = []
        moran_autocorr_values = []

        #iterate through protein sequences and calculate MAuto values, append to list
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Autocorrelation", file=sys.stdout):
            moranautocorr = Autocorrelation.CalculateMoranAuto(seq, aa_encoding,aa_name)
            moran_autocorr.append(moranautocorr)

        #get keys/column names generated from the descriptor calculation
        for name in range(0,len(aa_name)):
          moran_autocorr_keys.append(list(moran_autocorr[0][aa_name[name]].keys()))

        #flatten column names/keys into 1 1-D list
        moran_autocorr_keys = utils.flatten(moran_autocorr_keys)

        '''
        Reformatting the output of the descriptors from the PyBioMed package.
        The PyBioMed descriptor function returns a dict of dicts of the resultant
        values. For uniformity with the rest of the descriptor calculations,
        this output is transformed such that it can be output into a dataframe.
        '''
        for x in range(0,len(list(moran_autocorr))):
            for y in range(0,len(list(moran_autocorr[x].values()))):
                moran_autocorr_values.append(
                list((list((list(moran_autocorr)[x]).values()))[y].values())
                )

        #flatten list of lists of the autocorrelation values
        moran_autocorr_values = utils.flatten(moran_autocorr_values)

        #reshape into N x M array where N is the number of protein sequences and M
        #   is the number of AA properties used in the correlation
        moran_autocorr_values_ = np.reshape(moran_autocorr_values,
            (len(self.protein_seqs),len(moran_autocorr_keys)))

        #convert Autocorrelation array values into a dataframe
        moran_autocorr_df = pd.DataFrame(data=moran_autocorr_values_, columns=moran_autocorr_keys)

        self.moran_autocorrelation = moran_autocorr_df  #set descriptor attribute

        return moran_autocorr_df

    def get_geary_autocorrelation(self,aa_encoding=Autocorrelation._AAProperty,
        aa_name=Autocorrelation._AAPropertyName):
        """
        *refer to NMBAuto docstring for autocorrelation descriptipn.
        Geary Autocorrelation (GAuto) utilizes the square-difference of property
        values instead of vector-products (of property values or deviations)[3].

        Parameters
        ----------
        aa_encoding : list of dicts (default = Autocorrelation._AAProperty )
            **refer to NMBAuto doctring.
        aa_name : list of str (default = Autocorrelation._AAPropertyName)
            **refer to NMBAuto doctring.

        Returns
        -------
        geary_autocorr_df : pd.DataFrame
            dataframe of GAuto values for all protein sequences. DataFrame will
            be of the shape N x 240, where N is the number of protein sequences and
            240 is the number of features calcualted from the descriptor (30 features
            per property - using 8 properties.)
        """
        print('\nGetting Geary Autocorrelation Descriptors...')
        print('######################################\n')

        #initialise lists used to store the autocorrelation values and keys
        geary_autocorr = []
        geary_autocorr_keys = []
        geary_autocorr_values = []

        #iterate through protein sequences and calculate GAuto values, append to list
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Autocorrelation", file=sys.stdout):
            gearyautocorr = Autocorrelation.CalculateGearyAuto(seq, aa_encoding,aa_name)
            geary_autocorr.append(gearyautocorr)

        #get keys/column names generated from the descriptor calculation
        for name in range(0,len(aa_name)):
          geary_autocorr_keys.append(list(geary_autocorr[0][aa_name[name]].keys()))

        #flatten column names/keys into 1 1-D list
        geary_autocorr_keys = utils.flatten(geary_autocorr_keys)

        '''
        Reformatting the output of the descriptors from the PyBioMed package.
        The PyBioMed descriptor function returns a dict of dicts of the resultant
        values. For uniformity with the rest of the descriptor calculations,
        this output is transformed such that it can be output into a dataframe.
        '''
        for x in range(0,len(list(geary_autocorr))):
            for y in range(0,len(list(geary_autocorr[x].values()))):
                geary_autocorr_values.append(
                list((list((list(geary_autocorr)[x]).values()))[y].values())
                )

        #flatten list of lists of the autocorrelation values
        geary_autocorr_values = utils.flatten(geary_autocorr_values)

        #reshape into N x M array where N is the number of protein sequences and M
        #   is the number of AA properties used in the correlation
        geary_autocorr_values_ = np.reshape(geary_autocorr_values, (len(self.protein_seqs),len(geary_autocorr_keys)))

        #convert GAuto array values into a dataframe
        geary_autocorr_df = pd.DataFrame(data=geary_autocorr_values_, columns=geary_autocorr_keys)

        self.geary_autocorrelation = geary_autocorr_df  #set descriptor attribute

        return geary_autocorr_df

    def get_ctd(self):
        """
        Calculate Composition, transition and distribution (CTD) of protein sequences.
        Composition is the number of amino acids of a particular property (e.g., hydrophobicity)
        divided by the total number of amino acids in a protein sequence. Transition
        characterizes the percent frequency with which amino acids of a particular
        property is followed by amino acids of a different property. Distribution
        measures the chain length within which the first, 25%, 50%, 75%, and 100% of
        the amino acids of a particular property are located, respectively [4].
        CTD functionality in the PyBioMed package uses the properties:
        Polarizability, Solvent Accessibility, Secondary Structure, Charge,
        Polarity, Normalized VDWV, Hydrophobicity. The output will be of shape
        N x 147 where N is the number of protein sequences. 21/147 will be
        composition, 21/147 will be transition and the remaining 105 are distribution.

        Returns
        -------
        ctd_df : pd.DataFrame
            dataframe of CTD descriptor values for all protein sequences. DataFrame will
            be of the shape N x 147, where N is the number of protein sequences and
            147 is the number of features calcualted from the descriptors.
        """
        print('\nGetting CTD Descriptors...')
        print('##########################\n')

        ctd = []

        #get feature names for CTD descriptor
        keys = list((CTD.CalculateCTD(self.protein_seqs[0])).keys())

        #iterate through sequences, calculating the CTD values using PyBioMed Package
        time.sleep(1)
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="CTD", file=sys.stdout):
            ctd_=CTD.CalculateCTD(seq)
            ctd.append(list(ctd_.values()))

        #convert calculated ctd values into dataframe
        ctd_df = pd.DataFrame(data=ctd, columns=keys)

        self.CTD = ctd_df                   #set descriptor attributes
        self.composition = ctd_df.iloc[:,:21]
        self.transition = ctd_df.iloc[:, 21: 42]
        self.distribution = ctd_df.iloc[:, 42: ]

        return ctd_df

    def get_conjoint_triad(self):
        """
        Calculate Conjoint Triad features (CTriad) for the protein sequences. CTF
        mainly considers neighbor relationships in protein sequences by encoding
        each protein sequence using the triad (continuous three amino acids)
        frequency distribution extracted from a 7-letter reduced alphabet [5]. This
        descriptor calculates 343 different features (7x7x7), with the output
        being of shape N x 343 where N is the number of sequences.

        Returns
        -------
        ct_df : pd.DataFrame
            dataframe of CTriad descriptor values for all protein sequences. DataFrame
            will be of the shape N x 343, where N is the number of protein sequences and
            343 is the number of features calcualted from the descriptor.
        """
        print('\nGetting Conjoint Triad Descriptors...')
        print('##################################\n')

        ct = []

        #iterate through sequences, calculating the CTriad values using PyBioMed Package
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Conjoint Triad", file=sys.stdout):
            con_Tri = {}
            protein_num = ConjointTriad._Str2Num(seq)
            for i in range(7):
                for j in range(7):
                    for k in range(7):
                        temp = str(i) + str(j) + str(k)
                        con_Tri[temp] = protein_num.count(temp)
            ct.append(list(con_Tri.values()))

        #get feature/column names for CTriad descriptor
        keys = con_Tri[0].keys()

        # #iterate through sequences, calculating the descriptor values using PyBioMed Package
        # for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,
        #     desc="Conjoint Triad", file=sys.stdout):
        #     conTri=ConjointTriad.CalculateConjointTriad(seq)
        #     ct.append(list(conTri.values()))

        #convert calculated CTriad values into dataframe
        ct_df = pd.DataFrame(data=ct, columns=keys)

        self.conjoint_triad = ct_df         #set descriptor attribute

        return ct_df

    def get_seq_order_coupling_number(self, maxlag=30):
        """
        Calculate Sequence Order Coupling Number (SOCNum) features for the protein sequences.
        Sequence Order Coupling Number computes the dissimilarity between amino acid
        pairs. The distance between amino acid pairs is determined by d which varies
        between 1 to nlag. For each d, it computes the sum of the dissimilarities
        of all amino pairs [6].
*
        1 value is calculated per sequence so the output will
        be of shape N x 1 where N is the number of sequences.

        Parameters
        ----------
        maxlag : int (default = 30)
            maxlag is the maximum lag and the length of the protein should be larger
            than maxlag. default is 30.

        Returns
        -------
        seq_order_df : pd.DataFrame
            dataframe of SOCNum descriptor values for all protein sequences. DataFrame
            will be of the shape N x X, where N is the number of protein sequences and
            X is the number of features calcualted from the descriptor.
        """
        print('\nGetting Sequence Order Coupling Descriptors...')
        print('#############################################\n')

        seq_order = []

        #iterate through sequences, calculating the SOCNum values using PyBioMed Package
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,desc="Sequence Order Coupling Number", file=sys.stdout):
            # seq_order_num = QuasiSequenceOrder.GetSequenceOrderCouplingNumber(seq)
            seq_order_num = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(seq,maxlag=maxlag)
            seq_order.append(seq_order_num)

        #convert calculated SOCNum values into dataframe
        seq_order_df = pd.DataFrame(data=seq_order)

        self.seq_order_coupling_number = seq_order_df   #set descriptor attribute

        return seq_order_df

    def get_quasi_seq_order(self, maxlag=30, weight=0.1):
        """
        Calculate Quasi Sequence Order features for the protein sequences.
        The quasi-sequence-order descriptors were proposed by K.C. Chou, et.al. [7]
        They are derived from the distance matrix between the 20 amino acids. The
        Scheider-Wrede physicochemical distance matrix and Grantham chemical
        distance matrix used by Grantham et. al. are both used in the calculation
        of the descriptor.100 values are calculated per sequence, thus generating
        an output of N x 100 where N is the numnber of sequences.

        Parameters
        ----------
        maxlag : int (default = 30)
            **refer to seq-order-coupling-number docstring.
        weight : float (default = 0.1)
            weight factor, can select any value within the region from 0.05 to 0.7.

        Returns
        -------
        quasi_seq_order_df : pd.DataFrame
            dataframe of quasi-sequence-order descriptor values for the
            protein sequences.
        """
        print('\nGetting Quasi Sequence Order Descriptors...')
        print('###########################################\n')

        #if maxlag greater than length of protein, set to default value of 30
        if (maxlag >= self.protein_seqs[0]):
            maxlag=30

        quasi_seq_order = []
        #get feature names for QuasiSeqOrder descriptor
        keys = list((QuasiSequenceOrder.GetQuasiSequenceOrder(self.protein_seqs[0])).keys())

        #iterate through sequences, calculating the descriptor values using PyBioMed Package
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,
            desc="Quasi Sequence Order", file=sys.stdout):
            quasi_seq=QuasiSequenceOrder.GetQuasiSequenceOrder(seq,maxlag=maxlag, weight=weight)
            quasi_seq_order.append(list(quasi_seq.values()))

        #convert calculated quasi_seq_order values into dataframe
        quasi_seq_order_df = pd.DataFrame(data=quasi_seq_order, columns=keys)

        self.quasi_seq_order = quasi_seq_order_df    #set descriptor attribute

        return quasi_seq_order_df

    def get_pseudo_AAC(self, lambda_=30, weight=0.05, AAP=[PseudoAAC._Hydrophobicity,
            PseudoAAC._hydrophilicity, PseudoAAC._residuemass]):
        """
        Calculate pseudo Amino Acid Composition features for the protein sequences.
        Similar to the quasi-sequence order descriptor, the pseudo amino acid descriptor is
        made up of a 50-dimensional vector in which the first 20 components reflect the
        effect of the amino acid composition and the remaining 30 components reflect
        the effect of sequence order. 50 values will be calculated per sequence,
        thus generating an output of N x 50 where N is the number of sequences.

        [X] <- pseudo aa refer
        Parameters
        ----------
        lambda_ : int (default = 30)
            Lambda factor reflects the rank of correlation and is a non-Negative integer.
            The value should NOT be larger than the length of input protein
            sequences. When lambda=0, the output is the 20-D amino acid composition.
        weight : float (default = 0.05)
            weight factor is designed for the users to put weight on the additional
            psuedo AAC components with respect to the conventional AA components.
            The user can select any value within the region from 0.05 to 0.7.
        AAP : list (default = AAP=[PseudoAAC._Hydrophobicity,
                PseudoAAC._hydrophilicity, PseudoAAC._residuemass])
            list contraining various physiochemical properties of the sequences.
            properites will be a list of dicts, in the form:
            'Property_Name': Value (float). By default, 3 properties are used:
            hydrophobicity, hydrophillicity and residue mass of the amino acids.
            Refer to PyProtein module in PyBioMed package for more info:
            https://github.com/gadsbyfly/PyBioMed/tree/master/PyBioMed/PyProtein

        Returns
        -------
        psuedo_AAComp_df : pd.DataFrame
            dataframe of pseudo amino acid composition descriptor values
            for the protein sequences.
        """
        print('\nGetting Psuedo Amino Acid Composition Descriptors...')
        print('####################################################\n')

        #if lambda not a non-negative int then set to default value of 30
        if not (lamba_>=0):
            lambda_ = 30

        #if weight not b/w 0.05 and 0.7 then set to default value of 0.05
        if (weight<0.05 or weight>0.7):
            weight = 0.05

        psuedo_AA_Comp = []
        #get feature names for PsuedoAAComp descriptor
        keys = (list(PseudoAAC.GetPseudoAAC(self.protein_seqs[0], AAP=AAP).keys()))

        #iterate through sequences, calculating the descriptor values using PyBioMed Package
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,
            desc="Psuedo Amino Acid Composition", file=sys.stdout):
            psuedo_AA=PseudoAAC.GetPseudoAAC(seq, lamda=lambda_, weight=weight, AAP=AAP)
            psuedo_AA_Comp.append(list(psuedo_AA.values()))

        #convert calculated PsuedoAAComp values into dataframe
        psuedo_AAComp_df = pd.DataFrame(data=psuedo_AA_Comp, columns=keys)

        self.pseudo_AAC = psuedo_AAComp_df      #set descriptor attribute

        return psuedo_AAComp_df

    def get_amp_pseudo_AAC(self, lambda_=30, weight=0.5):
        """
        Calculate Amphiphilic (Type II) Pseudo amino acid composition features
        for the protein sequences.
        Amphiphilic pseudo amino acid composition has the same form as the amino acid
        compositon, but contains much more information that is related to the sequence
        order of a protein and the distribution of the hydrophobic and hydrophilic
        amino acids along its chain. 80 descriptor values will be generated, producing
        an output of the shape N x 80 where N is the number of sequences and 80
        is the number of features calculated from descriptor.

        Parameters
        ----------
        lambda_ : int (default = 30)
            **refer to Pseudo Amino Acid Composition docstring
        weight : float (default = 0.5)
            **refer to Pseudo Amino Acid Composition docstring

        Returns
        -------
        amp_pseudo_AAComp_df : pd.DataFrame
            dataframe of Amphiphilic pseudo amino acid composition descriptor
            values for the protein sequences.
        """
        print('\nGetting Amphiphilic Amino Acid Composition Descriptors...')
        print('#########################################################\n')

        #if lambda not a non-negative int then set to default value of 30
        if not (lamba_>=0):
            lambda_ = 30

        #if weight not b/w 0.05 and 0.7 then set to default value of 0.05
        if (weight<0.05 or weight>0.7):
            weight = 0.05

        amp_pseudo_AAComp = []
        #get column names/keys for descriptor
        keys = list((PseudoAAC.GetAPseudoAAC(self.protein_seqs[0])).keys())

        #iterate through sequences, calculating the descriptor values using PyBioMed Package
        for seq in tqdm(self.protein_seqs,unit=" sequences",position=0,
            desc="Ampiphillic Amino Acid Composition", file=sys.stdout):
            amp_psuedo=PseudoAAC.GetAPseudoAAC(seq, lamda=lambda_, weight=weight)
            amp_pseudo_AAComp.append(list(amp_psuedo.values()))

        #convert calculated Ampiphillic Psuedo_AAComp values into dataframe
        amp_pseudo_AAComp_df = pd.DataFrame(data=amp_pseudo_AAComp, columns=keys)

        self.amp_pseudo_AAC = amp_pseudo_AAComp_df  #set descriptor attribute

        return amp_pseudo_AAComp_df

    def get_descriptor_encoding(self,descriptor):
        """
        Get the protein descriptor values of a specified input descriptor. If the
        sought descriptor has already been calculated then its attribute is returned,
        else the descriptor is calculated using its get_descriptor function.

        Parameters
        ----------
        descriptor : str
            name of descriptor to return. Method can accept the approximate name
            of the descriptor, e.g. 'aa_comp'/'aa_compo' etc will return the
            'aa_composition' descriptor.

        Returns
        -------
        desc_encoding : pd.DataFrame/None
            dataframe of descriptor values, calculated from the descriptor
            instances' protein_seqs attribute. None returned if no matching descriptor.
        """
        #validate input descriptor is a valid available descriptor, get its closest match
        valid_desc = self.valid_descriptors()
        desc_matches = get_close_matches(descriptor,valid_desc,cutoff=0.4)

        if desc_matches!=[]:
            desc = desc_matches[0]
        else:
            raise ValueError('Could not find a match for the input descriptor {} \
                in available valid models: {}'.format(descriptor, valid_desc))

        #if sought descriptor attribute dataframe is empty, call the descriptor's
        #   get_descriptor() function, set desc_encoding to descriptor attribute
        if desc == 'aa_composition':
            if (getattr(self, desc).empty):
                self.get_aa_composition()
            desc_encoding = self.aa_composition
        elif desc == 'dipeptide_composition':
            if (getattr(self, desc).empty):
                self.get_dipeptide_composition()
            desc_encoding = self.dipeptide_composition
        elif desc == 'tripeptide_composition':
            if (getattr(self, desc).empty):
                self.get_tripeptide_composition()
            desc_encoding = self.tripeptide_composition
        elif desc == 'normalized_moreaubroto_autocorrelation':
            if (getattr(self, desc).empty):
              self.get_norm_moreaubroto_autocorrelation()
            desc_encoding = self.normalized_moreaubroto_autocorrelation
        elif desc == 'moran_autocorrelation':
            if (getattr(self, desc).empty):
              self.get_moran_autocorrelation()
            desc_encoding = self.moran_autocorrelation
        elif desc == 'geary_autocorrelation':
            if (getattr(self, desc).empty):
              self.get_geary_autocorrelation()
            desc_encoding = self.geary_autocorrelation
        elif desc == 'CTD':
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.CTD
        elif desc == 'composition':
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.composition
        elif desc == 'transition':
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.transition
        elif desc == 'distribution':
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.distribution
        elif desc == 'conjoint_triad':
            if (getattr(self, desc).empty):
              self.get_conjoint_triad()
            desc_encoding = self.conjoint_triad
        elif desc == 'seq_order_coupling_number':
            if (getattr(self, desc).empty):
              self.get_seq_order_coupling_number()
            desc_encoding = self.seq_order_coupling_number
        elif desc == 'quasi_seq_order':
            if (getattr(self, desc).empty):
              self.get_quasi_seq_order()
            desc_encoding = self.quasi_seq_order
        elif desc == 'pseudo_AAC':
            if (getattr(self, desc).empty):
              self.get_pseudo_AAC()
            desc_encoding = self.pseudo_AAC
        elif desc == 'amp_pseudo_AAC':
            if (getattr(self, desc).empty):
              self.get_amp_pseudo_AAC()
            desc_encoding = self.amp_pseudo_AAC
        else:
          desc_encoding = None      #no matching descriptor

        return desc_encoding

    def all_descriptors_list(self, desc_combo=1):
       """
       Get list of all available descriptor attributes. Using the desc_combo
       input parameter you can get the list of all descriptors, all combinations
       of 2 descriptors or all combinations of 3 descriptors. Default of 1 will
       mean a list of available descriptor attributes will be returned.

       Parameters
       ----------
       desc_combo : int (default = 1)
            combination of descriptors to return. A value of 2 or 3 will return
            all combinations of 2 or 3 descriptor attributes.

       Returns
       -------
       all_descriptors : list
            list of available descriptor attributes.
       """
       #filter out class attributes that are not any of the desired descriptors
       all_descriptors = list(filter(lambda x: x.startswith('_'), list(self.__dict__.keys())))
       all_descriptors = list(filter(lambda x: not x.startswith('_all_desc'), all_descriptors))

       #get all combinations of 2 or 3 descriptors.
       if desc_combo == 2:
           all_descriptors = list(itertools.combinations(all_descriptors, 2))
       elif desc_combo == 3:
           all_descriptors = list(itertools.combinations(all_descriptors, 3))
       else:
           pass     #if desc_combo not equal to 2 or 3 then use default all_descriptors

       return all_descriptors

    def get_all_descriptors(self):
        """
        Calculate all individual descriptor values, concatenating each descriptor
        dataframe into one storing all descriptors.

        Returns
        -------
        all_desc_df : pd.DataFrame
            concatenated dataframe of all individual descriptors. Output will be
            of the shape N x 10,030, where N is the number of sequences and
            10,030 is the total number of descriptor features calculated.
        """
        print('Calculating all Descriptor values....\n')

        #if descriptor attribute DF is empty then call get_descriptor function
        if (getattr(self, "aa_composition").empty):
                self.aa_composition = self.get_aa_composition()

        if (getattr(self, "dipeptide_composition").empty):
                self.dipeptide_composition = self.get_dipeptide_composition()

        if (getattr(self, "tripeptide_composition").empty):
                self.tripeptide_composition = self.get_tripeptide_composition()

        if (getattr(self, "normalized_moreaubroto_autocorrelation").empty):
            self.normalized_moreaubroto_autocorrelation = self.get_norm_moreaubroto_autocorrelation()

        if (getattr(self, "moran_autocorrelation").empty):
            self.moran_autocorrelation = self.get_moran_autocorrelation()

        if (getattr(self, "geary_autocorrelation").empty):
            self.geary_autocorrelation = self.get_geary_autocorrelation()

        if (getattr(self, "CTD").empty):
                self.CTD = self.get_ctd()

        if (getattr(self, "conjoint_triad").empty):
                self.conjoint_triad = self.get_conjoint_triad()

        if (getattr(self, "seq_order_coupling_number").empty):
                self.seq_order_coupling_number = self.get_seq_order_coupling_number()

        if (getattr(self, "quasi_seq_order").empty):
                self.quasi_seq_order = self.get_quasi_seq_order()

        if (getattr(self, "pseudo_AAC").empty):
                self.pseudo_AAC = self.get_pseudo_AAC()

        if (getattr(self, "amp_pseudo_AAC").empty):
                self.amp_pseudo_AAC = self.get_amp_pseudo_AAC()

        #append all calculated descriptors to list
        all_desc = [self.aa_composition, self.dipeptide_composition, self.tripeptide_composition,
                           self.normalized_moreaubroto_autocorrelation, self.moran_autocorrelation,
                           self.geary_autocorrelation, self.composition, self.transition,
                           self.distribution, self.conjoint_triad, self.seq_order_coupling_number,
                           self.quasi_seq_order, self.pseudo_AAC, self.amp_pseudo_AAC]

        #concatenate individual descriptor dataframe attributes
        all_desc_df = pd.concat(all_desc, axis = 1)

        self.all_descriptors = all_desc_df

        return all_desc_df

    def valid_descriptors(self):
        """
        Get a list of all valid descriptors available in the module.

        Returns
        -------
        valid_desc : list
            list of all valid descriptors that the module supports.
        """
        valid_desc = [

        'aa_composition', 'dipeptide_composition', 'tripeptide_composition', \
        'norm_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation', \
        'ctd', 'composition', 'transition', 'distribution', 'conjoint_triad', \
        'seq_order_coupling_number','quasi_seq_order_descriptors',\
        'pseudo_aa_comp', 'amphipilic_pseudo_aa_comp'

        ]
        return valid_desc

######################          Getters & Setters          ######################

    @property
    def all_desc(self):
        return self._all_desc

    @all_desc.setter
    def all_desc(self, val):
        self._all_desc = val

    @property
    def aa_composition(self):
        return self._aa_composition

    @aa_composition.setter
    def aa_composition(self, val):
        self._aa_composition = val

    @property
    def dipeptide_composition(self):
        return self._dipeptide_composition

    @dipeptide_composition.setter
    def dipeptide_composition(self, val):
        self._dipeptide_composition = val

    @property
    def tripeptide_composition(self):
        return self._tripeptide_composition

    @tripeptide_composition.setter
    def tripeptide_composition(self, val):
        self._tripeptide_composition = val

    @property
    def normalized_moreaubroto_autocorrelation(self):
        return self._normalized_moreaubroto_autocorrelation

    @normalized_moreaubroto_autocorrelation.setter
    def normalized_moreaubroto_autocorrelation(self, val):
        self._normalized_moreaubroto_autocorrelation = val

    @property
    def moran_autocorrelation(self):
        return self._moran_autocorrelation

    @moran_autocorrelation.setter
    def moran_autocorrelation(self, val):
        self._moran_autocorrelation = val

    @property
    def geary_autocorrelation(self):
        return self._geary_autocorrelation

    @geary_autocorrelation.setter
    def geary_autocorrelation(self, val):
        self._geary_autocorrelation = val

    @property
    def CTD(self):
        return self._CTD

    @CTD.setter
    def CTD(self, val):
        self._CTD = val

    @property
    def composition(self):
        return self._composition

    @composition.setter
    def composition(self, val):
        self._composition = val

    @property
    def transition(self):
        return self._transition

    @transition.setter
    def transition(self, val):
        self._transition = val

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        self._distribution = val

    @property
    def conjoint_triad(self):
        return self._conjoint_triad

    @conjoint_triad.setter
    def conjoint_triad(self, val):
        self._conjoint_triad = val

    @property
    def seq_order_coupling_number(self):
        return self._seq_order_coupling_number

    @seq_order_coupling_number.setter
    def seq_order_coupling_number(self, val):
        self._seq_order_coupling_number = val

    @property
    def quasi_seq_order(self):
        return self._quasi_seq_order

    @quasi_seq_order.setter
    def quasi_seq_order(self, val):
        self._quasi_seq_order = val

    @property
    def pseudo_AAC(self):
        return self._pseudo_AAC

    @pseudo_AAC.setter
    def pseudo_AAC(self, val):
        self._pseudo_AAC = val

    @property
    def amp_pseudo_AAC(self):
        return self._amp_pseudo_AAC

    @amp_pseudo_AAC.setter
    def amp_pseudo_AAC(self, val):
        self._amp_pseudo_AAC = val

    @property
    def all_descriptors(self):
        return self._all_descriptors

    @all_descriptors.setter
    def all_descriptors(self, val):
        self._all_descriptors = val

    @all_descriptors.deleter
    def all_descriptors(self):
        """ Delete all descriptor attribute dataframes """
        del self._all_descriptors
        del self._aa_composition
        del self._dipeptide_composition
        del self._tripeptide_composition
        del self._normalized_moreaubroto_autocorrelation
        del self._moran_autocorrelation
        del self._geary_autocorrelation
        del self._CTD
        del self._transition
        del self._composition
        del self._distribution
        del self._conjoint_triad
        del self._seq_order_coupling_number
        del self._quasi_seq_order
        del self._pseudo_AAC
        del self._amp_pseudo_AAC

################################################################################

    def __str__(self):
        return "Descriptor(Num Sequences: {}, Using All Descriptors: {})".format(
            self.num_seqs, self.all_desc)

    def __repr__(self):
        return ('<Descriptor: {}>'.format(self))

    def __len__(self):
        return len(self.all_descriptors)

    def __shape__(self):
        return self.all_descriptors.shape
