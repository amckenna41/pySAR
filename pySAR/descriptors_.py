################################################################################
#################                  Descriptors                 #################
################################################################################

import pandas as pd
import numpy as np
from difflib import get_close_matches
import json
from json import JSONDecodeError
import itertools
from tqdm import tqdm

from .globals_ import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from .aaindex import  AAIndex
from .model import Model
from .pyDSP import PyDSP
from .evaluate import Evaluate
from .utils import *
from .descriptors.autocorrelation import moran_autocorrelation, geary_autocorrelation, norm_moreaubroto_autocorrelation
from .descriptors.composition import AAComposition, DipeptideComposition, TripeptideComposition, \
    sequenceOrderCorrelationFactor, pseudoAAC, amphiphilicPseudoAAC
from .descriptors.conjointTriad import conjoint_triad
from .descriptors.ctd import ctd_composition, ctd_transition, ctd_distribution, ctd_
from .descriptors.quasiSequenceOrder import quasi_sequence_order, seq_order_coupling_number

class Descriptors():
    """
    Class for calculating a wide variety of protein physiochemical and structural descriptors. 
    These descriptors have been used in a wide variety of Bioinformaitcs applications including:
    protein strucutral and functional class prediction, protein-protein interactions,
    subcellular location, secondary structure prediction etc. They represent the different
    structural, functional & interaction profiles of proteins by exploring the
    features in the groups of composition, correlation, distribution of the constituent
    residues and their biochemical and physiochemical properties.

    This class allows calculation of the following descriptors: amino acid compostion (AAComp),
    dipeptide composition (DPComp), tripeptide composition (TPComp), normalized Moreau-Broto
    autocorrelation (NMBAuto), Moran autocorrelation (MAuto), Geary autocorrelation (GAuto),
    Composition (C), Transition (T), Distribution (D), CTD, Conjoint Triad (CTriad), sequence
    order coupling number (SOCNum), Quasi-sequence-order (QSOrder), Pseudo amino-acid
    composition - type 1 (PAAcomp) and Amphiphilic pseudo amino-acid composition - type 2 (APAAComp). 
    The descriptors directory contains the individual modules and functions used to calculate
    these descriptors. 

    Similar to other classes in pySAR, this class works via configuration files which contain
    the values for all the potential parameters of each descriptor. An array of protein sequences 
    can be passed into the class, but if none are then they will be imported from the dataset using
    the dataset parameters in the config file. By default, the class will look for a descriptors csv
    which is a file of the pre-calcualted descriptor values for the specified dataset, if this file 
    doesn't exist, or the parameter value is blank, then each descriptor will be calculated using 
    its respective function, depending on whether its respective parameter is set in the config.

    It is reccomended that with every new dataset, the Descriptors class should be instantiated 
    with the all_desc value set to 1 in the config file. This will calculate all the descriptor
    values for the dataset of protein sequences, storing the result in a csv file, meaning that
    this file can be used for future use. 

    Attributes
    ----------
    :desc_config: str
        path to configuration file which will contain the various parameter values for all
        descriptors. If invalid value input then error will be raised.
    :protein_seqs : np.ndarray
        array of protein sequences that descriptors will be calculated for. If set to none, 
        protein sequences will be imported from the specified dataset in the config file.

    References
    ----------
    [1]: Dong, J., Yao, ZJ., Zhang, L. et al. PyBioMed: a python library for
         various molecular representations of chemicals, proteins and DNAs and
         their interactions. J Cheminform 10, 16 (2018).
         https://doi.org/10.1186/s13321-018-0270-2
    [2]: Reczko, M. and Bohr, H. (1994) The DEF data base of sequence based protein
         fold class predictions. Nucleic Acids Res, 22, 3616-3619.
    [3]: Hua, S. and Sun, Z. (2001) Support vector machine approach for protein
         subcellular localization prediction. Bioinformatics, 17, 721-728.
    [4]: Broto P, Moreau G, Vandicke C: Molecular structures: perception,
         autocorrelation descriptor and SAR studies. Eur J Med Chem 1984, 19: 71–78.
    [5]: Ong, S.A., Lin, H.H., Chen, Y.Z. et al. Efficacy of different protein
         descriptors in predicting protein functional families. BMC Bioinformatics
         8, 300 (2007). https://doi.org/10.1186/1471-2105-8-300
    [6]: Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim. Prediction
         of protein folding class using global description of amino acid sequence.
         Proc.Natl. Acad.Sci.USA, 1995, 92, 8700-8704.
    [7]: Juwen Shen, Jian Zhang, Xiaomin Luo, Weiliang Zhu, Kunqian Yu, Kaixian Chen,
         Yixue Li, Huanliang Jiang. Predicting proten-protein interactions based only
         on sequences inforamtion. PNAS. 2007 (104) 4337-4341.
    [8]: Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
         Quasi-Sequence-Order Effect. Biochemical and Biophysical Research
         Communications 2000, 278, 477-483.<- quasi-seq-order refernece
    [9]: Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using
         Pseudo-Amino Acid Composition. PROTEINS: Structure, Function, and
         Genetics, 2001, 43: 246-255.
    [10]: Kuo-Chen Chou. Using amphiphilic pseudo amino acid composition to predict enzyme
          subfamily classes. Bioinformatics, 2005,21,10-19.
    """
    def __init__(self, desc_config="", protein_seqs=None):

        self.desc_config = desc_config
        self.protein_seqs = protein_seqs
        self.parameters = {}

        desc_config_filepath = ""
        #open json config file
        if not isinstance(desc_config, str) or desc_config is None:
            raise TypeError('JSON config file must be a filepath of type string, got type {}'.format(type(desc_config)))
        if os.path.isfile(self.desc_config):
            desc_config_filepath = self.desc_config
        elif os.path.isfile(os.path.join('config', self.desc_config)):
            desc_config_filepath = os.path.join('config', self.desc_config)
        else:
            raise OSError('JSON config file not found at path: {}.'.format(desc_config_filepath))
        try:
            with open(desc_config_filepath) as f:
                self.parameters = json.load(f)
        except:
            raise JSONDecodeError('Error parsing config JSON file: {}.'.format(desc_config_filepath))

        #set descriptor parameters
        self.dataset_parameters = self.parameters["dataset"][0]
        self.descr_config = self.parameters["descriptors"]
        self.descr_parameters = self.parameters["descriptor_parameters"]
        self.all_desc = self.descr_config[0]["descriptors"]["all_desc"]
        
        if not isinstance(self.protein_seqs, pd.Series):
            if (self.protein_seqs is None or self.protein_seqs == ""): 
                dataset_filepath = ""
                #open dataset and read protein seqs if protein_seqs is empty/None
                if os.path.isfile(self.dataset_parameters["dataset"]):
                    dataset_filepath = self.dataset_parameters["dataset"]
                elif os.path.isfile(os.path.join(DATA_DIR, self.dataset_parameters["dataset"])):
                    dataset_filepath = os.path.join(DATA_DIR, self.dataset_parameters["dataset"])
                else:
                    raise OSError('Dataset file not found at path: {}.'.format(dataset_filepath))

                #read in dataset csv
                try:
                    data = pd.read_csv(dataset_filepath, sep=",", header=0)
                    self.protein_seqs = data[self.dataset_parameters["sequence_col"]]
                except:
                    raise IOError('Error opening dataset file: {}'.format(dataset_filepath))
            else: 
                #if 1 protein sequence (1 string) input then convert to pandas Series object
                if isinstance(self.protein_seqs, str):
                    self.protein_seqs = pd.Series(self.protein_seqs)

                #only the sequences should be passed in, not all columns in a dataset etc.
                if isinstance(self.protein_seqs, pd.DataFrame) and \
                    len(self.protein_seqs.columns) > 1:
                    raise ValueError('The full dataset must not be passed in, only the \
                        columns containing the protein sequences.')

        #remove any gaps from protein sequences
        self.protein_seqs = remove_gaps(self.protein_seqs)

        #validate that all input protein sequences are valid and only contain
        #  valid amino acids, if not then raise ValueError
        invalid_seqs = valid_sequence(self.protein_seqs)
        if invalid_seqs!=None:
            raise ValueError('Invalid Amino Acids found in protein sequence dataset: {}'.
                format(invalid_seqs))

        #get the total number of inputted protein sequences
        self.num_seqs = len(self.protein_seqs)

        #initialise all descriptor attributes to empty dataframes
        self.aa_composition = pd.DataFrame()
        self.dipeptide_composition = pd.DataFrame()
        self.tripeptide_composition = pd.DataFrame()
        self.normalized_moreaubroto_autocorrelation = pd.DataFrame()
        self.moran_autocorrelation = pd.DataFrame()
        self.geary_autocorrelation = pd.DataFrame()
        self.ctd = pd.DataFrame()
        self.comp = pd.DataFrame()
        self.transition = pd.DataFrame()
        self.distribution = pd.DataFrame()
        self.conjoint_triad = pd.DataFrame()
        self.seq_order_coupling_number = pd.DataFrame()
        self.quasi_seq_order = pd.DataFrame()
        self.pseudo_aa_composition = pd.DataFrame()
        self.amp_pseudo_aa_composition = pd.DataFrame()
        self.all_descriptors = pd.DataFrame()

        #try importing descriptors csv with pre-calculated descriptor values,
        #  if not found then calculate all descriptors if all_desc is true
        if os.path.isfile((os.path.join(DATA_DIR, self.descr_config[0]["descriptors_csv"]))):
            self.import_descriptors()
            #get the total number of inputted protein sequences
            self.num_seqs = self.all_descriptors.shape[0]
        else:
            #if all_desc parameter true then calculate all descriptor values and store in their respective attributes
            if (self.all_desc):
                self.all_descriptors = self.get_all_descriptors()
                #save all calculated descriptor values for next time
                self.all_descriptors.to_csv(os.path.join(DATA_DIR, self.descr_config[0]["descriptors_csv"]), index=0)

        #create dictionary of descriptors and their associated groups
        keys = self.all_descriptors_list()
        values = ["Composition"]*3 + ["Autocorrelation"]*3 + ["CTD"]*4 + ["Conjoint Triad"] + \
            ["Quasi-Sequence-Order"]*2 + ["Pseudo Composition"]*2
        self.descriptor_groups = dict(zip(keys,values))

        #get shape of descriptors
        self.shape = self.all_descriptors.shape

    def import_descriptors(self):
        """
        By default, the class will search for a file in the DATA_DIR called
        whatever the "descriptors_csv" parameter in the config file is set to. This
        file contains ALL of the pre-calculated descriptor values.
        This function parses this csv file and sets the descriptor instance variables
        to the correct feature values. The descriptors file should contain the values
        of all possible 15 descriptor parameters for the N protein sequences.
        To create a descriptors file, create an instance of the desc class,
        with the parameter all_desc=True in the config file, this will calculate
        all descriptors and store them to the DATA_DIR in a csv. The aim of this 
        function is to simply import the descriptors during each use of pySAR rather than
        calculating each time.

        Parameters
        ----------
        None
        """
        descr_filepath = os.path.join(DATA_DIR, self.descr_config[0]["descriptors_csv"])

        #verify descriptors csv exists
        if not (os.path.isfile(descr_filepath)):
            raise OSError('Descriptors csv file does not exist, at filepath: {}'.format(descr_filepath))

        #import descriptors csv as dataframe
        try:
            descriptor_df = pd.read_csv(descr_filepath)
        except IOError:
            print('Error reading descriptors csv file: {}.'.format(descr_filepath))

        #replacing any +/- infinity or NAN values with 0
        descriptor_df.replace([np.inf, -np.inf], np.nan)
        descriptor_df = descriptor_df.fillna(0)

        #calculate dimension of each descriptor in the csv
        #pull each descriptor value from the csv according to its dimension, setting
        #the values to the class instance variables
        aa_composition_dim = (0,20)
        self.aa_composition = descriptor_df.iloc[:,aa_composition_dim[0]:aa_composition_dim[1]]

        dipeptide_composition_dim = (20,420)
        self.dipeptide_composition = descriptor_df.iloc[:,dipeptide_composition_dim[0]:dipeptide_composition_dim[1]]

        tripeptide_composition_dim = (420,8420)
        self.tripeptide_composition = descriptor_df.iloc[:,tripeptide_composition_dim[0]:tripeptide_composition_dim[1]]

        #dimension of autocorrelation descriptors depends on the max lag value and number of properties
        norm_moreaubroto_dim = (8420,
            8420 + (self.descr_parameters[0]["normalized_moreaubroto_autocorrelation"][0]["lag"]*len(self.descr_parameters[0]["normalized_moreaubroto_autocorrelation"][0]["properties"])))
        self.normalized_moreaubroto_autocorrelation = descriptor_df.iloc[:,norm_moreaubroto_dim[0]:norm_moreaubroto_dim[1]]

        moran_auto_dim = (norm_moreaubroto_dim[1], norm_moreaubroto_dim[1] +
            (self.descr_parameters[0]["moran_autocorrelation"][0]["lag"]*len(self.descr_parameters[0]["moran_autocorrelation"][0]["properties"])))
        self.moran_autocorrelation = descriptor_df.iloc[:,moran_auto_dim[0]: moran_auto_dim[1]]

        geary_auto_dim = (moran_auto_dim[1], moran_auto_dim[1] +
            (self.descr_parameters[0]["geary_autocorrelation"][0]["lag"]*len(self.descr_parameters[0]["geary_autocorrelation"][0]["properties"])))
        self.geary_autocorrelation = descriptor_df.iloc[:,geary_auto_dim[0]:geary_auto_dim[1]]

        ctd_dim = (geary_auto_dim[1], geary_auto_dim[1]+147)
        self.ctd =  descriptor_df.iloc[:,ctd_dim[0]:ctd_dim[1]]

        composition_dim = (ctd_dim[1], ctd_dim[1]+21)
        self.comp = descriptor_df.iloc[:,composition_dim[0]:composition_dim[1]]
    
        transition_dim = (composition_dim[1], composition_dim[1]+21)
        self.transition = descriptor_df.iloc[:,transition_dim[0]:transition_dim[1]]

        distribution_dim = (transition_dim[1], transition_dim[1]+105)
        self.distribution = descriptor_df.iloc[:,distribution_dim[0]:distribution_dim[1]]

        conjoint_triad_dim = (distribution_dim[1], distribution_dim[1]+343)
        self.conjoint_triad = descriptor_df.iloc[:,conjoint_triad_dim[0]:conjoint_triad_dim[1]]

        #dimension of SOCNum depends on the maximum lag value 
        seq_order_coupling_number_dim = (conjoint_triad_dim[1],
            conjoint_triad_dim[1] + (self.descr_parameters[0]["seq_order_coupling_number"][0]["lag"])*2)
        self.seq_order_coupling_number = descriptor_df.iloc[:,seq_order_coupling_number_dim[0]:seq_order_coupling_number_dim[1]]

        quasi_seq_order_dim = (seq_order_coupling_number_dim[1], seq_order_coupling_number_dim[1] + 100)
        self.quasi_seq_order = descriptor_df.iloc[:,quasi_seq_order_dim[0]:quasi_seq_order_dim[1]]

        pseudo_aa_composition_dim = (quasi_seq_order_dim[1], quasi_seq_order_dim[1] + 50)
        self.pseudo_aa_composition = descriptor_df.iloc[:,pseudo_aa_composition_dim[0]:pseudo_aa_composition_dim[1]]

        amp_pseudo_aa_composition_dim = (pseudo_aa_composition_dim[1], pseudo_aa_composition_dim[1] + 80)
        self.amp_pseudo_aa_composition = descriptor_df.iloc[:,amp_pseudo_aa_composition_dim[0]:amp_pseudo_aa_composition_dim[1]]

        self.all_descriptors = descriptor_df.iloc[:,:]

        #**need to fix pseudo_aa_composition and amp_pseudo_aa_composition dims

    def get_aa_composition(self):
        """
        Calculate Amino Acid Composition (AAComp) of protein sequences using the
        respective function in the composition.py module in the descriptors directory.

        Returns
        -------
        :aa_composition : pd.Dataframe
            pandas dataframe of AAComp for protein sequence. Dataframe will
            be of the shape N x 20, where 20 is the number of features 
            calculated from the descriptor (for the 20 amino acids).
        """
        print('\nGetting AA Composition Descriptors...')
        print('#####################################\n')

        #if attribute already calculated & not empty then return it
        if not self.aa_composition.empty:
            return self.aa_composition

        #initialise dataframe
        aa_comp_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            aa_comp_seq = AAComposition(seq)
            aa_comp_df = pd.concat([aa_comp_df, aa_comp_seq])

        self.aa_composition = aa_comp_df

        return self.aa_composition

    def get_dipeptide_composition(self):
        """
        Calculate Dipeptide Composition (DPComp) of protein sequences using the
        respective function in the composition.py module in the descriptors directory.

        Returns
        -------
        :dipeptide_composition : pd.Dataframe
            pandas Dataframe of dipeptide composition for protein sequence. Dataframe will
            be of the shape N x 400, where 400 is the number of features calculated 
            from the descriptor (20^2 for the 20 canonical amino acids).
        """
        print('\nGetting Dipeptide Composition Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not self.dipeptide_composition.empty:
            return self.dipeptide_composition

        #initialise dataframe
        dipeptide_comp_df = pd.DataFrame()
        
        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            dipeptide_comp_seq = DipeptideComposition(seq)
            dipeptide_comp_df = pd.concat([dipeptide_comp_df, dipeptide_comp_seq])

        self.dipeptide_composition = dipeptide_comp_df

        return self.dipeptide_composition

    def get_tripeptide_composition(self):
        """ 
        Calculate Tripeptide Composition (TPComp) of protein sequences using the
        respective function in the composition.py module in the descriptors directory.

        Returns
        -------
        :tripeptide_composition : pd.Dataframe
            pandas Dataframe of tripeptide composition for protein sequence. Dataframe will
            be of the shape N x 8000, where 8000 is the number of features calculated 
            from the descriptor (20^3 for the 20 canonical amino acids).
        """
        print('\nGetting Tripeptide Composition Descriptors...')
        print('#############################################\n')

        #if attribute already calculated & not empty then return it
        if not self.tripeptide_composition.empty:
            return self.tripeptide_composition

        #initialise dataframe
        tripeptide_comp_df = pd.DataFrame()
        
        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            tripeptide_comp_seq = TripeptideComposition(seq)
            tripeptide_comp_df = pd.concat([tripeptide_comp_df, tripeptide_comp_seq])

        self.tripeptide_composition = tripeptide_comp_df

        return self.tripeptide_composition

    def get_norm_moreaubroto_autocorrelation(self):
        """
        Calculate Normalized Moreau-Broto Autocorrelation (NMBAuto) of protein sequences using the
        respective function in the autocorrelation.py module in the descriptors directory.

        Returns
        -------
        :normalized_moreaubroto_autocorrelation : pd.Dataframe
            pandas Dataframe of NMBAuto values for protein sequence. Output will
            be of the shape N x M, where M is the number of features calculated from
            the descriptor. By default, the shape will be N x 240 (30 features per 
            property - using 8 properties).
        """
        print('\nGetting Normalized Moreaubroto Autocorrelation Descriptors...')
        print('#############################################################\n')

        #if attribute already calculated & not empty then return it
        if not self.normalized_moreaubroto_autocorrelation.empty:
            return self.normalized_moreaubroto_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.descr_parameters[0]["normalized_moreaubroto_autocorrelation"][0]["lag"]
        properties = self.descr_parameters[0]["normalized_moreaubroto_autocorrelation"][0]["properties"]

        #initialise dataframe
        norm_moreaubroto_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            norm_moreaubroto_seq = norm_moreaubroto_autocorrelation(seq, lag=lag, properties=properties)
            norm_moreaubroto_df = pd.concat([norm_moreaubroto_df, norm_moreaubroto_seq])
            
        self.normalized_moreaubroto_autocorrelation = norm_moreaubroto_df

        return self.normalized_moreaubroto_autocorrelation

    def get_moran_autocorrelation(self):
        """
        Calculate Moran Autocorrelation (MAuto) of protein sequences using the
        respective function in the autocorrelation.py module in the descriptors directory.

        Returns
        -------
        :moran_autocorrelation : pd.DataFrame
            pandas Dataframe of MAuto values for protein sequence. Output will
            be of the shape N x M, where M is the number of features calculated from
            the descriptor. By default, the shape will be N x 240 (30 features per 
            property - using 8 properties).
        """
        print('\nGetting Moran Autocorrelation Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not self.moran_autocorrelation.empty:
            return self.moran_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.descr_parameters[0]["moran_autocorrelation"][0]["lag"]
        properties = self.descr_parameters[0]["moran_autocorrelation"][0]["properties"]

        #initialise dataframe
        moran_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            moran_seq = moran_autocorrelation(seq, lag=lag, properties=properties)
            moran_df = pd.concat([moran_df, moran_seq])

        self.moran_autocorrelation = moran_df

        return self.moran_autocorrelation 
        
    def get_geary_autocorrelation(self):
        """
        Calculate Geary Autocorrelation (GAuto) of protein sequences using the
        respective function in the autocorrelation.py module in the descriptors directory.

        Returns
        -------
        :geary_autocorrelation : pd.DataFrame
            pandas Dataframe of GAuto values for protein sequence. Output will
            be of the shape N x M, where M is the number of features calculated from
            the descriptor. By default, the shape will be N x 240 (30 features per 
            property - using 8 properties).
        """
        print('\nGetting Geary Autocorrelation Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not self.geary_autocorrelation.empty:
            return self.geary_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.descr_parameters[0]["geary_autocorrelation"][0]["lag"]
        properties = self.descr_parameters[0]["geary_autocorrelation"][0]["properties"]

        #initialise dataframe
        geary_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            geary_seq = geary_autocorrelation(seq, lag=lag, properties=properties)
            geary_df = pd.concat([geary_df, geary_seq])

        self.geary_autocorrelation = geary_df

        return self.geary_autocorrelation 

    def get_composition(self):
        """ 
        Calculate Composition (C_CTD) of protein sequences using the
        respective function in the ctd.py module in the descriptors directory.

        Returns
        -------
        :composition : pd.DataFrame
            pandas dataframe of C_CTD values for protein sequence. Output will
            be of the shape N x 3, where 3 is the number of features calculated from
            the descriptor.
        """
        print('\nGetting Composition (CTD) Descriptors...')
        print('#########################################\n')

        #if attribute already calculated & not empty then return it
        if not self.comp.empty:
            return self.comp

        #get descriptor-specific parameters from config file
        property = self.descr_parameters[0]["composition"][0]["property"]

        #initialise dataframe
        comp_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            comp_seq = ctd_composition(seq, property=property)
            comp_df = pd.concat([comp_df, comp_seq])

        self.comp = comp_df

        return self.comp

    def get_transition(self):
        """ 
        Calculate Transition (T_CTD) of protein sequences using the
        respective function in the ctd.py module in the descriptors directory.

        Returns
        -------
        :transition : pd.Dataframe
            pandas Dataframe of T_CTD values for protein sequence. Output will
            be of the shape N x 1, where 3 is the number of features calculated from
            the descriptor.
        """
        print('\nGetting Transition (CTD) Descriptors...')
        print('#########################################\n')

        #if attribute already calculated & not empty then return it
        if not self.transition.empty:
            return self.transition

        #get descriptor-specific parameters from config file
        property = self.descr_parameters[0]["transition"][0]["property"]

        #initialise dataframe
        transition_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            transition_seq = ctd_transition(seq, property=property)
            transition_df = pd.concat([transition_df, transition_seq])

        self.transition = transition_df

        return self.transition

    def get_distribution(self):
        """ 
        Calculate Distribution (D_CTD) of protein sequences using the
        respective function in the ctd.py module in the descriptors directory.

        Returns
        -------
        :distribution : pd.Dataframe
            pandas Dataframe of D_CTD values for protein sequence. Output will
            be of the shape N x 15, where 15 is the number of features calculated from
            the descriptor.
        """
        print('\nGetting Distribution (CTD) Descriptors...')
        print('#########################################\n')

        #if attribute already calculated & not empty then return it
        if not self.distribution.empty:
            return self.distribution

        #get descriptor-specific parameters from config file
        property = self.descr_parameters[0]["distribution"][0]["property"]

        #initialise dataframe
        distribution_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            distribution_seq = ctd_distribution(seq, property=property)
            distribution_df = pd.concat([distribution_df, distribution_seq])

        self.distribution = distribution_df

        return self.distribution

    def get_ctd(self):
        """
        Calculate CTD of protein sequences using the respective function in 
        the ctd.py module in the descriptors directory.

        Returns
        -------
        :ctd : pd.Series
            pandas Series of CTD values for protein sequence. Output will
            be of the shape N x 147, where 147 is the number of features calculated from
            the descriptor.
        """
        print('\nGetting CTD Descriptors...')
        print('##########################\n')

        #if attribute already calculated & not empty then return it
        if not self.ctd.empty:
            return self.ctd

        #get descriptor-specific parameters from config file
        property = self.descr_parameters[0]["ctd"][0]["property"]
        all_ctd = self.descr_parameters[0]["ctd"][0]["all"]

        #initialise dataframe
        ctd_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            ctd_seq = ctd_(seq, property=property, all_ctd=all_ctd)
            ctd_df = pd.concat([ctd_df, ctd_seq])

        self.ctd = ctd_df

        return self.ctd

    def get_conjoint_triad(self):
        """
        Calculate Conjoint Triad (CTriad) of protein sequences using the respective function in 
        the conjointTriad.py module in the descriptors directory.

        Returns
        -------
        :conjoint_triad : pd.Dataframe
            pandas Dataframe of CTriad descriptor values for all protein sequences. Dataframe
            will be of the shape N x 343, where 343 is the number of features calculated 
            from the descriptor for a sequence.
        """
        print('\nGetting Conjoint Triad Descriptors...')
        print('#####################################\n')

        #if attribute already calculated & not empty then return it
        if not self.conjoint_triad.empty:
            return self.conjoint_triad

        #initialise dataframe
        conjoint_triad_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            conjoint_triad_seq = conjoint_triad(seq)
            conjoint_triad_df = pd.concat([conjoint_triad_df, conjoint_triad_seq])

        self.conjoint_triad = conjoint_triad_df

        return self.conjoint_triad

    def get_seq_order_coupling_number(self):
        """
        Calculate Sequence Order Coupling number (SOCNum) of protein sequences 
        using the respective function in the quasiSequenceOrder.py module in the descriptors directory.

        Returns
        -------
        :seq_order_df : pd.Dataframe
            Dataframe of SOCNum descriptor values for all protein sequences. Output
            will be of the shape N x M, where N is the number of protein sequences and
            M is the number of features calculated from the descriptor (calculated as
            N * 2 where N = lag).
        """
        print('\nGetting Sequence Order Coupling Descriptors...')
        print('##############################################\n')

        #if attribute already calculated & not empty then return it
        if not self.seq_order_coupling_number.empty:
            return self.seq_order_coupling_number

        #initialise dataframe
        seq_order_coupling_number_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.descr_parameters[0]["seq_order_coupling_number"][0]["lag"]
        distance_matrix = self.descr_parameters[0]["seq_order_coupling_number"][0]["distance_matrix"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            seq_order_coupling_number_seq = seq_order_coupling_number(seq, lag=lag, distance_matrix=distance_matrix)
            seq_order_coupling_number_df = pd.concat([seq_order_coupling_number_df, seq_order_coupling_number_seq])

        self.seq_order_coupling_number = seq_order_coupling_number_df

        return self.seq_order_coupling_number

    def get_quasi_seq_order(self):
        """
        Calculate Quasi Sequence Order Coupling number (QSOrder) of protein sequences 
        using the respective function in the quasiSequenceOrder.py module in the descriptors directory.

        Returns
        -------
        :quasi_seq_order_df : pd.Dataframe
            Dataframe of quasi-sequence-order descriptor values for the
            protein sequences, with output shape N x 100 where N is the number
            of sequences and 100 the number of calculated features.
        """
        print('\nGetting Quasi Sequence Order Descriptors...')
        print('###########################################\n')

        #if attribute already calculated & not empty then return it
        if not self.quasi_seq_order.empty:
            return self.quasi_seq_order

        #initialise dataframe
        quasi_seq_order_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.descr_parameters[0]["quasi_seq_order"][0]["lag"]
        weight = self.descr_parameters[0]["quasi_seq_order"][0]["weight"]
        distance_matrix = self.descr_parameters[0]["quasi_seq_order"][0]["distance_matrix"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            quasi_seq_order_seq = seq_order_coupling_number(seq, lag=lag, distance_matrix=distance_matrix)
            quasi_seq_order_df = pd.concat([quasi_seq_order_df, quasi_seq_order_seq])

        self.quasi_seq_order = quasi_seq_order_df

        return self.quasi_seq_order

    def get_pseudo_aa_composition(self):
        """
        Calculate Pseudo Amino Acid Composition (PAAComp) of protein sequences 
        using the respective function in the composition.py module in the descriptors directory.

        Returns
        -------
        :pseudo_aa_composition : pd.Dataframe
            Dataframe of pseudo amino acid composition descriptor values
            for the protein sequences of output shape N x 50.
        """
        print('\nGetting Pseudo Amino Acid Composition Descriptors...')
        print('####################################################\n')

        #if attribute already calculated & not empty then return it
        if not self.pseudo_aa_composition.empty:
            return self.pseudo_aa_composition

        #initialise dataframe
        pseudo_aacomp_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lamda = self.descr_parameters[0]["pseudo_aa_composition"][0]["lambda"]
        weight = self.descr_parameters[0]["pseudo_aa_composition"][0]["weight"]
        properties = self.descr_parameters[0]["pseudo_aa_composition"][0]["properties"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            pseudo_aacomp_seq = pseudoAAC(seq, lamda=lamda, weight=weight, properties=properties)
            pseudo_aacomp_df = pd.concat([pseudo_aacomp_df, pseudo_aacomp_seq])

        self.pseudo_aa_composition = pseudo_aacomp_df

        return self.pseudo_aa_composition
        
    def get_amp_pseudo_aa_composition(self):
        """
        Calculate Amphiphilic Pseudo Amino Acid Composition (APAAComp) of protein sequences 
        using the respective function in the composition.py module in the descriptors directory.

        Returns
        -------
        :amp_pseudo_aa_composition_df : pd.Dataframe
            Dataframe of Amphiphilic pseudo amino acid composition descriptor
            values for the protein sequences of output shape N x 80.
        """
        print('\nGetting Amphiphilic Pseudo Amino Acid Composition Descriptors...')
        print('###############################################################\n')

        #if attribute already calculated & not empty then return it
        if not self.amp_pseudo_aa_composition.empty:
            return self.amp_pseudo_aa_composition

        #get descriptor-specific parameters from config file
        lamda = self.parameters["amp_pseudo_aa_composition"]["lambda"]
        weight = self.parameters["amp_pseudo_aa_composition"]["weight"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            amp_pseudo_aacomp_seq = amphiphilicPseudoAAC(seq, lamda=lamda, weight=weight)
            amp_pseudo_aacomp_df = pd.concat([amp_pseudo_aacomp_df, amp_pseudo_aacomp_seq])

        self.amp_pseudo_aa_composition = amp_pseudo_aacomp_df

        return self.amp_pseudo_aa_composition
    
    def get_descriptor_encoding(self, descriptor):
        """
        Get the protein descriptor values of a specified input descriptor. If the
        sought descriptor has already been calculated then its attribute is returned,
        else the descriptor is calculated using its get_descriptor function.

        Parameters
        ----------
        :descriptor : str
            name of descriptor to return. Method can accept the approximate name
            of the descriptor, e.g. 'aa_comp'/'aa_compo' etc will return the
            'aa_composition' descriptor. This functionality is realised using
            the difflib library and its built-in get_close_matches function.

        Returns
        -------
        :desc_encoding : pd.DataFrame/None
            dataframe of descriptor values, calculated from the descriptor
            instances' protein sequences. None returned if no matching descriptor found.
        """
        #remove any detected whitespace from input parameter
        try:
          descriptor = descriptor.strip()
        except:
          raise TypeError('Input parameter {} is not of correct datatype string, got {}' \
            .format(descriptor, type(descriptor)))

        #validate input descriptor is a valid available descriptor, get its closest match
        desc_matches = get_close_matches(descriptor,self.valid_descriptors(),cutoff=0.4)
        if desc_matches!=[]:
            desc = desc_matches[0]  #set desc to closest descriptor match found
        else:
            raise ValueError('Could not find a match for the input descriptor {} \
                in available valid models:\n {}.'.format(descriptor, self.valid_descriptors()))

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
        elif desc == 'ctd':
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.ctd
        elif desc == 'comp':
            if (getattr(self, desc).empty):
              self.get_composition()
            desc_encoding = self.comp
        elif desc == 'transition':
            if (getattr(self, desc).empty):
              self.get_transition()
            desc_encoding = self.transition
        elif desc == 'distribution':
            if (getattr(self, desc).empty):
              self.get_distribution()
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
        elif desc == 'pseudo_aa_composition':
            if (getattr(self, desc).empty):
              self.get_pseudo_aa_composition()
            desc_encoding = self.pseudo_aa_composition
        elif desc == 'amp_pseudo_aa_composition':
            if (getattr(self, desc).empty):
              self.get_amp_pseudo_aa_composition()
            desc_encoding = self.amp_pseudo_aa_composition
        else:
          desc_encoding = None           #no matching descriptor

        return desc_encoding

    def all_descriptors_list(self, desc_combo=1):
       """
       Get list of all available descriptor attributes. Using the desc_combo
       input parameter you can get the list of all descriptors, all combinations
       of 2 descriptors or all combinations of 3 descriptors. Default of 1 will
       mean a list of all available descriptor attributes will be returned.

       Parameters
       ----------
       :desc_combo : int (default=1)
            combination of descriptors to return. A value of 2 or 3 will return
            all combinations of 2 or 3 descriptor attributes etc.

       Returns
       -------
       :all_descriptors : list
            list of available descriptor attributes.
       """
       #filter out class attributes that are not any of the desired descriptors
       all_descriptors = list(filter(lambda x: x.startswith('_'), list(self.__dict__.keys())))
       all_descriptors = list(filter(lambda x: not x.startswith('_all_desc'), all_descriptors))
       all_descriptors = [de[1:] for de in all_descriptors]

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
        Dataframe into one storing all descriptors.

        Returns
        -------
        :all_desc_df : pd.DataFrame
            concatenated dataframe of all individual descriptors. Using the default
            attributes and their associated values, the output will be of the shape
            N x 9920. The 2nd dimension is dependant on several additional parameters
            of some descriptors, including the number of properties and max lag for the
            Autocorrelation descriptors, the max lag for the SOCNum, the number of properties
            and lamda for PAAComp and the lambda for APAAComp. 9920 features are calculated 
            in total when using the default parameters of each descriptor and N represents 
            the number of sequences.
        """
        print('Calculating all descriptor values....\n')
        print('#####################################\n')

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

        if (getattr(self, "ctd").empty):
                self.ctd = self.get_ctd()

        if (getattr(self, "comp").empty):
                self.ctd = self.get_composition()

        if (getattr(self, "transition").empty):
            self.ctd = self.get_transition()
        
        if (getattr(self, "distribution").empty):
            self.ctd = self.get_distribution()

        if (getattr(self, "conjoint_triad").empty):
                self.conjoint_triad = self.get_conjoint_triad()

        if (getattr(self, "seq_order_coupling_number").empty):
                self.seq_order_coupling_number = self.get_seq_order_coupling_number()

        if (getattr(self, "quasi_seq_order").empty):
                self.quasi_seq_order = self.get_quasi_seq_order()

        if (getattr(self, "pseudo_aa_composition").empty):
                self.pseudo_aa_composition = self.get_pseudo_aa_composition()

        # if (getattr(self, "amp_pseudo_aa_composition").empty):
        #         self.amp_pseudo_aa_composition = self.get_amp_pseudo_aa_composition()

        #append all calculated descriptors to list
        all_desc = [
            self.aa_composition, self.dipeptide_composition, self.tripeptide_composition,
            self.normalized_moreaubroto_autocorrelation, self.moran_autocorrelation,
            self.geary_autocorrelation, self.comp, self.transition,
            self.distribution, self.ctd, self.conjoint_triad, self.seq_order_coupling_number,
            self.quasi_seq_order, self.pseudo_aa_composition
            ]
        # self.amp_pseudo_aa_composition
        #concatenate individual descriptor dataframe attributes
        all_desc_df = pd.concat(all_desc, axis = 1)
        self.all_descriptors = all_desc_df     

        return all_desc_df

    def valid_descriptors(self):
        """
        Get a list of all valid descriptors available in the Descriptors class.

        Returns
        -------
        :valid_desc : list
            list of all valid descriptors that the class supports.
        """
        valid_desc = [
            'aa_composition', 'dipeptide_composition', 'tripeptide_composition',
            'normalized_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation',
            'ctd', 'comp', 'transition', 'distribution', 'conjoint_triad',
            'seq_order_coupling_number','quasi_seq_order',
            'pseudo_aa_composition', 'amp_pseudo_aa_composition'
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
    def ctd(self):
        return self._ctd

    @ctd.setter
    def ctd(self, val):
        self._ctd = val

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, val):
        self._comp = val

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
    def pseudo_aa_composition(self):
        return self._pseudo_aa_composition

    @pseudo_aa_composition.setter
    def pseudo_aa_composition(self, val):
        self._pseudo_aa_composition = val

    @property
    def amp_pseudo_aa_composition(self):
        return self._amp_pseudo_aa_composition

    @amp_pseudo_aa_composition.setter
    def amp_pseudo_aa_composition(self, val):
        self._amp_pseudo_aa_composition = val

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
        del self._ctd
        del self._transition
        del self._comp
        del self._distribution
        del self._conjoint_triad
        del self._seq_order_coupling_number
        del self._quasi_seq_order
        del self._pseudo_aa_composition
        del self._amp_pseudo_aa_composition

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

    # Methods
    # -------
    # import_descriptors(descriptor_file)
    # get_aa_composition()
    # get_dipeptide_composition()
    # get_tripeptide_composition()
    # get_norm_moreaubroto_autocorrelation()
    # get_moran_autocorrelation()
    # get_geary_autocorrelation()
    # ......