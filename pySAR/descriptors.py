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

from .globals_ import DATA_DIR
from .utils import *
import protpy as protpy

class Descriptors():
    """
    Class for calculating a wide variety of protein physiochemical, biochemical and structural 
    descriptors. These descriptors have been used in a wide variety of Bioinformaitcs 
    applications including: protein strucutral and functional class prediction, 
    protein-protein interactions, subcellular location, secondary structure prediction etc. 
    They represent the different structural, functional & interaction profiles of proteins 
    by exploring the features in the groups of composition, correlation and distribution 
    of the constituent residues and their biochemical and physiochemical properties.

    A custom-built software package was created to generate these descriptors - protpy, which
    is also open-souurce and available here: https://github.com/amckenna41/protpy. The package
    takes 1 or more protein sequences, returning the respective descriptor values in a Pandas
    DataFrame. protpy and this class allows calculation of the following descriptors: Amino 
    Acid Compostion (AAComp), Dipeptide Composition (DPComp), Tripeptide Composition (TPComp), 
    MoreauBroto Autocorrelation (MBAuto), Moran Autocorrelation (MAuto), Geary Autocorrelation 
    (GAuto), Composition (C), Transition (T), Distribution (D), CTD, Conjoint Triad (CTriad), 
    Sequence Order Coupling Number (SOCN), Quasi Sequence Order (QSO), Pseudo Amino Acid 
    Composition - type 1 (PAAcomp) and Amphiphilic Pseudo Amino Acid Composition - type 2 (APAAComp). 

    Similar to other classes in pySAR, this class works via configuration files which contain
    the values for all the potential parameters, if applicable, of each descriptor. By default, 
    the class will look for a descriptors csv which is a file of the pre-calcualted descriptor 
    values for the specified dataset, if this file doesn't exist, or the parameter value is blank, 
    then each descriptor will have to be calculated using  its respective function.

    It is reccomended that with every new dataset, the Descriptors class should be instantiated 
    with the "all_desc" parameter set to 1 in the config file. This will calculate all the descriptor
    values for the dataset of protein sequences, storing the result in a csv file, meaning that
    this file can be used for future use and the descriptors will not have to be recalculated.

    Parameters
    ----------
    :config_file: str
        path to configuration file which will contain the various parameter values for all
        descriptors. If invalid value input then error will be raised.
    :protein_seqs : np.ndarray
        array of protein sequences that descriptors will be calculated for. If set to none 
        or empty then error will be raised.

    References
    ----------
    [1]  Dong, J., Yao, ZJ., Zhang, L. et al. PyBioMed: a python library for
         various molecular representations of chemicals, proteins and DNAs and
         their interactions. J Cheminform 10, 16 (2018).
         https://doi.org/10.1186/s13321-018-0270-2
    [2]  Reczko, M. and Bohr, H. (1994) The DEF data base of sequence based protein
         fold class predictions. Nucleic Acids Res, 22, 3616-3619.
    [3]  Hua, S. and Sun, Z. (2001) Support vector machine approach for protein
         subcellular localization prediction. Bioinformatics, 17, 721-728.
    [4]  Broto P, Moreau G, Vandicke C: Molecular structures: perception,
         autocorrelation descriptor and SAR studies. Eur J Med Chem 1984, 19: 71–78.
    [5]  Ong, S.A., Lin, H.H., Chen, Y.Z. et al. Efficacy of different protein
         descriptors in predicting protein functional families. BMC Bioinformatics
         8, 300 (2007). https://doi.org/10.1186/1471-2105-8-300
    [6]  Inna Dubchak, Ilya Muchink, Stephen R.Holbrook and Sung-Hou Kim. Prediction
         of protein folding class using global description of amino acid sequence.
         Proc.Natl. Acad.Sci.USA, 1995, 92, 8700-8704.
    [7]  Juwen Shen, Jian Zhang, Xiaomin Luo, Weiliang Zhu, Kunqian Yu, Kaixian Chen,
         Yixue Li, Huanliang Jiang. Predicting proten-protein interactions based only
         on sequences inforamtion. PNAS. 2007 (104) 4337-4341.
    [8]  Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating
         Quasi-Sequence-Order Effect. Biochemical and Biophysical Research
         Communications 2000, 278, 477-483.
    [9]  Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using
         Pseudo-Amino Acid Composition. PROTEINS: Structure, Function, and
         Genetics, 2001, 43: 246-255.
    [10] Kuo-Chen Chou. Using amphiphilic pseudo amino acid composition to predict enzyme
          subfamily classes. Bioinformatics, 2005,21,10-19.
    [11] J. Shen et al., “Predicting protein-protein interactions based only on sequences
         information,” Proc. Natl. Acad. Sci. U. S. A., vol. 104, no. 11, pp. 4337–4341, 2007.
    [12] Gisbert Schneider and Paul Wrede. The Rational Design of Amino Acid Sequences
         by Artifical Neural Networks and Simulated Molecular Evolution: Do Novo Design
         of an Idealized Leader Cleavge Site. Biophys Journal, 1994, 66, 335-344.
    [13]  Grantham, R. (1974-09-06). "Amino acid difference formula to help explain protein
         evolution". Science. 185 (4154): 862–864. Bibcode:1974Sci...185..862G.
         doi:10.1126/science.185.4154.862. ISSN 0036-8075. PMID 4843792. S2CID 35388307.   
    """
    def __init__(self, config_file="", protein_seqs=None):

        self.config_file = config_file
        self.protein_seqs = protein_seqs
        self.parameters = {}

        desc_config_filepath = ""

        #import congfig file, raise error if invalid path
        if not (isinstance(self.config_file, str) or (self.config_file is None)):
            raise TypeError('JSON config file must be a filepath of type string, got type {}.'.format(type(config_file)))
        if (os.path.splitext(self.config_file)[1] == ''):
            self.config_file = self.config_file + '.json' #append extension if only filename input        
        if (os.path.isfile(self.config_file)):
            desc_config_filepath = self.config_file
        elif (os.path.isfile(os.path.join('config', self.config_file))):
            desc_config_filepath = os.path.join('config', self.config_file)
        else:
            raise OSError('JSON config file not found at path: {}.'.format(self.config_file))

        #open json file and read config parameters
        try:
            with open(desc_config_filepath) as f:
                self.parameters = json.load(f)
        except:
            raise JSONDecodeError('Error parsing config JSON file: {}.'.format(desc_config_filepath))
        
        #create instance of Map class so parameters in config can be accessed via dot notation
        self.dataset_parameters = Map(self.parameters["dataset"])
        self.desc_parameters = Map(self.parameters["descriptors"])

        #parameter of whether to read in / use all available descriptors 
        self.all_desc = self.desc_parameters.all_desc 
        
        #create data directory if doesnt exist
        if not (os.path.isdir(DATA_DIR)):
            os.makedirs(DATA_DIR)

        #import protein sequences from dataset if not directly specified in protein_seqs input param
        if not (isinstance(self.protein_seqs, pd.Series)):
            if (self.protein_seqs is None or self.protein_seqs == ""): 
                dataset_filepath = ""
                #open dataset and read protein seqs if protein_seqs is empty/None
                if (os.path.isfile(self.dataset_parameters["dataset"])):
                    dataset_filepath = self.dataset_parameters["dataset"]
                elif (os.path.isfile(os.path.join(DATA_DIR, self.dataset_parameters["dataset"]))):
                    dataset_filepath = os.path.join(DATA_DIR, self.dataset_parameters["dataset"])
                else:
                    raise OSError('Dataset file not found at path: {}.'.format(dataset_filepath))

                #read in dataset csv from filepath mentioned in config 
                try:
                    data = pd.read_csv(dataset_filepath, sep=",", header=0)
                    self.protein_seqs = data[self.dataset_parameters["sequence_col"]]
                except:
                    raise IOError('Error opening dataset file: {}.'.format(dataset_filepath))
            else: 
                #if 1 protein sequence (1 string) input then convert to pandas Series object
                if (isinstance(self.protein_seqs, str)):
                    self.protein_seqs = pd.Series(self.protein_seqs)

                #only the sequences should be passed in, not all columns in a dataset etc.
                if (isinstance(self.protein_seqs, pd.DataFrame) and \
                    len(self.protein_seqs.columns) > 1):
                    raise ValueError("The full dataset must not be passed in, only the"
                        " columns containing the protein sequences.")

        #remove any gaps from protein sequences
        self.protein_seqs = remove_gaps(self.protein_seqs)

        #validate that all input protein sequences are valid and only contain
        #valid amino acids, if not then raise ValueError
        invalid_seqs = valid_sequence(self.protein_seqs)
        if (invalid_seqs != None):
            raise ValueError('Invalid Amino Acids found in protein sequence dataset: {}.'.
                format(invalid_seqs))

        #get the total number of inputted protein sequences
        self.num_seqs = len(self.protein_seqs)

        #initialise all descriptor attributes to empty dataframes
        self.amino_acid_composition = pd.DataFrame()
        self.dipeptide_composition = pd.DataFrame()
        self.tripeptide_composition = pd.DataFrame()
        self.moreaubroto_autocorrelation = pd.DataFrame()
        self.moran_autocorrelation = pd.DataFrame()
        self.geary_autocorrelation = pd.DataFrame()
        self.ctd = pd.DataFrame()
        self.ctd_composition = pd.DataFrame()
        self.ctd_transition = pd.DataFrame()
        self.ctd_distribution = pd.DataFrame()
        self.conjoint_triad = pd.DataFrame()
        self.sequence_order_coupling_number = pd.DataFrame()
        self.quasi_sequence_order = pd.DataFrame()
        self.pseudo_amino_acid_composition = pd.DataFrame()
        self.amphiphilic_pseudo_amino_acid_composition = pd.DataFrame()
        self.all_descriptors = pd.DataFrame()
        
        #append extension if just the filename input as descriptors csv
        if ((self.desc_parameters.descriptors_csv != '' and self.desc_parameters.descriptors_csv != None) 
            and (os.path.splitext(self.desc_parameters.descriptors_csv)[1] == '')):
            self.desc_parameters.descriptors_csv = self.desc_parameters.descriptors_csv + ".csv"

        #try importing descriptors csv with pre-calculated descriptor values
        if (os.path.isfile(self.desc_parameters.descriptors_csv) or 
            os.path.isfile((os.path.join(DATA_DIR, self.desc_parameters.descriptors_csv)))):
            self.import_descriptors(self.desc_parameters.descriptors_csv)
            #get the total number of inputted protein sequences
            self.num_seqs = self.all_descriptors.shape[0]
        # else:
            #if all_desc parameter true then calculate all descriptor values and store in their respective attributes
            # if (self.all_desc):
            #     self.all_descriptors = self.get_all_descriptors()
            #     #save all calculated descriptor values for next time
            #     if (self.desc_config.descriptors_csv == "" or self.desc_config.descriptors_csv == None):
            #         self.desc_config.descriptors_csv = "descriptors_output.csv"
            #     self.all_descriptors.to_csv(os.path.join(DATA_DIR, self.desc_config.descriptors_csv), index=0)

        #create dictionary of descriptors and their associated groups
        keys = self.all_descriptors_list()
        values = ["Composition"]*3 + ["Autocorrelation"]*3 + ["CTD"]*4 + ["Conjoint Triad"] + \
            ["Sequence Order"]*2 + ["Pseudo Composition"]*2
        self.descriptor_groups = dict(zip(keys,values))

        #get shape of descriptors
        self.shape = self.all_descriptors.shape

        #list of available protein descriptors
        self.valid_descriptors = [
            'amino_acid_composition', 'dipeptide_composition', 'tripeptide_composition',
            'moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation',
            'ctd', 'ctd_composition', 'ctd_transition', 'ctd_distribution', 'conjoint_triad',
            'sequence_order_coupling_number','quasi_sequence_order',
            'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition'
        ]

    def import_descriptors(self, descriptor_filepath=""):
        """
        Import descriptors from descriptors csv, setting the class attrbutes to their values.
        It is reccommended that after calculating the descriptors for a dataset of sequences 
        that the calculated values are exported to a csv; this means they don't need to be 
        recalculated each time. The all_descriptors class attribute is a dataframe of all 
        concatenated descriptors from the csv.

        Parameters
        ----------
        :descriptor_filepath : str 
            filepath to pre-calculated descriptor csv file.

        Returns
        -------
        None
        """
        #raise type error if filepath parameter isnt string
        if not (isinstance(descriptor_filepath, str)):
            raise TypeError("Filepath input parameter should be type str, got {}.".format(type(descriptor_filepath)))

        #verify descriptors csv exists at filepath
        if not (os.path.isfile(descriptor_filepath)):
            descriptor_filepath = os.path.join(DATA_DIR, descriptor_filepath)
            if not (os.path.isfile(descriptor_filepath)):
                raise OSError('Descriptors csv file does not exist at filepath: {}.'.format(descriptor_filepath))

        #import descriptors csv as dataframe
        try:
            descriptor_df = pd.read_csv(descriptor_filepath)
        except IOError:
            print('Error reading descriptors csv file: {}.'.format(descriptor_filepath))

        #replacing any +/- infinity or NAN values with 0
        descriptor_df.replace([np.inf, -np.inf], np.nan)
        descriptor_df = descriptor_df.fillna(0)

        '''
        calculate dimension of each descriptor in the csv according to the properties of each
        descriptor, pull each descriptor value from the csv according to its dimension, 
        setting the values to the class instance variables
        '''
        amino_acid_composition_dim = (0, 20)
        self.amino_acid_composition = descriptor_df.iloc[:,amino_acid_composition_dim[0]:amino_acid_composition_dim[1]]

        dipeptide_composition_dim = (20, 420)
        self.dipeptide_composition = descriptor_df.iloc[:,dipeptide_composition_dim[0]:dipeptide_composition_dim[1]]

        tripeptide_composition_dim = (420, 8420)
        self.tripeptide_composition = descriptor_df.iloc[:,tripeptide_composition_dim[0]:tripeptide_composition_dim[1]]

        #dimension of autocorrelation descriptors depends on the lag value and number of properties
        norm_moreaubroto_dim = (8420,
            8420 + (self.desc_parameters.moreaubroto_autocorrelation["lag"] * len(self.desc_parameters.moreaubroto_autocorrelation["properties"])))
        self.moreaubroto_autocorrelation = descriptor_df.iloc[:,norm_moreaubroto_dim[0]:norm_moreaubroto_dim[1]]

        moran_auto_dim = (norm_moreaubroto_dim[1], norm_moreaubroto_dim[1] +
            (self.desc_parameters.moran_autocorrelation["lag"] * len(self.desc_parameters.moran_autocorrelation["properties"])))
        self.moran_autocorrelation = descriptor_df.iloc[:,moran_auto_dim[0]: moran_auto_dim[1]]

        geary_auto_dim = (moran_auto_dim[1], moran_auto_dim[1] +
            (self.desc_parameters.geary_autocorrelation["lag"] * len(self.desc_parameters.geary_autocorrelation["properties"])))
        self.geary_autocorrelation = descriptor_df.iloc[:,geary_auto_dim[0]:geary_auto_dim[1]]

        #get CTD parameters from config to determine the dimensions of the CTD descriptors
        ctd_property = self.desc_parameters.ctd["property"]
        if not (isinstance(ctd_property, list)):
            ctd_property = ctd_property.split(',')
        ctd_all_ctd = self.desc_parameters.ctd["all"]
        
        #if using all properties in CTD calculation, 147 features generated, 21 features per 7 properties
        if (ctd_all_ctd):
            ctd_dim = (geary_auto_dim[1], geary_auto_dim[1]+147) #21 CTD features per 7 properties = 147
            ctd_comp_dim = (geary_auto_dim[1], geary_auto_dim[1] + 21) #3 CTD_Comp features per 7 properties = 21
            ctd_trans_dim = (ctd_comp_dim[1], ctd_comp_dim[1] + 21) #3 CTD_Trans features per 7 properties = 21
            ctd_distr_dim = (ctd_trans_dim[1], ctd_trans_dim[1] + 105) #15 CTD_Distr features per 7 properties = 105
        #only using a pre-determined list of physiochemical properties, 21 features per property
        else: 
            ctd_comp_dim = (geary_auto_dim[1], geary_auto_dim[1] + (len(ctd_property) * 3)) #3 CTD_Comp features per property
            ctd_trans_dim = (ctd_comp_dim[1], ctd_comp_dim[1] + (len(ctd_property) * 3)) #3 CTD_Trans features per property
            ctd_distr_dim = (ctd_trans_dim[1], ctd_trans_dim[1] + (len(ctd_property) * 15)) #15 CTD_Distr features per property
            ctd_dim = (geary_auto_dim[1], ctd_distr_dim[1]) #21 CTD features per property
        
        self.ctd =  descriptor_df.iloc[:,ctd_dim[0]:ctd_dim[1]]  

        self.ctd_composition = descriptor_df.iloc[:,ctd_comp_dim[0]:ctd_comp_dim[1]]
    
        self.ctd_transition = descriptor_df.iloc[:,ctd_trans_dim[0]:ctd_trans_dim[1]]

        self.ctd_distribution = descriptor_df.iloc[:,ctd_distr_dim[0]:ctd_distr_dim[1]]

        conjoint_triad_dim = (ctd_distr_dim[1], ctd_distr_dim[1]+343)

        self.conjoint_triad = descriptor_df.iloc[:,conjoint_triad_dim[0]:conjoint_triad_dim[1]]
        
        socn_lag = self.desc_parameters.sequence_order_coupling_number["lag"]
        socn_distance_matrix = self.desc_parameters.sequence_order_coupling_number["distance_matrix"]

        #if no distance matrix speciifed in config then both are used for descriptor calculation
        if (socn_distance_matrix == "" or socn_distance_matrix == None):
            socn_dim = (conjoint_triad_dim[1], conjoint_triad_dim[1] + (socn_lag * 2))
        #distance matrix specified in config
        else:
            socn_dim = (conjoint_triad_dim[1], conjoint_triad_dim[1] + socn_lag)

        self.sequence_order_coupling_number = descriptor_df.iloc[:,socn_dim[0]:socn_dim[1]]

        quasi_seq_order_lag = self.desc_parameters.quasi_sequence_order["lag"]
        quasi_seq_order_dist_matrix = self.desc_parameters.quasi_sequence_order["distance_matrix"]

        #if no distance matrix speciifed in config then both are used for descriptor calculation
        if (quasi_seq_order_dist_matrix == "" or quasi_seq_order_dist_matrix == None):
            quasi_seq_order_dim = (socn_dim[1], socn_dim[1] + ((quasi_seq_order_lag+20) * 2))
        #distance matrix specified in config
        else:
            quasi_seq_order_dim = (socn_dim[1], socn_dim[1] + (quasi_seq_order_lag+20))

        self.quasi_sequence_order = descriptor_df.iloc[:,quasi_seq_order_dim[0]:quasi_seq_order_dim[1]]

        paac_lambda = self.desc_parameters.pseudo_amino_acid_composition["lambda"]
        
        pseudo_amino_acid_composition_dim = (quasi_seq_order_dim[1], quasi_seq_order_dim[1] + (20 + paac_lambda))
        self.pseudo_amino_acid_composition = descriptor_df.iloc[:,pseudo_amino_acid_composition_dim[0]:pseudo_amino_acid_composition_dim[1]]

        apaac_lambda = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["lambda"]
     
        amphiphilic_pseudo_amino_acid_composition_dim = (pseudo_amino_acid_composition_dim[1], 
            pseudo_amino_acid_composition_dim[1] + (20 + (2*apaac_lambda)))
        self.amphiphilic_pseudo_amino_acid_composition = descriptor_df.iloc[:,amphiphilic_pseudo_amino_acid_composition_dim[0]:
            amphiphilic_pseudo_amino_acid_composition_dim[1]]

        self.all_descriptors = descriptor_df.iloc[:,:]

    def get_amino_acid_composition(self):
        """
        Calculate Amino Acid Composition (AAComp) of protein sequence using the
        custom-built protpy package. AAComp describes the fraction of each amino 
        acid type within a protein sequence, and is calculated as:

        AA_Comp(s) = AA(t)/N(s)

        where AA_Comp(s) is the AAComp of protein sequence s, AA(t) is the number
        of amino acid types t (where t = 1,2,..,20) and N(s) is the length of the
        sequence s. 

        Parameters
        ----------
        None

        Returns
        -------
        :amino_acid_composition : pd.Dataframe
            pandas dataframe of AAComp for protein sequence. Dataframe will
            be of the shape N x 20, where N is the number of protein sequences
            and 20 is the number of features calculated from the descriptor 
            (for the 20 canonical amino acids).
        """
        print('\nGetting Amino Acid Composition Descriptors...')
        print('#############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.amino_acid_composition.empty):
            return self.amino_acid_composition

        #initialise dataframe
        aa_comp_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            aa_comp_seq = protpy.amino_acid_composition(seq)
            aa_comp_df = pd.concat([aa_comp_df, aa_comp_seq])

        self.amino_acid_composition = aa_comp_df

        return self.amino_acid_composition

    def get_dipeptide_composition(self):
        """
        Calculate Dipeptide Composition (DPComp) for protein sequence using
        the custom-built protpy package. Dipeptide composition is the fraction 
        of each dipeptide type within a protein sequence. With dipeptides 
        being of length 2 and there being 20 canonical amino acids this creates 
        20^2 different combinations, thus a 400-Dimensional vector will be produced 
        such that:

        DPComp(s,t) = AA(s,t) / N -1

        where DPComp(s,t) is the dipeptide composition of the protein sequence
        for amino acid type s and t (where s and t = 1,2,..,20), AA(s,t) is the number
        of dipeptides represented by amino acid type s and t and N is the total number
        of dipeptides.

        Parameters
        ----------
        None

        Returns
        -------
        :dipeptide_composition : pd.Dataframe
            pandas Dataframe of dipeptide composition for protein sequence. Dataframe will
            be of the shape N x 400, where N is the number of protein sequences and 400 is 
            the number of features calculated from the descriptor (20^2 for the 20 canonical 
            amino acids).
        """
        print('\nGetting Dipeptide Composition Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.dipeptide_composition.empty):
            return self.dipeptide_composition

        #initialise dataframe
        dipeptide_comp_df = pd.DataFrame()
        
        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            dipeptide_comp_seq = protpy.dipeptide_composition(seq)
            dipeptide_comp_df = pd.concat([dipeptide_comp_df, dipeptide_comp_seq])

        self.dipeptide_composition = dipeptide_comp_df

        return self.dipeptide_composition

    def get_tripeptide_composition(self):
        """ 
        Calculate Tripeptide Composition (TPComp) of protein sequence using
        custom-built protpy package. Tripeptide composition is the fraction of 
        each tripeptide type within a protein sequence. With tripeptides being 
        of length 3 and there being 20 canonical amino acids this creates 20^3 
        different combinations, thus a 8000-Dimensional vector will be produced 
        such that:

        TPComp(s,t,u) = AA(s,t,u) / N -1

        where TPComp(s,t,u) is the tripeptide composition of the protein sequence
        for amino acid type s, t and u (where s, t and u = 1,2,..,20), AA(s,t,u) is
        the number of tripeptides represented by amino acid type s and t, and N is
        the total number of tripeptides.

        Parameters
        ----------
        None

        Returns
        -------
        :tripeptide_composition : pd.Dataframe
            pandas Dataframe of tripeptide composition for protein sequence. Dataframe will
            be of the shape N x 8000, where N is the number of protein sequences and 8000 is 
            the number of features calculated from the descriptor (20^3 for the 20 canonical 
            amino acids).
        """
        print('\nGetting Tripeptide Composition Descriptors...')
        print('#############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.tripeptide_composition.empty):
            return self.tripeptide_composition

        #initialise dataframe
        tripeptide_comp_df = pd.DataFrame()
        
        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            tripeptide_comp_seq = protpy.tripeptide_composition(seq)
            tripeptide_comp_df = pd.concat([tripeptide_comp_df, tripeptide_comp_seq])

        self.tripeptide_composition = tripeptide_comp_df

        return self.tripeptide_composition

    def get_moreaubroto_autocorrelation(self):
        """
        Calculate MoreauBrotoAuto Autocorrelation (MBAuto) descriptor using
        custom-built protpy package. Autocorrelation descriptors are a class 
        of topological descriptors, also known as molecular connectivity indices, that 
        describe the level of correlation between two objects (protein or peptide sequences) 
        in terms of their specific structural or physicochemical properties, which are
        defined based on the distribution of amino acid properties along the sequence.
        By default, 8 amino acid properties are used for deriving the descriptors. The 
        derivations and detailed explanations of this type of descriptor is outlind in 
        [4]. The MBAuto descriptor is a type of Autocorrelation descriptor that uses
        the property values as the basis for measurement. Each autocorrelation will
        generate the number of features depending on the lag value and number of
        properties input with total features = lag * number of properties. The 
        autocorrelation values can also be normalized if the "normalize" parameter
        is set in the config file. Using the default 8 properties with default lag 
        value of 30, 240 features are generated, the default 8 properties are:

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
        None

        Returns
        -------
        :moreaubroto_autocorrelation : pd.Dataframe
            pandas Dataframe of MBAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences and 
            M is the number of features calculated from the descriptor, calculated 
            as lag * number of properties. By default, the shape will be N x 240 
            (30 features per property - using 8 properties, with lag=30).
        """
        print('\nGetting Moreaubroto Autocorrelation Descriptors...')
        print('##################################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.moreaubroto_autocorrelation.empty):
            return self.moreaubroto_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.moreaubroto_autocorrelation["lag"]
        properties = self.desc_parameters.moreaubroto_autocorrelation["properties"]
        normalize = self.desc_parameters.moreaubroto_autocorrelation["normalize"]

        #initialise dataframe
        norm_moreaubroto_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            norm_moreaubroto_seq = protpy.moreaubroto_autocorrelation(seq, lag=lag, 
                properties=properties, normalize=normalize)
            norm_moreaubroto_df = pd.concat([norm_moreaubroto_df, norm_moreaubroto_seq])
            
        self.moreaubroto_autocorrelation = norm_moreaubroto_df

        return self.moreaubroto_autocorrelation

    def get_moran_autocorrelation(self):
        """
        Calculate Moran autocorrelation (MAuto) of protein sequences using the custom-built
        protpy package. MAuto utilizes property deviations from the average values.
        **refer to MBAuto docstring for autocorrelation description.

        Parameters
        ----------
        None

        Returns
        -------
        :moran_autocorrelation : pd.DataFrame
            pandas Dataframe of MAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences
            and M is the number of features calculated from the descriptor, 
            calculated as lag * number of properties. By default, the shape 
            will be N x 240 (30 features per property - using 8 properties, 
            with lag=30).
        """
        print('\nGetting Moran Autocorrelation Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.moran_autocorrelation.empty):
            return self.moran_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.moran_autocorrelation["lag"]
        properties = self.desc_parameters.moran_autocorrelation["properties"]
        normalize = self.desc_parameters.moran_autocorrelation["normalize"]

        #initialise dataframe
        moran_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            moran_seq = protpy.moran_autocorrelation(seq, lag=lag, 
                properties=properties, normalize=normalize)
            moran_df = pd.concat([moran_df, moran_seq])

        self.moran_autocorrelation = moran_df

        return self.moran_autocorrelation 

    def get_geary_autocorrelation(self):
        """
        Calculate Geary Autocorrelation (GAuto) of protein sequences using the
        custom-built protpy package. GAuto utilizes the square-difference of 
        property values instead of vector-products (of property values or 
        deviations).  
        **refer to MBAuto docstring for autocorrelation description.

        Parameters
        ----------
        None

        Returns
        -------
        :geary_autocorrelation : pd.DataFrame
            pandas Dataframe of GAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences and 
            M is the number of features calculated from the descriptor, calculated 
            as lag * number of properties. By default, the shape will be N x 240 
            (30 features per property - using 8 properties, with lag=30).
        """
        print('\nGetting Geary Autocorrelation Descriptors...')
        print('############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.geary_autocorrelation.empty):
            return self.geary_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.geary_autocorrelation["lag"]
        properties = self.desc_parameters.geary_autocorrelation["properties"]
        normalize = self.desc_parameters.geary_autocorrelation["normalize"]

        #initialise dataframe
        geary_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            geary_seq = protpy.geary_autocorrelation(seq, lag=lag, 
                properties=properties, normalize=normalize)
            geary_df = pd.concat([geary_df, geary_seq])

        self.geary_autocorrelation = geary_df

        return self.geary_autocorrelation 

    def get_ctd_composition(self):
        """ 
        Calculate Composition (C_CTD) physiochemical/structural descriptor
        of protein sequences from the calculated CTD descriptor.

        Parameters
        ----------
        None

        Returns
        -------
        :ctd_composition : pd.DataFrame
            pandas dataframe of C_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physiochemical properties * 3), with 3 
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 3. 
        """
        print('\nGetting Composition (CTD) Descriptors...')
        print('########################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.ctd_composition.empty):
            return self.ctd_composition
        
        #calculate ctd descriptor if not already calculated
        if (self.ctd.empty):
            self.ctd = self.get_ctd()

        #initialise dataframe
        comp_df = pd.DataFrame()

        #get ctd properties used for calculating descriptor
        ctd_property = self.desc_parameters.ctd["property"]
        if not (isinstance(ctd_property, list)):
            ctd_property = ctd_property.split(',')
        all_ctd = self.desc_parameters.ctd["all"]

        #get composition descriptor from CTD dataframe, dependant on number of props,
        #3 features per property
        if (all_ctd):
            comp_df = self.ctd.iloc[:,0:21]
        else:
            comp_df = self.ctd.iloc[:,0:3 * len(ctd_property)]
            
        self.ctd_composition = comp_df

        return self.ctd_composition
  
    def get_ctd_transition(self):
        """ 
        Calculate Transition (T_CTD) physiochemical/structural descriptor of 
        protein sequences from the calculated CTD descriptor.
        
        Parameters
        ----------
        None

        Returns
        -------
        :ctd_transition : pd.Dataframe
            pandas Dataframe of T_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physiochemical properties * 3), with 3 
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 3. 
        """
        print('\nGetting Transition (CTD) Descriptors...')
        print('#######################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.ctd_transition.empty):
            return self.ctd_transition

        #calculate ctd descriptor if not already calculated
        if (self.ctd.empty):
            self.ctd = self.get_ctd()

        #initialise dataframe
        transition_df = pd.DataFrame()

        #get ctd properties used for calculating descriptor
        ctd_property = self.desc_parameters.ctd["property"]
        if not (isinstance(ctd_property, list)):
            ctd_property = ctd_property.split(',')
        all_ctd = self.desc_parameters.ctd["all"]

        #get transition descriptor from CTD dataframe, dependant on number of props,
        #3 features per property
        if (all_ctd):
            transition_df = self.ctd.iloc[:,21:42]
        else:
            transition_df = self.ctd.iloc[:,3 * len(ctd_property):(3 * len(ctd_property) * 2)]
        
        self.ctd_transition = transition_df

        return self.ctd_transition

    def get_ctd_distribution(self):
        """ 
        Calculate Distribution (D_CTD) physiochemical/structural descriptor of 
        protein sequences from the calculated CTD descriptor.

        Parameters
        ----------
        None

        Returns
        -------
        :ctd_distribution : pd.Dataframe
            pandas Dataframe of D_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physiochemical properties * 15), with 15
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 15. 
        """
        print('\nGetting Distribution (CTD) Descriptors...')
        print('#########################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.ctd_distribution.empty):
            return self.ctd_distribution

        #calculate ctd descriptor if not already calculated
        if (self.ctd.empty):
            self.ctd = self.get_ctd()

        #initialise dataframe
        distribution_df = pd.DataFrame()

        #get ctd properties used for calculating descriptor
        ctd_property = self.desc_parameters.ctd["property"]
        if not (isinstance(ctd_property, list)):
            ctd_property = ctd_property.split(',')
        all_ctd = self.desc_parameters.ctd["all"]

        #get distribution descriptor from CTD dataframe, dependant on number of props,
        #15 features per property
        if (all_ctd):
            distribution_df = self.ctd.iloc[:,42:]
        else:
            distribution_df = self.ctd.iloc[:,2 * (3 * len(ctd_property)):]
        
        self.ctd_distribution = distribution_df

        return self.ctd_distribution

    def get_ctd(self):
        """
        Calculate all CTD (Composition, Transition, Distribution) 
        physiochemical/structural descriptor of protein sequences using the 
        custom-built protpy package. 

        Parameters
        ----------
        None

        Returns
        -------
        :ctd : pd.Series
            pandas Series of CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein 
            sequences and M is (number of physiochemical properties * 21),
            with 21 being the number of features calculated for each of the
            CTD descriptors per property. Using all properties will generate
            an output of N x 147, by default the "hydrophobicity"
            property is used, generating an output of N x 21. 
        """
        print('\nGetting CTD Descriptors...')
        print('##########################\n')

        #if attribute already calculated & not empty then return it
        if not (self.ctd.empty):
            return self.ctd

        #get descriptor-specific parameters from config file
        ctd_property = self.desc_parameters.ctd["property"]
        all_ctd = self.desc_parameters.ctd["all"]

        #initialise dataframe
        ctd_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            ctd_seq = protpy.ctd_(seq, property=ctd_property, all_ctd=all_ctd)
            ctd_df = pd.concat([ctd_df, ctd_seq])

        self.ctd = ctd_df

        return self.ctd

    def get_conjoint_triad(self):
        """
        Calculate Conjoint Triad (CTriad) of protein sequences using the custom-built
        protpy package. The descriptor mainly considers neighbor relationships in 
        protein sequences by encoding each protein sequence using the triad (continuous 
        three amino acids) frequency distribution extracted from a 7-letter reduced 
        alphabet [11]. CTriad calculates 343 different features (7x7x7), with the 
        output being of shape N x 343 where N is the number of sequences.

        Parameters
        ----------
        None

        Returns
        -------
        :conjoint_triad : pd.Dataframe
            pandas Dataframe of CTriad descriptor values for all protein sequences. Dataframe
            will be of the shape N x 343, where N is the number of protein sequences and 343 
            is the number of features calculated from the descriptor for a sequence.
        """
        print('\nGetting Conjoint Triad Descriptors...')
        print('#####################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.conjoint_triad.empty):
            return self.conjoint_triad

        #initialise dataframe
        conjoint_triad_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            conjoint_triad_seq = protpy.conjoint_triad(seq)
            conjoint_triad_df = pd.concat([conjoint_triad_df, conjoint_triad_seq])

        self.conjoint_triad = conjoint_triad_df

        return self.conjoint_triad

    def get_sequence_order_coupling_number(self):
        """
        Calculate Sequence Order Coupling Number (SOCN) features for input protein sequence
        using custom-built protpy package. SOCN computes the dissimilarity between amino acid
        pairs. The distance between amino acid pairs is determined by d which varies
        between 1 to lag. For each d, it computes the sum of the dissimilarities
        of all amino acid pairs. The number of output features can be calculated as N * 2,
        where N = lag, by default this value is 30 which generates an output of M x 60 
        where M is the number of protein sequenes. 

        Parameters
        ----------
        None

        Returns
        -------
        :sequence_order_coupling_number_df : pd.Dataframe
            Dataframe of SOCN descriptor values for all protein sequences. Output
            will be of the shape N x M, where N is the number of protein sequences and
            M is the number of features calculated from the descriptor (calculated as
            N * 2 where N = lag).
        """
        print('\nGetting Sequence Order Coupling Descriptors...')
        print('##############################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.sequence_order_coupling_number.empty):
            return self.sequence_order_coupling_number

        #initialise dataframe
        sequence_order_coupling_number_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.sequence_order_coupling_number["lag"]
        distance_matrix = self.desc_parameters.sequence_order_coupling_number["distance_matrix"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            #if no distance matrix present in config then calculate SOCN using both matrices
            if (distance_matrix == "" or distance_matrix == None):
                sequence_order_coupling_number_seq = protpy.sequence_order_coupling_number_all(seq, lag=lag)
            else:
                sequence_order_coupling_number_seq = protpy.sequence_order_coupling_number(seq, lag=lag, distance_matrix=distance_matrix)

            #concat sequence's descriptor output to dataframe
            sequence_order_coupling_number_df = pd.concat([sequence_order_coupling_number_df, sequence_order_coupling_number_seq])

        self.sequence_order_coupling_number = sequence_order_coupling_number_df

        return self.sequence_order_coupling_number

    def get_quasi_sequence_order(self):
        """
        Calculate Quasi Sequence Order features for the protein sequences using the
        custom-built protpy package.The quasi-sequence-order descriptors were proposed 
        by K.C. Chou, et.al. [10]. They are derived from the distance matrix between 
        the 20 amino acids. By default, the Scheider-Wrede physicochemical distance 
        matrix was used. Also utilised in the descriptor calculation is the Grantham 
        chemical distance matrix. Both of these matrices are used by Grantham et. al. 
        in the calculation of the descriptor [13]. 100 values are calculated per 
        sequence, thus generating an output of N x 100 per sequence, where N is the 
        number of protein sequences.

        Parameters
        ----------
        None

        Returns
        -------
        :quasi_sequence_order_df : pd.Dataframe
            Dataframe of quasi-sequence-order descriptor values for the
            protein sequences, with output shape N x 100 where N is the number
            of sequences and 100 the number of calculated features.
        """
        print('\nGetting Quasi Sequence Order Descriptors...')
        print('###########################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.quasi_sequence_order.empty):
            return self.quasi_sequence_order

        #initialise dataframe
        quasi_sequence_order_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.quasi_sequence_order["lag"]
        weight = self.desc_parameters.quasi_sequence_order["weight"]
        distance_matrix = self.desc_parameters.quasi_sequence_order["distance_matrix"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            #if no distance matrix present in config then calculate quasi seq order using both matrices
            if (distance_matrix == "" or distance_matrix == None):
                quasi_sequence_order_seq = protpy.quasi_sequence_order_all(seq, lag=lag, weight=weight)
            else:
                quasi_sequence_order_seq = protpy.quasi_sequence_order(seq, lag=lag, weight=weight, 
                    distance_matrix=distance_matrix)

            #concat sequence's descriptor output to dataframe
            quasi_sequence_order_df = pd.concat([quasi_sequence_order_df, quasi_sequence_order_seq])

        self.quasi_sequence_order = quasi_sequence_order_df

        return self.quasi_sequence_order

    def get_pseudo_amino_acid_composition(self):
        """
        Calculate Pseudo Amino Acid Composition (PAAComp) descriptor using custom-built protpy 
        package. PAAComp combines the vanilla amino acid composition descriptor with additional 
        local features, such as correlation between residues of a certain distance, as amino 
        acid composition doesn't take into accont sequence order info. The pseudo components 
        of the descriptor are a series rank-different correlation factors [10]. The first 20 
        components are a weighted sum of the amino acid composition and 30 are physiochemical 
        square correlations as dictated by the lambda and properties parameters. This generates 
        an output of [(20 + λ), 1] = 50 x 1 when using the default lambda of 30. By default, 
        the physiochemical properties used are hydrophobicity and hydrophillicity, with a lambda 
        of 30 and weight of 0.05.

        Parameters
        ----------
        None

        Returns
        -------
        :pseudo_amino_acid_composition_df : pd.Dataframe
            Dataframe of pseudo amino acid composition descriptor values for the protein sequences 
            of output shape N x (20 + λ), where N is the number of protein sequences. With 
            default lambda of 30, the output shape will be N x 50.
        """
        print('\nGetting Pseudo Amino Acid Composition Descriptors...')
        print('####################################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.pseudo_amino_acid_composition.empty):
            return self.pseudo_amino_acid_composition

        #initialise dataframe
        pseudo_amino_acid_composition_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lamda = self.desc_parameters.pseudo_amino_acid_composition["lambda"]
        weight = self.desc_parameters.pseudo_amino_acid_composition["weight"]
        properties = self.desc_parameters.pseudo_amino_acid_composition["properties"]

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            pseudo_amino_acid_composition_seq = protpy.pseudo_amino_acid_composition(seq, lamda=lamda, 
                weight=weight, properties=properties)
            pseudo_amino_acid_composition_df = pd.concat([pseudo_amino_acid_composition_df, pseudo_amino_acid_composition_seq])

        self.pseudo_amino_acid_composition = pseudo_amino_acid_composition_df

        return self.pseudo_amino_acid_composition
        
    def get_amphiphilic_pseudo_amino_acid_composition(self):
        """
        Calculate Amphiphilic Pseudo Amino Acid Composition (APAAComp) of protein sequences 
        using custom-built protpy package. APAAComp has the same form as the amino acid 
        composition, but contains much more information that is related to the sequence 
        order of a protein and the distribution of the hydrophobic and hydrophilic amino 
        acids along its chain. The first 20 numbers in the descriptor are the components 
        of the conventional amino acid composition; the next 2*lambda numbers are a set of 
        correlation factors that reflect different hydrophobicity and hydrophilicity 
        distribution patterns along a protein chain.

        Parameters
        ----------
        None

        Returns
        -------
        :amphiphilic_pseudo_amino_acid_composition_df : pd.Dataframe
            Dataframe of Amphiphilic pseudo amino acid composition descriptor values for 
            the protein sequences of output shape N x 80, where N is the number of 
            protein sequences and 80 is calculated as (20 + 2*lambda).
        """
        print('\nGetting Amphiphilic Pseudo Amino Acid Composition Descriptors...')
        print('################################################################\n')

        #if attribute already calculated & not empty then return it
        if not (self.amphiphilic_pseudo_amino_acid_composition.empty):
            return self.amphiphilic_pseudo_amino_acid_composition

        #get descriptor-specific parameters from config file
        lamda = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["lambda"]
        weight = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["weight"]

        #initialise dataframe
        amphiphilic_pseudo_amino_acid_composition_df = pd.DataFrame()

        #calculate descriptor value, concatenate descriptor values
        for seq in self.protein_seqs:
            amphiphilic_pseudo_amino_acid_composition_seq = protpy.amphiphilic_pseudo_amino_acid_composition(seq, 
                lamda=lamda, weight=weight)
            amphiphilic_pseudo_amino_acid_composition_df = pd.concat([amphiphilic_pseudo_amino_acid_composition_df, 
                amphiphilic_pseudo_amino_acid_composition_seq])

        self.amphiphilic_pseudo_amino_acid_composition = amphiphilic_pseudo_amino_acid_composition_df

        return self.amphiphilic_pseudo_amino_acid_composition

    def get_all_descriptors(self, export=False):
        """
        Calculate all individual descriptor values, concatenating each descriptor
        Dataframe into one storing all descriptors. The number of descriptor
        features calculated is dependant on several additional parameters of some 
        descriptors, including the number of properties and max lag for the 
        Autocorrelation, SOCN and QSO and the number of properties and lamda for 
        PAAComp and the lambda for APAAComp. To export all descriptors to a csv 
        set export=True when calling the function, this saves having to recalculate
        all the descriptor values when using them in multiple encoding processes, 
        and the descriptors can be imported using the import_descriptors function.

        Parameters
        ----------
        :export : bool (default=False)
            if true then all calculated descriptors from the protpy package will be 
            exported to a CSV. This allows for pre-calculated descriptors for a 
            dataset to be easily imported and not have to be recalculated again.

        Returns
        -------
        :all_descriptor_df : pd.DataFrame
            concatenated dataframe of all individual descriptors. Using the default
            attributes and their associated values, the output will be of the shape
            N x 9714, where N is the number of protein sequences and 9714 is the 
            number of descriptor features. 
        """
        #if descriptor attribute DF is empty then call its respective get_descriptor function
        if (getattr(self, "amino_acid_composition").empty):
            self.amino_acid_composition = self.get_amino_acid_composition()

        if (getattr(self, "dipeptide_composition").empty):
                self.dipeptide_composition = self.get_dipeptide_composition()

        if (getattr(self, "tripeptide_composition").empty):
                self.tripeptide_composition = self.get_tripeptide_composition()

        if (getattr(self, "moreaubroto_autocorrelation").empty):
            self.moreaubroto_autocorrelation = self.get_moreaubroto_autocorrelation()

        if (getattr(self, "moran_autocorrelation").empty):
            self.moran_autocorrelation = self.get_moran_autocorrelation()

        if (getattr(self, "geary_autocorrelation").empty):
            self.geary_autocorrelation = self.get_geary_autocorrelation()

        if (getattr(self, "ctd").empty):
                self.ctd = self.get_ctd()

        if (getattr(self, "ctd_composition").empty):
                self.ctd_composition = self.get_ctd_composition()

        if (getattr(self, "ctd_transition").empty):
            self.ctd_transition = self.get_ctd_transition()
        
        if (getattr(self, "ctd_distribution").empty):
            self.ctd_distribution = self.get_ctd_distribution()

        if (getattr(self, "conjoint_triad").empty):
                self.conjoint_triad = self.get_conjoint_triad()

        if (getattr(self, "sequence_order_coupling_number").empty):
                self.sequence_order_coupling_number = self.get_sequence_order_coupling_number()

        if (getattr(self, "quasi_sequence_order").empty):
                self.quasi_sequence_order = self.get_quasi_sequence_order()

        if (getattr(self, "pseudo_amino_acid_composition").empty):
                self.pseudo_amino_acid_composition = self.get_pseudo_amino_acid_composition()

        if (getattr(self, "amphiphilic_pseudo_amino_acid_composition").empty):
                self.amphiphilic_pseudo_amino_acid_composition = self.get_amphiphilic_pseudo_amino_acid_composition()

        #append all calculated descriptors to list
        all_desc = [
            self.amino_acid_composition, self.dipeptide_composition, self.tripeptide_composition,
            self.moreaubroto_autocorrelation, self.moran_autocorrelation,
            self.geary_autocorrelation, self.ctd_composition, self.ctd_transition,
            self.ctd_distribution, self.conjoint_triad, self.sequence_order_coupling_number,
            self.quasi_sequence_order, self.pseudo_amino_acid_composition, self.amphiphilic_pseudo_amino_acid_composition
            ]

        #concatenate individual descriptor dataframe attributes
        all_descriptor_df = pd.concat(all_desc, axis = 1)
        self.all_descriptors = all_descriptor_df     

        #export pre-calculated descriptor values to a csv, use default name if parameter empty
        if (export):
            if (self.desc_config.descriptors_csv == "" or self.desc_config.descriptors_csv == None):
                self.desc_config.descriptors_csv = "descriptors_output.csv"            
            
            self.all_descriptors.to_csv(os.path.join(DATA_DIR, self.desc_config.descriptors_csv), index=0)

        return all_descriptor_df

    def get_descriptor_encoding(self, descriptor):
        """
        Get the protein descriptor values of a specified input descriptor. If the
        sought descriptor has already been calculated then its attribute is returned,
        else the descriptor is calculated using its get_descriptor function.

        Parameters
        ----------
        :descriptor : str
            name of descriptor to return. Method can accept the approximate name
            of the descriptor, e.g. 'amino_comp'/'aa_composition' etc will return 
            the 'amino_acid_composition' descriptor. This functionality is realised 
            using the difflib library and its built-in get_close_matches function.

        Returns
        -------
        :desc_encoding : pd.DataFrame/None
            dataframe of matching descriptor attribute. None returned if no matching 
            descriptor found.
        """
        #input descriptor parameter should be a string
        if not(isinstance(descriptor, str)):
            raise TypeError('Input parameter {} is not of correct datatype string, got {}.'.
                format(descriptor, type(descriptor))) 

        #remove any whitespace from input parameter, replace spaces with underscores and lowercase
        descriptor = descriptor.strip().replace(' ', '_').lower()

        #validate input descriptor is a valid available descriptor, get its closest match
        desc_matches = get_close_matches(descriptor, self.valid_descriptors, cutoff=0.6)
        if (desc_matches != []):
            desc = desc_matches[0]  #set desc to closest descriptor match found
        else:
            raise ValueError("Could not find a match for the input descriptor ({}) in"
                " available valid models:\n {}.".format(descriptor, self.valid_descriptors))

        #if sought descriptor attribute dataframe is empty, call the descriptor's
        #  get_descriptor() function, set desc_encoding to descriptor attribute
        if (desc == 'amino_acid_composition'):
            if (getattr(self, desc).empty):
                self.get_amino_acid_composition()
            desc_encoding = self.amino_acid_composition

        elif (desc == 'dipeptide_composition'):
            if (getattr(self, desc).empty):
                self.get_dipeptide_composition()
            desc_encoding = self.dipeptide_composition

        elif (desc == 'tripeptide_composition'):
            if (getattr(self, desc).empty):
                self.get_tripeptide_composition()
            desc_encoding = self.tripeptide_composition

        elif (desc == 'moreaubroto_autocorrelation'):
            if (getattr(self, desc).empty):
              self.get_moreaubroto_autocorrelation()
            desc_encoding = self.moreaubroto_autocorrelation
            
        elif (desc == 'moran_autocorrelation'):
            if (getattr(self, desc).empty):
              self.get_moran_autocorrelation()
            desc_encoding = self.moran_autocorrelation

        elif (desc == 'geary_autocorrelation'):
            if (getattr(self, desc).empty):
              self.get_geary_autocorrelation()
            desc_encoding = self.geary_autocorrelation

        elif (desc == 'ctd'):
            if (getattr(self, desc).empty):
              self.get_ctd()
            desc_encoding = self.ctd

        elif (desc == 'ctd_composition'):
            if (getattr(self, desc).empty):
              self.get_ctd_composition()
            desc_encoding = self.ctd_composition

        elif (desc == 'ctd_transition'):
            if (getattr(self, desc).empty):
              self.get_ctd_transition()
            desc_encoding = self.ctd_transition

        elif (desc == 'ctd_distribution'):
            if (getattr(self, desc).empty):
              self.get_ctd_distribution()
            desc_encoding = self.ctd_distribution

        elif (desc == 'conjoint_triad'):
            if (getattr(self, desc).empty):
              self.get_conjoint_triad()
            desc_encoding = self.conjoint_triad

        elif (desc == 'sequence_order_coupling_number'):
            if (getattr(self, desc).empty):
              self.get_sequence_order_coupling_number()
            desc_encoding = self.sequence_order_coupling_number

        elif (desc == 'quasi_sequence_order'):
            if (getattr(self, desc).empty):
              self.get_quasi_sequence_order()
            desc_encoding = self.quasi_sequence_order

        elif (desc == 'pseudo_amino_acid_composition'):
            if (getattr(self, desc).empty):
              self.get_pseudo_amino_acid_composition()
            desc_encoding = self.pseudo_amino_acid_composition

        elif (desc == 'amphiphilic_pseudo_amino_acid_composition'):
            if (getattr(self, desc).empty):
              self.get_amphiphilic_pseudo_amino_acid_composition()
            desc_encoding = self.amphiphilic_pseudo_amino_acid_composition
        else:
          desc_encoding = None           #no matching descriptor found

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

       #get all combinations of 2 or 3 descriptors
       if (desc_combo == 2):
           all_descriptors = list(itertools.combinations(all_descriptors, 2))
       elif (desc_combo == 3):
           all_descriptors = list(itertools.combinations(all_descriptors, 3))
       else:
           pass     #if desc_combo not equal to 2 or 3 then use default all_descriptors

       return all_descriptors

######################          Getters & Setters          ######################

    @property
    def all_desc(self):
        return self._all_desc

    @all_desc.setter
    def all_desc(self, val):
        self._all_desc = val

    @property
    def amino_acid_composition(self):
        return self._amino_acid_composition

    @amino_acid_composition.setter
    def amino_acid_composition(self, val):
        self._amino_acid_composition = val

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
    def moreaubroto_autocorrelation(self):
        return self._moreaubroto_autocorrelation

    @moreaubroto_autocorrelation.setter
    def moreaubroto_autocorrelation(self, val):
        self._moreaubroto_autocorrelation = val

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
    def ctd_composition(self):
        return self._ctd_composition

    @ctd_composition.setter
    def ctd_composition(self, val):
        self._ctd_composition = val

    @property
    def ctd_transition(self):
        return self._ctd_transition

    @ctd_transition.setter
    def ctd_transition(self, val):
        self._ctd_transition = val

    @property
    def ctd_distribution(self):
        return self._ctd_distribution

    @ctd_distribution.setter
    def ctd_distribution(self, val):
        self._ctd_distribution = val

    @property
    def conjoint_triad(self):
        return self._conjoint_triad

    @conjoint_triad.setter
    def conjoint_triad(self, val):
        self._conjoint_triad = val

    @property
    def sequence_order_coupling_number(self):
        return self._sequence_order_coupling_number

    @sequence_order_coupling_number.setter
    def sequence_order_coupling_number(self, val):
        self._sequence_order_coupling_number = val

    @property
    def quasi_sequence_order(self):
        return self._quasi_sequence_order

    @quasi_sequence_order.setter
    def quasi_sequence_order(self, val):
        self._quasi_sequence_order = val

    @property
    def pseudo_amino_acid_composition(self):
        return self._pseudo_amino_acid_composition

    @pseudo_amino_acid_composition.setter
    def pseudo_amino_acid_composition(self, val):
        self._pseudo_amino_acid_composition = val

    @property
    def amphiphilic_pseudo_amino_acid_composition(self):
        return self._amphiphilic_pseudo_amino_acid_composition

    @amphiphilic_pseudo_amino_acid_composition.setter
    def amphiphilic_pseudo_amino_acid_composition(self, val):
        self._amphiphilic_pseudo_amino_acid_composition = val

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
        del self._amino_acid_composition
        del self._dipeptide_composition
        del self._tripeptide_composition
        del self._moreaubroto_autocorrelation
        del self._moran_autocorrelation
        del self._geary_autocorrelation
        del self._ctd
        del self._ctd_transition
        del self._ctd_composition
        del self._ctd_distribution
        del self._conjoint_triad
        del self._sequence_order_coupling_number
        del self._quasi_sequence_order
        del self._pseudo_amino_acid_composition
        del self._amphiphilic_pseudo_amino_acid_composition

    def __str__(self):
        return '''{}\nAmino Acid Composition: {}\nDipeptide Composition: {}\
            \nTripeptide Composition: {}\nMoreauBroto Autocorrelation: {}\
            \nMoran Autocorrelation: {}\nGeary Autocorrelation: {}\
            \nCTD: {}\nConjoint Triad: {}\nSequence Order Coupling Number: {}\
            \nQuasi Sequence Order: {}\nPseudo Amino Acid Composition: {}\
            \nAmphipilic Pseudo Amino Acid Composition: {}'''.format(
            self.shape, self.amino_acid_composition.shape, self.dipeptide_composition.shape,
            self.tripeptide_composition.shape, self.moreaubroto_autocorrelation.shape,
            self.moran_autocorrelation.shape, self.geary_autocorrelation.shape, self.ctd.shape,
            self.conjoint_triad.shape, self.sequence_order_coupling_number.shape, 
            self.quasi_sequence_order.shape, self.pseudo_amino_acid_composition.shape, 
            self.amphiphilic_pseudo_amino_acid_composition.shape)

    def __repr__(self):
        return ('<Descriptor: {}>'.format(self))

    def __len__(self):
        return len(self.all_descriptors)

    def __shape__(self):
        return self.all_descriptors.shape