################################################################################
#################                  Descriptors                 #################
################################################################################

from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from difflib import get_close_matches
import json
import os
import itertools
import time
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import valid_sequence, remove_gaps, Map
import protpy as protpy

# Descriptor feature dimension constants
AA_COUNT = 20
DIPEPTIDE_FEATURES = 20 ** 2  # 400
TRIPEPTIDE_FEATURES = 20 ** 3  # 8000
CONJOINT_TRIAD_FEATURES = 343

class DescriptorType(Enum):
    """Enumeration of available protein descriptor types."""
    AMINO_ACID_COMPOSITION = 'amino_acid_composition'
    DIPEPTIDE_COMPOSITION = 'dipeptide_composition'
    TRIPEPTIDE_COMPOSITION = 'tripeptide_composition'
    GRAVY = 'gravy'
    AROMATICITY = 'aromaticity'
    INSTABILITY_INDEX = 'instability_index'
    ISOELECTRIC_POINT = 'isoelectric_point'
    MOLECULAR_WEIGHT = 'molecular_weight'
    CHARGE_DISTRIBUTION = 'charge_distribution'
    HYDROPHOBIC_POLAR_CHARGED_COMPOSITION = 'hydrophobic_polar_charged_composition'
    SECONDARY_STRUCTURE_PROPENSITY = 'secondary_structure_propensity'
    KMER_COMPOSITION = 'kmer_composition'
    REDUCED_ALPHABET_COMPOSITION = 'reduced_alphabet_composition'
    MOTIF_COMPOSITION = 'motif_composition'
    AMINO_ACID_PAIR_COMPOSITION = 'amino_acid_pair_composition'
    ALIPHATIC_INDEX = 'aliphatic_index'
    EXTINCTION_COEFFICIENT = 'extinction_coefficient'
    BOMAN_INDEX = 'boman_index'
    AGGREGATION_PROPENSITY = 'aggregation_propensity'
    HYDROPHOBIC_MOMENT = 'hydrophobic_moment'
    SHANNON_ENTROPY = 'shannon_entropy'
    MOREAUBROTO_AUTOCORRELATION = 'moreaubroto_autocorrelation'
    MORAN_AUTOCORRELATION = 'moran_autocorrelation'
    GEARY_AUTOCORRELATION = 'geary_autocorrelation'
    CTD = 'ctd'
    CTD_COMPOSITION = 'ctd_composition'
    CTD_TRANSITION = 'ctd_transition'
    CTD_DISTRIBUTION = 'ctd_distribution'
    CONJOINT_TRIAD = 'conjoint_triad'
    SEQUENCE_ORDER_COUPLING_NUMBER = 'sequence_order_coupling_number'
    QUASI_SEQUENCE_ORDER = 'quasi_sequence_order'
    PSEUDO_AMINO_ACID_COMPOSITION = 'pseudo_amino_acid_composition'
    AMPHIPHILIC_PSEUDO_AMINO_ACID_COMPOSITION = 'amphiphilic_pseudo_amino_acid_composition'

class Descriptors():
    """
    Class for calculating a wide variety of protein physicochemical, biochemical and structural 
    descriptors. These descriptors have been used in a wide variety of Bioinformatics 
    applications including: protein structural and functional class prediction, 
    protein-protein interactions, subcellular location, secondary structure prediction, among
    many more. They represent the different structural, functional & interaction profiles of 
    proteins by exploring the features in the groups of composition, correlation and distribution 
    of the constituent residues and their biochemical and physicochemical properties.

    A custom-built software package was created to generate these descriptors - protpy, which
    is also open-source and available here: https://github.com/amckenna41/protpy. The package
    takes 1 or more protein sequences, returning the respective descriptor values in a Pandas
    DataFrame. protpy and this class allows calculation of the following descriptors: Amino 
    Acid Composition (AAComp), Dipeptide Composition (DPComp), Tripeptide Composition (TPComp), 
    MoreauBroto Autocorrelation (MBAuto), Moran Autocorrelation (MAuto), Geary Autocorrelation 
    (GAuto), Composition (CTD_C), Transition (CTD_T), Distribution (CTD_D), CTD, Conjoint Triad 
    (CTriad), Sequence Order Coupling Number (SOCN), Quasi Sequence Order (QSO), Pseudo Amino Acid 
    Composition - type 1 (PAAcomp), Amphiphilic Pseudo Amino Acid Composition - type 2 (APAAComp),
    GRAVY, Aromaticity, Instability Index, Isoelectric Point, Molecular Weight, Charge Distribution,
    Hydrophobic/Polar/Charged Composition (HPC), Secondary Structure Propensity (SSP), k-mer
    Composition, Reduced Alphabet Composition, Motif Composition, Amino Acid Pair Composition,
    Aliphatic Index, Extinction Coefficient, Boman Index, Aggregation Propensity, Hydrophobic
    Moment, and Shannon Entropy.

    Similar to other classes in pySAR, this class works via configuration files which contain
    the values for all the potential parameters, if applicable, of each descriptor. By default, 
    the class will look for a descriptors csv which is a file of the pre-calculated descriptor 
    values for the specified dataset, if this file doesn't exist, or the parameter value is blank, 
    then each descriptor will have to be calculated using its respective function.

    During initialization, input sequences are normalized by removing gaps and then validated
    against canonical amino acids before descriptor generation begins.

    This class is also designed to feed descriptor feature matrices directly into downstream
    Encoding and PySAR workflows for model training and evaluation.

    It is recommended that with every new dataset, the Descriptors class should be instantiated 
    with the "all_desc" parameter set to 1 in the config file. This will calculate all the descriptor
    values for the dataset of protein sequences, storing the result in a csv file, meaning that
    this file can be used for future use and the descriptors will not have to be recalculated each 
    time. This csv file will be saved to the path and filename according to the "descriptors_csv"
    parameter in the config file.

    Parameters
    ==========
    :config_file: str
        path to configuration file which will contain the various parameter values for all
        descriptors. If invalid value input then error will be raised.
    :protein_seqs: pd.Series or str
        protein sequences to calculate descriptors for. A single sequence string is converted
        internally to a pandas Series. If None or empty, sequences are loaded from the dataset
        path in the configuration.
    **kwargs: dict
        keyword argument names and values for the dataset filename/path and the descriptors 
        csv path parameters. The keywords should be the same name and form of those in the 
        configuration file. The keyword values input take precedence over those in the config files.

    Attributes
    ==========
    :amino_acid_composition: pd.DataFrame
        Amino acid composition descriptor (20 features)
    :dipeptide_composition: pd.DataFrame
        Dipeptide composition descriptor (400 features)
    :tripeptide_composition: pd.DataFrame
        Tripeptide composition descriptor (8000 features)
    :moreaubroto_autocorrelation: pd.DataFrame
        Moreaubroto autocorrelation descriptor (240 features)
    :moran_autocorrelation: pd.DataFrame
        Moran autocorrelation descriptor (240 features)
    :geary_autocorrelation: pd.DataFrame
        Geary autocorrelation descriptor (240 features)
    :ctd: pd.DataFrame
        Composition-Transition-Distribution descriptor
    :conjoint_triad: pd.DataFrame
        Conjoint triad descriptor (343 features)
    :pseudo_amino_acid_composition: pd.DataFrame
        Pseudo amino acid composition descriptor
    :amphiphilic_pseudo_amino_acid_composition: pd.DataFrame
        Amphiphilic pseudo amino acid composition descriptor
    :gravy: pd.DataFrame
        GRAVY (Grand Average of Hydropathy) descriptor (1 feature)
    :aromaticity: pd.DataFrame
        Aromaticity descriptor (1 feature)
    :instability_index: pd.DataFrame
        Instability Index descriptor (1 feature)
    :isoelectric_point: pd.DataFrame
        Isoelectric Point descriptor (1 feature)
    :molecular_weight: pd.DataFrame
        Molecular Weight descriptor (1 feature)
    :charge_distribution: pd.DataFrame
        Charge Distribution descriptor (3 features)
    :hydrophobic_polar_charged_composition: pd.DataFrame
        Hydrophobic/Polar/Charged Composition descriptor (3 features)
    :secondary_structure_propensity: pd.DataFrame
        Secondary Structure Propensity descriptor (3 features)
    :kmer_composition: pd.DataFrame
        k-mer Composition descriptor (20^k features, default 400)
    :reduced_alphabet_composition: pd.DataFrame
        Reduced Alphabet Composition descriptor (alphabet_size features, default 6)
    :motif_composition: pd.DataFrame
        Motif Composition descriptor (8 features by default)
    :amino_acid_pair_composition: pd.DataFrame
        Amino Acid Pair Composition descriptor (400 features)
    :aliphatic_index: pd.DataFrame
        Aliphatic Index descriptor (1 feature)
    :extinction_coefficient: pd.DataFrame
        Extinction Coefficient descriptor (2 features)
    :boman_index: pd.DataFrame
        Boman Index descriptor (1 feature)
    :aggregation_propensity: pd.DataFrame
        Aggregation Propensity descriptor (2 features)
    :hydrophobic_moment: pd.DataFrame
        Hydrophobic Moment descriptor (2 features)
    :shannon_entropy: pd.DataFrame
        Shannon Entropy descriptor (1 feature)
    :all_descriptors: pd.DataFrame
        Concatenated dataframe of all calculated descriptors
    :valid_descriptors: list
        List of all available descriptor names
    :descriptor_groups: dict
        Mapping of descriptor names to their functional groups
    :num_seqs: int
        Total number of input protein sequences
    :protein_seqs: pd.Series
        Loaded protein sequences with gaps removed

    Methods
    =======
    import_descriptors()
        Import pre-calculated descriptors from CSV file
    get_amino_acid_composition()
        Calculate amino acid composition for all sequences
    get_dipeptide_composition()
        Calculate dipeptide composition for all sequences
    get_tripeptide_composition()
        Calculate tripeptide composition for all sequences
    get_moreaubroto_autocorrelation()
        Calculate Moreau-Broto autocorrelation descriptor
    get_moran_autocorrelation()
        Calculate Moran autocorrelation descriptor
    get_geary_autocorrelation()
        Calculate Geary autocorrelation descriptor
    get_ctd()
        Calculate CTD descriptor
    get_ctd_composition()
        Calculate CTD composition descriptor
    get_ctd_transition()
        Calculate CTD transition descriptor
    get_ctd_distribution()
        Calculate CTD distribution descriptor
    get_conjoint_triad()
        Calculate conjoint triad descriptor
    get_sequence_order_coupling_number()
        Calculate sequence order coupling number descriptor
    get_quasi_sequence_order()
        Calculate quasi sequence order descriptor
    get_pseudo_amino_acid_composition()
        Calculate pseudo amino acid composition descriptor
    get_amphiphilic_pseudo_amino_acid_composition()
        Calculate amphiphilic pseudo amino acid composition descriptor
    get_gravy()
        Calculate GRAVY (Grand Average of Hydropathy) descriptor
    get_aromaticity()
        Calculate Aromaticity descriptor
    get_instability_index()
        Calculate Instability Index descriptor
    get_isoelectric_point()
        Calculate Isoelectric Point descriptor
    get_molecular_weight()
        Calculate Molecular Weight descriptor
    get_charge_distribution()
        Calculate Charge Distribution descriptor
    get_hydrophobic_polar_charged_composition()
        Calculate Hydrophobic/Polar/Charged Composition descriptor
    get_secondary_structure_propensity()
        Calculate Secondary Structure Propensity descriptor
    get_kmer_composition()
        Calculate k-mer Composition descriptor
    get_reduced_alphabet_composition()
        Calculate Reduced Alphabet Composition descriptor
    get_motif_composition()
        Calculate Motif Composition descriptor
    get_amino_acid_pair_composition()
        Calculate Amino Acid Pair Composition descriptor
    get_aliphatic_index()
        Calculate Aliphatic Index descriptor
    get_extinction_coefficient()
        Calculate Extinction Coefficient descriptor
    get_boman_index()
        Calculate Boman Index descriptor
    get_aggregation_propensity()
        Calculate Aggregation Propensity descriptor
    get_hydrophobic_moment()
        Calculate Hydrophobic Moment descriptor
    get_shannon_entropy()
        Calculate Shannon Entropy descriptor
    get_all_descriptors()
        Calculate all descriptors and return a concatenated dataframe
    get_descriptor_encoding()
        Resolve a descriptor name and return its encoding dataframe
    all_descriptors_list()
        Return descriptor names or combinations of descriptor names
    validate_descriptors()
        Validate descriptor names exist in valid descriptors list
    validate_sequences()
        Validate sequences contain only canonical amino acids
    get_descriptor_info()
        Get metadata about a specific descriptor
    reset_descriptors()
        Clear all descriptor DataFrames to empty state
    clear_cache()
        Free memory from cached descriptor metadata
    get_descriptor_columns()
        Get column names for a calculated descriptor
    __str__()
        Return a human-readable string summary of descriptor shapes
    __repr__()
        Return the object representation string
    __len__()
        Return number of rows in all_descriptors
    __shape__()
        Return shape of all_descriptors
    __sizeof__()
        Return memory footprint of all_descriptors

    Raises
    ======
    :TypeError
        If config_file is not a string or protein sequences are invalid type
    :OSError
        If config file or dataset file not found at specified path
    :InvalidSequenceError
        If protein sequences contain non-canonical amino acids
    :InvalidDescriptorError
        If requesting a non-existent descriptor
    :DescriptorConfigError
        If configuration JSON file is invalid or malformed

    Examples
    ========
    >>> from pySAR.descriptors import Descriptors
    >>> desc = Descriptors(config_file='config/thermostability.json')
    >>> 
    >>> # Calculate single descriptor
    >>> aa_comp = desc.get_amino_acid_composition()
    >>> 
    >>> # Calculate multiple descriptors
    >>> desc.get_dipeptide_composition()
    >>> desc.get_moran_autocorrelation()
    >>> 
    >>> # Get all descriptors at once
    >>> all_desc = desc.get_all_descriptors()
    >>> alldescs.shape
    (261, 10572)
    >>> 
    >>> # Get descriptor information
    >>> info = desc.get_descriptor_info('amino_acid_composition')
    >>> info['feature_count']
    20
    >>> 
    >>> # Get columns for a descriptor
    >>> columns = desc.get_descriptor_columns('dipeptide_composition')
    >>> len(columns)
    400

    Notes
    =====
    - Tripeptide and pseudo-amino acid composition descriptors are computationally expensive
      and may take significant time to calculate on large datasets
    - Pre-calculating all descriptors and exporting to CSV (via 'all_desc' config parameter)
      is recommended to avoid recalculation
    - The descriptor_feature_count property is cached for performance
    - Memory usage scales with dataset size and number of descriptors calculated
    - Protein sequences must contain only standard 20 amino acids (A-W, excluding B, O, U, Z)

    References
    ==========
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
    [13] Grantham, R. (1974-09-06). "Amino acid difference formula to help explain protein
         evolution". Science. 185 (4154): 862–864. Bibcode:1974Sci...185..862G.
         doi:10.1126/science.185.4154.862. ISSN 0036-8075. PMID 4843792. S2CID 35388307.   
    [14] B. Hollas, “An analysis of the autocorrelation descriptor for molecules,” J. Math. Chem., 
        vol. 33, no. 2, pp. 91–101, 2003.
    """
    def __init__(self,
                 config_file: str = "",
                 protein_seqs: Optional[Union[pd.Series, str]] = None,
                 n_jobs: int = 1,
                 **kwargs) -> None:

        self.config_file = config_file
        self.protein_seqs = protein_seqs
        self.n_jobs = max(1, int(n_jobs))
        self.kwargs = locals()['kwargs'] #get any keyword argument variables of class
        self.config_parameters = {}

        desc_config_filepath = ""

        #import config file, raise error if invalid path
        if not (isinstance(self.config_file, str) or (self.config_file is None)):
            raise TypeError(f'JSON config file must be a filepath of type string, got type {type(config_file)}.')
        if (os.path.splitext(self.config_file)[1] == ''):
            self.config_file = self.config_file + '.json' #append extension if only filename input        
        if (os.path.isfile(self.config_file)):
            desc_config_filepath = self.config_file
        elif (os.path.isfile(os.path.join('config', self.config_file))):
            desc_config_filepath = os.path.join('config', self.config_file)
        else:
            raise OSError(f'JSON config file not found at path: {self.config_file}.')

        #open json file and read config parameters
        try:
            with open(desc_config_filepath) as f:
                self.config_parameters = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            raise DescriptorConfigError(f'Error parsing config JSON file {desc_config_filepath}: {e}')
        
        #create instance of Map class so parameters in config can be accessed via dot notation
        self.dataset_parameters = Map(self.config_parameters["dataset"])
        self.desc_parameters = Map(self.config_parameters["descriptors"])
        
        #set dataset and descriptors csv filepath from kwargs, if applicable, or the config file values
        self.dataset_filepath = self.kwargs.get('dataset') if 'dataset' in self.kwargs else self.dataset_parameters["dataset"]
        self.descriptors_csv = self.kwargs.get('descriptors_csv') if 'descriptors_csv' in self.kwargs else self.desc_parameters.descriptors_csv

        #import protein sequences from dataset if not directly specified in protein_seqs input param
        if not (isinstance(self.protein_seqs, pd.Series)):
            if (self.protein_seqs is None or self.protein_seqs == ""): 
                #open dataset and read protein seqs if protein_seqs is empty/None
                if not (os.path.isfile(self.dataset_filepath)):
                    raise OSError(f'Dataset file not found at path: {self.dataset_filepath}.')

                #read in dataset csv from filepath mentioned in config 
                try:
                    data = pd.read_csv(self.dataset_filepath, sep=",", header=0)
                    self.protein_seqs = data[self.dataset_parameters["sequence_col"]]
                except (FileNotFoundError, IOError, KeyError, pd.errors.ParserError) as e:
                    raise DescriptorError(f'Error opening dataset file {self.dataset_filepath}: {e}')
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

        #validate that all input protein sequences are valid and only contain valid amino acids, if not then raise ValueError
        invalid_seqs = valid_sequence(self.protein_seqs)
        if (invalid_seqs is not None):
            raise InvalidSequenceError(f'Invalid Amino Acids found in protein sequence dataset: {invalid_seqs}.')

        #get the total number of inputted protein sequences
        self.num_seqs = len(self.protein_seqs)

        #initialise all descriptor attributes to empty dataframes
        self._init_descriptor_attrs()

        #append extension if just the filename input as descriptors csv
        if ((self.descriptors_csv != '' and self.descriptors_csv is not None) 
            and (os.path.splitext(self.descriptors_csv)[1] == '')):
            self.descriptors_csv = self.descriptors_csv + ".csv"

        #try importing descriptors csv with pre-calculated descriptor values
        if (os.path.isfile(self.descriptors_csv)):
            self.import_descriptors(self.descriptors_csv)
            #get the total number of inputted protein sequences
            self.num_seqs = self.all_descriptors.shape[0]

        #create dictionary of descriptors and their associated groups
        keys = self.all_descriptors_list()
        # 21 Composition (3 original + 18 new) + 3 Autocorrelation + 4 CTD + 1 Conjoint Triad + 2 Sequence Order + 2 Pseudo Composition
        values = (["Composition"] * 21 + ["Autocorrelation"] * 3 + ["CTD"] * 4 +
                  ["Conjoint Triad"] + ["Sequence Order"] * 2 + ["Pseudo Composition"] * 2)
        self.descriptor_groups = dict(zip(keys,values))

        #get shape of descriptors
        self.shape = self.all_descriptors.shape

        #list of available protein descriptors
        self.valid_descriptors = [
            'amino_acid_composition', 'dipeptide_composition', 'tripeptide_composition',
            'gravy', 'aromaticity', 'instability_index', 'isoelectric_point', 'molecular_weight',
            'charge_distribution', 'hydrophobic_polar_charged_composition',
            'secondary_structure_propensity', 'kmer_composition', 'reduced_alphabet_composition',
            'motif_composition', 'amino_acid_pair_composition', 'aliphatic_index',
            'extinction_coefficient', 'boman_index', 'aggregation_propensity',
            'hydrophobic_moment', 'shannon_entropy',
            'moreaubroto_autocorrelation', 'moran_autocorrelation', 'geary_autocorrelation',
            'ctd', 'ctd_composition', 'ctd_transition', 'ctd_distribution', 'conjoint_triad',
            'sequence_order_coupling_number', 'quasi_sequence_order',
            'pseudo_amino_acid_composition', 'amphiphilic_pseudo_amino_acid_composition'
        ]

    def _init_descriptor_attrs(self) -> None:
        """ Set all 34 descriptor attributes to empty DataFrames. Called from __init__ and reset_descriptors. """
        self.amino_acid_composition = pd.DataFrame()
        self.dipeptide_composition = pd.DataFrame()
        self.tripeptide_composition = pd.DataFrame()
        self.gravy = pd.DataFrame()
        self.aromaticity = pd.DataFrame()
        self.instability_index = pd.DataFrame()
        self.isoelectric_point = pd.DataFrame()
        self.molecular_weight = pd.DataFrame()
        self.charge_distribution = pd.DataFrame()
        self.hydrophobic_polar_charged_composition = pd.DataFrame()
        self.secondary_structure_propensity = pd.DataFrame()
        self.kmer_composition = pd.DataFrame()
        self.reduced_alphabet_composition = pd.DataFrame()
        self.motif_composition = pd.DataFrame()
        self.amino_acid_pair_composition = pd.DataFrame()
        self.aliphatic_index = pd.DataFrame()
        self.extinction_coefficient = pd.DataFrame()
        self.boman_index = pd.DataFrame()
        self.aggregation_propensity = pd.DataFrame()
        self.hydrophobic_moment = pd.DataFrame()
        self.shannon_entropy = pd.DataFrame()
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

    def import_descriptors(self, descriptor_filepath: str = "") -> None:
        """
        Import descriptors from descriptors csv, setting the class attributes to their values.
        It is recommended that after calculating the descriptors for a dataset of sequences
        that the calculated values are exported to a csv; this means they don't need to be
        recalculated each time. The all_descriptors class attribute is a dataframe of all
        concatenated descriptors from the csv.

        Parameters
        ==========
        :descriptor_filepath: str
            filepath to pre-calculated descriptor csv file.

        Returns
        =======
        None
        """
        #raise type error if filepath parameter isn't string
        if not (isinstance(descriptor_filepath, str)):
            raise TypeError(f"Filepath input parameter should be type str, got {type(descriptor_filepath)}.")

        #verify descriptors csv exists at filepath
        if not (os.path.isfile(descriptor_filepath)):
            raise OSError(f'Descriptors csv file does not exist at filepath: {descriptor_filepath}.')

        #import descriptors csv as dataframe
        try:
            descriptor_df = pd.read_csv(descriptor_filepath, low_memory=False)
        except (FileNotFoundError, IOError, pd.errors.ParserError) as e:
            raise DescriptorError(f'Error reading descriptors csv file {descriptor_filepath}: {e}')

        #replacing any +/- infinity or NAN values with 0
        descriptor_df = descriptor_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Dummy sequence covering all 20 canonical AAs, long enough for lag-dependent descriptors (lag<=30)
        _DUMMY = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

        def _select_cols(df, cols):
            """Return df subset to cols that exist; return empty DataFrame if none found."""
            present = [c for c in cols if c in df.columns]
            return df[present] if present else pd.DataFrame()

        def _protpy_cols(fn, **kwargs):
            """Call a protpy function on the dummy sequence and return its column names."""
            return list(fn(_DUMMY, **kwargs).columns)

        # Amino acid, dipeptide and tripeptide composition
        self.amino_acid_composition = _select_cols(descriptor_df, _protpy_cols(protpy.amino_acid_composition))
        self.dipeptide_composition = _select_cols(descriptor_df, _protpy_cols(protpy.dipeptide_composition))
        self.tripeptide_composition = _select_cols(descriptor_df, _protpy_cols(protpy.tripeptide_composition))

        # Physicochemical descriptors (no extra parameters)
        self.gravy = _select_cols(descriptor_df, _protpy_cols(protpy.gravy))
        self.aromaticity = _select_cols(descriptor_df, _protpy_cols(protpy.aromaticity))
        self.instability_index = _select_cols(descriptor_df, _protpy_cols(protpy.instability_index))
        self.isoelectric_point = _select_cols(descriptor_df, _protpy_cols(protpy.isoelectric_point))
        self.molecular_weight = _select_cols(descriptor_df, _protpy_cols(protpy.molecular_weight))
        self.aliphatic_index = _select_cols(descriptor_df, _protpy_cols(protpy.aliphatic_index))
        self.boman_index = _select_cols(descriptor_df, _protpy_cols(protpy.boman_index))
        self.shannon_entropy = _select_cols(descriptor_df, _protpy_cols(protpy.shannon_entropy))
        self.hydrophobic_polar_charged_composition = _select_cols(descriptor_df, _protpy_cols(protpy.hydrophobic_polar_charged_composition))
        self.secondary_structure_propensity = _select_cols(descriptor_df, _protpy_cols(protpy.secondary_structure_propensity))
        self.extinction_coefficient = _select_cols(descriptor_df, _protpy_cols(protpy.extinction_coefficient))
        self.amino_acid_pair_composition = _select_cols(descriptor_df, _protpy_cols(protpy.amino_acid_pair_composition))

        # Physicochemical descriptors (parameterized)
        ph_params = getattr(self.desc_parameters, 'charge_distribution', {})
        ph = ph_params.get('ph', 7.4) if ph_params else 7.4
        self.charge_distribution = _select_cols(descriptor_df, _protpy_cols(protpy.charge_distribution, ph=ph))

        kmer_params = getattr(self.desc_parameters, 'kmer_composition', {})
        k = kmer_params.get('k', 2) if kmer_params else 2
        self.kmer_composition = _select_cols(descriptor_df, _protpy_cols(protpy.kmer_composition, k=k))

        rac_params = getattr(self.desc_parameters, 'reduced_alphabet_composition', {})
        alphabet_size = rac_params.get('alphabet_size', 6) if rac_params else 6
        self.reduced_alphabet_composition = _select_cols(descriptor_df, _protpy_cols(protpy.reduced_alphabet_composition, alphabet_size=alphabet_size))

        motif_params = getattr(self.desc_parameters, 'motif_composition', {})
        motifs = (motif_params.get('motifs', None) if motif_params else None) or None
        self.motif_composition = _select_cols(descriptor_df, _protpy_cols(protpy.motif_composition, motifs=motifs))

        agg_params = getattr(self.desc_parameters, 'aggregation_propensity', {})
        self.aggregation_propensity = _select_cols(descriptor_df, _protpy_cols(
            protpy.aggregation_propensity,
            window=agg_params.get('window', 5) if agg_params else 5,
            hydrophobicity_threshold=agg_params.get('hydrophobicity_threshold', 2.0) if agg_params else 2.0,
            charge_threshold=agg_params.get('charge_threshold', 1) if agg_params else 1
        ))

        hm_params = getattr(self.desc_parameters, 'hydrophobic_moment', {})
        self.hydrophobic_moment = _select_cols(descriptor_df, _protpy_cols(
            protpy.hydrophobic_moment,
            window=hm_params.get('window', 11) if hm_params else 11,
            angle=hm_params.get('angle', 100) if hm_params else 100
        ))

        # Autocorrelation descriptors
        self.moreaubroto_autocorrelation = _select_cols(descriptor_df, _protpy_cols(
            protpy.moreaubroto_autocorrelation,
            lag=self.desc_parameters.moreaubroto_autocorrelation["lag"],
            properties=self.desc_parameters.moreaubroto_autocorrelation["properties"],
            normalize=self.desc_parameters.moreaubroto_autocorrelation["normalize"]
        ))
        self.moran_autocorrelation = _select_cols(descriptor_df, _protpy_cols(
            protpy.moran_autocorrelation,
            lag=self.desc_parameters.moran_autocorrelation["lag"],
            properties=self.desc_parameters.moran_autocorrelation["properties"],
            normalize=self.desc_parameters.moran_autocorrelation["normalize"]
        ))
        self.geary_autocorrelation = _select_cols(descriptor_df, _protpy_cols(
            protpy.geary_autocorrelation,
            lag=self.desc_parameters.geary_autocorrelation["lag"],
            properties=self.desc_parameters.geary_autocorrelation["properties"],
            normalize=self.desc_parameters.geary_autocorrelation["normalize"]
        ))

        # CTD - derive expected columns from protpy then split sub-components by name prefix
        ctd_property = self.desc_parameters.ctd["property"]
        all_ctd = self.desc_parameters.ctd["all"]
        _ctd_all_cols = _protpy_cols(protpy.ctd_, property=ctd_property, all_ctd=all_ctd)
        self.ctd = _select_cols(descriptor_df, _ctd_all_cols)

        _present_ctd = [c for c in _ctd_all_cols if c in descriptor_df.columns]
        _ctd_c = [c for c in _present_ctd if 'CTD_C_' in c]
        _ctd_t = [c for c in _present_ctd if 'CTD_T_' in c]
        _ctd_d = [c for c in _present_ctd if 'CTD_D_' in c]
        self.ctd_composition = descriptor_df[_ctd_c] if _ctd_c else pd.DataFrame()
        self.ctd_transition = descriptor_df[_ctd_t] if _ctd_t else pd.DataFrame()
        self.ctd_distribution = descriptor_df[_ctd_d] if _ctd_d else pd.DataFrame()

        # Conjoint Triad
        self.conjoint_triad = _select_cols(descriptor_df, _protpy_cols(protpy.conjoint_triad))

        # Sequence Order Coupling Number
        socn_lag = self.desc_parameters.sequence_order_coupling_number["lag"]
        socn_dm = self.desc_parameters.sequence_order_coupling_number["distance_matrix"]
        if not socn_dm:
            _socn_cols = _protpy_cols(protpy.sequence_order_coupling_number_all, lag=socn_lag)
        else:
            _socn_cols = _protpy_cols(protpy.sequence_order_coupling_number, lag=socn_lag, distance_matrix=socn_dm)
        self.sequence_order_coupling_number = _select_cols(descriptor_df, _socn_cols)

        # Quasi Sequence Order
        qso_lag = self.desc_parameters.quasi_sequence_order["lag"]
        qso_weight = self.desc_parameters.quasi_sequence_order["weight"]
        qso_dm = self.desc_parameters.quasi_sequence_order["distance_matrix"]
        if not qso_dm:
            _qso_cols = _protpy_cols(protpy.quasi_sequence_order_all, lag=qso_lag, weight=qso_weight)
        else:
            _qso_cols = _protpy_cols(protpy.quasi_sequence_order, lag=qso_lag, weight=qso_weight, distance_matrix=qso_dm)
        self.quasi_sequence_order = _select_cols(descriptor_df, _qso_cols)

        # Pseudo Amino Acid Composition
        paac_lamda = self.desc_parameters.pseudo_amino_acid_composition["lambda"]
        paac_weight = self.desc_parameters.pseudo_amino_acid_composition["weight"]
        paac_props = self.desc_parameters.pseudo_amino_acid_composition["properties"]
        self.pseudo_amino_acid_composition = _select_cols(descriptor_df, _protpy_cols(
            protpy.pseudo_amino_acid_composition, lamda=paac_lamda, weight=paac_weight, properties=paac_props
        ))

        # Amphiphilic Pseudo Amino Acid Composition
        apaac_lamda = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["lambda"]
        apaac_weight = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["weight"]
        self.amphiphilic_pseudo_amino_acid_composition = _select_cols(descriptor_df, _protpy_cols(
            protpy.amphiphilic_pseudo_amino_acid_composition, lamda=apaac_lamda, weight=apaac_weight
        ))

        self.all_descriptors = descriptor_df

    def validate_descriptors(self, descriptors: Union[str, List[str]]) -> List[str]:
        """
        Validate that requested descriptors exist in the valid descriptors list.
        
        Parameters
        ==========
        :descriptors: str or list of str
            Descriptor name(s) to validate
        
        Returns
        =======
        :List[str]
            List of validated descriptor names
        
        Raises
        ======
        :TypeError
            If descriptors is not a string or list of strings
        :InvalidDescriptorError
            If any invalid descriptors are requested
        """
        if isinstance(descriptors, str):
            descriptors = [descriptors]
        elif not isinstance(descriptors, list):
            raise TypeError(
                f"Descriptors must be a string or list of strings, got {type(descriptors)}."
            )

        if not all(isinstance(descriptor, str) for descriptor in descriptors):
            raise TypeError("All descriptor names must be strings.")
        
        invalid = set(descriptors) - set(self.valid_descriptors)
        if invalid:
            raise InvalidDescriptorError(f"Invalid descriptors requested: {invalid}. "
                f"Valid descriptors: {self.valid_descriptors}")
        
        return descriptors

    def validate_sequences(self, seqs: Optional[pd.Series] = None) -> bool:
        """
        Validate all sequences contain only valid amino acids.
        
        Parameters
        ==========
        :seqs: pd.Series, optional
            Sequences to validate. If None, uses self.protein_seqs
        
        Returns
        =======
        :bool
            True if all sequences are valid
        
        Raises
        ======
        :InvalidSequenceError
            If invalid amino acids found
        """
        seqs = seqs if seqs is not None else self.protein_seqs
        invalid = valid_sequence(seqs)
        
        if invalid is not None:
            raise InvalidSequenceError(f"Invalid amino acids found: {invalid}")
        
        return True

    @property
    @lru_cache(maxsize=1)
    def descriptor_feature_count(self) -> Dict[str, int]:
        """
        Get count of features in each descriptor (cached for performance).
        
        Returns
        =======
        :Dict[str, int]
            Dictionary mapping descriptor names to feature counts
        """
        counts = {
            'amino_acid_composition': AA_COUNT,
            'dipeptide_composition': DIPEPTIDE_FEATURES,
            'tripeptide_composition': TRIPEPTIDE_FEATURES,
        }
        
        # Autocorrelation counts depend on lag and properties
        if not self.moreaubroto_autocorrelation.empty:
            counts['moreaubroto_autocorrelation'] = self.moreaubroto_autocorrelation.shape[1]
        if not self.moran_autocorrelation.empty:
            counts['moran_autocorrelation'] = self.moran_autocorrelation.shape[1]
        if not self.geary_autocorrelation.empty:
            counts['geary_autocorrelation'] = self.geary_autocorrelation.shape[1]
        
        # CTD counts
        if not self.ctd.empty:
            counts['ctd'] = self.ctd.shape[1]
            counts['ctd_composition'] = self.ctd_composition.shape[1]
            counts['ctd_transition'] = self.ctd_transition.shape[1]
            counts['ctd_distribution'] = self.ctd_distribution.shape[1]
        
        counts['conjoint_triad'] = CONJOINT_TRIAD_FEATURES
        
        # Sequence order counts
        if not self.sequence_order_coupling_number.empty:
            counts['sequence_order_coupling_number'] = self.sequence_order_coupling_number.shape[1]
        if not self.quasi_sequence_order.empty:
            counts['quasi_sequence_order'] = self.quasi_sequence_order.shape[1]
        
        # Pseudo composition counts
        if not self.pseudo_amino_acid_composition.empty:
            counts['pseudo_amino_acid_composition'] = self.pseudo_amino_acid_composition.shape[1]
        if not self.amphiphilic_pseudo_amino_acid_composition.empty:
            counts['amphiphilic_pseudo_amino_acid_composition'] = self.amphiphilic_pseudo_amino_acid_composition.shape[1]
        
        return counts

    def get_amino_acid_composition(self) -> pd.DataFrame:
        """
        Calculate Amino Acid Composition (AAComp) of protein sequence using the
        custom-built protpy package. AAComp describes the fraction of each amino 
        acid type within a protein sequence, and is calculated as:

        AA_Comp(s) = AA(t)/N(s)

        where AA_Comp(s) is the AAComp of protein sequence s, AA(t) is the number
        of amino acid types t (where t = 1,2,..,20) and N(s) is the length of the
        sequence s. 

        Parameters
        ==========
        None

        Returns
        =======
        :amino_acid_composition: pd.Dataframe
            pandas dataframe of AAComp for protein sequence. Dataframe will
            be of the shape N x 20, where N is the number of protein sequences
            and 20 is the number of features calculated from the descriptor 
            (for the 20 canonical amino acids).
        """
        #if attribute already calculated & not empty then return it
        if not self.amino_acid_composition.empty:
            return self.amino_acid_composition

        #calculate descriptor value for each sequence using helper method
        self.amino_acid_composition = self._calculate_descriptor_batch(
            protpy.amino_acid_composition,
            desc_name="Amino Acid Composition"
        )

        return self.amino_acid_composition

    def get_dipeptide_composition(self) -> pd.DataFrame:
        """
        Calculate Dipeptide Composition (DPComp) for protein sequence using
        the custom-built protpy package. Dipeptide composition is the fraction 
        of each dipeptide type within a protein sequence. With dipeptides 
        being of length 2 and there being 20 canonical amino acids, this creates 
        20^2 different combinations, thus a 400-Dimensional vector will be produced 
        such that:

        DPComp(s,t) = AA(s,t) / N -1

        where DPComp(s,t) is the dipeptide composition of the protein sequence
        for amino acid type s and t (where s and t = 1,2,..,20), AA(s,t) is the number
        of dipeptides represented by amino acid type s and t and N is the total number
        of dipeptides.

        Parameters
        ==========
        None

        Returns
        =======
        :dipeptide_composition: pd.Dataframe
            pandas Dataframe of dipeptide composition for protein sequence. Dataframe will
            be of the shape N x 400, where N is the number of protein sequences and 400 is 
            the number of features calculated from the descriptor (20^2 for the 20 canonical 
            amino acids).
        """
        #if attribute already calculated & not empty then return it
        if not self.dipeptide_composition.empty:
            return self.dipeptide_composition

        #calculate descriptor value using helper method
        self.dipeptide_composition = self._calculate_descriptor_batch(
            protpy.dipeptide_composition,
            desc_name="Dipeptide Composition"
        )

        return self.dipeptide_composition

    def get_tripeptide_composition(self) -> pd.DataFrame:
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
        ==========
        None

        Returns
        =======
        :tripeptide_composition: pd.Dataframe
            pandas Dataframe of tripeptide composition for protein sequence. Dataframe will
            be of the shape N x 8000, where N is the number of protein sequences and 8000 is 
            the number of features calculated from the descriptor (20^3 for the 20 canonical 
            amino acids).
        """
        #if attribute already calculated & not empty then return it
        if not self.tripeptide_composition.empty:
            return self.tripeptide_composition

        #calculate descriptor value using helper method
        self.tripeptide_composition = self._calculate_descriptor_batch(
            protpy.tripeptide_composition,
            desc_name="Tripeptide Composition"
        )

        return self.tripeptide_composition

    def get_gravy(self) -> pd.DataFrame:
        """
        Calculate the Grand Average of Hydropathy (GRAVY) for protein sequences using
        the protpy package. GRAVY is the mean of Kyte-Doolittle hydropathy values across
        all residues. A positive value indicates overall hydrophobicity; a negative value
        indicates overall hydrophilicity.

        Parameters
        ==========
        None

        Returns
        =======
        :gravy: pd.DataFrame
            Dataframe of GRAVY values, shape N x 1 where N is the number of sequences.
        """
        # return cached result if already computed
        if not self.gravy.empty:
            return self.gravy

        # calculate GRAVY for all sequences
        self.gravy = self._calculate_descriptor_batch(
            protpy.gravy,
            desc_name="GRAVY"
        )
        return self.gravy

    def get_aromaticity(self) -> pd.DataFrame:
        """
        Calculate Aromaticity for protein sequences using the protpy package.
        Aromaticity is the fraction of aromatic residues (F, W, Y, H) in the sequence.

        Parameters
        ==========
        None

        Returns
        =======
        :aromaticity: pd.DataFrame
            Dataframe of Aromaticity values, shape N x 1 where N is the number of sequences.
        """
        # return cached result if already computed
        if not self.aromaticity.empty:
            return self.aromaticity

        # calculate aromaticity for all sequences
        self.aromaticity = self._calculate_descriptor_batch(
            protpy.aromaticity,
            desc_name="Aromaticity"
        )
        return self.aromaticity

    def get_instability_index(self) -> pd.DataFrame:
        """
        Calculate the Instability Index for protein sequences using the protpy package.
        Based on dipeptide instability weight values (DIWV). Values below 40 indicate a
        stable protein; 40 or above indicates instability.

        Parameters
        ==========
        None

        Returns
        =======
        :instability_index: pd.DataFrame
            Dataframe of InstabilityIndex values, shape N x 1.
        """
        # return cached result if already computed
        if not self.instability_index.empty:
            return self.instability_index

        # calculate instability index for all sequences
        self.instability_index = self._calculate_descriptor_batch(
            protpy.instability_index,
            desc_name="Instability Index"
        )
        return self.instability_index

    def get_isoelectric_point(self) -> pd.DataFrame:
        """
        Calculate the Isoelectric Point for protein sequences using the protpy package.
        The isoelectric point is the estimated pH at which the protein carries no net
        charge, calculated iteratively using standard pKa values for ionisable residues.

        Parameters
        ==========
        None

        Returns
        =======
        :isoelectric_point: pd.DataFrame
            Dataframe of IsoelectricPoint values, shape N x 1.
        """
        # return cached result if already computed
        if not self.isoelectric_point.empty:
            return self.isoelectric_point

        # calculate isoelectric point for all sequences
        self.isoelectric_point = self._calculate_descriptor_batch(
            protpy.isoelectric_point,
            desc_name="Isoelectric Point"
        )
        return self.isoelectric_point

    def get_molecular_weight(self) -> pd.DataFrame:
        """
        Calculate the Molecular Weight for protein sequences using the protpy package.
        Average molecular weight calculated from residue masses, corrected for water
        lost at each peptide bond.

        Parameters
        ==========
        None

        Returns
        =======
        :molecular_weight: pd.DataFrame
            Dataframe of MolecularWeight values (Da), shape N x 1.
        """
        # return cached result if already computed
        if not self.molecular_weight.empty:
            return self.molecular_weight

        # calculate molecular weight for all sequences
        self.molecular_weight = self._calculate_descriptor_batch(
            protpy.molecular_weight,
            desc_name="Molecular Weight"
        )
        return self.molecular_weight

    def get_charge_distribution(self) -> pd.DataFrame:
        """
        Calculate Charge Distribution for protein sequences using the protpy package.
        Computes positive, negative, and net charge contributions of ionisable residues
        at a given pH using the Henderson-Hasselbalch equation.

        Parameters
        ==========
        None

        Returns
        =======
        :charge_distribution: pd.DataFrame
            Dataframe of charge values, shape N x 3 (PositiveCharge, NegativeCharge, NetCharge).
        """
        # return cached result if already computed
        if not self.charge_distribution.empty:
            return self.charge_distribution

        # get pH parameter from config, falling back to physiological default
        ph_params = getattr(self.desc_parameters, 'charge_distribution', {})
        ph = ph_params.get('ph', 7.4) if ph_params else 7.4

        # calculate charge distribution for all sequences
        self.charge_distribution = self._calculate_descriptor_batch(
            protpy.charge_distribution,
            desc_name="Charge Distribution",
            ph=ph
        )
        return self.charge_distribution

    def get_hydrophobic_polar_charged_composition(self) -> pd.DataFrame:
        """
        Calculate Hydrophobic/Polar/Charged Composition (HPC) for protein sequences
        using the protpy package. Computes the percentage of residues belonging to each
        of three physicochemical groups: hydrophobic (A, C, F, I, L, M, V, W, Y),
        polar (G, N, Q, S, T), and charged (D, E, H, K, R).

        Parameters
        ==========
        None

        Returns
        =======
        :hydrophobic_polar_charged_composition: pd.DataFrame
            Dataframe of HPC values, shape N x 3 (Hydrophobic, Polar, Charged).
        """
        # return cached result if already computed
        if not self.hydrophobic_polar_charged_composition.empty:
            return self.hydrophobic_polar_charged_composition

        # calculate HPC composition for all sequences
        self.hydrophobic_polar_charged_composition = self._calculate_descriptor_batch(
            protpy.hydrophobic_polar_charged_composition,
            desc_name="Hydrophobic/Polar/Charged Composition"
        )
        return self.hydrophobic_polar_charged_composition

    def get_secondary_structure_propensity(self) -> pd.DataFrame:
        """
        Calculate Secondary Structure Propensity (SSP) for protein sequences using the
        protpy package. Computes average Chou-Fasman propensity values for alpha-helix,
        beta-sheet, and random coil conformations across all residues.

        Parameters
        ==========
        None

        Returns
        =======
        :secondary_structure_propensity: pd.DataFrame
            Dataframe of SSP values, shape N x 3 (Helix, Sheet, Coil).
        """
        # return cached result if already computed
        if not self.secondary_structure_propensity.empty:
            return self.secondary_structure_propensity

        # calculate secondary structure propensity for all sequences
        self.secondary_structure_propensity = self._calculate_descriptor_batch(
            protpy.secondary_structure_propensity,
            desc_name="Secondary Structure Propensity"
        )
        return self.secondary_structure_propensity

    def get_kmer_composition(self) -> pd.DataFrame:
        """
        Calculate k-mer Composition for protein sequences using the protpy package.
        Computes the frequency of all possible k-length residue subsequences, expressed
        as a percentage of total k-mers.

        Parameters
        ==========
        None

        Returns
        =======
        :kmer_composition: pd.DataFrame
            Dataframe of k-mer composition values, shape N x 20^k (e.g. N x 400 for k=2).
        """
        # return cached result if already computed
        if not self.kmer_composition.empty:
            return self.kmer_composition

        # get k-mer length from config, defaulting to 2 (dipeptide)
        kmer_params = getattr(self.desc_parameters, 'kmer_composition', {})
        k = kmer_params.get('k', 2) if kmer_params else 2

        # calculate k-mer composition for all sequences
        self.kmer_composition = self._calculate_descriptor_batch(
            protpy.kmer_composition,
            desc_name="k-mer Composition",
            k=k
        )
        return self.kmer_composition

    def get_reduced_alphabet_composition(self) -> pd.DataFrame:
        """
        Calculate Reduced Alphabet Composition for protein sequences using the protpy
        package. Computes amino acid composition after mapping residues to a reduced
        alphabet of physicochemical groups. Supported alphabet sizes: 2, 3, 4, 6.

        Parameters
        ==========
        None

        Returns
        =======
        :reduced_alphabet_composition: pd.DataFrame
            Dataframe of reduced composition values, shape N x alphabet_size.
        """
        # return cached result if already computed
        if not self.reduced_alphabet_composition.empty:
            return self.reduced_alphabet_composition

        # get alphabet size from config, defaulting to 6 groups
        rac_params = getattr(self.desc_parameters, 'reduced_alphabet_composition', {})
        alphabet_size = rac_params.get('alphabet_size', 6) if rac_params else 6

        # calculate reduced alphabet composition for all sequences
        self.reduced_alphabet_composition = self._calculate_descriptor_batch(
            protpy.reduced_alphabet_composition,
            desc_name="Reduced Alphabet Composition",
            alphabet_size=alphabet_size
        )
        return self.reduced_alphabet_composition

    def get_motif_composition(self) -> pd.DataFrame:
        """
        Calculate Motif Composition for protein sequences using the protpy package.
        Counts occurrences (including overlapping) of biological sequence motifs matched
        via regular expressions. Uses 8 built-in motifs by default; a custom dict of
        name->pattern mappings can be supplied via config.

        Parameters
        ==========
        None

        Returns
        =======
        :motif_composition: pd.DataFrame
            Dataframe of motif counts, shape N x len(motifs).
        """
        # return cached result if already computed
        if not self.motif_composition.empty:
            return self.motif_composition

        # get custom motifs from config; None causes protpy to use built-in defaults
        motif_params = getattr(self.desc_parameters, 'motif_composition', {})
        motifs = motif_params.get('motifs', None) if motif_params else None
        # treat empty list/dict as None to trigger built-in default motifs
        if not motifs:
            motifs = None

        # calculate motif composition for all sequences
        self.motif_composition = self._calculate_descriptor_batch(
            protpy.motif_composition,
            desc_name="Motif Composition",
            motifs=motifs
        )
        return self.motif_composition

    def get_amino_acid_pair_composition(self) -> pd.DataFrame:
        """
        Calculate Amino Acid Pair Composition for protein sequences using the protpy
        package. Computes the frequency of all 400 residue-pair combinations with
        column names annotated by the physicochemical class of each residue.

        Parameters
        ==========
        None

        Returns
        =======
        :amino_acid_pair_composition: pd.DataFrame
            Dataframe of pair composition values, shape N x 400.
        """
        # return cached result if already computed
        if not self.amino_acid_pair_composition.empty:
            return self.amino_acid_pair_composition

        # calculate amino acid pair composition for all sequences
        self.amino_acid_pair_composition = self._calculate_descriptor_batch(
            protpy.amino_acid_pair_composition,
            desc_name="Amino Acid Pair Composition"
        )
        return self.amino_acid_pair_composition

    def get_aliphatic_index(self) -> pd.DataFrame:
        """
        Calculate the Aliphatic Index for protein sequences using the protpy package.
        Measures the relative volume occupied by aliphatic side chains (Ala, Val, Ile,
        Leu). Higher values indicate greater thermostability.

        Parameters
        ==========
        None

        Returns
        =======
        :aliphatic_index: pd.DataFrame
            Dataframe of AliphaticIndex values, shape N x 1.
        """
        # return cached result if already computed
        if not self.aliphatic_index.empty:
            return self.aliphatic_index

        # calculate aliphatic index for all sequences
        self.aliphatic_index = self._calculate_descriptor_batch(
            protpy.aliphatic_index,
            desc_name="Aliphatic Index"
        )
        return self.aliphatic_index

    def get_extinction_coefficient(self) -> pd.DataFrame:
        """
        Calculate the Extinction Coefficient for protein sequences using the protpy
        package. Computes the molar extinction coefficient at 280 nm from the number of
        Trp (W), Tyr (Y), and Cys (C) residues. Reported for reduced and oxidized states.

        Parameters
        ==========
        None

        Returns
        =======
        :extinction_coefficient: pd.DataFrame
            Dataframe of extinction coefficient values, shape N x 2
            (ExtCoeff_Reduced, ExtCoeff_Oxidized).
        """
        # return cached result if already computed
        if not self.extinction_coefficient.empty:
            return self.extinction_coefficient

        # calculate extinction coefficient for all sequences
        self.extinction_coefficient = self._calculate_descriptor_batch(
            protpy.extinction_coefficient,
            desc_name="Extinction Coefficient"
        )
        return self.extinction_coefficient

    def get_boman_index(self) -> pd.DataFrame:
        """
        Calculate the Boman Index for protein sequences using the protpy package.
        Sum of solubility values for amino acids divided by sequence length, predicting
        potential for protein-protein interactions.

        Parameters
        ==========
        None

        Returns
        =======
        :boman_index: pd.DataFrame
            Dataframe of BomanIndex values, shape N x 1.
        """
        # return cached result if already computed
        if not self.boman_index.empty:
            return self.boman_index

        # calculate Boman index for all sequences
        self.boman_index = self._calculate_descriptor_batch(
            protpy.boman_index,
            desc_name="Boman Index"
        )
        return self.boman_index

    def get_aggregation_propensity(self) -> pd.DataFrame:
        """
        Calculate Aggregation Propensity for protein sequences using the protpy package.
        Estimates aggregation-prone regions via a sliding-window approach combining
        Kyte-Doolittle hydrophobicity and charge neutrality. Returns the count of
        qualifying windows and the fraction of the sequence covered.

        Parameters
        ==========
        None

        Returns
        =======
        :aggregation_propensity: pd.DataFrame
            Dataframe of aggregation values, shape N x 2
            (AggregProneRegions, AggregProneFraction).
        """
        # return cached result if already computed
        if not self.aggregation_propensity.empty:
            return self.aggregation_propensity

        # get sliding-window parameters from config, using standard defaults otherwise
        agg_params = getattr(self.desc_parameters, 'aggregation_propensity', {})
        window = agg_params.get('window', 5) if agg_params else 5
        hydrophobicity_threshold = agg_params.get('hydrophobicity_threshold', 2.0) if agg_params else 2.0
        charge_threshold = agg_params.get('charge_threshold', 1) if agg_params else 1

        # calculate aggregation propensity for all sequences
        self.aggregation_propensity = self._calculate_descriptor_batch(
            protpy.aggregation_propensity,
            desc_name="Aggregation Propensity",
            window=window,
            hydrophobicity_threshold=hydrophobicity_threshold,
            charge_threshold=charge_threshold
        )
        return self.aggregation_propensity

    def get_hydrophobic_moment(self) -> pd.DataFrame:
        """
        Calculate Hydrophobic Moment for protein sequences using the protpy package.
        Computes the mean and maximum hydrophobic moment across sliding windows using
        the Eisenberg hydrophobicity scale and a helical-wheel projection. Captures
        amphipathicity of putative helix segments.

        Parameters
        ==========
        None

        Returns
        =======
        :hydrophobic_moment: pd.DataFrame
            Dataframe of hydrophobic moment values, shape N x 2
            (HydrophobicMoment_Mean, HydrophobicMoment_Max).
        """
        # return cached result if already computed
        if not self.hydrophobic_moment.empty:
            return self.hydrophobic_moment

        # get window and helical angle from config, using Eisenberg scale defaults
        hm_params = getattr(self.desc_parameters, 'hydrophobic_moment', {})
        window = hm_params.get('window', 11) if hm_params else 11
        angle = hm_params.get('angle', 100) if hm_params else 100

        # calculate hydrophobic moment for all sequences
        self.hydrophobic_moment = self._calculate_descriptor_batch(
            protpy.hydrophobic_moment,
            desc_name="Hydrophobic Moment",
            window=window,
            angle=angle
        )
        return self.hydrophobic_moment

    def get_shannon_entropy(self) -> pd.DataFrame:
        """
        Calculate Shannon Entropy for protein sequences using the protpy package.
        An information-theoretic measure of amino acid diversity in a sequence computed
        as H = -sum(p_i * log2(p_i)). A value of 0 means a completely repetitive
        sequence; the theoretical maximum of ~4.322 bits corresponds to a perfectly
        uniform distribution across all 20 canonical amino acids.

        Parameters
        ==========
        None

        Returns
        =======
        :shannon_entropy: pd.DataFrame
            Dataframe of ShannonEntropy values, shape N x 1.
        """
        # return cached result if already computed
        if not self.shannon_entropy.empty:
            return self.shannon_entropy

        # calculate Shannon entropy for all sequences
        self.shannon_entropy = self._calculate_descriptor_batch(
            protpy.shannon_entropy,
            desc_name="Shannon Entropy"
        )
        return self.shannon_entropy

    def get_moreaubroto_autocorrelation(self) -> pd.DataFrame:
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

        AccNo. CIDH920105 - Normalized Average Hydrophobicity Scales.
        AccNo. BHAR880101 - Average Flexibility Indices.
        AccNo. CHAM820101 - Polarizability Parameter.
        AccNo. CHAM820102 - Free Energy of Solution in Water, kcal/mole.
        AccNo. CHOC760101 - Residue Accessible Surface Area in Tripeptide.
        AccNo. BIGC670101 - Residue Volume.
        AccNo. CHAM810101 - Steric Parameter.
        AccNo. DAYM780201 - Relative Mutability.

        Parameters
        ==========
        None

        Returns15
        =======
        :moreaubroto_autocorrelation: pd.Dataframe
            pandas Dataframe of MBAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences and 
            M is the number of features calculated from the descriptor, calculated 
            as lag * number of properties. By default, the shape will be N x 240 
            (30 features per property - using 8 properties, with lag=30).
        """
        #if attribute already calculated & not empty then return it
        if not self.moreaubroto_autocorrelation.empty:
            return self.moreaubroto_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.moreaubroto_autocorrelation["lag"]
        properties = self.desc_parameters.moreaubroto_autocorrelation["properties"]
        normalize = self.desc_parameters.moreaubroto_autocorrelation["normalize"]

        #calculate descriptor value using helper method
        self.moreaubroto_autocorrelation = self._calculate_descriptor_batch(
            protpy.moreaubroto_autocorrelation,
            desc_name="MoreauBroto Autocorrelation",
            lag=lag,
            properties=properties,
            normalize=normalize
        )

        return self.moreaubroto_autocorrelation

    def get_moran_autocorrelation(self) -> pd.DataFrame:
        """
        Calculate Moran autocorrelation (MAuto) of protein sequences using the custom-built
        protpy package. MAuto utilizes property deviations from the average values.
        **refer to MBAuto docstring for autocorrelation description.

        Parameters
        ==========
        None

        Returns
        =======
        :moran_autocorrelation: pd.DataFrame
            pandas Dataframe of MAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences
            and M is the number of features calculated from the descriptor, 
            calculated as lag * number of properties. By default, the shape 
            will be N x 240 (30 features per property - using 8 properties, 
            with lag=30).
        """
        #if attribute already calculated & not empty then return it
        if not self.moran_autocorrelation.empty:
            return self.moran_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.moran_autocorrelation["lag"]
        properties = self.desc_parameters.moran_autocorrelation["properties"]
        normalize = self.desc_parameters.moran_autocorrelation["normalize"]

        #calculate descriptor value using helper method
        self.moran_autocorrelation = self._calculate_descriptor_batch(
            protpy.moran_autocorrelation,
            desc_name="Moran Autocorrelation",
            lag=lag,
            properties=properties,
            normalize=normalize
        )

        return self.moran_autocorrelation 

    def get_geary_autocorrelation(self) -> pd.DataFrame:
        """
        Calculate Geary Autocorrelation (GAuto) of protein sequences using the
        custom-built protpy package. GAuto utilizes the square-difference of 
        property values instead of vector-products (of property values or 
        deviations).  
        **refer to MBAuto docstring for autocorrelation description.

        Parameters
        ==========
        None

        Returns
        =======
        :geary_autocorrelation: pd.DataFrame
            pandas Dataframe of GAuto values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences and 
            M is the number of features calculated from the descriptor, calculated 
            as lag * number of properties. By default, the shape will be N x 240 
            (30 features per property - using 8 properties, with lag=30).
        """
        #if attribute already calculated & not empty then return it
        if not self.geary_autocorrelation.empty:
            return self.geary_autocorrelation

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.geary_autocorrelation["lag"]
        properties = self.desc_parameters.geary_autocorrelation["properties"]
        normalize = self.desc_parameters.geary_autocorrelation["normalize"]

        #calculate descriptor value using helper method
        self.geary_autocorrelation = self._calculate_descriptor_batch(
            protpy.geary_autocorrelation,
            desc_name="Geary Autocorrelation",
            lag=lag,
            properties=properties,
            normalize=normalize
        )

        return self.geary_autocorrelation 

    def get_ctd_composition(self) -> pd.DataFrame:
        """ 
        Calculate Composition (C_CTD) physicochemical/structural descriptor
        of protein sequences from the calculated CTD descriptor. Composition 
        is determined as the number of amino acids of a particular property 
        divided by total number of amino acids,

        Parameters
        ==========
        None

        Returns
        =======
        :ctd_composition: pd.DataFrame
            pandas dataframe of C_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physicochemical properties * 3), with 3 
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 3. 
        """
        #if attribute already calculated & not empty then return it
        if not (self.ctd_composition.empty):
            return self.ctd_composition
        
        #calculate ctd descriptor if not already calculated
        if (self.ctd.empty):
            self.ctd = self.get_ctd()

        #initialise dataframe
        comp_df = pd.DataFrame()

        #get ctd properties  used for calculating descriptor
        ctd_property = self.desc_parameters.ctd["property"]
        if not (isinstance(ctd_property, list)):
            ctd_property = ctd_property.split(',')
        all_ctd = self.desc_parameters.ctd["all"]

        #get composition descriptor from CTD dataframe, dependant on number of props, 3 features per property
        if (all_ctd):
            comp_df = self.ctd.iloc[:,0:21]
        else:
            comp_df = self.ctd.iloc[:,0:3 * len(ctd_property)]
            
        self.ctd_composition = comp_df

        return self.ctd_composition
  
    def get_ctd_transition(self) -> pd.DataFrame:
        """ 
        Calculate Transition (T_CTD) physicochemical/structural descriptor of 
        protein sequences from the calculated CTD descriptor. Transition is 
        determined as the number of transitions from a particular property to 
        different property divided by (total number of amino acids − 1).
        
        Parameters
        ==========
        None

        Returns
        =======
        :ctd_transition: pd.Dataframe
            pandas Dataframe of T_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physicochemical properties * 3), with 3 
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 3. 
        """
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

        #get transition descriptor from CTD dataframe, dependant on number of props, 3 features per property
        if (all_ctd):
            transition_df = self.ctd.iloc[:,21:42]
        else:
            transition_df = self.ctd.iloc[:,3 * len(ctd_property):(3 * len(ctd_property) * 2)]
        
        self.ctd_transition = transition_df

        return self.ctd_transition

    def get_ctd_distribution(self) -> pd.DataFrame:
        """ 
        Calculate Distribution (D_CTD) physicochemical/structural descriptor of 
        protein sequences from the calculated CTD descriptor. Distribution is 
        the chain length within which the first, 25%, 50%, 75% and 100% of the 
        amino acids of a particular property are located.

        Parameters
        ==========
        None

        Returns
        =======
        :ctd_distribution: pd.Dataframe
            pandas Dataframe of D_CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein sequences 
            and M is the (number of physicochemical properties * 15), with 15
            features being calculated per property. By default the 
            "hydrophobicity" property will be used, generating an output of 
            N x 15. 
        """
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

        #get distribution descriptor from CTD dataframe, dependant on number of props, 15 features per property
        if (all_ctd):
            distribution_df = self.ctd.iloc[:,42:]
        else:
            distribution_df = self.ctd.iloc[:,2 * (3 * len(ctd_property)):]
        
        self.ctd_distribution = distribution_df

        return self.ctd_distribution

    def get_ctd(self) -> pd.DataFrame:
        """
        Calculate all CTD (Composition, Transition, Distribution) 
        physicochemical/structural descriptor of protein sequences using the 
        custom-built protpy package. 

        Parameters
        ==========
        None

        Returns
        =======
        :ctd: pd.Series
            pandas Series of CTD values for protein sequence. Output will
            be of the shape N x M, where N is the number of protein 
            sequences and M is (number of physicochemical properties * 21),
            with 21 being the number of features calculated for each of the
            CTD descriptors per property. Using all properties will generate
            an output of N x 147, by default the "hydrophobicity"
            property is used, generating an output of N x 21. 
        """
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

        self.ctd = ctd_df.reset_index(drop=True)

        return self.ctd

    def get_conjoint_triad(self) -> pd.DataFrame:
        """
        Calculate Conjoint Triad (CTriad) of protein sequences using the custom-built
        protpy package. The descriptor mainly considers neighbour relationships in 
        protein sequences by encoding each protein sequence using the triad (continuous 
        three amino acids) frequency distribution extracted from a 7-letter reduced 
        alphabet [11]. CTriad calculates 343 different features (7x7x7), with the 
        output being of shape N x 343 where N is the number of sequences.

        Parameters
        ==========
        None

        Returns
        =======
        :conjoint_triad: pd.Dataframe
            pandas Dataframe of CTriad descriptor values for all protein sequences. Dataframe
            will be of the shape N x 343, where N is the number of protein sequences and 343 
            is the number of features calculated from the descriptor for a sequence.
        """
        #if attribute already calculated & not empty then return it
        if not (self.conjoint_triad.empty):
            return self.conjoint_triad

        #initialise dataframe
        conjoint_triad_df = pd.DataFrame()

        #calculate descriptor value, for each sequence, concatenate descriptor values
        for seq in self.protein_seqs:
            conjoint_triad_seq = protpy.conjoint_triad(seq)
            conjoint_triad_df = pd.concat([conjoint_triad_df, conjoint_triad_seq])

        self.conjoint_triad = conjoint_triad_df.reset_index(drop=True)

        return self.conjoint_triad

    def get_sequence_order_coupling_number(self) -> pd.DataFrame:
        """
        Calculate Sequence Order Coupling Number (SOCN) features for input protein sequence
        using custom-built protpy package. SOCN computes the dissimilarity between amino acid
        pairs. The distance between amino acid pairs is determined by d which varies between 
        1 to lag. For each d, it computes the sum of the dissimilarities of all amino acid 
        pairs. The number of output features can be calculated as N * 2, where N = lag, by 
        default this value is 30 which generates an output of M x 60 where M is the number 
        of protein sequenes. 

        Parameters
        ==========
        None

        Returns
        =======
        :sequence_order_coupling_number_df: pd.Dataframe
            Dataframe of SOCN descriptor values for all protein sequences. Output
            will be of the shape N x M, where N is the number of protein sequences and
            M is the number of features calculated from the descriptor (calculated as
            N * 2 where N = lag).
        """
        #if attribute already calculated & not empty then return it
        if not (self.sequence_order_coupling_number.empty):
            return self.sequence_order_coupling_number

        #initialise dataframe
        sequence_order_coupling_number_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.sequence_order_coupling_number["lag"]
        distance_matrix = self.desc_parameters.sequence_order_coupling_number["distance_matrix"]

        #calculate descriptor value, for each sequence, concatenate descriptor values
        for seq in self.protein_seqs:
            #if no distance matrix present in config then calculate SOCN using both matrices
            if (distance_matrix == "" or distance_matrix is None):
                sequence_order_coupling_number_seq = protpy.sequence_order_coupling_number_all(seq, lag=lag)
            else:
                sequence_order_coupling_number_seq = protpy.sequence_order_coupling_number(seq, lag=lag, distance_matrix=distance_matrix)

            #concat sequence's descriptor output to dataframe
            sequence_order_coupling_number_df = pd.concat([sequence_order_coupling_number_df, sequence_order_coupling_number_seq])

        self.sequence_order_coupling_number = sequence_order_coupling_number_df.reset_index(drop=True)

        return self.sequence_order_coupling_number

    def get_quasi_sequence_order(self) -> pd.DataFrame:
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
        ==========
        None

        Returns
        =======
        :quasi_sequence_order_df: pd.Dataframe
            Dataframe of quasi-sequence-order descriptor values for the
            protein sequences, with output shape N x 100 where N is the number
            of sequences and 100 the number of calculated features.
        """
        #if attribute already calculated & not empty then return it
        if not (self.quasi_sequence_order.empty):
            return self.quasi_sequence_order

        #initialise dataframe
        quasi_sequence_order_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lag = self.desc_parameters.quasi_sequence_order["lag"]
        weight = self.desc_parameters.quasi_sequence_order["weight"]
        distance_matrix = self.desc_parameters.quasi_sequence_order["distance_matrix"]

        #calculate descriptor value, for each sequene, concatenate descriptor values
        for seq in self.protein_seqs:
            #if no distance matrix present in config then calculate quasi seq order using both matrices
            if (distance_matrix == "" or distance_matrix is None):
                quasi_sequence_order_seq = protpy.quasi_sequence_order_all(seq, lag=lag, weight=weight)
            else:
                quasi_sequence_order_seq = protpy.quasi_sequence_order(seq, lag=lag, weight=weight, 
                    distance_matrix=distance_matrix)

            #concat sequence's descriptor output to dataframe
            quasi_sequence_order_df = pd.concat([quasi_sequence_order_df, quasi_sequence_order_seq])

        self.quasi_sequence_order = quasi_sequence_order_df.reset_index(drop=True)

        return self.quasi_sequence_order

    def get_pseudo_amino_acid_composition(self) -> pd.DataFrame:
        """
        Calculate Pseudo Amino Acid Composition (PAAComp) descriptor using custom-built protpy 
        package. PAAComp combines the vanilla amino acid composition descriptor with additional 
        local features, such as correlation between residues of a certain distance, as amino 
        acid composition doesn't take into account sequence order info. The pseudo components 
        of the descriptor are a series rank-different correlation factors [10]. The first 20 
        components are a weighted sum of the amino acid composition and 30 are physicochemical 
        square correlations as dictated by the lambda and properties parameters. This generates 
        an output of [(20 + λ), 1] = 50 x 1 when using the default lambda of 30. By default, 
        the physicochemical properties used are hydrophobicity and hydrophillicity, with a lambda 
        of 30 and weight of 0.05.

        Parameters
        ==========
        None

        Returns
        =======
        :pseudo_amino_acid_composition_df: pd.Dataframe
            Dataframe of pseudo amino acid composition descriptor values for the protein sequences 
            of output shape N x (20 + λ), where N is the number of protein sequences. With 
            default lambda of 30, the output shape will be N x 50.
        """
        #if attribute already calculated & not empty then return it
        if not (self.pseudo_amino_acid_composition.empty):
            return self.pseudo_amino_acid_composition

        #initialise dataframe
        pseudo_amino_acid_composition_df = pd.DataFrame()

        #get descriptor-specific parameters from config file
        lamda = self.desc_parameters.pseudo_amino_acid_composition["lambda"]
        weight = self.desc_parameters.pseudo_amino_acid_composition["weight"]
        properties = self.desc_parameters.pseudo_amino_acid_composition["properties"]

        #calculate descriptor value, for each sequence, concatenate descriptor values,
        #tqdm loop to visualise progress as descriptor can take some time to execute
        for seq in tqdm(self.protein_seqs, unit=" sequence", position=0, 
            desc="PAAComp", mininterval=30, ncols=90):
            pseudo_amino_acid_composition_seq = protpy.pseudo_amino_acid_composition(seq, lamda=lamda, 
                weight=weight, properties=properties)
            pseudo_amino_acid_composition_df = pd.concat([pseudo_amino_acid_composition_df, pseudo_amino_acid_composition_seq])

        self.pseudo_amino_acid_composition = pseudo_amino_acid_composition_df.reset_index(drop=True)

        return self.pseudo_amino_acid_composition
        
    def get_amphiphilic_pseudo_amino_acid_composition(self) -> pd.DataFrame:
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
        ==========
        None

        Returns
        =======
        :amphiphilic_pseudo_amino_acid_composition_df: pd.Dataframe
            Dataframe of Amphiphilic pseudo amino acid composition descriptor values for 
            the protein sequences of output shape N x 80, where N is the number of 
            protein sequences and 80 is calculated as (20 + 2*lambda).
        """
        #if attribute already calculated & not empty then return it
        if not (self.amphiphilic_pseudo_amino_acid_composition.empty):
            return self.amphiphilic_pseudo_amino_acid_composition

        #get descriptor-specific parameters from config file
        lamda = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["lambda"]
        weight = self.desc_parameters.amphiphilic_pseudo_amino_acid_composition["weight"]

        #initialise dataframe
        amphiphilic_pseudo_amino_acid_composition_df = pd.DataFrame()

        #calculate descriptor value, for each sequence, concatenate descriptor values, 
        #tqdm loop to visualise progress as descriptor can take some time to execute
        for seq in tqdm(self.protein_seqs, unit=" sequence", position=0, 
            desc="APAAComp", mininterval=30, ncols=90):
            amphiphilic_pseudo_amino_acid_composition_seq = protpy.amphiphilic_pseudo_amino_acid_composition(seq, 
                lamda=lamda, weight=weight)
            amphiphilic_pseudo_amino_acid_composition_df = pd.concat([amphiphilic_pseudo_amino_acid_composition_df, 
                amphiphilic_pseudo_amino_acid_composition_seq])

        self.amphiphilic_pseudo_amino_acid_composition = amphiphilic_pseudo_amino_acid_composition_df.reset_index(drop=True)

        return self.amphiphilic_pseudo_amino_acid_composition


    def get_descriptor_encoding(self, descriptor: str) -> Optional[pd.DataFrame]:
        """
        Get the protein descriptor values of a specified input descriptor. If the
        sought descriptor has already been calculated then its attribute is returned,
        else the descriptor is calculated using its get_descriptor function.

        Parameters
        ==========
        :descriptor: str
            name of descriptor to return. Method can accept the approximate name
            of the descriptor, e.g. 'amino_comp'/'aa_composition' etc will return 
            the 'amino_acid_composition' descriptor. This functionality is realised 
            using the difflib library and its built-in get_close_matches function.

        Returns
        =======
        :desc_encoding: pd.DataFrame or None
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
            raise ValueError(f"Could not find a match for the input descriptor {descriptor} in"
                f" list of available valid models:\n{self.valid_descriptors}.")

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

        elif (desc == 'gravy'):
            if (getattr(self, desc).empty):
                self.get_gravy()
            desc_encoding = self.gravy

        elif (desc == 'aromaticity'):
            if (getattr(self, desc).empty):
                self.get_aromaticity()
            desc_encoding = self.aromaticity

        elif (desc == 'instability_index'):
            if (getattr(self, desc).empty):
                self.get_instability_index()
            desc_encoding = self.instability_index

        elif (desc == 'isoelectric_point'):
            if (getattr(self, desc).empty):
                self.get_isoelectric_point()
            desc_encoding = self.isoelectric_point

        elif (desc == 'molecular_weight'):
            if (getattr(self, desc).empty):
                self.get_molecular_weight()
            desc_encoding = self.molecular_weight

        elif (desc == 'charge_distribution'):
            if (getattr(self, desc).empty):
                self.get_charge_distribution()
            desc_encoding = self.charge_distribution

        elif (desc == 'hydrophobic_polar_charged_composition'):
            if (getattr(self, desc).empty):
                self.get_hydrophobic_polar_charged_composition()
            desc_encoding = self.hydrophobic_polar_charged_composition

        elif (desc == 'secondary_structure_propensity'):
            if (getattr(self, desc).empty):
                self.get_secondary_structure_propensity()
            desc_encoding = self.secondary_structure_propensity

        elif (desc == 'kmer_composition'):
            if (getattr(self, desc).empty):
                self.get_kmer_composition()
            desc_encoding = self.kmer_composition

        elif (desc == 'reduced_alphabet_composition'):
            if (getattr(self, desc).empty):
                self.get_reduced_alphabet_composition()
            desc_encoding = self.reduced_alphabet_composition

        elif (desc == 'motif_composition'):
            if (getattr(self, desc).empty):
                self.get_motif_composition()
            desc_encoding = self.motif_composition

        elif (desc == 'amino_acid_pair_composition'):
            if (getattr(self, desc).empty):
                self.get_amino_acid_pair_composition()
            desc_encoding = self.amino_acid_pair_composition

        elif (desc == 'aliphatic_index'):
            if (getattr(self, desc).empty):
                self.get_aliphatic_index()
            desc_encoding = self.aliphatic_index

        elif (desc == 'extinction_coefficient'):
            if (getattr(self, desc).empty):
                self.get_extinction_coefficient()
            desc_encoding = self.extinction_coefficient

        elif (desc == 'boman_index'):
            if (getattr(self, desc).empty):
                self.get_boman_index()
            desc_encoding = self.boman_index

        elif (desc == 'aggregation_propensity'):
            if (getattr(self, desc).empty):
                self.get_aggregation_propensity()
            desc_encoding = self.aggregation_propensity

        elif (desc == 'hydrophobic_moment'):
            if (getattr(self, desc).empty):
                self.get_hydrophobic_moment()
            desc_encoding = self.hydrophobic_moment

        elif (desc == 'shannon_entropy'):
            if (getattr(self, desc).empty):
                self.get_shannon_entropy()
            desc_encoding = self.shannon_entropy

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

    def all_descriptors_list(self, desc_combo: int = 1) -> Union[List[str], List[Tuple[str, ...]]]:
       """
       Get list of all available descriptor attributes. Using the desc_combo
       input parameter you can get the list of all descriptors, all combinations
       of 2 descriptors or all combinations of 3 descriptors. Default of 1 will
       mean a list of all available descriptor attributes will be returned. With 
       there being 33 descriptors, 528 and 5456 combinations of 2 and 3 descriptors
       will be returned if desc_combo=2 or desc_combo=3, respectively.

       Parameters
       ==========
       :desc_combo: int (default=1)
            combination of descriptors to return. A value of 2 or 3 will return
            all combinations of 2 or 3 descriptor attributes etc.

       Returns
       =======
       :all_descriptors: List[str] or List[Tuple[str, ...]]
            list of available descriptor attributes, or list of tuples of descriptor combinations.
       """
       #filter out class attributes that are not any of the desired descriptors
       all_descriptors = [k[1:] for k in self.__dict__.keys()
                          if k.startswith('_') and not k.startswith('_all_desc')]

       #get all combinations of 2 or 3 descriptors
       if (desc_combo == 2):
           all_descriptors = list(itertools.combinations(all_descriptors, 2))
       elif (desc_combo == 3):
           all_descriptors = list(itertools.combinations(all_descriptors, 3))
       else:
           pass     #if desc_combo not equal to 2 or 3 then use default all_descriptors

       return all_descriptors

    def _calculate_descriptor_batch(self,
                                   descriptor_func: Callable,
                                   desc_name: str = "",
                                   **kwargs) -> pd.DataFrame:
        """
        Generic helper method to calculate descriptors for all sequences, preventing code repetition.
        Uses self.n_jobs threads to parallelise across sequences when n_jobs > 1.

        Parameters
        ==========
        :descriptor_func: Callable
            Function to calculate descriptor (e.g., protpy.amino_acid_composition)
        :desc_name: str
            Name of descriptor for progress tracking
        :kwargs: dict
            Additional keyword arguments to pass to descriptor function

        Returns
        =======
        :pd.DataFrame
            Dataframe with calculated descriptor values for all sequences
        """
        seqs = list(self.protein_seqs)

        if self.n_jobs <= 1:
            iterator = tqdm(seqs, desc=f"Computing {desc_name}", ncols=90) if desc_name else seqs
            # accumulate results in a list to avoid O(n²) repeated concat
            desc_list = [descriptor_func(seq, **kwargs) for seq in iterator]
        else:
            desc_list = [None] * len(seqs)
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(descriptor_func, seq, **kwargs): i
                           for i, seq in enumerate(seqs)}
                progress = tqdm(as_completed(futures), total=len(seqs),
                                desc=f"Computing {desc_name}", ncols=90) if desc_name else as_completed(futures)
                for future in progress:
                    i = futures[future]
                    desc_list[i] = future.result()

        return pd.concat(desc_list, ignore_index=False).reset_index(drop=True)
        
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
    def gravy(self):
        return self._gravy

    @gravy.setter
    def gravy(self, val):
        self._gravy = val

    @property
    def aromaticity(self):
        return self._aromaticity

    @aromaticity.setter
    def aromaticity(self, val):
        self._aromaticity = val

    @property
    def instability_index(self):
        return self._instability_index

    @instability_index.setter
    def instability_index(self, val):
        self._instability_index = val

    @property
    def isoelectric_point(self):
        return self._isoelectric_point

    @isoelectric_point.setter
    def isoelectric_point(self, val):
        self._isoelectric_point = val

    @property
    def molecular_weight(self):
        return self._molecular_weight

    @molecular_weight.setter
    def molecular_weight(self, val):
        self._molecular_weight = val

    @property
    def charge_distribution(self):
        return self._charge_distribution

    @charge_distribution.setter
    def charge_distribution(self, val):
        self._charge_distribution = val

    @property
    def hydrophobic_polar_charged_composition(self):
        return self._hydrophobic_polar_charged_composition

    @hydrophobic_polar_charged_composition.setter
    def hydrophobic_polar_charged_composition(self, val):
        self._hydrophobic_polar_charged_composition = val

    @property
    def secondary_structure_propensity(self):
        return self._secondary_structure_propensity

    @secondary_structure_propensity.setter
    def secondary_structure_propensity(self, val):
        self._secondary_structure_propensity = val

    @property
    def kmer_composition(self):
        return self._kmer_composition

    @kmer_composition.setter
    def kmer_composition(self, val):
        self._kmer_composition = val

    @property
    def reduced_alphabet_composition(self):
        return self._reduced_alphabet_composition

    @reduced_alphabet_composition.setter
    def reduced_alphabet_composition(self, val):
        self._reduced_alphabet_composition = val

    @property
    def motif_composition(self):
        return self._motif_composition

    @motif_composition.setter
    def motif_composition(self, val):
        self._motif_composition = val

    @property
    def amino_acid_pair_composition(self):
        return self._amino_acid_pair_composition

    @amino_acid_pair_composition.setter
    def amino_acid_pair_composition(self, val):
        self._amino_acid_pair_composition = val

    @property
    def aliphatic_index(self):
        return self._aliphatic_index

    @aliphatic_index.setter
    def aliphatic_index(self, val):
        self._aliphatic_index = val

    @property
    def extinction_coefficient(self):
        return self._extinction_coefficient

    @extinction_coefficient.setter
    def extinction_coefficient(self, val):
        self._extinction_coefficient = val

    @property
    def boman_index(self):
        return self._boman_index

    @boman_index.setter
    def boman_index(self, val):
        self._boman_index = val

    @property
    def aggregation_propensity(self):
        return self._aggregation_propensity

    @aggregation_propensity.setter
    def aggregation_propensity(self, val):
        self._aggregation_propensity = val

    @property
    def hydrophobic_moment(self):
        return self._hydrophobic_moment

    @hydrophobic_moment.setter
    def hydrophobic_moment(self, val):
        self._hydrophobic_moment = val

    @property
    def shannon_entropy(self):
        return self._shannon_entropy

    @shannon_entropy.setter
    def shannon_entropy(self, val):
        self._shannon_entropy = val

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
        del self._gravy
        del self._aromaticity
        del self._instability_index
        del self._isoelectric_point
        del self._molecular_weight
        del self._charge_distribution
        del self._hydrophobic_polar_charged_composition
        del self._secondary_structure_propensity
        del self._kmer_composition
        del self._reduced_alphabet_composition
        del self._motif_composition
        del self._amino_acid_pair_composition
        del self._aliphatic_index
        del self._extinction_coefficient
        del self._boman_index
        del self._aggregation_propensity
        del self._hydrophobic_moment
        del self._shannon_entropy
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

    def __str__(self) -> str:
        return f'''{self.shape}
Amino Acid Composition: {self.amino_acid_composition.shape}
Dipeptide Composition: {self.dipeptide_composition.shape}
Tripeptide Composition: {self.tripeptide_composition.shape}
GRAVY: {self.gravy.shape}
Aromaticity: {self.aromaticity.shape}
Instability Index: {self.instability_index.shape}
Isoelectric Point: {self.isoelectric_point.shape}
Molecular Weight: {self.molecular_weight.shape}
Charge Distribution: {self.charge_distribution.shape}
Hydrophobic/Polar/Charged Composition: {self.hydrophobic_polar_charged_composition.shape}
Secondary Structure Propensity: {self.secondary_structure_propensity.shape}
k-mer Composition: {self.kmer_composition.shape}
Reduced Alphabet Composition: {self.reduced_alphabet_composition.shape}
Motif Composition: {self.motif_composition.shape}
Amino Acid Pair Composition: {self.amino_acid_pair_composition.shape}
Aliphatic Index: {self.aliphatic_index.shape}
Extinction Coefficient: {self.extinction_coefficient.shape}
Boman Index: {self.boman_index.shape}
Aggregation Propensity: {self.aggregation_propensity.shape}
Hydrophobic Moment: {self.hydrophobic_moment.shape}
Shannon Entropy: {self.shannon_entropy.shape}
MoreauBroto Autocorrelation: {self.moreaubroto_autocorrelation.shape}
Moran Autocorrelation: {self.moran_autocorrelation.shape}
Geary Autocorrelation: {self.geary_autocorrelation.shape}
CTD: {self.ctd.shape}
Conjoint Triad: {self.conjoint_triad.shape}
Sequence Order Coupling Number: {self.sequence_order_coupling_number.shape}
Quasi Sequence Order: {self.quasi_sequence_order.shape}
Pseudo Amino Acid Composition: {self.pseudo_amino_acid_composition.shape}
Amphiphilic Pseudo Amino Acid Composition: {self.amphiphilic_pseudo_amino_acid_composition.shape}'''

    def get_all_descriptors(self, export: bool = False, descriptors_export_filename: str = "",
                            descriptors: Optional[List[str]] = None, verbose: bool = False,
                            sequence_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate all individual descriptor values and return them as one concatenated
        DataFrame.

        ``ctd`` is intentionally excluded from the default concatenation because its
        columns are fully covered by ``ctd_composition``, ``ctd_transition`` and
        ``ctd_distribution``; including it would produce duplicate columns in the output.
        Users can still call ``get_ctd()`` directly to obtain the combined CTD frame.

        Parameters
        ==========
        :export: bool (default=False)
            If True, write the concatenated DataFrame to a CSV file.  Pre-calculating
            and exporting descriptors is recommended so they don't need to be
            recomputed on every run; import them later via ``import_descriptors()``.
        :descriptors_export_filename: str
            Path/filename for the exported CSV.  Falls back to the ``descriptors_csv``
            config parameter, then to ``"descriptors_output.csv"`` if both are empty.
        :descriptors: list of str, optional
            Specific subset of descriptor names to calculate.  If None, all valid
            descriptors except ``'ctd'`` are calculated.
        :verbose: bool (default=False)
            If True, print progress messages and timing information during calculation.
        :sequence_col: str, optional (default=None)
            Column name in the dataset to prepend as the first column of the output
            DataFrame/CSV, so each row can be identified (e.g. ``'name'`` for the
            thermostability dataset).  Requires the class to have been initialised
            with a dataset file.  Ignored if None.

        Returns
        =======
        :all_descriptor_df: pd.DataFrame
            Concatenated DataFrame of all requested descriptor values.  Using default
            config attributes the output will be of shape N x 10572+, where N is the
            number of protein sequences.
        """
        start = time.time()

        if descriptors is None:
            # Exclude 'ctd' — its features are fully covered by ctd_composition,
            # ctd_transition, and ctd_distribution, so including it would duplicate
            # all CTD columns in the exported CSV.
            descriptors = [d for d in self.valid_descriptors if d != 'ctd']
        else:
            descriptors = self.validate_descriptors(descriptors)

        if verbose:
            print(f'Calculating {len(descriptors)} descriptors for {len(self.sequences)} sequences '
                  f'(n_jobs={self.n_jobs})...')

        if self.n_jobs > 1:
            pending = [(desc, getattr(self, f'get_{desc}')) for desc in descriptors
                       if getattr(self, desc).empty]
            if verbose:
                print(f'{len(pending)} descriptors need computing ({len(descriptors) - len(pending)} already cached).')
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(getter): desc for desc, getter in pending}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   unit=" descriptor", desc="Descriptors", ncols=90,
                                   disable=not verbose):
                    desc = futures[future]
                    setattr(self, desc, future.result())
        else:
            for i, desc in enumerate(tqdm(descriptors, unit=" descriptor", position=0,
                                          desc="Descriptors", mininterval=30, ncols=90,
                                          disable=not verbose), start=1):
                if getattr(self, desc).empty:
                    if verbose:
                        print(f'  [{i}/{len(descriptors)}] Computing {desc}...')
                    getattr(self, f'get_{desc}')()

        elapsed = time.time() - start
        if verbose:
            print(f'All descriptors calculated in {elapsed/60:.2f} minutes.')

        all_descriptor_df = pd.concat(
            [getattr(self, desc) for desc in descriptors],
            axis=1
        )
        self.all_descriptors = all_descriptor_df

        if sequence_col is not None:
            if not self.dataset_filepath or not os.path.isfile(self.dataset_filepath):
                raise DescriptorError(
                    f"sequence_col='{sequence_col}' requires a dataset file, but none is available.")
            dataset = pd.read_csv(self.dataset_filepath, sep=",", header=0)
            if sequence_col not in dataset.columns:
                raise ValueError(
                    f"Column '{sequence_col}' not found in dataset. "
                    f"Available columns: {list(dataset.columns)}")
            id_col = dataset[sequence_col].reset_index(drop=True)
            all_descriptor_df = pd.concat([id_col, all_descriptor_df.reset_index(drop=True)], axis=1)

        if export:
            out_path = descriptors_export_filename or self.descriptors_csv or "descriptors_output.csv"
            if os.path.splitext(os.path.basename(out_path))[1] == "":
                out_path += ".csv"
            all_descriptor_df.to_csv(out_path, index=False)

        return all_descriptor_df

    def get_descriptor_info(self, descriptor_name: str) -> Dict[str, Any]:
        """
        Get metadata and information about a specific descriptor.
        
        Parameters
        ==========
        :descriptor_name: str
            Name of the descriptor
        
        Returns
        =======
        :Dict[str, Any]
            Dictionary with descriptor metadata including name, feature count, group, and parameters
        """
        self.validate_descriptors(descriptor_name)
        
        descriptor_info = {
            'name': descriptor_name,
            'group': self.descriptor_groups.get(descriptor_name, 'Unknown'),
            'feature_count': self.descriptor_feature_count.get(descriptor_name, 0),
            'parameters': {},
        }
        
        # Add parameters if available
        if hasattr(self.desc_parameters, descriptor_name):
            parameters = getattr(self.desc_parameters, descriptor_name)
            if isinstance(parameters, dict):
                descriptor_info['parameters'] = dict(parameters)
            elif hasattr(parameters, '__dict__'):
                descriptor_info['parameters'] = vars(parameters)
        
        return descriptor_info

    def reset_descriptors(self) -> None:
        """
        Reset all descriptor attributes to empty DataFrames.
        Clears all calculated descriptor values without affecting configuration.
        
        Parameters
        ==========
        None
        
        Returns
        =======
        None
        """
        self._init_descriptor_attrs()

    def clear_cache(self) -> None:
        """
        Clear cached descriptor metadata to free memory.
        Useful after major descriptor calculations or when memory is constrained.
        
        Parameters
        ==========
        None
        
        Returns
        =======
        None
        """
        if hasattr(self.descriptor_feature_count, 'cache_clear'):
            self.descriptor_feature_count.fget.cache_clear()

    def get_descriptor_columns(self, descriptor: str) -> List[str]:
        """
        Get list of column names for a specific descriptor.
        
        Parameters
        ==========
        :descriptor: str
            Name of the descriptor (e.g., 'amino_acid_composition')
        
        Returns
        =======
        :List[str]
            List of column names in the descriptor DataFrame
        
        Raises
        ======
        :InvalidDescriptorError
            If descriptor name is invalid
        :ValueError
            If descriptor has not been calculated yet
        """
        # Validate descriptor name
        self.validate_descriptors(descriptor)
        
        # Get the descriptor dataframe attribute
        desc_attr = getattr(self, descriptor, None)
        
        if desc_attr is None or desc_attr.empty:
            raise ValueError(f"Descriptor '{descriptor}' has not been calculated yet. "
                           f"Call get_{descriptor}() first.")
        
        return desc_attr.columns.tolist()

    def __repr__(self) -> str:
        return f'<Descriptor: {self}>'

    def __len__(self) -> int:
        return len(self.all_descriptors)

    def __shape__(self) -> Tuple[int, int]:
        return self.all_descriptors.shape

    def __sizeof__(self) -> int:
        """ Get size of all_descriptors object that stores all descriptor values. """
        return self.all_descriptors.__sizeof__()

class DescriptorError(Exception):
    """Base exception for descriptor operations."""
    pass


class InvalidSequenceError(DescriptorError):
    """Raised when sequence contains invalid amino acids."""
    pass


class DescriptorConfigError(DescriptorError):
    """Raised when config file is invalid or malformed."""
    pass


class InvalidDescriptorError(DescriptorError):
    """Raised when requesting non-existent descriptor."""
    pass