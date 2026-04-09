Descriptors
===========

The ``Descriptors`` class (``pySAR/descriptors.py``) calculates a comprehensive set of
physicochemical, biochemical, and structural protein descriptors. These 33 descriptors
span composition, autocorrelation, CTD, conjoint triad, sequence order, and pseudo
amino acid composition and produce over 10,000 features in total when all are calculated.

Descriptors are calculated via `protpy <https://github.com/amckenna41/protpy>`_, a
purpose-built open-source package for protein feature engineering. Input sequences must
contain only the 20 canonical amino acids; gaps are stripped automatically on
initialisation.

.. code-block:: python

    from pySAR.descriptors import Descriptors

    desc = Descriptors(config_file="config/thermostability.json")

    # calculate a single descriptor
    aa_comp = desc.get_amino_acid_composition()   # shape: (N, 20)

    # calculate all descriptors at once
    all_desc = desc.get_all_descriptors()         # shape: (N, 10572+)

----

Instantiation
-------------

``Descriptors.__init__(config_file, protein_seqs=None, **kwargs)``

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``config_file``
     - —
     - Path to the JSON configuration file. The ``.json`` extension is appended automatically if omitted.
   * - ``protein_seqs``
     - ``None``
     - Protein sequences as a ``pd.Series`` or a single string. If ``None`` or empty, sequences are loaded from the dataset path specified in the config.
   * - ``**kwargs``
     - —
     - Keyword arguments (``dataset``, ``descriptors_csv``) that override the corresponding config file values.

On construction the class:

1. Parses the config JSON and loads dataset/descriptor parameters.
2. Reads protein sequences from the dataset CSV if not directly supplied.
3. Strips gaps and validates all sequences against the 20 canonical amino acids.
4. Attempts to import pre-calculated descriptor values from the ``descriptors_csv`` path, if it exists.

Importing pre-calculated descriptors is strongly recommended for large datasets — set
``all_desc: 1`` in the ``[descriptors]`` config section on first run to generate the
CSV, then subsequent runs will load from it directly without recalculating.

----

Descriptor Groups
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Descriptors
   * - Composition
     - ``amino_acid_composition``, ``dipeptide_composition``, ``tripeptide_composition``, ``gravy``, ``aromaticity``, ``instability_index``, ``isoelectric_point``, ``molecular_weight``, ``charge_distribution``, ``hydrophobic_polar_charged_composition``, ``secondary_structure_propensity``, ``kmer_composition``, ``reduced_alphabet_composition``, ``motif_composition``, ``amino_acid_pair_composition``, ``aliphatic_index``, ``extinction_coefficient``, ``boman_index``, ``aggregation_propensity``, ``hydrophobic_moment``, ``shannon_entropy``
   * - Autocorrelation
     - ``moreaubroto_autocorrelation``, ``moran_autocorrelation``, ``geary_autocorrelation``
   * - CTD
     - ``ctd``, ``ctd_composition``, ``ctd_transition``, ``ctd_distribution``
   * - Conjoint Triad
     - ``conjoint_triad``
   * - Sequence Order
     - ``sequence_order_coupling_number``, ``quasi_sequence_order``
   * - Pseudo Composition
     - ``pseudo_amino_acid_composition``, ``amphiphilic_pseudo_amino_acid_composition``

----

Composition Descriptors
-----------------------

Composition descriptors capture the amino acid content and physicochemical properties of
a sequence without considering positional information.

Amino Acid Composition
~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_amino_acid_composition()`` | **Features:** 20

The fraction of each of the 20 canonical amino acid types within a sequence:

.. math::

   \text{AAComp}(t) = \frac{AA(t)}{N}

where $AA(t)$ is the count of amino acid type $t$ and $N$ is the total sequence length.

.. code-block:: python

    aa_comp = desc.get_amino_acid_composition()   # shape: (N, 20)

Dipeptide Composition
~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_dipeptide_composition()`` | **Features:** 400 (20²)

The fraction of each of the 400 possible dipeptide types:

.. math::

   \text{DPComp}(s,t) = \frac{AA(s,t)}{N - 1}

where $AA(s,t)$ is the count of dipeptide type $(s, t)$ and $N-1$ is the total number
of dipeptides in the sequence.

.. code-block:: python

    dp_comp = desc.get_dipeptide_composition()    # shape: (N, 400)

Tripeptide Composition
~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_tripeptide_composition()`` | **Features:** 8000 (20³)

The fraction of each of the 8,000 possible tripeptide types. Computationally
expensive on large datasets; pre-calculation and CSV caching is recommended.

.. code-block:: python

    tp_comp = desc.get_tripeptide_composition()   # shape: (N, 8000)

GRAVY
~~~~~

**Method:** ``get_gravy()`` | **Features:** 1

The Grand Average of Hydropathy (GRAVY) is the mean Kyte-Doolittle hydropathy score
across all residues. Positive values indicate overall hydrophobicity; negative values
indicate hydrophilicity.

.. code-block:: python

    gravy = desc.get_gravy()   # shape: (N, 1)

Aromaticity
~~~~~~~~~~~

**Method:** ``get_aromaticity()`` | **Features:** 1

Fraction of aromatic residues (F, W, Y, H) in the sequence.

.. code-block:: python

    arom = desc.get_aromaticity()   # shape: (N, 1)

Instability Index
~~~~~~~~~~~~~~~~~

**Method:** ``get_instability_index()`` | **Features:** 1

Computed from dipeptide instability weight values (DIWV). A value below 40 indicates
a stable protein; 40 or above suggests instability.

.. code-block:: python

    ii = desc.get_instability_index()   # shape: (N, 1)

Isoelectric Point
~~~~~~~~~~~~~~~~~

**Method:** ``get_isoelectric_point()`` | **Features:** 1

The estimated pH at which the protein carries no net charge, calculated iteratively
using standard pK\ :sub:`a` values for ionisable residues.

.. code-block:: python

    pi = desc.get_isoelectric_point()   # shape: (N, 1)

Molecular Weight
~~~~~~~~~~~~~~~~

**Method:** ``get_molecular_weight()`` | **Features:** 1

Average molecular weight (Da) calculated from residue masses, corrected for water
lost at each peptide bond.

.. code-block:: python

    mw = desc.get_molecular_weight()   # shape: (N, 1)

Charge Distribution
~~~~~~~~~~~~~~~~~~~

**Method:** ``get_charge_distribution()`` | **Features:** 3

Positive, negative, and net charge contributions of ionisable residues at a specified
pH using the Henderson-Hasselbalch equation (default pH 7.4). Output columns:
``PositiveCharge``, ``NegativeCharge``, ``NetCharge``.

Config parameter: ``charge_distribution.ph`` (default 7.4).

.. code-block:: python

    charge = desc.get_charge_distribution()   # shape: (N, 3)

Hydrophobic/Polar/Charged Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_hydrophobic_polar_charged_composition()`` | **Features:** 3

Percentage of residues belonging to each of three physicochemical groups:

- **Hydrophobic:** A, C, F, I, L, M, V, W, Y
- **Polar:** G, N, Q, S, T
- **Charged:** D, E, H, K, R

Output columns: ``Hydrophobic``, ``Polar``, ``Charged``.

.. code-block:: python

    hpc = desc.get_hydrophobic_polar_charged_composition()   # shape: (N, 3)

Secondary Structure Propensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_secondary_structure_propensity()`` | **Features:** 3

Average Chou-Fasman propensity values for alpha-helix, beta-sheet, and random-coil
conformations across all residues. Output columns: ``Helix``, ``Sheet``, ``Coil``.

.. code-block:: python

    ssp = desc.get_secondary_structure_propensity()   # shape: (N, 3)

k-mer Composition
~~~~~~~~~~~~~~~~~

**Method:** ``get_kmer_composition()`` | **Features:** 20\ :sup:`k` (default 400)

Frequency of all possible k-length residue subsequences expressed as a percentage of
total k-mers. Config parameter: ``kmer_composition.k`` (default 2, producing 400 features).

.. code-block:: python

    kmer = desc.get_kmer_composition()   # shape: (N, 400) with k=2

Reduced Alphabet Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_reduced_alphabet_composition()`` | **Features:** ``alphabet_size`` (default 6)

Amino acid composition after mapping residues to a reduced set of physicochemical
groups. Supported alphabet sizes: 2, 3, 4, 6. Config parameter:
``reduced_alphabet_composition.alphabet_size`` (default 6).

.. code-block:: python

    rac = desc.get_reduced_alphabet_composition()   # shape: (N, 6)

Motif Composition
~~~~~~~~~~~~~~~~~

**Method:** ``get_motif_composition()`` | **Features:** number of motifs (default 8)

Count of occurrences (including overlapping) of predefined biological sequence motifs,
matched by regular expression. Uses 8 built-in motifs by default; a custom
``name → pattern`` dict can be supplied via ``motif_composition.motifs`` in config.

.. code-block:: python

    motif = desc.get_motif_composition()   # shape: (N, 8)

Amino Acid Pair Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_amino_acid_pair_composition()`` | **Features:** 400

Frequency of all 400 residue-pair combinations, with column names annotated by the
physicochemical class of each residue.

.. code-block:: python

    pair = desc.get_amino_acid_pair_composition()   # shape: (N, 400)

Aliphatic Index
~~~~~~~~~~~~~~~

**Method:** ``get_aliphatic_index()`` | **Features:** 1

Relative volume occupied by aliphatic side chains (Ala, Val, Ile, Leu). Higher values
are associated with greater thermostability.

.. code-block:: python

    ai = desc.get_aliphatic_index()   # shape: (N, 1)

Extinction Coefficient
~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_extinction_coefficient()`` | **Features:** 2

Molar extinction coefficient at 280 nm derived from the number of Trp (W), Tyr (Y),
and Cys (C) residues. Reported for both reduced and oxidised states. Output columns:
``ExtCoeff_Reduced``, ``ExtCoeff_Oxidized``.

.. code-block:: python

    ec = desc.get_extinction_coefficient()   # shape: (N, 2)

Boman Index
~~~~~~~~~~~

**Method:** ``get_boman_index()`` | **Features:** 1

Sum of residue solubility values divided by sequence length. Predicts potential for
protein-protein interactions.

.. code-block:: python

    boman = desc.get_boman_index()   # shape: (N, 1)

Aggregation Propensity
~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_aggregation_propensity()`` | **Features:** 2

Identifies aggregation-prone regions via a sliding-window approach combining
Kyte-Doolittle hydrophobicity and charge neutrality. Output columns:
``AggregProneRegions`` (count of qualifying windows) and ``AggregProneFraction``
(fraction of sequence covered). Config parameters: ``aggregation_propensity.window``
(default 5), ``.hydrophobicity_threshold`` (default 2.0), ``.charge_threshold``
(default 1).

.. code-block:: python

    agg = desc.get_aggregation_propensity()   # shape: (N, 2)

Hydrophobic Moment
~~~~~~~~~~~~~~~~~~

**Method:** ``get_hydrophobic_moment()`` | **Features:** 2

Mean and maximum hydrophobic moment across sliding windows using the Eisenberg
hydrophobicity scale and a helical-wheel projection, capturing amphipathicity. Output
columns: ``HydrophobicMoment_Mean``, ``HydrophobicMoment_Max``. Config parameters:
``hydrophobic_moment.window`` (default 11), ``.angle`` (default 100).

.. code-block:: python

    hm = desc.get_hydrophobic_moment()   # shape: (N, 2)

Shannon Entropy
~~~~~~~~~~~~~~~

**Method:** ``get_shannon_entropy()`` | **Features:** 1

An information-theoretic measure of amino acid diversity:

.. math::

   H = -\sum_{i=1}^{20} p_i \log_2 p_i

A value of 0 indicates a completely repetitive sequence; the theoretical maximum of
~4.322 bits corresponds to a perfectly uniform distribution across all 20 amino acids.

.. code-block:: python

    se = desc.get_shannon_entropy()   # shape: (N, 1)

----

Autocorrelation Descriptors
----------------------------

Autocorrelation descriptors describe the level of correlation between two positions
in a sequence separated by a lag distance $d$, in terms of a specified physicochemical
property. Each of the three variants uses a different mathematical formulation. By
default, 8 physicochemical properties are used with a lag of 30, generating 240 features
per descriptor.

**Default properties (8):**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - AAIndex Accession
     - Property
   * - CIDH920105
     - Normalised Average Hydrophobicity
   * - BHAR880101
     - Average Flexibility Indices
   * - CHAM820101
     - Polarizability Parameter
   * - CHAM820102
     - Free Energy of Solution in Water (kcal/mol)
   * - CHOC760101
     - Residue Accessible Surface Area in Tripeptide
   * - BIGC670101
     - Residue Volume
   * - CHAM810101
     - Steric Parameter
   * - DAYM780201
     - Relative Mutability

Config parameters common to all three descriptors: ``lag`` (default 30),
``properties`` (list of AAIndex accession numbers), ``normalize`` (bool).

**Feature count formula:** ``lag × len(properties)`` → default 30 × 8 = **240**.

MoreauBroto Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_moreaubroto_autocorrelation()`` | **Features:** lag × properties (default 240)

Uses the raw property values of two residues separated by lag $d$:

.. math::

   \text{MBAuto}(d) = \sum_{i=1}^{N-d} P_i \cdot P_{i+d}

Config section: ``[moreaubroto_autocorrelation]``.

.. code-block:: python

    mb = desc.get_moreaubroto_autocorrelation()   # shape: (N, 240)

Moran Autocorrelation
~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_moran_autocorrelation()`` | **Features:** lag × properties (default 240)

Uses normalised deviations from the mean property value:

.. math::

   \text{MAuto}(d) = \frac{\frac{1}{N-d}\sum_{i=1}^{N-d}(P_i - \bar{P})(P_{i+d} - \bar{P})}{\frac{1}{N}\sum_{i=1}^{N}(P_i - \bar{P})^2}

Config section: ``[moran_autocorrelation]``.

.. code-block:: python

    moran = desc.get_moran_autocorrelation()   # shape: (N, 240)

Geary Autocorrelation
~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_geary_autocorrelation()`` | **Features:** lag × properties (default 240)

Uses squared differences between residue property values:

.. math::

   \text{GAuto}(d) = \frac{\frac{1}{2(N-d)}\sum_{i=1}^{N-d}(P_i - P_{i+d})^2}{\frac{1}{N-1}\sum_{i=1}^{N}(P_i - \bar{P})^2}

Config section: ``[geary_autocorrelation]``.

.. code-block:: python

    geary = desc.get_geary_autocorrelation()   # shape: (N, 240)

----

CTD Descriptors
---------------

CTD describes the amino acid composition within seven physicochemical property classes
(hydrophobicity, volume, polarity, polarisability, charge, secondary structure, solvent
accessibility). Each property divides the 20 amino acids into three classes (C1, C2, C3),
from which three sub-descriptors are computed.

Using all 7 properties generates **147 features** (21 per property).
A subset of properties can be specified via ``ctd.property`` in config.

CTD (Combined)
~~~~~~~~~~~~~~

**Method:** ``get_ctd()`` | **Features:** 147 (all 7 properties)

Contains all CTD sub-descriptors concatenated: Composition + Transition + Distribution.

.. code-block:: python

    ctd = desc.get_ctd()   # shape: (N, 147)

CTD Composition
~~~~~~~~~~~~~~~

**Method:** ``get_ctd_composition()`` | **Features:** 3 per property (21 total)

Fraction of residues in each of the three classes (C1, C2, C3) for each property.

.. code-block:: python

    ctd_c = desc.get_ctd_composition()   # shape: (N, 21)

CTD Transition
~~~~~~~~~~~~~~

**Method:** ``get_ctd_transition()`` | **Features:** 3 per property (21 total)

Fraction of transitions between pairs of property classes in the sequence
(C1↔C2, C1↔C3, C2↔C3).

.. code-block:: python

    ctd_t = desc.get_ctd_transition()   # shape: (N, 21)

CTD Distribution
~~~~~~~~~~~~~~~~

**Method:** ``get_ctd_distribution()`` | **Features:** 15 per property (105 total)

For each class, the sequence positions (as percentages of sequence length) of the 1st,
25th, 50th, 75th, and 100th occurrence of that class — capturing how each property
class is distributed along the sequence.

.. code-block:: python

    ctd_d = desc.get_ctd_distribution()   # shape: (N, 105)

----

Conjoint Triad
--------------

**Method:** ``get_conjoint_triad()`` | **Features:** 343 (7³)

Describes the neighbourhood environment of each residue by considering triplets of
adjacent residues, each residue grouped into one of 7 physicochemical classes. The
frequency of each of the 7³ = 343 possible triplet combinations is computed.

.. code-block:: python

    ct = desc.get_conjoint_triad()   # shape: (N, 343)

----

Sequence Order Descriptors
--------------------------

Sequence Order Coupling Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_sequence_order_coupling_number()`` | **Features:** ``lag`` or ``2 × lag``

Captures long-range interactions by summing the squared differences of a property
between residues $d$ positions apart up to a specified lag. If a single distance matrix
is given in config, ``lag`` features are produced; if no matrix is specified both the
Schneider-Wrede and Grantham matrices are used, producing ``2 × lag`` features.

Config section: ``[sequence_order_coupling_number]``, params: ``lag``, ``distance_matrix``.

.. code-block:: python

    socn = desc.get_sequence_order_coupling_number()

Quasi Sequence Order
~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_quasi_sequence_order()`` | **Features:** ``20 + lag`` or ``2 × (20 + lag)``

Extends amino acid composition with sequence-order correlation factors derived from
pairwise residue distance matrices. Feature count: ``20 + lag`` with one distance matrix,
or ``2 × (20 + lag)`` when both Schneider-Wrede and Grantham matrices are used.

Config section: ``[quasi_sequence_order]``, params: ``lag``, ``distance_matrix``.

.. code-block:: python

    qso = desc.get_quasi_sequence_order()

----

Pseudo Amino Acid Composition
------------------------------

Pseudo Amino Acid Composition (Type 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_pseudo_amino_acid_composition()`` | **Features:** ``20 + lambda``

Augments amino acid composition (20 features) with ``lambda`` sequence-order correlation
factors (correlation along the chain at lags 1 through ``lambda``), capturing both
composition and sequence-order information. Config section:
``[pseudo_amino_acid_composition]``, param: ``lambda``.

.. code-block:: python

    paac = desc.get_pseudo_amino_acid_composition()   # shape: (N, 20+lambda)

Amphiphilic Pseudo Amino Acid Composition (Type 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``get_amphiphilic_pseudo_amino_acid_composition()`` | **Features:** ``20 + 2 × lambda``

Extends PseAAC Type 1 by adding separate hydrophobicity and hydrophilicity correlation
factors for each lag, producing ``20 + 2 × lambda`` features. Designed to capture
amphipathic patterns. Config section:
``[amphiphilic_pseudo_amino_acid_composition]``, param: ``lambda``.

.. code-block:: python

    apaac = desc.get_amphiphilic_pseudo_amino_acid_composition()   # shape: (N, 20+(2*lambda))

----

All Descriptors Summary
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Descriptor
     - Features
     - Method
   * - Amino Acid Composition
     - 20
     - ``get_amino_acid_composition()``
   * - Dipeptide Composition
     - 400
     - ``get_dipeptide_composition()``
   * - Tripeptide Composition
     - 8000
     - ``get_tripeptide_composition()``
   * - GRAVY
     - 1
     - ``get_gravy()``
   * - Aromaticity
     - 1
     - ``get_aromaticity()``
   * - Instability Index
     - 1
     - ``get_instability_index()``
   * - Isoelectric Point
     - 1
     - ``get_isoelectric_point()``
   * - Molecular Weight
     - 1
     - ``get_molecular_weight()``
   * - Charge Distribution
     - 3
     - ``get_charge_distribution()``
   * - Hydrophobic/Polar/Charged Composition
     - 3
     - ``get_hydrophobic_polar_charged_composition()``
   * - Secondary Structure Propensity
     - 3
     - ``get_secondary_structure_propensity()``
   * - k-mer Composition
     - 20\ :sup:`k` (default 400)
     - ``get_kmer_composition()``
   * - Reduced Alphabet Composition
     - alphabet_size (default 6)
     - ``get_reduced_alphabet_composition()``
   * - Motif Composition
     - len(motifs) (default 8)
     - ``get_motif_composition()``
   * - Amino Acid Pair Composition
     - 400
     - ``get_amino_acid_pair_composition()``
   * - Aliphatic Index
     - 1
     - ``get_aliphatic_index()``
   * - Extinction Coefficient
     - 2
     - ``get_extinction_coefficient()``
   * - Boman Index
     - 1
     - ``get_boman_index()``
   * - Aggregation Propensity
     - 2
     - ``get_aggregation_propensity()``
   * - Hydrophobic Moment
     - 2
     - ``get_hydrophobic_moment()``
   * - Shannon Entropy
     - 1
     - ``get_shannon_entropy()``
   * - MoreauBroto Autocorrelation
     - lag × props (default 240)
     - ``get_moreaubroto_autocorrelation()``
   * - Moran Autocorrelation
     - lag × props (default 240)
     - ``get_moran_autocorrelation()``
   * - Geary Autocorrelation
     - lag × props (default 240)
     - ``get_geary_autocorrelation()``
   * - CTD
     - 147
     - ``get_ctd()``
   * - CTD Composition
     - 21
     - ``get_ctd_composition()``
   * - CTD Transition
     - 21
     - ``get_ctd_transition()``
   * - CTD Distribution
     - 105
     - ``get_ctd_distribution()``
   * - Conjoint Triad
     - 343
     - ``get_conjoint_triad()``
   * - Sequence Order Coupling Number
     - lag or 2×lag
     - ``get_sequence_order_coupling_number()``
   * - Quasi Sequence Order
     - 20+λ or 2×(20+λ)
     - ``get_quasi_sequence_order()``
   * - Pseudo Amino Acid Composition
     - 20+λ
     - ``get_pseudo_amino_acid_composition()``
   * - Amphiphilic Pseudo Amino Acid Composition
     - 20+2λ
     - ``get_amphiphilic_pseudo_amino_acid_composition()``

----

Utility Methods
---------------

``get_all_descriptors()``
    Calculates every descriptor in sequence and returns a concatenated DataFrame of all
    features. Also exports to the ``descriptors_csv`` path if configured.

    .. code-block:: python

        all_desc = desc.get_all_descriptors()   # shape: (N, ~10572 with defaults)

``get_descriptor_encoding(descriptor)``
    Resolves a descriptor name (with fuzzy matching) and returns its feature DataFrame.
    Useful when the descriptor name is read from config or supplied at runtime.

    .. code-block:: python

        df = desc.get_descriptor_encoding("moran")   # resolves to moran_autocorrelation

``all_descriptors_list()``
    Returns the list of all 33 descriptor names.

``validate_descriptors(descriptors)``
    Validates that all names in a list (or single string) are recognised descriptor
    names. Raises ``InvalidDescriptorError`` for any unknown names.

``get_descriptor_info(name)``
    Returns a metadata dict for ``name`` including ``feature_count``, ``group``, and
    the associated ``get_*`` method.

``reset_descriptors()``
    Clears all descriptor DataFrames back to empty state, freeing memory without
    re-instantiating the class.

``get_descriptor_columns(name)``
    Returns the column names of the calculated DataFrame for descriptor ``name``.

----

Pre-calculated Descriptors
---------------------------

For any new dataset it is recommended to calculate all descriptors once and cache them
to a CSV file, which is then loaded automatically on subsequent runs:

1. Set ``all_desc: 1`` and ``descriptors_csv: "data/descriptors_<dataset>.csv"`` in the
   ``[descriptors]`` config section.
2. Run once — all descriptor values are calculated and written to the CSV.
3. On every subsequent run, the CSV is detected and imported automatically —
   no recalculation required.

Pre-calculated descriptor CSVs for the bundled example datasets are included in
``data/`` and ``example_datasets/``.

----

Config File
-----------

All descriptor parameters are set under the ``[descriptors]`` key in the pySAR config JSON:

.. code-block:: json

    {
        "descriptors": {
            "descriptors_csv": "data/descriptors_thermostability.csv",
            "all_desc": 1,
            "descriptor": "amino_acid_composition",
            "moreaubroto_autocorrelation": {
                "lag": 30,
                "properties": ["CIDH920105","BHAR880101","CHAM820101","CHAM820102",
                               "CHOC760101","BIGC670101","CHAM810101","DAYM780201"],
                "normalize": 0
            },
            "moran_autocorrelation": {
                "lag": 30,
                "properties": ["CIDH920105","BHAR880101","CHAM820101","CHAM820102",
                               "CHOC760101","BIGC670101","CHAM810101","DAYM780201"],
                "normalize": 0
            },
            "geary_autocorrelation": {
                "lag": 30,
                "properties": ["CIDH920105","BHAR880101","CHAM820101","CHAM820102",
                               "CHOC760101","BIGC670101","CHAM810101","DAYM780201"],
                "normalize": 0
            },
            "ctd": {
                "property": ["hydrophobicity","volume","polarity","polarizability",
                             "charge","secondaryStructure","solventAccessibility"],
                "all": 1
            },
            "conjoint_triad": {},
            "sequence_order_coupling_number": {
                "lag": 30,
                "distance_matrix": ""
            },
            "quasi_sequence_order": {
                "lag": 30,
                "distance_matrix": ""
            },
            "pseudo_amino_acid_composition": {
                "lambda": 30,
                "weight": 0.05
            },
            "amphiphilic_pseudo_amino_acid_composition": {
                "lambda": 30,
                "weight": 0.05
            }
        }
    }

See the `CONFIG.md <https://github.com/amckenna41/pySAR/blob/master/CONFIG.md>`_ file
and the example config files for the full list of available parameters:

- `thermostability.json <https://github.com/amckenna41/pySAR/blob/master/config/thermostability.json>`_
- `absorption.json <https://github.com/amckenna41/pySAR/blob/master/config/absorption.json>`_
- `enantioselectivity.json <https://github.com/amckenna41/pySAR/blob/master/config/enantioselectivity.json>`_
- `localization.json <https://github.com/amckenna41/pySAR/blob/master/config/localization.json>`_
