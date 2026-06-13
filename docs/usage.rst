Usage
=====

.. _installation:

Installation
------------

Install the latest release via ``pip``:

.. code-block:: console

   pip install pySAR

Alternatively, clone the repository and install from source:

.. code-block:: console

   git clone -b master https://github.com/amckenna41/pySAR.git
   cd pySAR
   pip install .

Configuration Files
-------------------

pySAR is driven by JSON configuration files. Each dataset requires its own config file that
specifies the dataset path, activity column, encoding parameters, and descriptor parameters.
The config files are stored in the ``config/`` directory of the project. See
`CONFIG.md <https://github.com/amckenna41/pySAR/blob/master/CONFIG.md>`_
for a full description of all available parameters.

Config files are passed to the ``PySAR``, ``Encoding``, or ``Descriptors`` classes via the
``config_file`` parameter. All parameter **names** must remain unchanged; only their **values**
should be edited. Any unused parameter can be set to ``null``. Parameters can alternatively be
passed directly as ``**kwargs`` to each class.

Four example config files are provided in the
`config/ <https://github.com/amckenna41/pySAR/tree/master/config>`_ directory, one per
supported dataset:
`thermostability.json <https://github.com/amckenna41/pySAR/blob/master/config/thermostability.json>`_,
`enantioselectivity.json <https://github.com/amckenna41/pySAR/blob/master/config/enantioselectivity.json>`_,
`localization.json <https://github.com/amckenna41/pySAR/blob/master/config/localization.json>`_,
and
`absorption.json <https://github.com/amckenna41/pySAR/blob/master/config/absorption.json>`_.

The config file is divided into four top-level sections:

**dataset**
  Defines the input data. ``dataset`` is the path to the sequence/activity file; ``sequence_col``
  names the column holding protein sequences; ``activity`` names the target activity column.

**model**
  Specifies the regression algorithm (e.g. ``plsregression``, ``randomforest``, ``svr``),
  optional hyperparameters, and the train/test split ratio (``test_split``, default ``0.2``).

**descriptors**
  Controls which protein descriptors are calculated and their metaparameters (lag values,
  properties, window sizes, etc.). ``descriptors_csv`` can point to a pre-calculated descriptor
  CSV to skip recomputation on repeated runs.

**pyDSP**
  Governs optional Digital Signal Processing applied to AAI-encoded sequences before model
  training. Set ``use_dsp`` to ``1`` to enable; then configure the ``spectrum`` type
  (``power``, ``absolute``, ``real``, ``imaginary``), a convolutional ``window``
  (e.g. ``hamming``, ``blackman``, ``gaussian``), and an optional ``filter``
  (e.g. ``savgol``, ``medfilt``).

A full configuration file looks like:

.. code-block:: json

   {
     "dataset": {
       "dataset": "thermostability.txt",
       "sequence_col": "sequence",
       "activity": "T50"
     },
     "model": {
       "algorithm": "plsregression",
       "parameters": "",
       "test_split": 0.2
     },
     "descriptors": {
       "descriptors_csv": "descriptors_thermostability.csv",
       "moreaubroto_autocorrelation": {
         "lag": 30,
         "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
                        "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
         "normalize": 1
       },
       "ctd": {
         "property": "hydrophobicity",
         "all": 1
       },
       "pseudo_amino_acid_composition": {
         "lambda": 30,
         "weight": 0.05,
         "properties": []
       },
       "charge_distribution": { "ph": 7.4 },
       "kmer_composition": { "k": 2 },
       "reduced_alphabet_composition": { "alphabet_size": 6 },
       "motif_composition": { "motifs": null },
       "aggregation_propensity": {
         "window": 5,
         "hydrophobicity_threshold": 2.0,
         "charge_threshold": 1
       },
       "hydrophobic_moment": { "window": 11, "angle": 100 }
     },
     "pyDSP": {
       "use_dsp": 1,
       "spectrum": "power",
       "window": { "type": "hamming" },
       "filter": { "type": null }
     }
   }

Descriptor Encoding
-------------------

pySAR supports 33 protein descriptors via the ``Descriptors`` class. Descriptors are
calculated using the `protpy <https://github.com/amckenna41/protpy>`_ package (>=1.3.0).

**Initialising the Descriptors class:**

.. code-block:: python

   from pySAR.descriptors import Descriptors

   # Single-threaded (default)
   desc = Descriptors(config_file="config/thermostability.json")

   # Parallel computation across sequences and descriptor types
   desc = Descriptors(config_file="config/thermostability.json", n_jobs=8)

Composition Descriptors
~~~~~~~~~~~~~~~~~~~~~~~

**Amino Acid Composition** — frequency of each of the 20 canonical amino acids (N × 20):

.. code-block:: python

   aa_comp = desc.get_amino_acid_composition()
   print(aa_comp.shape)      # (261, 20)
   print(aa_comp.dtypes[0])  # float64

**Dipeptide Composition** — frequency of all 400 dipeptide pairs (N × 400):

.. code-block:: python

   dp_comp = desc.get_dipeptide_composition()
   print(dp_comp.shape)  # (261, 400)

**Tripeptide Composition** — frequency of all 8000 tripeptide combinations (N × 8000):

.. code-block:: python

   tp_comp = desc.get_tripeptide_composition()
   print(tp_comp.shape)  # (261, 8000)

**GRAVY** — Grand Average of Hydropathicity using Kyte-Doolittle values (N × 1):

.. code-block:: python

   gravy = desc.get_gravy()
   print(gravy.columns.tolist())  # ['GRAVY']

**Aromaticity** — fraction of aromatic residues (F, W, Y, H) in the sequence (N × 1):

.. code-block:: python

   arom = desc.get_aromaticity()
   print(arom.columns.tolist())  # ['Aromaticity']

**Instability Index** — DIWV-based stability score; values ≥ 40 indicate instability (N × 1):

.. code-block:: python

   instab = desc.get_instability_index()
   print(instab.columns.tolist())  # ['InstabilityIndex']

**Isoelectric Point** — estimated pH at which the protein carries no net charge (N × 1):

.. code-block:: python

   pi = desc.get_isoelectric_point()
   print(pi.columns.tolist())  # ['IsoelectricPoint']

**Molecular Weight** — average molecular weight in Daltons, corrected for peptide bonds (N × 1):

.. code-block:: python

   mw = desc.get_molecular_weight()
   print(mw.columns.tolist())  # ['MolecularWeight']

**Charge Distribution** — positive, negative, and net charge at a given pH (default 7.4) (N × 3):

.. code-block:: python

   charge = desc.get_charge_distribution()
   print(charge.columns.tolist())  # ['PositiveCharge', 'NegativeCharge', 'NetCharge']

**Hydrophobic/Polar/Charged Composition** — percentage of residues in each physicochemical
group (N × 3):

.. code-block:: python

   hpc = desc.get_hydrophobic_polar_charged_composition()
   print(hpc.columns.tolist())  # ['Hydrophobic', 'Polar', 'Charged']

**Secondary Structure Propensity** — mean Chou-Fasman propensity values for helix, sheet,
and coil conformations (N × 3):

.. code-block:: python

   ssp = desc.get_secondary_structure_propensity()
   print(ssp.columns.tolist())  # ['Helix', 'Sheet', 'Coil']

**k-mer Composition** — frequency of all 20^k subsequences; default k=2 gives 400 features
(N × 400 by default):

.. code-block:: python

   kmer = desc.get_kmer_composition()
   print(kmer.shape)  # (261, 400)

**Reduced Alphabet Composition** — amino acid composition after mapping residues to a reduced
physicochemical alphabet; default alphabet_size=6 (N × 6 by default):

.. code-block:: python

   rac = desc.get_reduced_alphabet_composition()
   print(rac.shape)  # (261, 6)

**Motif Composition** — count of 8 built-in biological sequence motifs (N × 8 by default):

.. code-block:: python

   motifs = desc.get_motif_composition()
   print(motifs.shape)  # (261, 8)

**Amino Acid Pair Composition** — frequency of all 400 residue-pair combinations with
physicochemical class annotations (N × 400):

.. code-block:: python

   pair_comp = desc.get_amino_acid_pair_composition()
   print(pair_comp.shape)  # (261, 400)

**Aliphatic Index** — relative volume of aliphatic side chains (Ala, Val, Ile, Leu);
higher values correlate with thermostability (N × 1):

.. code-block:: python

   ali = desc.get_aliphatic_index()
   print(ali.columns.tolist())  # ['AliphaticIndex']

**Extinction Coefficient** — molar extinction coefficient at 280 nm from Trp, Tyr, Cys
counts; reported for reduced and oxidised states (N × 2):

.. code-block:: python

   ext = desc.get_extinction_coefficient()
   print(ext.columns.tolist())  # ['ExtCoeff_Reduced', 'ExtCoeff_Oxidized']

**Boman Index** — sum of solubility values for all residues divided by sequence length;
predicts protein–protein interaction potential (N × 1):

.. code-block:: python

   boman = desc.get_boman_index()
   print(boman.columns.tolist())  # ['BomanIndex']

**Aggregation Propensity** — count and fraction of aggregation-prone windows identified via
a sliding-window Kyte-Doolittle + charge-neutrality heuristic (N × 2):

.. code-block:: python

   agg = desc.get_aggregation_propensity()
   print(agg.columns.tolist())  # ['AggregProneRegions', 'AggregProneFraction']

**Hydrophobic Moment** — mean and maximum hydrophobic moment across sliding helical-wheel
windows using the Eisenberg hydrophobicity scale (N × 2):

.. code-block:: python

   hm = desc.get_hydrophobic_moment()
   print(hm.columns.tolist())  # ['HydrophobicMoment_Mean', 'HydrophobicMoment_Max']

**Shannon Entropy** — information-theoretic measure of amino acid diversity;
0 = fully repetitive, ~4.322 bits = perfectly uniform over 20 amino acids (N × 1):

.. code-block:: python

   ent = desc.get_shannon_entropy()
   print(ent.columns.tolist())  # ['ShannonEntropy']

Autocorrelation Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autocorrelation descriptors encode the correlation between the physicochemical properties of
amino acid residues separated by a given sequence lag. All three variants use up to 8 AAIndex
properties and a default lag of 30, producing 240 features per descriptor (N × 240).

**MoreauBroto Autocorrelation** — measures the average product of property values at residues
separated by lag *d*. It captures the overall strength of property correlation across the
sequence without normalising by variance, making it sensitive to the absolute scale of the
chosen property:

.. code-block:: python

   mb = desc.get_moreaubroto_autocorrelation()
   print(mb.shape)  # (261, 240)

**Moran Autocorrelation** — a normalised variant of MoreauBroto that divides by the variance
of the property values across the sequence. This makes it scale-invariant and directly
comparable across different physicochemical properties, reflecting the spatial clustering of
similar residues:

.. code-block:: python

   ma = desc.get_moran_autocorrelation()
   print(ma.shape)  # (261, 240)

**Geary Autocorrelation** — measures the mean squared difference between property values at
residues separated by lag *d*, normalised by the overall variance. Unlike Moran, values close
to 0 indicate strong positive autocorrelation and values greater than 1 indicate negative
autocorrelation, making it sensitive to local dissimilarity along the chain:

.. code-block:: python

   ga = desc.get_geary_autocorrelation()
   print(ga.shape)  # (261, 240)

CTD Descriptors
~~~~~~~~~~~~~~~

**CTD** — combined Composition, Transition, and Distribution descriptor (N × 147):

.. code-block:: python

   ctd = desc.get_ctd()
   print(ctd.shape)  # (261, 147)

Sub-components can be accessed individually:

.. code-block:: python

   ctd_c = desc.get_ctd_composition()    # (261, 21)
   ctd_t = desc.get_ctd_transition()     # (261, 21)
   ctd_d = desc.get_ctd_distribution()   # (261, 105)

Conjoint Triad
~~~~~~~~~~~~~~

**Conjoint Triad** — considers neighbour relationships in protein 3D structure; produces
343 features (N × 343):

.. code-block:: python

   ct = desc.get_conjoint_triad()
   print(ct.shape)  # (261, 343)

Sequence Order Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sequence Order Coupling Number** — dissimilarity between amino acid pairs at varying
distances; default lag=30 gives 60 features. Can use Schneider-Wrede and/or Grantham
distance matrices (N × 60):

.. code-block:: python

   socn = desc.get_sequence_order_coupling_number()
   print(socn.shape)  # (261, 60)

**Quasi Sequence Order** — extends SOCN with amino acid composition; generates 100 features
by default (N × 100):

.. code-block:: python

   qso = desc.get_quasi_sequence_order()
   print(qso.shape)  # (261, 100)

Pseudo Composition Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pseudo Amino Acid Composition** — combines amino acid composition with physicochemical
correlation factors. Default generates 50 features (N × 50):

.. code-block:: python

   paac = desc.get_pseudo_amino_acid_composition()
   print(paac.shape)  # (261, 50)

**Amphiphilic Pseudo Amino Acid Composition** — extends PAAComp with hydrophobic and
hydrophilic distribution patterns along the chain. Default generates 80 features (N × 80):

.. code-block:: python

   apaac = desc.get_amphiphilic_pseudo_amino_acid_composition()
   print(apaac.shape)  # (261, 80)

Calculating All Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate all 33 descriptors at once and concatenate them into a single DataFrame:

.. code-block:: python

   all_desc = desc.get_all_descriptors()
   print(all_desc.shape)  # (261, <total_features>)

   # Export to CSV for future reuse (avoids recomputation)
   desc.get_all_descriptors(export=True, descriptors_export_filename="descriptors.csv")

Parallel Computation
~~~~~~~~~~~~~~~~~~~~

For large datasets, pass ``n_jobs`` to the ``Descriptors`` constructor to enable parallel
computation at two levels simultaneously:

- **Sequence-level** — each descriptor's ``_calculate_descriptor_batch`` distributes sequences
  across ``n_jobs`` threads using ``concurrent.futures.ThreadPoolExecutor``.
- **Descriptor-level** — ``get_all_descriptors`` submits all descriptor getters concurrently
  so multiple descriptor types are computed at the same time.

.. code-block:: python

   from pySAR.descriptors import Descriptors

   # Use 8 threads — sequences and descriptor types are both parallelised
   desc = Descriptors(config_file="config/thermostability.json", n_jobs=8)

   # All 33 descriptors computed in parallel; export for reuse
   all_desc = desc.get_all_descriptors(export=True, descriptors_export_filename="descriptors.csv")

The default ``n_jobs=1`` preserves the original single-threaded behaviour. Values less than
or equal to zero are silently clamped to 1.

AAI Encoding
------------

Encode sequences using physicochemical indices from the AAIndex1 database, combined
with Digital Signal Processing (DSP) features:

.. code-block:: python

   from pySAR.encoding import Encoding

   enc = Encoding(config_file="config/thermostability.json")

   # Encode using a single AAIndex record
   results = enc.aai_encoding(aai_indices="FAUJ880110", sort_by="R2")
   print(results[["Index", "R2", "RMSE"]])

   # Encode using multiple comma-separated AAIndex records
   results = enc.aai_encoding(aai_indices="FAUJ880110, BIGC670101", sort_by="MSE")

Descriptor Encoding
-------------------

Build predictive models using one or more protein descriptors as feature matrices:

.. code-block:: python

   from pySAR.encoding import Encoding

   enc = Encoding(config_file="config/thermostability.json")

   # Single descriptor
   results = enc.descriptor_encoding(descriptors="amino_acid_composition", desc_combo=1, sort_by="R2")

   # Multiple specific descriptors
   results = enc.descriptor_encoding(
       descriptors=["gravy", "molecular_weight", "charge_distribution"],
       desc_combo=1, sort_by="R2"
   )

   # All 33 descriptors (empty list = all)
   results = enc.descriptor_encoding(descriptors=[], desc_combo=1, sort_by="R2")
   print(len(results))  # 36

   # All combinations of 2 descriptors
   results = enc.descriptor_encoding(descriptors=[], desc_combo=2, sort_by="R2")
   print(len(results))  # 528

AAI + Descriptor Encoding
--------------------------

Combine AAI-encoded features with descriptor features:

.. code-block:: python

   from pySAR.encoding import Encoding

   enc = Encoding(config_file="config/thermostability.json")

   results = enc.aai_descriptor_encoding(
       aai_indices="FAUJ880110",
       descriptors="amino_acid_composition",
       desc_combo=1,
       sort_by="R2"
   )
   print(results[["Index", "Descriptor", "R2", "RMSE"]])

PySAR Workflow
--------------

The ``PySAR`` class provides the top-level workflow for building and evaluating models:

.. code-block:: python

   from pySAR.pySAR import PySAR

   # AAI encoding workflow
   pysar = PySAR(config_file="config/thermostability.json")
   results = pysar.encode_aai(aai_indices="FAUJ880110")

   # Descriptor encoding workflow
   results = pysar.encode_descriptor(descriptors="amino_acid_composition")

   # AAI + Descriptor encoding workflow
   results = pysar.encode_aai_descriptor(
       aai_indices="FAUJ880110",
       descriptors="amino_acid_composition"
   )

Predicting Activity for New Sequences
--------------------------------------

After calling any of the ``encode_*`` methods, use ``predict_activity()`` to generate
predictions for unseen protein sequences. The method re-encodes the new sequences using
the same strategy (AAI, descriptor, or combined) that was applied during training:

.. code-block:: python

   from pySAR.pySAR import PySAR

   pysar = PySAR(config_file="config/thermostability.json")
   pysar.encode_aai(aai_indices="FAUJ880110")

   # Predict for a single sequence
   pred = pysar.predict_activity("ACDEFGHIKLMNPQRSTVWY")
   print(pred)  # array([<predicted T50 value>])

   # Predict for multiple sequences
   new_seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSVISMLDLHPTQVSDFDFRDLHTGSQLAVICRPVGNLPNMDMREQAVEKRQRQAALQLQELQRESQ"]
   preds = pysar.predict_activity(new_seqs)
   print(preds.shape)  # (2,)

``predict_activity()`` raises ``RuntimeError`` if called before any ``encode_*`` method,
or ``ValueError`` if the input sequences contain invalid amino acids.

GPR Uncertainty Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the underlying regression model is a ``GaussianProcessRegressor``, pass
``return_uncertainty=True`` to ``predict_activity()`` to also receive the per-sequence
predictive standard deviation:

.. code-block:: python

   from pySAR.pySAR import PySAR

   pysar = PySAR(
       config_file="config/thermostability.json",
       algorithm="gaussianprocessregressor",
   )
   pysar.encode_aai(aai_indices="FAUJ880110")

   preds, std = pysar.predict_activity(new_seqs, return_uncertainty=True)
   print(preds)   # predicted activity values
   print(std)     # per-sequence standard deviation (uncertainty)

For non-GPR models the ``return_uncertainty`` flag is ignored and only ``preds`` is
returned.

Typed Configuration with PySARConfig
-------------------------------------

``PySARConfig`` is a typed dataclass that provides an IDE-friendly alternative to editing
raw JSON files. All fields mirror the keys in the JSON configuration files and can be
used as overrides via ``to_kwargs()``:

.. code-block:: python

   from pySAR import PySARConfig
   from pySAR.pySAR import PySAR

   cfg = PySARConfig(
       config_file="config/thermostability.json",
       algorithm="randomforest",
       test_split=0.1,
   )

   pysar = PySAR(cfg.config_file, **cfg.to_kwargs())
   pysar.encode_aai(aai_indices="FAUJ880110")

Fields set to ``None`` are omitted from ``to_kwargs()`` and therefore fall back to the
values defined in the JSON file. This lets you selectively override only the parameters
you want to change:

.. code-block:: python

   cfg = PySARConfig(
       config_file="config/thermostability.json",
       test_split=0.15,   # override test split only
   )

   from pySAR.encoding import Encoding
   enc = Encoding(cfg.config_file, **cfg.to_kwargs())

Available ``PySARConfig`` fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``config_file``
     - Path to the JSON config file (required to retain dataset/model defaults).
   * - ``dataset``
     - Path to the sequence/activity dataset.
   * - ``sequence_col``
     - Column name for protein sequences.
   * - ``activity_col``
     - Column name for activity/fitness values.
   * - ``algorithm``
     - Sklearn regression algorithm name (e.g. ``'randomforest'``).
   * - ``parameters``
     - Dict of algorithm-specific constructor kwargs.
   * - ``test_split``
     - Fraction of data reserved for testing.
   * - ``use_dsp``
     - Whether to apply the FFT/DSP pipeline.
   * - ``spectrum``
     - Spectrum type: ``'power'``, ``'real'``, ``'imaginary'``, or ``'absolute'``.
   * - ``window_type``
     - Window function for FFT (e.g. ``'hamming'``, ``'blackman'``).
   * - ``filter_type``
     - Post-FFT filter (e.g. ``'savgol'``, ``'medfilt'``).
   * - ``descriptors_csv``
     - Path to a pre-calculated descriptors CSV.

EncodingResult Return Type
---------------------------

All three ``Encoding`` sweep methods return a ``pandas.DataFrame`` by default. An
``EncodingResult`` dataclass is also available for convenient structured access:

.. code-block:: python

   from pySAR.encoding import Encoding, EncodingResult

   enc = Encoding(config_file="config/thermostability.json")
   df = enc.aai_encoding(aai_indices=["FAUJ880110", "BIGC670101"], sort_by="R2")

   # Wrap in EncodingResult for structured access
   result = EncodingResult.from_dataframe(df, elapsed_time=12.4)
   print(result.best_index)  # 'FAUJ880110' (or whichever has the highest R2)
   print(result.best_r2)     # 0.743
   print(result.elapsed_time)  # 12.4

Exporting the Best Model
-------------------------

Pass ``export_best_model=True`` to any ``Encoding`` sweep method to automatically
re-train the best-performing model and save it to ``<output_folder>/best_model.pkl``:

.. code-block:: python

   from pySAR.encoding import Encoding

   enc = Encoding(config_file="config/thermostability.json")
   df = enc.aai_encoding(
       aai_indices=["FAUJ880110", "BIGC670101", "GEIM800111"],
       sort_by="R2",
       output_folder="outputs/",
       export_best_model=True,
   )
   # outputs/best_model.pkl now contains the best fitted model + scaler

The saved pickle can be loaded back with ``Model.load()``:

.. code-block:: python

   from pySAR.model import Model

   best = Model.load("outputs/best_model.pkl")
   best.model_fitted()  # True

Overriding Config Parameters with ``**kwargs``
-----------------------------------------------

Every JSON configuration parameter can be overridden at construction time by passing it
as a keyword argument (``**kwargs``) to ``PySAR`` or ``Encoding``.  The keyword argument
takes precedence over whatever the config file specifies.  This is useful for quick
experiments without modifying the config file or for programmatic sweeps.

.. code-block:: python

   from pySAR.pySAR import PySAR

   # Use the config file but swap in a different algorithm and test split
   pysar = PySAR(
       config_file="config/thermostability.json",
       algorithm="randomforest",
       test_split=0.15,
   )

   # Override the sequence column name (fuzzy matching will resolve close names)
   pysar = PySAR(
       config_file="config/thermostability.json",
       sequence_col="sequences",   # close to 'sequence' — emits UserWarning, resolves automatically
   )

The following table lists all overridable keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Keyword
     - Type
     - Description
   * - ``dataset``
     - ``str``
     - Path to the sequence/activity data file (CSV or TXT).
   * - ``sequence_col``
     - ``str``
     - Column name for protein sequences. Fuzzy matching is applied when the
       exact name is not found; a ``UserWarning`` is emitted on a fuzzy match and
       a ``ValueError`` is raised when no match exists.
   * - ``activity_col``
     - ``str``
     - Column name for the target activity/fitness values.
   * - ``algorithm``
     - ``str``
     - Regression algorithm to use (e.g. ``'plsregression'``, ``'randomforest'``,
       ``'lasso'``).
   * - ``parameters``
     - ``dict``
     - Algorithm-specific constructor keyword arguments forwarded to sklearn.
   * - ``test_split``
     - ``float``
     - Fraction of data held out for testing (default ``0.2``).
   * - ``use_dsp``
     - ``bool``
     - Apply the FFT/DSP spectral pipeline to AAI encodings (default ``False``).
   * - ``spectrum``
     - ``str``
     - Spectrum type: ``'power'``, ``'real'``, ``'imaginary'``, or
       ``'absolute'`` (used when ``use_dsp=True``).
   * - ``window_type``
     - ``str``
     - Window function applied before FFT (e.g. ``'hamming'``, ``'blackman'``).
   * - ``filter_type``
     - ``str``
     - Post-FFT smoothing filter (``'savgol'``, ``'medfilt'``, ``'lfilter'``,
       or ``'hilbert'``).
   * - ``filter_parameters``
     - ``dict``
     - Extra parameters forwarded to the chosen filter function (e.g.
       ``{"window_length": 5, "polyorder": 2}`` for ``savgol``).
   * - ``descriptors_csv``
     - ``str``
     - Path to a pre-calculated descriptors CSV to avoid recomputing descriptors.

Any unrecognised keyword argument is silently ignored so that external tooling can pass
extra metadata without causing errors.

Reproducible Runs and Cross-Validation
----------------------------------------

All three ``encode_*`` methods accept ``random_state`` and ``cv`` keyword arguments:

- ``random_state`` (``int``, default ``None``) — seeds the train/test split for
  reproducible results.
- ``cv`` (``int``, default ``None``) — when set, runs k-fold cross-validation after
  fitting and logs ``CV R² mean ± std``.

.. code-block:: python

   from pySAR.pySAR import PySAR

   pysar = PySAR(config_file="config/thermostability.json")

   results = pysar.encode_aai(
       aai_indices="FAUJ880110",
       random_state=42,
       cv=5,
   )
   # Output includes: "# CV R2 (k=5): mean=0.7413, std=0.0321"

Structured Logging
--------------------

Pass a standard ``logging.Logger`` to ``PySAR.__init__`` to route all output through
your logging infrastructure instead of ``print()``:

.. code-block:: python

   import logging
   from pySAR.pySAR import PySAR

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger("pysar")

   pysar = PySAR(config_file="config/thermostability.json", logger=logger)
   pysar.encode_aai(aai_indices="FAUJ880110")
   # All encode/results output goes to the logger, not stdout

Saving and Loading Sessions
-----------------------------

After fitting a model, use ``save_session()`` to persist the entire ``PySAR`` state
(model, scaler, encoding strategy, configuration) to a pickle file:

.. code-block:: python

   from pySAR.pySAR import PySAR

   pysar = PySAR(config_file="config/thermostability.json")
   pysar.encode_aai(aai_indices="FAUJ880110")

   pysar.save_session("my_run.pkl")   # saves to my_run.pkl

Restore the session later and predict without re-training:

.. code-block:: python

   loaded = PySAR.load_session("my_run.pkl")
   preds = loaded.predict_activity(new_seqs)

.. warning::

   ``save_session()`` / ``load_session()`` use Python pickle.  Never load session
   files from untrusted or unverified sources — they can execute arbitrary code on
   deserialization.  ``load_session(allow_pickle=False)`` raises ``ValueError``
   and can be used to enforce this policy in code.

