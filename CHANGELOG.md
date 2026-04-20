# Changelog

## v2.5.1 - April 2026

### Changed
- `pySAR/globals_.py`: removed redundant `global` keyword declarations at module level — `global` is only meaningful inside functions, not at module scope.
- `pySAR/model.py`: `valid_models` list is now derived directly from `MODEL_CONSTRUCTORS.keys()` to eliminate duplication and prevent future sync issues.
- `pySAR/model.py`: `hyperparameter_tuning()` now emits a `UserWarning` when an invalid `cv` value is supplied before defaulting to 5, rather than silently resetting.
- `pySAR/pySAR.py`: replaced all `== None` / `!= None` comparisons with `is None` / `is not None` (PEP 8 E711).
- `pySAR/pySAR.py`: `encode_descriptor()` validation now also rejects empty list input (`[]`), consistent with `encode_aai()` and `encode_aai_descriptor()`.
- `pySAR/pySAR.py`: `encode_aai_descriptor()` validation now also rejects empty list input (`[]`) for both `aai_indices` and `descriptors`.
- `pySAR/pySAR.py`: all four string columns in `encode_aai_descriptor()` now use `pd.StringDtype()` consistently (previously `Index` used the bare `"string"` alias).
- `pySAR/evaluate.py`: `rpd_()` now reuses the already-computed `self.mse` attribute rather than calling `self.mse_()` a second time, avoiding a redundant `mean_squared_error` computation.
- `pyproject.toml`, `pySAR/__init__.py`, `docs/conf.py`: version bumped from `2.5.0` to `2.5.1`.

### Fixed
- `pySAR/model.py`: `save()` was silently swallowing `pickle.PickleError` with a `print()` statement. It now re-raises as `RuntimeError` so callers are aware of serialisation failures.
- `pySAR/pySAR.py`: removed leftover commented-out dead code `# aai_desc_df['Index'] = aai_desc_df['Index'].astype(pd.StringDtype())`.

### Removed
- `pySAR/descriptors.py`: removed unused `from json import JSONDecodeError` import — `json.JSONDecodeError` is accessed via the already-imported `json` module.
- `pySAR/utils.py`: removed unused `flatten()` function — it was never imported or called anywhere in the package.
- `pyproject.toml`, `README.md`, `docs/conf.py`, `PySAR.egg-info/`: removed `delayed` package dependency — it was listed as a requirement but never imported or used anywhere in the source code.

## v2.5.0 - April 2026

### Added
- `pySAR/descriptors.py`: added 18 new protein descriptors from protpy v1.3.0:
  - `gravy` — Grand Average of Hydropathicity (GRAVY) index (1 feature).
  - `aromaticity` — fraction of aromatic amino acids (1 feature).
  - `instability_index` — Guruprasad et al. instability index (1 feature).
  - `isoelectric_point` — theoretical isoelectric point (pI) (1 feature).
  - `molecular_weight` — molecular weight of the protein sequence (1 feature).
  - `charge_distribution` — positive, negative, and net charge at a given pH (3 features, default `ph=7.4`).
  - `hydrophobic_polar_charged_composition` — fractions of hydrophobic, polar, and charged residues (3 features).
  - `secondary_structure_propensity` — propensity for helix, sheet, and coil conformations (3 features).
  - `kmer_composition` — frequency of all k-mer subsequences (400 features at default `k=2`).
  - `reduced_alphabet_composition` — amino acid composition using a reduced alphabet (6 features at default `alphabet_size=6`).
  - `motif_composition` — counts of 8 built-in sequence motifs (8 features).
  - `amino_acid_pair_composition` — pairwise amino acid composition with physicochemical class labelling (400 features).
  - `aliphatic_index` — relative volume occupied by aliphatic side chains (1 feature).
  - `extinction_coefficient` — molar extinction coefficients for reduced and oxidised forms (2 features).
  - `boman_index` — Boman index (potential protein interaction index) (1 feature).
  - `aggregation_propensity` — number and fraction of aggregation-prone regions (2 features).
  - `hydrophobic_moment` — mean and maximum hydrophobic moment of the sequence (2 features, default `window=11`, `angle=100`).
  - `shannon_entropy` — Shannon entropy of amino acid composition (1 feature).
- `pySAR/descriptors.py`: updated `DescriptorType` enum from 15 to 33 entries to include all new descriptors.
- `pySAR/descriptors.py`: `valid_descriptors` list expanded from 15 to 33 entries.
- `pySAR/descriptors.py`: `descriptor_groups` updated — Composition group now contains 21 descriptors (3 original + 18 new).
- `config/*.json`: all 4 configuration files (`thermostability.json`, `absorption.json`, `enantioselectivity.json`, `localization.json`) updated with parameter sections for the new descriptors.
- `tests/test_descriptors.py`: added 18 new individual descriptor test methods covering shape, dtype, null/inf checks, and column naming for each new descriptor.
- `tests/test_descriptors.py`: updated `test_descriptor_groups`, `test_all_descriptors_list`, `test_valid_descriptors`, `test_descriptor_import`, and `test_get_all_descriptors` to reflect the expanded descriptor set.
- `pySAR/model.py`: `hyperparameter_tuning()` now raises `RuntimeError` with a clear message if called before `train_test_split()`.
- `pySAR/encoding.py`: added `threading.Lock` (`_cache_lock`) to guard `_aai_feature_cache` and `_descriptor_feature_cache` against race conditions when `n_jobs > 1`.
- `pySAR/pySAR.py`: added `UserWarning` in `preprocessing()` when missing or infinite activity values are silently replaced with `0`.
- `tests/test_pyDSP.py`: added `test_all_window_types` — systematically exercises all 18 supported window functions (`hamming`, `blackman`, `blackmanharris`, `bartlett`, `gaussian`, `kaiser`, `hann`, `barthann`, `bohman`, `chebwin`, `cosine`, `exponential`, `flattop`, `boxcar`, `nuttall`, `parzen`, `triang`, `tukey`).
- `tests/test_pyDSP.py`: added `test_inverse_fft` — verifies `inverse_fft()` returns an ndarray of the requested length.
- `tests/test_model.py`: added `test_hyperparameter_tuning_before_train_test_split_raises` — verifies `RuntimeError` is raised when `hyperparameter_tuning()` is called before `train_test_split()`.
- Added modern packaging metadata via `pyproject.toml`.
- Added new documentation files under `docs/`:
  - `docs/README.md`
  - `docs/conf.py`
  - `docs/index.rst`
  - `docs/usage.rst`
  - `docs/contributing.rst`
  - `docs/requirements.txt`
- Added `FILE_CHECKLIST.md` for repository/file tracking.
- Added descriptor constants and enum support in `pySAR/descriptors.py`:
  - feature-size constants for descriptor families
  - `DescriptorType` enum
- Added descriptor utility APIs in `pySAR/descriptors.py`:
  - `validate_descriptors`
  - `validate_sequences`
  - cached `descriptor_feature_count`
  - `get_descriptor_info`
  - `reset_descriptors`
  - `clear_cache`
  - `get_descriptor_columns`
- Added custom descriptor exceptions in `pySAR/descriptors.py`:
  - `DescriptorError`
  - `InvalidSequenceError`
  - `DescriptorConfigError`
  - `InvalidDescriptorError`
- Added extended encoding infrastructure in `pySAR/encoding.py`:
  - `MetricKey` and `SortKey` enums
  - structured logging support
  - helper methods for input validation, feature building, model execution, metric collection, and result formatting
  - per-index/per-descriptor caching
  - optional parallel execution
  - resume/checkpoint support
  - model limiting/sample controls

### Changed
- `requirements.txt`: bumped `protpy` dependency from `>=1.0.0` to `>=1.3.0`.
- `pySAR/encoding.py`: updated class and method docstrings to reflect 33 supported descriptors and the new combination counts (33 / 528 / 5456 for desc\_combo 1 / 2 / 3).
- `tests/test_descriptors.py`: updated descriptor count assertions — 15 → 33 valid descriptors, 3 → 21 Composition group members, `all_descriptors` shape 9714 → 10572 columns.
- `pySAR/pySAR.py`: renamed local variable `eval` → `evaluation` in `encode_aai()`, `encode_descriptor()`, and `encode_aai_descriptor()` to avoid shadowing the Python built-in.
- `pySAR/pySAR.py`: `encode_aai_descriptor()` now reuses the cached `self.descriptor` instance from `__init__` rather than creating a fresh `Descriptors` object on every call.
- `pySAR/encoding.py`: `descriptor_encoding()` and `aai_descriptor_encoding()` now reuse `self.descriptor` instead of instantiating a new `Descriptors` object, avoiding redundant CSV reloads.
- `pySAR/encoding.py`: aligned column order in `aai_descriptor_encoding()` so `MAE` precedes `RPD`, consistent with `aai_encoding()` and `descriptor_encoding()`.
- `pySAR/encoding.py`: updated the stale comment in `_apply_model_limit()` to accurately describe the deterministic first-N slicing (previously said "random sample").
- `pySAR/utils.py`: `zero_padding()` replaced O(n²) string `+=` concatenation with `.ljust(max_len, '0')`.
- `pySAR/utils.py`: `save_results()` no longer appends a datetime suffix to caller-supplied `output_folder` values; the path is now used as-is, consistent with the default `OUTPUT_FOLDER` path.
- `pySAR/pyDSP.py`: migrated FFT import from deprecated `scipy.fftpack.fft` to `scipy.fft.fft` (with `numpy.fft` as fallback).
- `pyproject.toml`: removed unused `requests` dependency.
- Migrated packaging flow from legacy `setup.py`/`setup.cfg` to `pyproject.toml`.
- Refactored `pySAR/descriptors.py` with broader type hints and improved API surface.
- Refactored `pySAR/encoding.py` to a helper-based architecture with cleaner execution flow and improved configurability.
- Updated CI workflows in `.github/workflows/`:
  - expanded Python matrix and dependency caching
  - added/updated security scanning steps
  - improved artifacts and summary output
  - updated deploy pipelines for TestPyPI and PyPI
- Updated Read the Docs configuration in `.readthedocs.yml`.
- Updated Sphinx build scripts in `docs/Makefile` and `docs/make.bat` to the current docs layout.
- Updated `tests/test_descriptors.py` with new/expanded tests for validation utilities and custom exceptions.
- Updated spelling and terminology in multiple files (for example, physiochemical -> physicochemical/physicochemical).
- Updated repository TODO/checklist content in `TODO.md` and related docs.

### Fixed
- `pySAR/utils.py`: fixed `remove_gaps()` silently concatenating all sequences into a single string when a plain `list` of multiple sequences was passed.
- `pySAR/pySAR.py`: removed dead commented-out line `#self.aai_indices = ""` in `encode_aai()`.
- Fixed descriptor validation behavior for mixed/invalid descriptor input types.
- Fixed descriptor metadata handling to return stable info fields (including `feature_count`) and safer parameter serialization.
- Fixed several descriptor/encoding messaging and formatting inconsistencies by adopting f-strings in updated modules.
- Fixed encoding workflow issues around feature construction, sorting behavior, and model execution consistency.
- Fixed docs/build path mismatches between Read the Docs configuration and Sphinx build scripts.

### Removed
- Removed legacy CircleCI config: `.circleci/config.yml`.
- Removed legacy docs source bootstrap files:
  - `docs/source/conf.py`
  - `docs/source/index.rst`
- Removed legacy packaging files:
  - `setup.py`
  - `setup.cfg`