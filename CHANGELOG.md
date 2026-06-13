# Changelog

## v2.5.2 - May 2026

### Added
- `pySAR/pySAR.py`: new `predict_activity(sequences, return_uncertainty=False)` method on `PySAR` — re-encodes unseen protein sequences using the same strategy applied during the most recent `encode_aai()`, `encode_descriptor()`, or `encode_aai_descriptor()` call, applies the stored scaler (if any), and returns predicted activity values as a `np.ndarray`. When `return_uncertainty=True` and the underlying model is a `GaussianProcessRegressor`, returns a `(predictions, std)` tuple.
- `pySAR/pySAR.py`: new `save_session(path)` method — serialises the entire fitted `PySAR` instance (model, scaler, encoding strategy, and all configuration attributes) to a pickle file for later restoration.
- `pySAR/pySAR.py`: new `load_session(path, allow_pickle=True)` classmethod — deserialises a session file written by `save_session()`; `allow_pickle=False` raises `ValueError` as a safety gate; emits a `UserWarning` about untrusted sources.
- `pySAR/pySAR.py`: `__init__` now accepts an optional `logger: logging.Logger` parameter; when provided, all output from `output_results()` is routed through the logger instead of `print()`, enabling structured logging in production pipelines.
- `pySAR/pySAR.py`: new `_log(message, level)` private helper — routes messages to `self.logger` when set, or falls back to `print()`.
- `pySAR/pySAR.py`: `encode_aai()`, `encode_descriptor()`, and `encode_aai_descriptor()` each now accept `random_state` and `cv` keyword arguments forwarded to `_fit_and_evaluate()` for reproducible train/test splits and optional cross-validation scoring.
- `pySAR/model.py`: new `cv_score(cv=5, metric='r2', n_jobs=None)` method on `Model` — evaluates the model using k-fold cross-validation on the full (X, Y) data without permanently altering the fitted state; uses a `deepcopy` of the model internally and returns a `np.ndarray` of per-fold scores.
- `pySAR/model.py`: new `@classmethod load(path)` on `Model` — deserialises a `.pkl` file created by `save()`, reconstructs a `Model` instance with both the fitted model and its scaler restored.
- `pySAR/config.py` *(new file)*: `PySARConfig` dataclass providing a typed, IDE-friendly alternative to raw JSON config files. All fields mirror the keys in the JSON configuration files; a `to_kwargs()` helper returns a dict of non-`None` overrides suitable for passing as `**kwargs` to `PySAR` or `Encoding`.
- `pySAR/encoding.py`: new `EncodingResult` dataclass — structured return type wrapping the results DataFrame from any `Encoding` encoding run with `metrics`, `best_index`, `best_r2`, `best_model_path`, and `elapsed_time` attributes. Constructable from any sorted results DataFrame via `EncodingResult.from_dataframe()`.
- `pySAR/encoding.py`: all three encoding methods (`aai_encoding`, `descriptor_encoding`, `aai_descriptor_encoding`) now accept an `export_best_model: bool = False` parameter; when `True` the best-performing model is re-trained and saved to `outputs/best_model.pkl`.
- `pySAR/__init__.py`: `SortKey` and `EncodingResult` are now exported from the top-level `pySAR` namespace.
- `pySAR/__init__.py`: `PySARConfig` is now exported from the top-level `pySAR` namespace.
- `docs/usage.rst`: new "Overriding Config Parameters with ``**kwargs``" section documenting all overridable keyword arguments accepted by `PySAR` and `Encoding` constructors, including fuzzy column-matching behaviour.

### Changed
- `pySAR/pySAR.py`: `__init__` refactored into four private helper methods — `_load_config()`, `_extract_config_params()`, `_load_data()`, and `_init_descriptors()` — reducing the constructor body and making each responsibility independently testable.
- `pySAR/pySAR.py`: new `_fit_and_evaluate(X)` private helper consolidates the repeated `Model → train_test_split → fit → predict → Evaluate` pipeline that was previously duplicated verbatim in `encode_aai`, `encode_descriptor`, and `encode_aai_descriptor`.
- `pySAR/pySAR.py`: `encode_aai`, `encode_descriptor`, and `encode_aai_descriptor` each replaced their O(n²) `pd.concat` loops (re-concatenating the growing DataFrame on every iteration) with a list accumulator and a single `pd.concat` call at the end of the loop.
- `pySAR/pySAR.py`: `get_aai_encoding()` inner loop replaced with a NumPy ASCII-ordinal lookup array — a 128-element float array is built once per index, and each sequence is vectorised with `np.frombuffer(seq.encode('ascii'), dtype=np.uint8)` as the index vector, removing the nested Python loop and significantly speeding up large-dataset encoding.
- `pySAR/pySAR.py`: each of the three `encode_*` methods now records the encoding strategy (`_encoding_type`, `_encoding_aai_indices`, `_encoding_descriptors`) used for the run so that `predict_activity()` can reproduce it faithfully.
- `pySAR/pySAR.py`: `preprocessing()` now requires an exact column match first; fuzzy matching is only tried as a fallback and emits a `UserWarning` when it is used, preventing silent column misidentification.
- `pySAR/pySAR.py`: removed `self.algorithm = repr(self.model)` from `encode_aai()`, `encode_descriptor()`, and `encode_aai_descriptor()` — that assignment was silently overwriting the user-configured algorithm name with the class `repr`.
- `pySAR/pySAR.py`: PEP 8 cleanup — removed redundant parentheses from `if (condition):` guards in all `encode_*` methods.
- `pySAR/pySAR.py`: `encode_aai()` DSP path now correctly forwards `spectrum`, `window_type`, and `filter_type` to `PyDSP`, fixing a bug where these parameters were silently ignored and DSP used config-file defaults even when `PySAR` was initialised with overrides.
- `pySAR/pySAR.py`: `preprocessing()` activity-column block is now correctly indented inside the `preprocessing()` method body; previously it appeared at module scope and was never executed.
- `pySAR/pySAR.py`: `encode_descriptor()` no longer sets `self.descriptors = None` at the end of the method, preserving the attribute for introspection.
- `pySAR/encoding.py`: `_get_aai_features` and `_get_descriptor_features` now use a per-key `concurrent.futures.Future` pattern to eliminate the TOCTOU race condition; when two threads simultaneously miss the cache for the same key, the second thread waits on the existing `Future` instead of recomputing the features independently.
- `pySAR/encoding.py`: `validate_inputs()` now uses `difflib.get_close_matches()` as a fuzzy-matching fallback for each invalid value; if a close match is found it is substituted with a `UserWarning`, and only values with no close match raise `ValueError`.
- `pySAR/encoding.py`: `format_and_save_results()` now normalises a `SortKey` enum instance passed as `sort_by` to its string `.value` before validation, so callers may pass either `SortKey.R2` or `"R2"` interchangeably.
- `pySAR/model.py`: `train_test_split()` now stores `self.scaler` (the fitted `StandardScaler`, or `None` when `scale=False`) so that the scaler is always available for subsequent `predict_activity()` calls.
- `pySAR/model.py`: `save()` now serialises a `{'model': ..., 'scaler': ...}` dict rather than the bare sklearn model object, ensuring the scaler is preserved alongside the model.
- `pySAR/model.py`: `load()` fixed — attributes were previously set with a leading underscore prefix (e.g. `instance._model_fit`) that did not match the attribute names used by `model_fitted()` and `predict()` (e.g. `self.model_fit`), causing loaded models to always report `model_fitted() == False` and fail on `predict()`.
- `pySAR/model.py`: `load()` now also initialises `X_train`, `X_test`, `Y_train`, `Y_test` to `None` so callers can detect they need to call `train_test_split()` first.
- `pySAR/model.py`: `train_test_split()` now raises `ValueError` when `test_split` is outside `(0, 1)` instead of silently resetting to 0.2.
- `pySAR/model.py`: `hyperparameter_tuning()` now returns `self.grid_result` so callers can access the `GridSearchCV` result object.
- `pySAR/evaluate.py`: `rpd_()` now returns `float('nan')` instead of `np.inf` when MSE is 0, preventing infinite values from appearing in output CSVs.
- `pySAR/plots.py`: `plot_reg()` now calls `get_output_folder()` and `get_current_datetime()` per invocation instead of using the module-level frozen constants `OUTPUT_FOLDER`/`CURRENT_DATETIME`, so each plot gets its own timestamped output directory.
- `pySAR/globals_.py`: `CURRENT_DATETIME` and `OUTPUT_FOLDER` are now generated fresh on each call via `get_current_datetime()` and `get_output_folder()` functions, preventing all runs in a long-lived Python session from sharing the same frozen timestamp.
- `pySAR/globals_.py`: legacy `CURRENT_DATETIME` and `OUTPUT_FOLDER` module-level constants now issue a `DeprecationWarning` when accessed via `__getattr__`; callers should use `get_current_datetime()` and `get_output_folder()` instead.
- `pySAR/utils.py`: `remove_gaps()` list/array branch now correctly strips gaps from each sequence individually (previously it joined all sequences into a single concatenated string before stripping).
- `docs/usage.rst`: corrected descriptor count from 36 to 33 in two places.

### Fixed
- `pySAR/utils.py`: `remove_gaps()` — passing a list of sequences no longer concatenates them into a single string; each sequence is processed independently and returned as its own element.
- `pySAR/utils.py`: `Map.__getattr__` now raises `AttributeError` for missing keys instead of returning `None`; this enables the standard `getattr(obj, key, default)` pattern to work correctly with `Map` instances.
- `pySAR/utils.py`: `save_results()` — `csv.DictWriter` is now opened with `newline=''` to prevent spurious blank rows in CSV output on Windows.
- `pySAR/globals_.py`: `__getattr__` deprecation warning was broken — `name` was lowercased before being compared to the uppercase key `CURRENT_DATETIME`, making the replacement string unreachable and always falling back to the generic message. Fixed to compare the original cased name.
- `pySAR/pyDSP.py`: `fft_power` was computing the magnitude spectrum (`np.abs(fft)`) instead of the power spectrum (`np.abs(fft)**2`). Fixed to correctly square the magnitudes.
- `pySAR/pyDSP.py`: window dispatch replaced a ~100-line if-elif chain with a `_WINDOW_DISPATCH` dict; unknown window names now surface a clear `KeyError` rather than silently producing no window.
- `pySAR/encoding.py`: resume-file read is now wrapped in `try/except`; a corrupt or unreadable resume CSV emits a `UserWarning` and restarts from scratch instead of propagating the exception.
- `pySAR/encoding.py`: parallel task results are now caught per-future; a single failed encoding task emits a `RuntimeWarning` and is skipped rather than aborting the entire concurrent run.
- `pySAR/model.py`: `load()` now accepts an `allow_pickle: bool = True` parameter; `allow_pickle=False` raises `ValueError`, and loading always emits a `UserWarning` advising against loading pickle files from untrusted sources.
- `pySAR/model.py`: `feature_selection()` now accepts a configurable `k` parameter for `selectkbest` and `chi2` methods, defaulting to the prior hard-coded values (1 and 2 respectively) when not supplied.
- `pySAR/pySAR.py`: `predict_activity()` AAI and descriptor encoding loops now use the list-accumulate-then-concat pattern, eliminating O(n²) DataFrame copies in the prediction path.
- `pySAR/pySAR.py`: `output_results()` now uses `self._log()` throughout, so output is routed to the configured logger rather than always printing to stdout.
- `pySAR/pySAR.py`: `preprocessing()` no longer silently matches the wrong column when the requested name is absent; an explicit `UserWarning` is now raised when fuzzy fallback is used.

### Tests
- `tests/test_pyDSP.py`: added `test_filter_medfilt`, `test_filter_hilbert`, `test_filter_lfilter_with_coefficients`, and `test_filter_lfilter_without_ba` covering all four supported filter types.
- `tests/test_pyDSP.py`: added `test_fft_power_is_magnitude_squared` — verifies `fft_power == |fft|^2`.
- `tests/test_model.py`: `test_load` now verifies that predictions from a round-tripped (save → load) model are numerically identical to those from the original model; added cases for `allow_pickle=False` raising `ValueError` and loading emitting a `UserWarning`.
- `tests/test_model.py`: `test_feature_selection` extended with `k=3` and `k=1` sub-cases for `selectkbest` and `chi2` to cover the new configurable `k` parameter.
- `tests/test_utils.py`: `test_remove_gaps` assertions corrected to match actual per-element behaviour for lists of individual characters; added test cases for lists of full protein sequence strings and numpy arrays of sequences.
- `tests/test_utils.py`: `test_map` extended with sub-cases verifying that missing attribute access raises `AttributeError` and that `getattr(map, missing_key, default)` returns the default correctly.
- `tests/test_pySAR.py`: added `test_preprocessing_fuzzy_column_matching` verifying that a close column name emits a `UserWarning` and resolves correctly, and that a completely unrecognised name raises `ValueError`.
- `tests/test_pySAR.py`: added `test_predict_activity_uncertainty` — verifies that `return_uncertainty=True` returns a `(predictions, std)` tuple for GPR models with non-negative std values.
- `tests/test_pySAR.py`: added `test_encode_aai_random_state_cv`, `test_encode_descriptor_random_state_cv`, and `test_encode_aai_descriptor_random_state_cv` — verify all three encode methods accept `random_state` and `cv` without raising.
- `tests/test_pySAR.py`: added `test_logger_parameter` — verifies that a custom `logging.Logger` passed to `PySAR.__init__` receives messages from `output_results()`.
- `tests/test_pySAR.py`: added `test_save_and_load_session`, `test_load_session_allow_pickle_false`, and `test_load_session_missing_file` — full round-trip save/load test plus error-path coverage.
- `tests/test_encoding.py`: added `test_get_aai_features_concurrent_cache` — five threads request the same AAI index concurrently; all must receive identical DataFrames (validates the TOCTOU fix).

## v2.5.2 - April 2026

### Changed
- `pySAR/globals_.py`: removed redundant `global` keyword declarations at module level — `global` is only meaningful inside functions, not at module scope.
- `pySAR/model.py`: `valid_models` list is now derived directly from `MODEL_CONSTRUCTORS.keys()` to eliminate duplication and prevent future sync issues.
- `pySAR/model.py`: `hyperparameter_tuning()` now emits a `UserWarning` when an invalid `cv` value is supplied before defaulting to 5, rather than silently resetting.
- `pySAR/pySAR.py`: replaced all `== None` / `!= None` comparisons with `is None` / `is not None` (PEP 8 E711).
- `pySAR/pySAR.py`: `encode_descriptor()` validation now also rejects empty list input (`[]`), consistent with `encode_aai()` and `encode_aai_descriptor()`.
- `pySAR/pySAR.py`: `encode_aai_descriptor()` validation now also rejects empty list input (`[]`) for both `aai_indices` and `descriptors`.
- `pySAR/pySAR.py`: all four string columns in `encode_aai_descriptor()` now use `pd.StringDtype()` consistently (previously `Index` used the bare `"string"` alias).
- `pySAR/evaluate.py`: `rpd_()` now reuses the already-computed `self.mse` attribute rather than calling `self.mse_()` a second time, avoiding a redundant `mean_squared_error` computation.
- `pyproject.toml`, `pySAR/__init__.py`, `docs/conf.py`: version bumped to `2.5.2`.

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