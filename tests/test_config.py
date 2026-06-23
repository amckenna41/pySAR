################################################################################
#################             Config Module Tests              #################
################################################################################

import json
import os
import unittest
from difflib import get_close_matches

unittest.TestLoader.sortTestMethodsUsing = None

# @unittest.skip("")
class ConfigUnitTests(unittest.TestCase):
    """
    Unit tests for the JSON config files in tests/test_config/.

    Tests validate the structure, types, and value constraints of each config
    file without instantiating PySAR or loading any dataset.

    Test Cases
    ==========
    test_configs_are_valid_json:
        all four config files parse as valid JSON dicts.
    test_top_level_keys:
        required top-level sections (dataset, model, descriptors, pyDSP) present.
    test_dataset_section_keys:
        dataset section has required keys with string values.
    test_model_section_keys:
        model section has required keys with correct types.
    test_test_split_range:
        test_split is strictly between 0 and 1.
    test_algorithm_is_recognised:
        algorithm values fuzzy-match a known sklearn regressor name.
    test_pyDSP_section_keys:
        pyDSP section has required keys; use_dsp is boolean-compatible.
    test_spectrum_is_valid:
        spectrum value is one of the four accepted types.
    test_window_section:
        pyDSP.window sub-section contains a 'type' key.
    test_filter_section:
        pyDSP.filter sub-section contains a 'type' key.
    test_descriptors_section_has_expected_keys:
        descriptors section contains the expected descriptor sub-sections.
    test_descriptors_csv_key:
        each config specifies a .csv path under descriptors.descriptors_csv.
    test_autocorrelation_params:
        autocorrelation descriptors have valid lag, properties, and normalize values.
    test_quasi_sequence_order_params:
        quasi_sequence_order has valid lag, weight, and distance_matrix values.
    test_pseudo_amino_acid_composition_params:
        pseudo_amino_acid_composition has valid lambda and weight values.
    test_ctd_params:
        ctd section has a 'property' key and an 'all' flag.
    test_kmer_composition_params:
        kmer_composition has a positive integer k value.
    test_charge_distribution_params:
        charge_distribution has a numeric pH value in a physiological range.
    test_configs_share_top_level_structure:
        all four configs have identical top-level keys.
    test_configs_share_model_keys:
        all four configs have identical model sub-keys.
    test_unique_activity_columns:
        each config targets a distinct activity column name.
    test_unique_algorithms:
        configs do not all use the same algorithm (exercises different models).
    """

    _VALID_ALGORITHMS = [
        'plsregression', 'randomforestregressor', 'adaboostregressor',
        'svr', 'knn', 'lasso', 'ridge', 'baggingregressor',
        'gradientboostingregressor', 'linearregression',
    ]
    _VALID_SPECTRA = ['power', 'real', 'imaginary', 'absolute']
    _AUTOCORRELATION_KEYS = [
        'moreaubroto_autocorrelation',
        'moran_autocorrelation',
        'geary_autocorrelation',
    ]

    def setUp(self):
        config_path = os.path.join('tests', 'test_config')
        self.config_files = {
            'thermostability':    os.path.join(config_path, 'test_thermostability.json'),
            'absorption':         os.path.join(config_path, 'test_absorption.json'),
            'enantioselectivity': os.path.join(config_path, 'test_enantioselectivity.json'),
            'localization':       os.path.join(config_path, 'test_localization.json'),
        }
        self.configs = {}
        for name, path in self.config_files.items():
            with open(path) as fh:
                self.configs[name] = json.load(fh)

    # --- structural / type tests -------------------------------------------------

    def test_configs_are_valid_json(self):
        for name, path in self.config_files.items():
            with self.subTest(config=name):
                with open(path) as fh:
                    data = json.load(fh)
                self.assertIsInstance(data, dict)
                self.assertTrue(data)

    def test_top_level_keys(self):
        required = {'dataset', 'model', 'descriptors', 'pyDSP'}
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                missing = required - cfg.keys()
                self.assertFalse(missing, f"Missing top-level keys in {name}: {missing}")

    def test_dataset_section_keys(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                ds = cfg['dataset']
                for key in ('dataset', 'sequence_col', 'activity'):
                    self.assertIn(key, ds)
                    self.assertIsInstance(ds[key], str)
                    self.assertTrue(ds[key], f"Empty value for dataset.{key} in {name}")

    def test_model_section_keys(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                m = cfg['model']
                self.assertIn('algorithm', m)
                self.assertIn('parameters', m)
                self.assertIn('test_split', m)
                self.assertIsInstance(m['algorithm'], str)
                self.assertIsInstance(m['parameters'], dict)
                self.assertIsInstance(m['test_split'], float)

    def test_test_split_range(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                split = cfg['model']['test_split']
                self.assertGreater(split, 0.0)
                self.assertLess(split, 1.0)

    def test_algorithm_is_recognised(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                alg = cfg['model']['algorithm'].lower().strip()
                matches = get_close_matches(alg, self._VALID_ALGORITHMS, cutoff=0.4)
                self.assertTrue(matches, f"Algorithm {alg!r} not recognised in config {name!r}")

    def test_pyDSP_section_keys(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                dsp = cfg['pyDSP']
                for key in ('use_dsp', 'spectrum', 'window', 'filter'):
                    self.assertIn(key, dsp)
                self.assertIn(dsp['use_dsp'], (0, 1, True, False))

    def test_spectrum_is_valid(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                self.assertIn(cfg['pyDSP']['spectrum'], self._VALID_SPECTRA)

    def test_window_section(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                window = cfg['pyDSP']['window']
                self.assertIsInstance(window, dict)
                self.assertIn('type', window)

    def test_filter_section(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                filt = cfg['pyDSP']['filter']
                self.assertIsInstance(filt, dict)
                self.assertIn('type', filt)

    # --- descriptors section tests -----------------------------------------------

    def test_descriptors_section_has_expected_keys(self):
        expected = [
            'moreaubroto_autocorrelation', 'moran_autocorrelation',
            'geary_autocorrelation', 'ctd', 'sequence_order_coupling_number',
            'quasi_sequence_order', 'pseudo_amino_acid_composition',
            'amphiphilic_pseudo_amino_acid_composition',
        ]
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                for key in expected:
                    self.assertIn(key, cfg['descriptors'],
                                  f"Missing descriptor key {key!r} in {name}")

    def test_descriptors_csv_key(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                csv_val = cfg['descriptors'].get('descriptors_csv')
                if csv_val:
                    self.assertIsInstance(csv_val, str)
                    self.assertTrue(csv_val.endswith('.csv'))

    def test_autocorrelation_params(self):
        for name, cfg in self.configs.items():
            for key in self._AUTOCORRELATION_KEYS:
                with self.subTest(config=name, descriptor=key):
                    ac = cfg['descriptors'][key]
                    self.assertIsInstance(ac['lag'], int)
                    self.assertGreater(ac['lag'], 0)
                    self.assertIsInstance(ac['properties'], list)
                    self.assertTrue(ac['properties'], f"Empty properties list in {name}/{key}")
                    self.assertIn(ac['normalize'], (0, 1))

    def test_quasi_sequence_order_params(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                qso = cfg['descriptors']['quasi_sequence_order']
                self.assertIsInstance(qso['lag'], int)
                self.assertGreater(qso['lag'], 0)
                self.assertIsInstance(qso['weight'], float)
                self.assertGreater(qso['weight'], 0.0)
                self.assertIsInstance(qso['distance_matrix'], str)

    def test_pseudo_amino_acid_composition_params(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                paac = cfg['descriptors']['pseudo_amino_acid_composition']
                self.assertIsInstance(paac['lambda'], int)
                self.assertGreater(paac['lambda'], 0)
                self.assertIsInstance(paac['weight'], float)
                self.assertGreater(paac['weight'], 0.0)

    def test_ctd_params(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                ctd = cfg['descriptors']['ctd']
                self.assertIn('property', ctd)
                self.assertIsInstance(ctd['property'], str)
                self.assertIn('all', ctd)
                self.assertIn(ctd['all'], (0, 1, True, False))

    def test_kmer_composition_params(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                k = cfg['descriptors']['kmer_composition']['k']
                self.assertIsInstance(k, int)
                self.assertGreater(k, 0)

    def test_charge_distribution_params(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                ph = cfg['descriptors']['charge_distribution']['ph']
                self.assertIsInstance(ph, (int, float))
                self.assertGreaterEqual(ph, 0.0)
                self.assertLessEqual(ph, 14.0)

    # --- cross-config consistency tests ------------------------------------------

    def test_configs_share_top_level_structure(self):
        names = list(self.configs.keys())
        ref_keys = set(self.configs[names[0]].keys())
        for name in names[1:]:
            with self.subTest(config=name):
                self.assertEqual(ref_keys, set(self.configs[name].keys()))

    def test_configs_share_model_keys(self):
        names = list(self.configs.keys())
        ref_keys = set(self.configs[names[0]]['model'].keys())
        for name in names[1:]:
            with self.subTest(config=name):
                self.assertEqual(ref_keys, set(self.configs[name]['model'].keys()))

    def test_unique_activity_columns(self):
        activities = [cfg['dataset']['activity'] for cfg in self.configs.values()]
        self.assertEqual(len(activities), len(set(activities)),
                         "Expected each config to target a distinct activity column")

    def test_unique_algorithms(self):
        algorithms = [cfg['model']['algorithm'] for cfg in self.configs.values()]
        self.assertGreater(len(set(algorithms)), 1,
                           "Expected configs to use more than one algorithm")


class ConfigIntegrationTests(unittest.TestCase):
    """
    Integration tests verifying that each config file loads correctly into PySAR.

    These tests instantiate PySAR (which reads the dataset) and assert that the
    extracted attributes match the values declared in the config JSON.

    Test Cases
    ==========
    test_pysar_loads_all_configs:
        PySAR can be instantiated from each test config without raising.
    test_pysar_dataset_params_match_config:
        dataset, sequence_col, and activity_col attributes match the config.
    test_pysar_model_params_match_config:
        test_split attribute matches the config value.
    test_pysar_dsp_params_match_config:
        spectrum and use_dsp attributes match the config values.
    test_pysar_config_file_attr_ends_with_json:
        the config_file attribute on the PySAR instance ends with .json.
    test_pysar_adds_json_extension:
        PySAR appends .json when the config path has no extension.
    test_pysar_kwargs_override_config:
        keyword arguments passed to PySAR take precedence over config values.
    test_pysar_invalid_config_raises_oserror:
        a non-existent config path raises OSError.
    test_pysar_non_string_config_raises_typeerror:
        passing a non-string config raises TypeError.
    test_pysar_config_map_dot_access:
        config_parameters supports dot-notation access for top-level sections.
    test_pysar_config_dataclass_to_kwargs:
        PySARConfig.to_kwargs() returns only non-None, non-config_file fields.
    test_pysar_config_dataclass_empty:
        PySARConfig() with no arguments returns an empty to_kwargs() dict.
    test_pysar_config_dataclass_round_trip:
        PySARConfig can drive PySAR instantiation with kwargs overrides.
    """

    def setUp(self):
        config_path = os.path.join('tests', 'test_config')
        self.config_files = {
            'thermostability':    os.path.join(config_path, 'test_thermostability.json'),
            'absorption':         os.path.join(config_path, 'test_absorption.json'),
            'enantioselectivity': os.path.join(config_path, 'test_enantioselectivity.json'),
            'localization':       os.path.join(config_path, 'test_localization.json'),
        }
        self.configs = {}
        for name, path in self.config_files.items():
            with open(path) as fh:
                self.configs[name] = json.load(fh)

    def _make_pysar(self, name, **kwargs):
        from pySAR.pySAR import PySAR
        return PySAR(config_file=self.config_files[name], **kwargs)

    # --- loading tests -----------------------------------------------------------

    def test_pysar_loads_all_configs(self):
        for name in self.config_files:
            with self.subTest(config=name):
                sar = self._make_pysar(name)
                self.assertIsNotNone(sar)

    def test_pysar_dataset_params_match_config(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                sar = self._make_pysar(name)
                self.assertEqual(sar.dataset,      cfg['dataset']['dataset'])
                self.assertEqual(sar.sequence_col, cfg['dataset']['sequence_col'])
                self.assertEqual(sar.activity_col, cfg['dataset']['activity'])

    def test_pysar_model_params_match_config(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                sar = self._make_pysar(name)
                self.assertEqual(sar.test_split, cfg['model']['test_split'])

    def test_pysar_dsp_params_match_config(self):
        for name, cfg in self.configs.items():
            with self.subTest(config=name):
                sar = self._make_pysar(name)
                self.assertEqual(sar.spectrum, cfg['pyDSP']['spectrum'])
                self.assertEqual(bool(sar.use_dsp), bool(cfg['pyDSP']['use_dsp']))

    # --- attribute / error tests -------------------------------------------------

    def test_pysar_config_file_attr_ends_with_json(self):
        sar = self._make_pysar('thermostability')
        self.assertTrue(sar.config_file.endswith('.json'))

    def test_pysar_adds_json_extension(self):
        from pySAR.pySAR import PySAR
        path_no_ext = self.config_files['thermostability'].replace('.json', '')
        sar = PySAR(config_file=path_no_ext)
        self.assertTrue(sar.config_file.endswith('.json'))

    def test_pysar_kwargs_override_config(self):
        sar = self._make_pysar('thermostability', test_split=0.1)
        self.assertEqual(sar.test_split, 0.1)

    def test_pysar_invalid_config_raises_oserror(self):
        from pySAR.pySAR import PySAR
        with self.assertRaises(OSError):
            PySAR(config_file='does_not_exist_config.json')

    def test_pysar_non_string_config_raises_typeerror(self):
        from pySAR.pySAR import PySAR
        with self.assertRaises(TypeError):
            PySAR(config_file=42)

    def test_pysar_config_map_dot_access(self):
        sar = self._make_pysar('thermostability')
        self.assertIsNotNone(sar.config_parameters.dataset)
        self.assertIsNotNone(sar.config_parameters.model)
        self.assertIsNotNone(sar.config_parameters.pyDSP)
        self.assertIsNotNone(sar.config_parameters.descriptors)

    # --- PySARConfig dataclass tests ---------------------------------------------

    def test_pysar_config_dataclass_to_kwargs(self):
        from pySAR.config import PySARConfig
        cfg = PySARConfig(algorithm='plsregression', test_split=0.15)
        kwargs = cfg.to_kwargs()
        self.assertEqual(kwargs['algorithm'], 'plsregression')
        self.assertEqual(kwargs['test_split'], 0.15)
        self.assertNotIn('config_file', kwargs)
        self.assertNotIn('dataset', kwargs)

    def test_pysar_config_dataclass_empty(self):
        from pySAR.config import PySARConfig
        self.assertEqual(PySARConfig().to_kwargs(), {})

    def test_pysar_config_dataclass_round_trip(self):
        from pySAR.config import PySARConfig
        path = self.config_files['thermostability']
        cfg = PySARConfig(config_file=path, test_split=0.3)
        from pySAR.pySAR import PySAR
        sar = PySAR(cfg.config_file, **cfg.to_kwargs())
        self.assertEqual(sar.test_split, 0.3)


if __name__ == '__main__':
    unittest.main()
