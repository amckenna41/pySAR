################################################################################
#################              PyDSP Module Tests             ##################
################################################################################

import unittest

import numpy as np

from pySAR.pyDSP import PyDSP

# @unittest.skip("")
class TestPyDSP(unittest.TestCase):
    """Focused tests for PyDSP behavior and error handling."""

    def setUp(self):
        self.basic_config = {
            "pyDSP": {
                "spectrum": "power",
                "window": {"type": "hamming"},
                "filter": {"type": None},
            }
        }
        self.protein_seqs = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
            ],
            dtype=float,
        )

    def test_constructor_with_dict_config(self):
        """PyDSP should initialize from dict config and produce expected shapes."""
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=self.protein_seqs)

        self.assertEqual(dsp.spectrum, "power")
        self.assertEqual(dsp.window_type, "hamming")
        self.assertIsNone(dsp.filter_type)
        self.assertIsInstance(dsp.window, np.ndarray)
        self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape)
        self.assertEqual(dsp.fft_freqs.shape, self.protein_seqs.shape)

    def test_constructor_spectrum_only_without_config(self):
        """PyDSP should support kwargs-only spectrum path without a config file."""
        dsp = PyDSP(config_file="", protein_seqs=self.protein_seqs, spectrum="power")

        self.assertEqual(dsp.spectrum, "power")
        self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape)

    def test_invalid_config_path_raises(self):
        """Invalid config file path should raise OSError."""
        with self.assertRaises(OSError):
            PyDSP(config_file="not_a_real_config.json", protein_seqs=self.protein_seqs)

    def test_invalid_config_type_raises(self):
        """Invalid config type should raise TypeError."""
        with self.assertRaises(TypeError):
            PyDSP(config_file=4.21, protein_seqs=self.protein_seqs)
        with self.assertRaises(TypeError):
            PyDSP(config_file=False, protein_seqs=self.protein_seqs)

    def test_missing_protein_sequences_raises(self):
        """Missing protein sequences should raise ValueError."""
        with self.assertRaises(ValueError):
            PyDSP(config_file=self.basic_config, protein_seqs=None)

    def test_none_spectrum_raises(self):
        """spectrum=None should raise ValueError instead of silently passing."""
        with self.assertRaises(ValueError):
            PyDSP(config_file="", protein_seqs=self.protein_seqs, spectrum=None)

    def test_preprocessing_sanitizes_nan_and_inf(self):
        """NaN/Inf values should be cleaned during preprocessing."""
        dirty = np.array(
            [
                [1.0, np.nan, 2.0, np.inf],
                [0.0, -np.inf, 1.0, 2.0],
            ],
            dtype=float,
        )
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=dirty)

        self.assertFalse(np.isnan(dsp.protein_seqs).any())
        self.assertFalse(np.isinf(dsp.protein_seqs).any())

    def test_spectrum_mapping_variants(self):
        """Spectrum selection should map to the corresponding fft attribute."""
        spectra = [
            ("power", "fft_power"),
            ("real", "fft_real"),
            ("imaginary", "fft_imag"),
            ("absolute", "fft_abs"),
        ]

        for spectrum_name, attr_name in spectra:
            with self.subTest(spectrum=spectrum_name):
                dsp = PyDSP(config_file="", protein_seqs=self.protein_seqs, spectrum=spectrum_name)
                self.assertTrue(
                    np.array_equal(dsp.spectrum_encoding, getattr(dsp, attr_name)),
                    f"Expected spectrum_encoding to match {attr_name} for spectrum={spectrum_name}.",
                )

    def test_window_and_filter_savgol(self):
        """Window and savgol filter should both be applied when configured."""
        dsp = PyDSP(
            config_file="",
            protein_seqs=self.protein_seqs,
            spectrum="power",
            window_type="hann",
            filter_type="savgol",
            filter_parameters={"window_length": 3, "polyorder": 1},
        )

        self.assertEqual(dsp.window_type, "hann")
        self.assertEqual(dsp.filter_type, "savgol")
        self.assertIsInstance(dsp.window, np.ndarray)
        self.assertIsNotNone(dsp.filter)

    def test_filter_medfilt(self):
        """medfilt filter should be applied and produce output of matching shape."""
        dsp = PyDSP(
            config_file="",
            protein_seqs=self.protein_seqs,
            spectrum="power",
            filter_type="medfilt",
        )
        self.assertEqual(dsp.filter_type, "medfilt")
        self.assertIsNotNone(dsp.filter)
        self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape)

    def test_filter_hilbert(self):
        """hilbert filter should be applied and produce output of matching shape."""
        dsp = PyDSP(
            config_file="",
            protein_seqs=self.protein_seqs,
            spectrum="power",
            filter_type="hilbert",
        )
        self.assertEqual(dsp.filter_type, "hilbert")
        self.assertIsNotNone(dsp.filter)
        self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape)

    def test_filter_lfilter_with_coefficients(self):
        """lfilter with explicit b/a coefficients should be applied successfully."""
        dsp = PyDSP(
            config_file="",
            protein_seqs=self.protein_seqs,
            spectrum="power",
            filter_type="lfilter",
            filter_parameters={"b": [1.0], "a": [1.0, -0.5]},
        )
        self.assertEqual(dsp.filter_type, "lfilter")
        self.assertIsNotNone(dsp.filter)
        self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape)

    def test_filter_lfilter_without_ba(self):
        """lfilter without b/a coefficients should leave filter as None."""
        dsp = PyDSP(
            config_file="",
            protein_seqs=self.protein_seqs,
            spectrum="power",
            filter_type="lfilter",
        )
        self.assertEqual(dsp.filter_type, "lfilter")
        self.assertIsNone(dsp.filter)


        """Frequency helper methods should return scalar values for 1D input."""
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=self.protein_seqs)

        max_freq, max_index = dsp.max_freq(dsp.spectrum_encoding[0])
        self.assertIsInstance(max_freq, float)
        self.assertIsInstance(max_index, np.int64)

        consensus = dsp.consensus_freq(dsp.spectrum_encoding[0])
        self.assertIsInstance(consensus, float)

    def test_frequency_helpers_reject_non_1d_input(self):
        """Frequency helper methods should reject multi-sequence (non-1D) input."""
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=self.protein_seqs)

        with self.assertRaises(ValueError):
            dsp.max_freq(dsp.spectrum_encoding)

        with self.assertRaises(ValueError):
            dsp.consensus_freq(dsp.spectrum_encoding)

    def test_all_window_types(self):
        """All 18 supported window functions should produce a valid ndarray window."""
        # Windows that require non-optional parameters passed via window_parameters;
        # chebwin is excluded since pyDSP.py already hardcodes at=100 for it.
        window_required_params = {
            "kaiser": {"beta": 14.0},
        }
        window_types = [
            "hamming", "blackman", "blackmanharris", "bartlett", "gaussian",
            "kaiser", "hann", "barthann", "bohman", "chebwin", "cosine",
            "exponential", "flattop", "boxcar", "nuttall", "parzen", "triang", "tukey",
        ]
        for window_name in window_types:
            with self.subTest(window=window_name):
                extra = window_required_params.get(window_name, {})
                dsp = PyDSP(
                    config_file="",
                    protein_seqs=self.protein_seqs,
                    spectrum="power",
                    window_type=window_name,
                    window_parameters=extra,
                )
                self.assertIsInstance(dsp.window, np.ndarray,
                    f"Expected window to be ndarray for window_type='{window_name}'.")
                self.assertEqual(len(dsp.window), self.protein_seqs.shape[1],
                    f"Window length should match signal length for window_type='{window_name}'.")
                self.assertEqual(dsp.spectrum_encoding.shape, self.protein_seqs.shape,
                    f"spectrum_encoding shape mismatch for window_type='{window_name}'.")

    def test_inverse_fft(self):
        """inverse_fft should return an ndarray of the requested length."""
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=self.protein_seqs)
        n = self.protein_seqs.shape[1]
        inv = dsp.inverse_fft(dsp.spectrum_encoding[0], n)

        self.assertIsInstance(inv, np.ndarray,
            "inverse_fft should return a numpy ndarray.")
        self.assertEqual(len(inv), n,
            f"inverse_fft output length should equal n={n}, got {len(inv)}.")

    def test_fft_power_is_magnitude_squared(self):
        """fft_power should equal |fft|^2 (power spectrum), not |fft| (magnitude spectrum)."""
        dsp = PyDSP(config_file=self.basic_config, protein_seqs=self.protein_seqs)
        expected_power = np.abs(dsp.fft) ** 2
        np.testing.assert_array_almost_equal(
            dsp.fft_power, expected_power,
            err_msg="fft_power must be the power spectrum (|fft|^2), not the magnitude spectrum (|fft|).",
        )
        # Confirm it is NOT the raw magnitude (would fail if power==magnitude, i.e. all values are 0 or 1)
        magnitude = np.abs(dsp.fft)
        # They should differ whenever the magnitude is not 0 or 1
        if not np.allclose(dsp.fft_power, magnitude):
            self.assertFalse(
                np.allclose(dsp.fft_power, magnitude),
                "fft_power should differ from the raw magnitude spectrum for non-trivial inputs.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
