################################################################################
#################              Plots Module Tests             #################
################################################################################

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg")

import pySAR.plots as plots


# @unittest.skip("")
class PlotTests(unittest.TestCase):
    """
    Test suite for testing plots module and functionality in pySAR package.

    Test Cases
    ==========
    test_plot_reg_creates_file_default_output:
        testing file creation when default output folder is used.
    test_plot_reg_creates_file_custom_output:
        testing file creation when custom output folder is used.
    test_plot_reg_input_validation:
        testing input validation for empty/mismatched/non-finite arrays and invalid r2.
    test_plot_reg_filename_handling:
        testing filename validation and automatic .png extension appending.
    test_plot_reg_show_plot_branch:
        testing show_plot=True branch without opening interactive UI.
    """

    def setUp(self):
        """Create deterministic arrays and temporary directories for test outputs."""
        self.rng = np.random.default_rng(42)
        self.y_true = self.rng.random(20)
        self.y_pred = self.rng.random(20)
        self.temp_root = tempfile.mkdtemp(prefix="test_plots_")
        self.default_output = os.path.join(self.temp_root, "default_output")
        self.custom_base = os.path.join(self.temp_root, "custom_output")

    def _patched_globals(self):
        return patch.multiple(
            plots,
            OUTPUT_FOLDER=self.default_output,
            CURRENT_DATETIME="2026-03-23_00-00-00",
        )

    def test_plot_reg_creates_file_default_output(self):
        """Testing file creation and returned path for default output folder."""
        with self._patched_globals():
            save_path = plots.plot_reg(self.y_true, self.y_pred, 0.55, output_folder="", show_plot=False)
            self.assertTrue(os.path.isfile(save_path), f"Expected saved plot file at {save_path}.")
            self.assertEqual(
                os.path.dirname(save_path),
                self.default_output,
                f"Expected default output directory {self.default_output}, got {os.path.dirname(save_path)}.")
            self.assertTrue(
                save_path.endswith("model_regression_plot.png"),
                f"Expected default filename model_regression_plot.png, got {save_path}.")

    def test_plot_reg_creates_file_custom_output(self):
        """Testing file creation and returned path for custom output folder."""
        with self._patched_globals():
            save_path = plots.plot_reg(
                self.y_true,
                self.y_pred,
                0.72,
                output_folder=self.custom_base,
                show_plot=False,
                filename="custom_plot.png",
            )
            expected_dir = f"{self.custom_base}_{plots.CURRENT_DATETIME}"
            self.assertTrue(os.path.isfile(save_path), f"Expected saved plot file at {save_path}.")
            self.assertEqual(
                os.path.dirname(save_path),
                expected_dir,
                f"Expected custom output directory {expected_dir}, got {os.path.dirname(save_path)}.")
            self.assertTrue(
                save_path.endswith("custom_plot.png"),
                f"Expected custom filename custom_plot.png, got {save_path}.")

    def test_plot_reg_input_validation(self):
        """Testing input validation and error raising for invalid inputs."""
        with self._patched_globals():
            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([]), np.array([]), 0.5)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0]), 0.5)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, np.nan]), np.array([1.0, 2.0]), 0.5)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, np.inf]), 0.5)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.nan)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.inf)

            with self.assertRaises(ValueError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), -np.inf)

            with self.assertRaises(TypeError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), None)

            with self.assertRaises(TypeError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), "not_a_number")

            with self.assertRaises(TypeError):
                plots.plot_reg(np.array(["a", "b"]), np.array([1.0, 2.0]), 0.5)

            with self.assertRaises(TypeError):
                plots.plot_reg(np.array([1.0, 2.0]), np.array(["a", "b"]), 0.5)

    def test_plot_reg_filename_handling(self):
        """Testing filename validation and extension auto-appending."""
        with self._patched_globals():
            save_path = plots.plot_reg(self.y_true, self.y_pred, 0.33, filename="plot_without_ext")
            self.assertTrue(os.path.isfile(save_path), f"Expected saved plot file at {save_path}.")
            self.assertTrue(
                save_path.endswith("plot_without_ext.png"),
                f"Expected .png extension to be appended, got {save_path}.")

            with self.assertRaises(ValueError):
                plots.plot_reg(self.y_true, self.y_pred, 0.33, filename="")

            with self.assertRaises(ValueError):
                plots.plot_reg(self.y_true, self.y_pred, 0.33, filename="   ")

    def test_plot_reg_show_plot_branch(self):
        """Testing show_plot=True branch without opening an interactive window."""
        with self._patched_globals():
            with patch("matplotlib.pyplot.show") as mock_show, patch("matplotlib.pyplot.pause") as mock_pause:
                save_path = plots.plot_reg(self.y_true, self.y_pred, 0.40, show_plot=True)
                self.assertTrue(os.path.isfile(save_path), f"Expected saved plot file at {save_path}.")
                mock_show.assert_called_once_with(block=False)
                mock_pause.assert_called_once_with(3)

    def test_plot_reg_output_folder_variants(self):
        """Testing output folder behavior for None and pathlib.Path inputs."""
        with self._patched_globals():
            none_path = plots.plot_reg(self.y_true, self.y_pred, 0.51, output_folder=None)
            self.assertEqual(
                os.path.dirname(none_path),
                self.default_output,
                f"Expected None output_folder to resolve to default output path, got {os.path.dirname(none_path)}.")

            path_base = Path(self.custom_base)
            path_save = plots.plot_reg(self.y_true, self.y_pred, 0.62, output_folder=path_base)
            expected_dir = f"{path_base}_{plots.CURRENT_DATETIME}"
            self.assertEqual(
                os.path.dirname(path_save),
                expected_dir,
                f"Expected Path output folder to resolve to {expected_dir}, got {os.path.dirname(path_save)}.")

    def tearDown(self):
        """Delete temporary directories and test artifacts."""
        shutil.rmtree(self.temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
