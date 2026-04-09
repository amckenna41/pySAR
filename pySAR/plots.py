################################################################################
#################                    Plots                     #################
################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from .globals_ import OUTPUT_FOLDER, CURRENT_DATETIME

def plot_reg(Y_true, Y_pred, r2, output_folder="", show_plot=False, filename="model_regression_plot.png"):
    """
    Plot regression plot of observed (Y_true) vs predicted activity values (Y_pred).

    Parameters
    ==========
    :Y_true: np.ndarray
        array of observed values.
    :Y_pred: np.ndarray
        array of predicted values.
    :r2: float
        r2 score value.
    :output_folder: str (default="")
        output folder to store regression plot to, if empty input it will be stored in 
        the OUTPUT_FOLDER global var.
    :show_plot: bool (default=False)
        whether to display plot or not when function is run, if False the plot is just
        saved to output folder. 
    :filename: str (default="model_regression_plot.png")
        output filename for saved plot image.

    Returns
    =======
    :save_path: str
        full output path of saved regression plot.
    """
    # Validate inputs and normalize to 1D float arrays for plotting.
    try:
        y_true = np.asarray(Y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(Y_pred, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError("Y_true and Y_pred must be numeric array-like inputs.")

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Y_true and Y_pred must be non-empty arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Y_true and Y_pred must have same length, got {y_true.shape[0]} and {y_pred.shape[0]}.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Y_true and Y_pred must contain only finite numeric values.")
    try:
        r2 = float(r2)
    except (TypeError, ValueError):
        raise TypeError(f"r2 must be a numeric value, got {type(r2)}.")

    if not np.isfinite(r2):
        raise ValueError(f"r2 must be a finite numeric value, got {r2}.")

    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("filename must be a non-empty string.")
    if Path(filename).suffix == "":
        filename = f"{filename}.png"

    # Resolve output folder and ensure it exists.
    if output_folder in ("", None):
        target_dir = Path(OUTPUT_FOLDER)
    else:
        target_dir = Path(f"{output_folder}_{CURRENT_DATETIME}")
    target_dir.mkdir(parents=True, exist_ok=True)

    save_path = target_dir / filename

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        # Plot predicted values against observed values to match axis labels.
        sns.regplot(x=y_pred, y=y_true, marker="+", truncate=False, fit_reg=True, ax=ax)
        r2_annotation = f"R2: {r2:.3f}"
        ax.text(0.15, 0.92, r2_annotation, ha="left", va="top", fontsize=15, color="green",
            fontweight="bold", transform=ax.transAxes)
        ax.set_xlabel("Predicted Value", fontdict=dict(weight="bold"), fontsize=12)
        ax.set_ylabel("Observed Value", fontdict=dict(weight="bold"), fontsize=12)
        ax.set_title("Observed vs Predicted values for protein activity", fontdict=dict(weight="bold"), fontsize=15)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show(block=False)
            plt.pause(3)

        return str(save_path)
    finally:
        plt.close(fig)