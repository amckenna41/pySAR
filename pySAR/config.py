################################################################################
#################                  PySARConfig                 #################
################################################################################

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class PySARConfig:
    """
    Typed configuration container for PySAR and Encoding.

    All parameters mirror the keys in the JSON configuration files so a
    ``PySARConfig`` instance can be used wherever a config filepath is accepted.
    Fields left as *None* fall back to the defaults encoded in the JSON file.

    Parameters
    ==========
    :config_file: str
        Path to the JSON configuration file.  When provided all other fields
        are used as overrides rather than replacements.
    :dataset: str
        Path to the CSV dataset of protein sequences and activity values.
    :sequence_col: str
        Name of the column in *dataset* that contains the protein sequences.
    :activity_col: str
        Name of the column in *dataset* that contains the activity/fitness values.
    :algorithm: str
        Sklearn regression algorithm name (e.g. ``'plsregression'``, ``'randomforest'``).
    :parameters: dict
        Keyword arguments forwarded to the sklearn model constructor.
    :test_split: float
        Fraction of data held back for testing (0 < test_split < 1).
    :use_dsp: bool
        Apply a DSP (FFT) pipeline to the AAI-encoded sequences before modelling.
    :spectrum: str
        Informational spectrum to use when *use_dsp* is True.
        One of ``'power'``, ``'real'``, ``'imaginary'``, ``'absolute'``.
    :window_type: str
        Window function to apply before the FFT (e.g. ``'hamming'``, ``'blackman'``).
    :filter_type: str
        Filter to apply after the FFT (e.g. ``'savgol'``, ``'medfilt'``).
    :descriptors_csv: str
        Path to a pre-calculated descriptors CSV file.  When provided the
        ``Descriptors`` class will import values directly rather than
        recomputing them.

    Usage
    =====
    >>> cfg = PySARConfig(
    ...     config_file="thermostability.json",
    ...     algorithm="randomforest",
    ...     test_split=0.1,
    ... )
    >>> from pySAR import PySAR
    >>> sar = PySAR(cfg.config_file, algorithm=cfg.algorithm, test_split=cfg.test_split)
    """

    config_file: str = ""
    dataset: Optional[str] = None
    sequence_col: Optional[str] = None
    activity_col: Optional[str] = None
    algorithm: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    test_split: Optional[float] = None
    use_dsp: Optional[bool] = None
    spectrum: Optional[str] = None
    window_type: Optional[str] = None
    filter_type: Optional[str] = None
    descriptors_csv: Optional[str] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Return a dict of non-None, non-config_file fields suitable for passing
        as ``**kwargs`` to :class:`~pySAR.pySAR.PySAR` or
        :class:`~pySAR.encoding.Encoding`.

        Returns
        =======
        :kwargs: dict
            Only fields that have been explicitly set (i.e. are not None) are
            included.  The ``config_file`` field is excluded since it is passed
            as a positional argument.
        """
        result: Dict[str, Any] = {}
        for field_name in (
            "dataset",
            "sequence_col",
            "activity_col",
            "algorithm",
            "parameters",
            "test_split",
            "use_dsp",
            "spectrum",
            "window_type",
            "filter_type",
            "descriptors_csv",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result
