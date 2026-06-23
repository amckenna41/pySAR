""" pySAR software metadata. """
__name__ = 'pySAR'
__version__ = "2.5.3"
__description__ = 'A Python package used to analysis Sequence Activity Relationships (SARs) of protein sequences and their mutants using Machine Learning.'
__author__ = 'AJ McKenna: https://github.com/amckenna41'
__authorEmail__ = 'amckenna41@qub.ac.uk'
__maintainer__ = "AJ McKenna"
__license__ = 'MIT'
__url__ = 'https://github.com/amckenna41/pySAR'
__download_url__ = "https://github.com/amckenna41/pySAR/archive/refs/heads/main.zip"
__status__ = "Production"
__keywords__ = ["bioinformatics", "protein engineering", "python", "pypi", "machine learning", \
        "directed evolution", "drug discovery", "sequence activity relationships", "SAR", "aaindex", "protpy", "protein descriptors"]
__test_suite__ = "tests"

from .encoding import SortKey, EncodingResult
from .config import PySARConfig

__all__ = [
    '__version__',
    '__description__',
    '__author__',
    '__authorEmail__',
    '__maintainer__',
    '__license__',
    '__url__',
    '__download_url__',
    '__status__',
    '__keywords__',
    '__test_suite__',
    'SortKey',
    'EncodingResult',
    'PySARConfig',
]