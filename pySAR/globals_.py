################################################################################
##############                 Global Variables                  ###############
################################################################################

import os
import warnings
from datetime import datetime

#output dir is the default directory used to store all outputs generated
OUTPUT_DIR = 'outputs'

def get_current_datetime() -> str:
    """Return a fresh timestamp string for the current moment."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_output_folder() -> str:
    """Return a fresh output folder path using the current timestamp."""
    return os.path.join(OUTPUT_DIR, f'model_output_{get_current_datetime()}')

# Legacy module-level constants; access is intercepted via __getattr__ to issue
# a DeprecationWarning.  Use get_current_datetime() / get_output_folder() instead.
_LEGACY_CONSTANTS = {
    'CURRENT_DATETIME': lambda: get_current_datetime(),
    'OUTPUT_FOLDER': lambda: get_output_folder(),
}

def __getattr__(name: str):
    """Warn when legacy constants are accessed and return a fresh value."""
    if name in _LEGACY_CONSTANTS:
        replacement = 'get_current_datetime()' if name == 'CURRENT_DATETIME' else 'get_output_folder()'
        warnings.warn(
            f"globals_.{name} is deprecated in pySAR 2.5.3. "
            f"Use globals_.{replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_CONSTANTS[name]()
    raise AttributeError(f"module 'pySAR.globals_' has no attribute {name!r}")