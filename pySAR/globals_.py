################################################################################
##############                 Global Variables                  ###############
################################################################################

import os
from datetime import datetime

NOW = datetime.now()

#output dir is the default directory used to store all outputs generated
global OUTPUT_DIR
OUTPUT_DIR = 'outputs'

#current datetime appended to output assets & directories to uniquely identify them
global CURRENT_DATETIME
CURRENT_DATETIME = NOW.strftime('%Y-%m-%d_%H-%M-%S')

#output folder is the default folder within the OUTPUT_DIR used to store all
#outputs generated from one run of the program.
global OUTPUT_FOLDER
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f'model_output_{CURRENT_DATETIME}')