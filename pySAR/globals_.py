################################################################################
##############                 Global Variables                  ###############
################################################################################

import os
from datetime import datetime

#data dir is the default directory used to store all data required for pySAR
global DATA_DIR
DATA_DIR = 'data'

#output dir is the default directory used to store all outputs generated
global OUTPUT_DIR
OUTPUT_DIR = 'outputs'

#current datetime appended to output assets & directories to uniquely identify them
global CURRENT_DATETIME
CURRENT_DATETIME = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H_%M')))

#output folder is the default folder within the OUTPUT_DIR used to store all
#   outputs generated from one run of the program.
global OUTPUT_FOLDER
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, 'model_output_' + CURRENT_DATETIME)
