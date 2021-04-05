
#########################################################################
###                         Global Variables                          ###
#########################################################################

from datetime import date
import time
import os
from datetime import datetime

#data dir is the default directory used to store all data required for project
DATA_DIR = 'data'

#output dir is the default directory used to store all outputs generated
OUTPUT_DIR = 'outputs'

#current datetime appended to output assets to uniquely identify them
CURRENT_DATETIME = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

#output folder is the default folder within the OUTPUT_DIR used to store all \
#   outputs generated from one run of the program.
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, 'model_output_'+CURRENT_DATETIME)
