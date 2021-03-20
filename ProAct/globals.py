
from datetime import date
import time
import os
from datetime import datetime

DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'

CURRENT_DATETIME = str(datetime.date(datetime.now())) + \
    '_' + str((datetime.now().strftime('%H:%M')))

OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, 'model_output_'+CURRENT_DATETIME)
