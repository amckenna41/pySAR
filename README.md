
![alt text](https://raw.githubusercontent.com/amckenna41/pySAR/main/pySAR.png)

# pySAR <a name="TOP"></a>
![PyPI](https://img.shields.io/pypi/v/pySAR)
[![pytest](https://github.com/amckenna41/pySAR/workflows/Building%20and%20Testing%20%F0%9F%90%8D/badge.svg)](https://github.com/amckenna41/pySAR/actions?query=workflowBuilding%20and%20Testing%20%F0%9F%90%8D)
![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)
[![PythonV](https://img.shields.io/pypi/pyversions/pySAR?logo=2)](https://pypi.org/project/pySAR/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/github/workflow/status/amckenna41/pySAR/Deploy%20to%20PyPI%20%F0%9F%93%A6)](https://github.com/amckenna41/pySAR/actions)
![Traivs](https://img.shields.io/travis/amckenna41/pySAR)
[![Issues](https://img.shields.io/github/issues/amckenna41/pySAR)](https://github.com/amckenna41/pySAR/issues)
[![Size](https://img.shields.io/github/repo-size/amckenna41/pySAR)](https://github.com/amckenna41/pySAR)
[![Commits](https://img.shields.io/github/commit-activity/w/amckenna41/pySAR)](https://github.com/amckenna41/pySAR)

#everything is italisz


pySAR is a Python library for analysing Sequence Activity Relationships (SARs) of protein sequences. pySAR offers extensive and verbose functionalities that allow you to numerically encode a dataset of protein sequences using a large abundance of available methodologies and features. The software uses physiochemical and biochemical features from the Amino Acid Index (AAI) database as well as allowing for the calculation of a range of structural protein descriptors.<br>
After finding the optimal technique and feature set at which to encode your dataset of sequences, pySAR can then be used to build a predictive regression model with the training data being that of the encoded sequences and training labels being the experimentally pre-calculated activity values for each protein sequence. The model can then be used to predict the activity/fitness value of a new unseen sequence.

Table of Contents
-----------------

  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Directory Folders](#directories)
  * [Tests](#tests)
  * [Contact](#contact)

Requirements
------------
<!-- ## System Requirements ## -->

* [Python][python] >= 3.6
* [numpy][numpy] >= 1.16.0
* [pandas][pandas] >= 1.1.0
* [sklearn][sklearn] >= 0.24
* [scipy][scipy] >= 1.4.1
* [tqdm][tqdm] >= 4.55.0
* [seaborn][seaborn] >= 0.11.1


<!-- ## Installation -->
Installation
-----------------
Install the latest version of pySAR using pip:

```bash
pi3 install pySAR
```

Install by cloning repository:
```bash
git clone https://github.com/amckenna41/pySAR.git
python3 setup.py install
```


Usage
-----

### Building predictive model from AAI and or protein descriptors:
e.g: the below code will build a PlsRegression model using the AAI index CIDH920105 and the 'amino acid composition' descriptor. The index is passed through a DSP pipeline and is transformed into its informational protein spectra using the power spectra, with a hamming window function applied to the output of the FFT. The concatenated features from the AAI index and the descriptor will be used as the feature data in building the PLS model.

```python
#first-party imports
from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
import utils as utils
from plots import plot_reg
import descriptors as desc

#create instance of PySAR class
pySAR = PySAR(dataset="dataset.txt",activity="activity",algorithm="PlsRegression")
"""
PySAR parameters:

dataset : str (default = "")
    full path to dataset or name of dataset if it is stored in DATA_DIR.
seq_col : str (default = "sequence")
    name of column in dataset that stores the protein sequences. By default
    the class will look for a column called 'sequence'.
activity : str (default = "")
    name of activity column in dataset.
algorithm : str (default = "")
    name of regression model to use for building the predictive models, class
    will accept full name or approximate name of model e.g "PLSReg", "plsregg" and
    "PLSRegression" will all build a PlsRegression model.
parameters : dict (default = {})
    dictionary of parameters to use for the predictive model. By default the
    default parameters of the model will be used.
test_split : float (default = 0.2)
    specifies the proportion of the dataset to use for testing. By default a
    80:20 split will be used, meaning 80% of the data will be used for training
    and 20% for testing.
descriptors_csv : str (default = "descriptors.csv")
    csv file storing the pre-calculated descriptor values for the sequences
    in the dataset. By default the class will look for a file named
    "descriptors.csv" in the DATA_DIR and will use its contents as the
    descriptor features, instead of having to recalculate all descriptors for the dataset.
"""
#encode protein sequences using both the CIDH920105 index + aa_composition descriptor.
results_df = pySAR.encode_aai_desc(indices="CIDH920105", descriptors="aa_composition",
  spectrum="power", window="hamming")
```
Output results showing AAI index and its category as well as all the associated metric values for each predictive model:
|    | Index      | Category   |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:-----------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | CHOP780206 | sec_struct | 0.62737  | 3.85619 | 14.8702 | 1.63818 | 3.16755 |        0.713467 |


### Encoding using all 566 AAIndex indices
Encoding protein sequences in dataset using all 566 indices in the AAI database. At each iteration, the encoded sequences using the indices from the AAI will be used to generate an protein spectra using the imaginary spectrum with a blackman window function applied, this will then be used as feature data to build a predictive model that can be used for accurate prediction of the sought activity value of unseen protein sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.

```python

#create instance of Encoding class, inherits from pySAR class, using RandomForest algorithm
#   with 200 estimators and a max_depth of 50.
encoding = Encoding(dataset="dataset.txt", activity="activity_col",
  algorithm="RandomForest", parameters={"n_estimators":"200","max_depth":"50"})

#encode sequences using all indices in the AAI
aai_encoding = encoding.aai_encoding(spectrum='imaginary', window='blackman')

```
Output results showing AAI index and its category as well as all the associated metric values for each predictive model:
|    | Index      | Category   |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:-----------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | CHOP780206 | sec_struct | 0.62737  | 3.85619 | 14.8702 | 1.63818 | 3.16755 |        0.713467 |
|  1 | QIAN880131 | sec_struct | 0.626689 | 3.90576 | 15.255  | 1.63668 | 3.09849 |        0.631582 |
|  2 | QIAN880118 | sec_struct | 0.625156 | 3.99581 | 15.9665 | 1.63333 | 3.32038 |        0.625897 |
|  3 | PRAM900104 | sec_struct | 0.615866 | 3.90389 | 15.2403 | 1.61346 | 3.24906 |        0.617799 |
| .. | .......... | .......... | ........ | ....... | ....... | ....... | ....... | ............... |

### Encoding using list of 4 AAI indices, with no DSP functionalities
Same procedure as prior, except 4 indices from the AAI are being specifically input into the function, with the encoded sequence output being concatenated together and used as feature data to build the predictive PlsRegression model with its default parameters.

```python

encoding = Encoding(dataset="dataset.txt", activity="activity_col",
  algorithm="PLSRegression", parameters={})

#encode sequences using 4 indices specified by user
aai_encoding = encoding.aai_encoding(use_dsp=False, aai_list=["PONP800102","RICJ880102","ROBB760107","KARS160113"])


```

### Encoding protein sequences using their calculated protein descriptors
Calculate the protein descriptor values for a dataset of protein sequences from the 15 available descriptors in the <em>descriptors</em> module. Use each descriptor as a feature set in the building of the predictive models used to predict the activity value of unseen sequences. By default, function will look for a file called 'descriptors.csv' that contains the pre-calculated descriptor values for a dataset, this filename can be changed according to <em>descriptors_csv</em> input parameter, if file is not found then all descriptor values will be calculated for dataset.
```python

#create instance of Encoding class using AdaBoost algorithm, using 100 estimaors & a learning rate of 1.5
encoding = Encoding(dataset="dataset.txt", activity="activity_col",algorithm="AdaBoost",   
       parameters={"n_estimators":100,"learning_rate":1.5}, descriptors_csv="descriptors.csv")

#building predictive models using all available descriptors
#   calculating evaluation metrics values for models and storing into desc_results_df DataFrame
desc_results_df = encoding.descriptor_encoding()

```

Output results showing the protein descriptor and its group as well as all the associated metric values for each predictive model:
|    | Descriptor              | Group           |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:------------------------|:----------------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | _distribution           | CTD             | 0.721885 | 3.26159 | 10.638  | 1.89621 | 2.60679 |        0.727389 |
|  1 | _geary_autocorrelation  | Autocorrelation | 0.648121 | 3.67418 | 13.4996 | 1.68579 | 2.82868 |        0.666745 |
|  2 | _tripeptide_composition | Composition     | 0.616577 | 3.3979  | 11.5457 | 1.61496 | 2.53736 |        0.675571 |
|  3 | _aa_composition         | Composition     | 0.612824 | 3.37447 | 11.3871 | 1.60711 | 2.79698 |        0.643864 |
|  4 | ......                  | ......          | ......   | ......  | ......  | ......  | ......  |        ......   |


### Encoding using AAI + protein descriptors
Encoding protein sequences in dataset using all 566 indices in the AAI database combined with protein descriptors. All 566 indices can be used in concatenation with 1, 2 or 3 descriptors. E.g: at each iteration the encoded sequences using the indices from the AAI will be used to generate an protein spectra using the power spectrum with no window function applied, this will then be combined with the feature set generated from the dataset's descriptor values and used to build a predictive model that can be used for accurate prediction of the sought activity value of unseen protein sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.
```python

#create instance of Encoding class using RF algorithm, using 100 estimaors with a learning rate of 1.5
encoding = Encoding(dataset="dataset.txt", activity="activity_col",algorithm="AdaBoost",   
       parameters={"n_estimators":100,"learning_rate":1.5}, descriptors_csv="descriptors.csv")

#building predictive models using all available aa_indices + combination of 2 descriptors,
#   calculating evaluation metrics values for models and storing into aai_desc_results_df DataFrame
aai_desc_results_df = encoding.aai_descriptor_encoding(desc_combo=2, spectrum='power', window=None)

```
Output results showing AAI index and its category, the protein descriptor and its group as well as the R2 and RMSE values for each predictive model:
|    | Index      | Category    | Descriptor                 | Descriptor Group     |       R2 |    RMSE |
|---:|:-----------|:------------|:---------------------------|:---------------------|---------:|--------:|
|  0 | ARGP820103 | composition | _conjoint_triad            | Conjoint Triad       | 0.72754  | 3.22135 |
|  1 | ARGP820101 | hydrophobic | _quasi_seq_order           | Quasi-Sequence-Order | 0.722284 | 3.30995 |
|  2 | ARGP820101 | hydrophobic | _seq_order_coupling_number | Quasi-Sequence-Order | 0.722158 | 3.34926 |
|  3 | ANDN920101 | observable  | _seq_order_coupling_number | Quasi-Sequence-Order | 0.70826  | 3.25232 |
|  4 | .....      | .....       | .....                      | .....                | .....    | .....   |


### Generate all protein descriptors
Functionality to calculate ALL 15 descriptor values for a dataset of protein sequences. Output values are stored in dataset set by <em>desc_dataset</em> input parameter. Output will be of the shape N x 9920, where N is the number of protein sequences in the dataset and 9920 is the total number of features calculated from all 15 descriptors. Moving forward pySAR will pull the descriptor values from this file rather than recalculating them.
```python

#calculating all descriptor values and storing in file names 'descriptors.csv'
#     all_desc = True means that all descriptors will be calculated, it is False by default.
desc = Descriptor(protein_seqs = data, desc_dataset = "descriptors.csv",
    all_desc=True)

```
### Get record from AAIndex database
The AAIndex class offers diverse functionalities for obtaining any element from any record in the database. Each record is stored in json format in a class attribute called <em>aaindex_json</em>. You can search for a particular record by its index code, description or reference. You can also get the index category and importantly its associated amino acid values.
```python

import aaindex
aai = AAIndex()

record = aai.get_record_from_code('CHOP780206')   #get full AAI record
category = aai.get_category_from_record('CHOP780206') #get record's category
values = aai.get_values_from_record('CHOP780206')    #get amino acid values from record
refs = aai.get_ref_from_record('CHOP780206')      #get references from record
num_record = aai.get_num_records()                #get total number of records
record_names = aai.get_record_names()             #get list of all record names
```

Directories
-----------------
* `/pySAR/PyBioMed` - package partially forked from https://github.com/gadsbyfly/PyBioMed, used in
the calculation of the protein descriptors.
* `/Results` - stores all calculated results that were generated for the research article, studying the SAR for a thermostability dataset.
* `/pySAR/tests` - unit and integration tests for pySAR.
* `/pySAR/data` - all required data and datasets are stored in this folder.


Tests
-----
To run all tests, from the main pySAR folder run:
```
python3 -m unittest discover
```

To run tests for specific module, from the main pySAR folder run:
```
python -m unittest tests.MODULE_NAME -v
```

Contact
-------
If you have any questions or comments, please contact amckenna41@qub.ac.uk or raise an issue on the [Issues][Issues] tab.

[Back to top](#TOP)

<!-- |Logo| image:: https://raw.githubusercontent.com/pySAR/pySAR/master/pySAR.png -->


[python]: https://www.python.org/downloads/release/python-360/
[numpy]: https://numpy.org/
[pandas]: https://pandas.pydata.org/
[sklearn]: https://scikit-learn.org/stable/
[scipy]: https://www.scipy.org/
[tqdm]: https://tqdm.github.io/
[seaborn]: https://seaborn.pydata.org/
[Issues]: https://github.com/amckenna41/pySAR/issues
