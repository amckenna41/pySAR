
![alt text](https://raw.githubusercontent.com/amckenna41/pySAR/main/pySAR.png)

# pySAR <a name="TOP"></a>
[![pytest](https://github.com/ray-project/tune-sklearn/workflows/Development/badge.svg)](https://github.com/ray-project/tune-sklearn/actions?query=workflow%3A%22Development%22)
![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)
![PythonV](https://img.shields.io/pypi/pyversions/Django?logo=2)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


pySAR is a Python library for analysing Sequence Activity Relationships (SARs) of protein sequences. pySAR offers extensive and verbose functionalities that allow you to numerically encode a dataset of protein sequences using a large abundance of available methodologies and features. The software uses physiochemical and biochemical features from the Amino Acid Index (AAI) database as well as allowing for the calculation of a range of structural protein descriptors.<br>
After finding the optimal technique and feature set at which to encode your dataset of sequences, pySAR can then be used to build a predictive regression model with the training data being that of the encoded sequences and training labels being the experimentally pre-calculated activity values for each protein sequence. The model can then be used to predict the activity/fitness value of a new unseen sequence.

## status
> Development Stage

## To DO List:
- [ ] Add Github Workflow CI thing
- [X] Add Category and Descriptor Group to pySAR results DF.
- [X] Condense comments in functions, remove some whitespace lines
- [ ] Add help function
- [ ] Mention that PyBioMed package duplicated here as it is not available via pyPI and would mean that user would have to install the full pybiomed zip
- [X] raise type errors instead of Value ?
- [X] index errors?
- [X] remove plot func from DSP
- [X] do StanardScaler after every AAIndex encoding and before model building####
- [ ] Change importing globals : import globals / globals.OUTPUT_DIR
- [X] Split up autocorrelation descriptors into their own functions
- [ ] Allow fasta file to be input to Descriptor class?
- [ ] github workflow with Twine that automatically published to pypi
- [ ] provide example script for running on GCP or AWS resources?
- [X] don't return None after raising an exception??
- [ ] add descriptions to each methods in each class
- [X] remove spacing in equals in keyword args in class/function defintiion
- [ ] setters and getters to Evaluate class? using @property
- [ ] add python version badge to readme
- [ ] add pypi badge to readme
- [ ] add introduction to readme
- [ ] add references to descriptor module
- [ ] integrate descriptor and AAIndex when using properties from AAIndex
- [X] look into setup.cfg or setup.py
- [ ] add distance matrices json to dara ? : https://github.com/MartinThoma/propy3/blob/master/propy/QuasiSequenceOrder.py
- [ ] split up QuasiSequenceOrder descriptor into its consitent quasi-seq-order
- [ ] in readme show example usage for each module/class
- [X] change AAI method names from get_feature etc to get_record...
- [X] change get_feature_names to get_feature_desc
- [ ] add AAI category to each AAI record
- [X] change all 'aa_index' to 'aaindex'
- [ ] Add assertion comments to each unit test, got X wanted Y..
- [ ] add test numbers in comments for each block of unit tests.
- [X] Go through each parameters list and refer to its previous reference rather than repeating it.
- [ ] add cutoff index/value again just for testing
- [ ] print out default parameters if using them.
- [X] remove verbose argument - dont need since tqdm prints progress bar
- [ ] add if __name__ == "main" to encoding and pySAR class.
- [ ] split function defs to two lines?
- [ ] publish to conda?
- [ ] pypi logo
- [X] license logo
- [ ] leave = False on 2nd loop
<!-- #maybe split up multiple descriptor names/categorties in results DF into seperate columns -->



## Installation

Install using pip:

```bash
pi3 install pySAR
```

## Usage

### Building predictive model from AAI and or protein descriptors:
e.g the below code will build a PlsRegression model using the AAI index CIDH920105 and the amino acid composition descriptor. The index is passed through a DSP pipeline and is transformed into its informational protein spectra using the power spectra, with a hamming window function applied to the output of the FFT.
spectrum after a window function is applied.

```python
#first-party imports
from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR
from aaindex import  AAIndex
from model import Model
from proDSP import ProDSP
from evaluate import Evaluate
import utils as utils
from plots import plot_reg
import descriptors as desc

pySAR = PySAR(dataset="dataset.txt",seq_col="sequence", activity="activity",algorithm = "PlsRegression", parameters={}, test_split=0.2)

results_df = pySAR.encode_aai_desc(indices="CIDH920105", descriptors="aa_composition", spectrum="power", window="hamming")

```

### Encoding using all 566 AAIndex indices
```python

#create instance of Encoding class, inherits from pySAR class
encoding = Encoding(dataset="dataset.txt", activity="activity_col",
  algorithm="RandomForest", parameters={"n_estimators":"200","max_depth":"50"})

aai_encoding = encoding.aai_encoding(spectrum='imaginary', window='blackman')

```
### Encoding using list of 4 AAIndex indices, with no DSP functionalities
```python

encoding = Encoding(dataset="dataset.txt", activity="activity_col",
  algorithm="PLSRegression", parameters={"":"","":"", })

aai_encoding = encoding.aai_encoding(use_dsp=False, aai_list=["PONP800102","RICJ880102","ROBB760107","KARS160113"])


```

### Encoding using protein descriptors
```python

encoding = Encoding(dataset="dataset.txt", activity="activity_col",
  algorithm="RandomForest", parameters={"":"","":"", }, descriptors_csv="descriptors.csv")

desc_encoding = encoding.desc_encoding(desc_combo = 2, verbose = True)
def descriptor_encoding(self, desc_list=None, desc_combo=1, verbose=True):


```
### Encoding using AAI + protein descriptors
```python

```
### Generate all protein descriptors

```python

  desc = Descriptor(protein_seqs = data, desc_dataset = "descriptors.csv",
      all_desc=True)

```
where protein_seqs is the dataset of protein sequences, desc_dataset is the name
of the ouput csv used to store the calculated descriptors of the protein sequences
and all_desc means that the class will get and calculate all descriptors.

### Get record from AAIndex database

```python

  desc = Descriptor(protein_seqs = data, desc_dataset = "descriptors.csv",
      all_desc=True)

```
## Output Results

| Descriptor  | Index |         | R2  | RMSE | MSE         
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |

## System Requirements ##

Python > 3.6
numpy >= 1.16.6
pandas >= 1.1.0
scikit-learn >= 0.24
scipy >= 1.4.1


## Running Tests ##
To run tests, from the main pySAR folder run:
```
python -m unittest tests.MODULE_NAME -v

```
MODULE_NAME ->

## Directory folders:

* `/pySAR/PyBioMed` - package partially forked from https://github.com/gadsbyfly/PyBioMed, used in
the calculation of the protein descriptors.
* `/Results` - stores all calculated results that were generated for the research article, studying the SAR for a thermostability dataset.
* `/pySAR/tests` - unit and integration tests for pySAR.
* `/pySAR/data` - all required data and datasets are stored in this folder.


# Contact

If you have any questions or comments, please contact: amckenna41@qub.ac.uk @

[Back to top](#TOP)

|Logo| image:: https://raw.githubusercontent.com/pySAR/pySAR/master/pySAR.png


Install required dependencies and packages:
```python
python setup.py install
```
