# pySAR <a name="TOP"></a>

| Logo |
## status
> Development Stage

## To DO List:
- [ ] Add Github Workflow CI thing
- [ ] Add Category and Descriptor Group to pySAR results DF.
- [ ] Condense comments in functions, remove some whitespace lines
- [ ] Add help function
- [ ] Mention that PyBioMed package duplicated here as it is not available via pyPI and would mean that user would have to install the full pybiomed zip
- [ ] raise type errors instead of Value ?
- [ ] index errors?
##def __repr__(self):
return '<Class Name: {}>'.format(self)
#try except instead of assert
- [ ] remove plot func from DSP
- [ ] dd StanardScaler after every AAIndex encoding and before model building####
- [ ] Change importing globals : import globals / globals.OUTPUT_DIR
- [X] Split up autocorrelation descriptors into their own functions
- [ ] Allow fasta file to be input to Descriptor class?
- [ ] github workflow with Twine that automatically published to pypi
- [ ] provide example script for running on GCP or AWS resources?
- [ ] don't return None after raising an exception??
- [ ] add descriptions to each methods in each class
- [ ] remove spacing in equals in keyword args in class/function defintiion
- [ ] setters and getters to Evaluate class? using @property
- [ ] add python version badge to readme
- [ ] add pypi badge to readme
- [ ] add introduction to readme
- [ ] add references to descriptor module
- [ ] integrate descriptor and AAIndex when using properties from AAIndex
- [ ] look into setup.cfg or setup.py
- [ ] add distance matrices json to dara ? : https://github.com/MartinThoma/propy3/blob/master/propy/QuasiSequenceOrder.py
- [ ] split up QuasiSequenceOrder descriptor into its consitent quasi-seq-order
- [ ] check conjoint triad feature is correct... (512-D but should be 343?) Overall descriptors should be 9920, not 10030
- [ ] add descriptor group and index category to output results DF.
- [ ] in readme show example usage for each module/class
- [ ] change AAI method names from get_feature etc to get_record...
- [ ] change get_feature_names to get_feature_desc etc, i dunno
- [ ] add AAI category to each AAI record
- [ ] change all 'aa_index' to 'aaindex'
- [ ] Add assertion comments to each unit test, got X wanted Y..
- [ ] add test numbers in comments for each block of unit tests.
- [ ] Go through each parameters list and refer to its previous reference rather than repeating it.
<!-- #maybe split up multiple descriptor names/categorties in results DF into seperate columns -->
[![pytest](https://github.com/ray-project/tune-sklearn/workflows/Development/badge.svg)](https://github.com/ray-project/tune-sklearn/actions?query=workflow%3A%22Development%22)

![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)

pySAR is a Python library for analysing the sequence activity relationship (SAR)
between proteins. pySAR allows the encoding of protein sequences using indices
from the AAIndex database [] via Digital Signal Processing transformations and
through specific physicochemical and structural protein descriptors.

## Installation

To install, clone this repository locally:

```bash
git clone https://github.com/amckenna41/pySAR.git
cd pySAR
```

Install required dependencies and packages:
```python
python setup.py install
```

## Usage

### Local imports
```python

from utils import *
from model import *
from proDSP import *
from evaluate import *
from plots import *
from descriptors import *
from pySAR import *
from encoding import *
```

### Building predictive model from AAI and or protein descriptors, e.g the below code
will build a PlsRegression model using the AAIndex Indices CIDH920105 & PALJ810116
and the amino acid composition descriptor. The 2 indices are encoded via the power
spectrum after a window function is applied.

```python

pySAR = PySAR(dataset="dataset.txt",seq_col="sequence", activity="activity", aa_indices=
    ["CIDH920105","PALJ810116"], window="hamming", filter="", spectrum="power", descriptors=
    ["aa_comp"] algorithm = "PlsRegression", parameters={}, test_split=0.2)


**when creaitn instance of pySAR



### Encoding using AAIndex indices
```python

aa_encoding = encoding.aai_encoding(aaindex,verbose=True)

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

### Encoding using protein descriptors


### Encoding using AAI + protein descriptors


### Generate all protein descriptors

```python

  desc = Descriptor(protein_seqs = data, desc_dataset = "descriptors.csv",
    all_desc=True)

```
where protein_seqs is the dataset of protein sequences, desc_dataset is the name
of the ouput csv used to store the calculated descriptors of the protein sequences
and all_desc means that the class will get and calculate all descriptors.


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
* `/Resuts` - stores all calculated results from the evaluation of a variety of protein
encoding strategies using pySAR.
* `/pySAR/tests` - unit and integration tests for pySAR
* `/pySAR/data` - all required data and datasets are stored in this folder.


# Contact

If you have any questions or comments, please contact: amckenna41@qub.ac.uk @

[Back to top](#TOP)

.. |Logo| image:: https://raw.githubusercontent.com/pySAR/pySAR/master/pySAR.png
