# pySAR <a name="TOP"></a>

| Logo |
## status
> Development Stage

[![pytest](https://github.com/ray-project/tune-sklearn/workflows/Development/badge.svg)](https://github.com/ray-project/tune-sklearn/actions?query=workflow%3A%22Development%22)

## Add Github Workflow CI thing
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

```
Alternatively, the system also supports inputting data via JSON or YAML format, with the
keys of the input file required to be the same as the input parameters of the pySAR class,
for example:
```json
{
  "dataset": "T50.txt",
  "activity": "T50",
  "sequence_col": "sequence",
  "aa_indices": "ARGP820101",
  "window": "hamming",
  "filter": "",
  "spectrum": "power",
  "descriptors":[
    "aa_compos"
  ],
  "algorithm":"Plsreg",
  "parameters":{},
  "test_split":0.2
}
```
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


## Directory folders:

* `/pySAR/PyBioMed` - package partially forked from https://github.com/gadsbyfly/PyBioMed, used in
the calculation of the protein descriptors.
* `/Resuts` - stores all calculated results from the evaluation of a variety of protein
encoding strategies using pySAR.
* `/pySAR/tests` - unit and integration tests for pySAR
* `/pySAR/data` - all required data and datasets are stored in this folder.



.. |Logo| image:: https://raw.githubusercontent.com/pySAR/pySAR/master/pySAR.png
