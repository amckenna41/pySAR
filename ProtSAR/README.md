# ProtSAR <a name="TOP"></a>

## status
> Development Stage

[![pytest](https://github.com/ray-project/tune-sklearn/workflows/Development/badge.svg)](https://github.com/ray-project/tune-sklearn/actions?query=workflow%3A%22Development%22)

![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)

ProtSAR is a Python library for analysing the sequence activity relationship (SAR)
between proteins. ProtSAR allows the encoding of protein sequences using indices
from the AAIndex database [] via Digital Signal Processing transformations and
through specific physicochemical and structural protein descriptors.

## Installation

To install, clone this repository locally:

```bash
git clone https://github.com/amckenna41/ProtSAR.git
cd ProtSAR
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
from ProtSAR import *
from encoding import *
```

### Building predictive model from AAI and or protein descriptors, e.g the below code
will build a PlsRegression model using the AAIndex Indices CIDH920105 & PALJ810116
and the amino acid composition descriptor. The 2 indices are encoded via the power
spectrum after a window function is applied.

```python

protSAR = ProtSAR(dataset="dataset.txt",seq_col="sequence", activity="activity", aa_indices=
    ["CIDH920105","PALJ810116"], window="hamming", filter="", spectrum="power", descriptors=
    ["aa_comp"] algorithm = "PlsRegression", parameters={}, test_split=0.2)

```
Alternatively, the system also supports inputting data via JSON or YAML format, with the
keys of the input file required to be the same as the input parameters of the ProtSAR class,
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
**when creaitn instance of ProtSAR



### Encoding using AAIndex indices
```python

aa_encoding = encoding.aai_encoding(aaindex,verbose=True)

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

### Encoding using protein descriptors


### Encoding using AAI + protein descriptors



## System Requirements ##

Python > 3.6
numpy >= 1.16.6
pandas >= 1.1.0
scikit-learn >= 0.24
scipy >= 1.4.1


## Directory folders:

* `/data` - stores all required feature data and datasets
* `/models` - stores model output
* `/tests` - tests for class methods and functions
