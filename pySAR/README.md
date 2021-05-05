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


### Generate all protein descriptors

```python

  desc = Descriptor(protein)

```

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






#
# #create instance of Descriptors class using data in instance variable
# descr = desc.Descriptors(self.data[self.seq_col])
#
# #get closest valid available descriptors from input descriptor parameter,
# #   if a list of descriptors passed in as the input parameter then get
# #       all valid descriptors in list
# if isinstance(self.descriptors, list):
#     for de in range(0,len(self.descriptors)):
#         desc_matches = get_close_matches(self.descriptors[de],
#             descr.valid_descriptors(),cutoff=0.4)
#         self.descriptors[de] = desc_matches[0]
# else:
#     desc_matches = get_close_matches(self.descriptors,descr.valid_descriptors(),cutoff=0.4)
#     self.descriptors = desc_matches[0]
#
# #initialise temp lists and DF to store encoded descriptor values
# encoded_desc_temp = []
# encoded_desc_vals = []
# encoded_desc_temp = pd.DataFrame()
#
# #if single descriptor passed in, not a list, get descriptor values for sequences
# if not isinstance(self.descriptors, list):
#     encoded_desc_temp = descr.get_descriptor_encoding(self.descriptors)
#
#     #raise value error if descriptor is empty
#     if (encoded_desc_temp.empty):
#         raise ValueError('Descriptors {} cannot be empty or None'.format(self.descriptors))
#
#     encoded_desc = encoded_desc_temp
#
# #if list of descriptors passed as input, iterate and get each descriptors' values
# else:
#     for d in range(0,len(self.descriptors)):
#       encoded_desc_temp = descr.get_descriptor_encoding(self.descriptors[d])
#
#         #raise value error if descriptor is empty
#       if (encoded_desc_temp.empty):
#           raise ValueError('Descriptors {} cannot be empty or None'.format(self.descriptors[d]))
#             # self.descriptors[d]))
#
#       encoded_desc_vals.append(encoded_desc_temp)
#       encoded_desc_temp = pd.DataFrame()   #reset to empty dataframe
#
#     #concatenate dataframes of descriptors
#     encoded_desc = pd.concat(encoded_desc_vals,axis=1)
#
# X = encoded_desc    #features for training




def get_all_metrics(self):
    """
    Calculate all metrics for inputted predicted and observed class labels,
    return a dict with the keys as the metrics names and values as the metric
    values.

    Returns
    -------
    all_metrics_dict : dict
        dictionary of all calculated metrics values.
    """
    #initialise keys and values for metrics dict
    keys = ['R2', 'RMSE', 'MSE', 'MAE', 'RPD','Explained Var']
    vals = [self.r2, self.rmse, self.mse, self.mae, self.rpd, self.explained_var]

    #zip keys and values into a dictionary
    all_metrics_dict = dict(zip(keys,vals))

    return all_metrics_dict

#aai_descriptor_encoding from encoding.py
    # desc_ = descriptor_list

    #concatenate each descriptor dataframe into one
    # if desc_combo == 2:
    #     descriptor_list_concat = np.concatenate((desc_[0],desc_[1]),axis = 1)
    # elif desc_combo == 3:
    #     descriptor_list_concat = np.concatenate((desc_[0],desc_[1],desc_[2]),axis = 1)
    #
    # desc_ = descriptor_list_concat


#descriptor_encoding from encoding.py
#initialise Descriptor object with protein sequences, set all_desc to calculate all descriptors
# desc = Descriptors(self.data[self.seq_col], all_desc = True)


#desc:

    def test_all_descriptors(self):
        """ Testing all descriptors functionality. """

        all_desc = [

        'aa_composition', 'dipeptide_composition', 'tripeptide_composition', \
        'normalized_moreaubroto_autocorrelation','moran_autocorrelation','geary_autocorrelation', \
        'ctd', 'composition', 'transition', 'distribution', 'conjoint_triad', \
        'seq_order_coupling_number','quasi_seq_order'

        ]
        desc = Descriptors(self.all_test_datasets[0])

        val_desc = desc.all_descriptors_list()

        self.assertEqual(len(val_desc), 15, 'Number of valid descriptors \
            should be 15, got {}'.format(len(val_desc)))
        self.assertIsInstance(val_desc, list, 'Valid Descriptors variable \
            should be a list, got {}'.format(type(val_desc)))
        self.assertNotIn("all_descriptors", val_desc,
            'all_descriptors attribute should not be be returned by the all_descriptors \
            list function')


        for d in range(0, len(val_desc)):

            self.assertIn('_'+val_desc[d], list(desc.__dict__.keys(),
                'Descriptor {} not found in available descriptor attributes: {}'
                .format(val_desc[d],list(desc.__dict__.keys()))
            self.assertIn(val_desc[d], all_desc,
                'Descriptor ({}) not found in all descriptors attribute: {}'.format(val_desc[d], all_desc))

        val_desc = desc.all_descriptors_list(desc_combo=2)

        self.assertEqual(len(val_desc), 105,
            'There should be 105 total descriptor combinations, got {}'.format(len(val_desc)))

        val_desc = desc.all_descriptors_list(desc_combo=3)

        self.assertEqual(len(val_desc), 455,
            'There should be 455 total descriptor combinations, got {}'.format(len(val_desc)))
