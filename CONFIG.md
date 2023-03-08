# Config file parameters <a name="TOP"></a>

pySAR works via configuration files that contain the plethora of parameters and variables available. The config files are in JSON format and broken into 5 different subsections: "dataset", "model", "descriptors", "descriptor_properties" and "pyDSP". "dataset" outlines parameters to do with the dataset, "model" consists of all ML model related parameters, "descriptors" specifies what protein physiochemical/structural descriptors to use, "descriptor_properties" is the more granular parameters of some of the aforementioned protein descriptors, with a selection of themn being further tuneable, and "pyDSP" is all parameters related to any of the DSP functionalities in pySAR. <br>

Example configuration file for thermostability.json used in research:

```json
{
    "dataset": [
      {
        "dataset": "thermostability.txt",
        "sequence_col": "sequence",
        "activity": "t50"
      }
  
    ],
  
    "model": [
      {
        "algorithm": "plsregression",
        "parameters": "",
        "test_split": 0.2
      }
  
    ],
    
    "descriptors":
      [
        {
          "descriptors_csv": "descriptors_thermostability.csv",
          "descriptors": {
            "all_desc": 0,
            "aa_composition": 1,
            "dipeptide_comp": 1,
            "tripeptide_comp": 1,
            "moreaubroto_autocorrelation": 1,
            "moran_autocorrelation": 1,
            "geary_autocorrelation": 1,
            "ctd": 1,
            "conjoint_triad": 1,
            "sequence_order_coupling_number": 1,
            "quasi_sequence_order": 1,
            "pseudo_amino_acid_composition": 1,
            "amphiphilic_pseudo_amino_acid_composition": 1
  
        }
  
        }
      ],
  
    "descriptor_properties":[{
  
      "moreaubroto_autocorrelation":[{
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
      }],
      "moran_autocorrelation":[{
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
      }],
      "geary_autocorrelation":[{
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
      }],
      "ctd":[{
        "property": "hydrophobicity",
        "all": 1
      }],
      "sequence_order_coupling_number":[{
        "lag": 30,
        "distance_matrix": "schneider-wrede-physiochemical-distance-matrix.json"
      }],
      "quasi_sequence_order":[{
        "lag": 30,
        "weight": 0.1,
        "distance_matrix": "schneider-wrede-physiochemical-distance-matrix.json"
      }],
      "pseudo_amino_acid_composition":[{
        "lambda": 30,
        "weight": 0.05,
        "properties": []
      }],
      "amphiphilic_pseudo_amino_acid_composition":[{
        "lambda": 30,
        "weight": 0.5
      }]
  
        }],

    "pyDSP":[
      {
        "use_dsp": 0,
        "spectrum": "power",
        "window": {
          "type": "hamming",
          "sym": 1,
          "beta": null,
          "alpha": null,
          "nbar": null,
          "sll": null,
          "norm": null
        },
        "filter": {
          "type": null,
          "window_length": 5,
          "polyorder": 2,
          "deriv": 0,
          "delta": 1,
          "mode": "interp"
        }
      }
    ]
  
  }
```
# Below is an explanation of each of the parameters within the JSON config files:

**Dataset Parameters:**
* `dataset[dataset]` - name of dataset.
* `dataset[sequence_col]` - name of sequence column in dataset holding protein sequences, if left blank 'sequence' will be used by default.
* `dataset[activity]` - name of protein activity column in dataset being studied.

**Model Parameters:**
* `model[algorithm]` - name of ML regression algorithm being utilised.
* `model[parameters]` - parameters of ML regression algorithm being utilised. If left blank, the algorithm's default parameters will be used.
* `model[test_split]` - training/test split for training the ML regression model, between 0 and 1. If left blank, a default split of 0.2 is used. 

**Descriptor Parameters:**
* `descriptors[descriptors_csv]` - path to csv file of pre-calculated descriptor values of a dataset, saves time having to recalculate the features each time.
* `descriptors[descriptors]` - nested json list of all available descriptors and whether they are to be set upon instantiation of the Descriptor class.
* `descriptors[descriptors][all_desc]` - set to 0 or 1 indicating if all descriptors are to be calculated and or imported upon instantiation of the Descriptor class. 
* `descriptors[descriptors][aa_composition]` - calculate or import from descriptors csv the Amino Acid composition descriptor values.
* `descriptors[descriptors][dipeptide_comp]` - calculate or import from descriptors csv the Dipeptide composition descriptor values.
* `descriptors[descriptors][tripeptide_comp]` - calculate or import from descriptors csv the Tripeptide composition descriptor values.
* `descriptors[descriptors][moreaubroto_autocorrelation]` - calculate or import from descriptors csv the Moreau-Broto Autocorreltaion descriptor values.
* `descriptors[descriptors][moran_autocorrelation]` - calculate or import from descriptors csv the Moran Autocorreltaion descriptor values.
* `descriptors[descriptors][geary_autocorrelation]` - calculate or import from descriptors csv the Geary Autocorreltaion descriptor values.
* `descriptors[descriptors][ctd]` - calculate or import from descriptors csv the Composition, Transition and Distribution from CTD descriptor values.
* `descriptors[descriptors][conjoint_triad]` - calculate or import from descriptors csv the Conjoint Triad descriptor values.
* `descriptors[descriptors][sequence_order_coupling_number]` - calculate or import from descriptors csv the Sequence Order Coupling number descriptor values.
* `descriptors[descriptors][quasi_sequence_order]` - calculate or import from descriptors csv the Quasi Sequence Order descriptor values.
* `descriptors[descriptors][pseudo_amino_acid_composition]` - calculate or import from descriptors csv the Pseudo Amino Acid Composition descriptor values.
* `descriptors[descriptors][amphiphilic_pseudo_amino_acid_composition]` - calculate or import from descriptors csv the Amphiphilic Pseudo Amino Acid Composition descriptor values.

**Descriptor Properties Parameters**
* `descriptor_properties[moreaubroto_autocorrelation][lag] / descriptor_properties[moran_autocorrelation][lag] / descriptor_properties[geary_autocorrelation][lag]` - The maximum lag value for each of the autocorrelation descriptors. If invalid value input then a default of 30 is used.
* `descriptor_properties[moreaubroto_autocorrelation][properties] / descriptor_properties[moran_autocorrelation][properties] / descriptor_properties[geary_autocorrelation][properties]` - List of protein physiochemical and structural descriptors used in the calculation of each of the autocorrelation descriptors, properties must be a lit of their AAIndex number/accession number. There must be a least 1 property value input.
* `descriptor_properties[moreaubroto_autocorrelation][normalize] / descriptor_properties[moran_autocorrelation][normalize] / descriptor_properties[geary_autocorrelation][normalize]` - rescale/normalize Autocorrelation values into range of 0-1.

* `descriptor_properties[ctd][property]` - list of 1 or more physiochemical properties to use when calculating CTD descriptors. List of available input properties: If no properties input then hydrophobicity used by default.
* `descriptor_properties[ctd][all]` - if True then all 7 of the available physiochemical descriptors will be used when calculating the CTD descriptors. Each proeprty generates 21 features so using all properties will output 147 features. Only 1 property used by default. 


* `descriptor_properties[sequence_order_coupling_number][maxlag]` - maximum lag; length of the protein must be not less than maxlag.
* `descriptor_properties[sequence_order_coupling_number][distance_matrix]` - physiochemical distance matrix for calculating sequence order coupling number.

* `descriptor_properties[quasi_sequence_order][maxlag]` - maximum lag; length of the protein must be not less than maxlag.
* `descriptor_properties[quasi_sequence_order][weight]` - weighting factor to use when calculating descriptor.
* `descriptor_properties[quasi_sequence_order][distance_matrix]` - path to physiochemical distance matrix for calculating quasi sequence order.

* `descriptor_properties[pseudo_amino_acid_composition][lambda]` - lamda parameter that reflects the rank correlation and should be a non-negative integer and not larger than the length of the protein sequence.
* `descriptor_properties[pseudo_amino_acid_composition][weight]` - weighting factor to use when calculating descriptor.
* `descriptor_properties[pseudo_amino_acid_composition][properties]` - 1 or more amino acid index properties from the AAI database used for calculating the sequence-order.

* `descriptor_properties[amphiphilic_pseudo_amino_acid_composition][lambda]` - lamda parameter that reflects the rank correlation and should be a non-negative integer and not larger than the length of the protein sequence.
* `descriptor_properties[amphiphilic_pseudo_amino_acid_composition][weight]` - weighting factor to use when calculating descriptor.


**DSP Parameters:**
* `pyDSP[use_dsp]` - whether or not to apply Digital Signal Processing (DSP) techniques to the features passed into the model. If true, the values of the next DSP parameters will be applied to the features. 
* `pyDSP[spectrum]` - which frequency output to use from the generated types of signals from DSP to use e.g power, absolute, imaginery, real. 
* `pyDSP[window]` - convolutional window to apply to the signal output, pySAR supports: hamming, blackman, blackmanharris, gaussian, bartlett, kaiser, barthann, bohman, chebwin, cosine, exponential, flattop, hann, boxcar, hanning, nuttall, parzen, triang, tukey.
* `pyDSP[filter]` - window filter to apply to the signal output, pySAR supports: savgol, medfilt, symiirorder1, lfilter, hilbert.
