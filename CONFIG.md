# Config file parameters <a name="TOP"></a>

pySAR works via configuration files that contain the plethora of parameters and variables available. The config files are in JSON format and broken into 5 different subsections: "dataset", "model", "descriptors", "descriptor_parameters" and "pyDSP". "dataset" outlines parameters to do with the dataset, "model" consists of all ML model related parameters, "descriptors" specifies what protein physiochemical/structural descriptors to use, "descriptor_parameters" is the more granular parameters of some of the aforementioned protein descriptors, with a selection of themn being further tuneable, and "pyDSP" is all parameters related to any of the DSP functionalities in pySAR. <br>

Example configuration file for thermostability.json used in research:

```json

"dataset": [
      {
        "dataset": "thermostability.txt",
        "sequence_col": "sequence",
        "activity": "t50"
      }
  
    ],
  
    "model": [
      {
        "algoritm": "plsregression",
        "parameters": "",
        "test_split": 0.2
      }
  
    ],
    
    "descriptors":
      [
        {
          "descriptors_csv": "descriptors.csv",
          "descriptors": {
            "all_desc": 0,
            "aa_composition": 1,
            "dipeptide_comp": 1,
            "tripeptide_comp": 1,
            "normalized_moreaubroto_autocorrelation": 1,
            "moran_autocorrelation": 1,
            "geary_autocorrelation": 1,
            "composition": 1,
            "transition": 1,
            "distribution": 1,
            "conjoint_triad": 1,
            "seq_order_coupling_number": 1,
            "quasi_seq_order": 1,
            "pseudo_aa_composition": 1,
            "amphiphilic_pseudo_aa_composition": 1
  
        }
  
        }
      ],
  
    "descriptor_parameters":[{
  
      "normalized_moreaubroto_autocorrelation":[{
        "lag":30,
        "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
      }],
      "moran_autocorrelation":[{
        "lag":30,
        "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
      }],
      "geary_autocorrelation":[{
        "lag":30,
        "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
      }],
      "composition":[{
        "property":"hydrophobicity"
      }],
      "transition":[{
        "property":"hydrophobicity"
      }],
      "distribution":[{
        "property":"hydrophobicity"
      }],
      "seq_order_coupling_number":[{
        "maxlag":30,
        "distance_matrix":"schneider-wrede-physicochemical-distance-matrix.json"
      }],
      "quasi_seq_order":[{
        "maxlag":30,
        "weight":0.1,
        "distance_matrix":"schneider-wrede-physicochemical-distance-matrix.json"
      }],
      "pseudo_aa_composition":[{
        "lambda":30,
        "weight":0.05,
        "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
      }]
  
        }],

    "pyDSP":[
      {
        "use_dsp": 1,
        "spectrum": "power",
        "window": "hamming",
        "filter": null,
        "convolution": null
      }
    ]
  
  }
  
```

Below is an explanation of each of the parameters within the JSON config files:

* `dataset[dataset]` - name of dataset.
* `dataset[sequence_col]` - name of sequence column in dataset holding protein sequences, if left blank 'sequence' will be used by default.
* `dataset[activity]` - name of protein activity column in dataset being studied.
* `model[algorithm]` - name of ML regression algorithm being utilised.
* `model[parameters]` - parameters of ML regression algorithm being utilised. If left blank, the algorithm's default parameters will be used.
* `model[test_split]` - training/test split for training the ML regression model, between 0 and 1. If left blank, a default split of 0.2 is used. 
* `descriptors[descriptors_csv]` - path to csv file of pre-calculated descriptor values of a dataset, saves time having to recalculate the features each time.
* `descriptors[descriptors]` - nested json list of all available descriptors and whether they are to be set upon instantiation of the Descriptor class.
* `descriptors[descriptors][all_desc]` - set to 0 or 1 indicating if all descriptors are to be calculated and or imported upon instantiation of the Descriptor class. 
* `descriptors[descriptors][aa_composition]` - calculate or import from descriptors csv the Amino Acid composition descriptor values.
* `descriptors[descriptors][dipeptide_comp]` - calculate or import from descriptors csv the Dipeptide composition descriptor values.
* `descriptors[descriptors][tripeptide_comp]` - calculate or import from descriptors csv the Tripeptide composition descriptor values.
* `descriptors[descriptors][normalized_moreaubroto_autocorrelation]` - calculate or import from descriptors csv the Normalized Moreau-Broto Autocorreltaion descriptor values.
* `descriptors[descriptors][moran_autocorrelation]` - calculate or import from descriptors csv the Moran Autocorreltaion descriptor values.
* `descriptors[descriptors][geary_autocorrelation]` - calculate or import from descriptors csv the Geary Autocorreltaion descriptor values.
* `descriptors[descriptors][composition]` - calculate or import from descriptors csv the Composition from CTD descriptor values.
* `descriptors[descriptors][transition]` - calculate or import from descriptors csv the Transition from CTD descriptor values.
* `descriptors[descriptors][distribution]` - calculate or import from descriptors csv the Distribution from CTD descriptor values.
* `descriptors[descriptors][conjoint_triad]` - calculate or import from descriptors csv the Conjoint Triad descriptor values.
* `descriptors[descriptors][seq_order_coupling_number]` - calculate or import from descriptors csv the Sequence Order Coupling number descriptor values.
* `descriptors[descriptors][quasi_seq_order]` - calculate or import from descriptors csv the Quasi Sequence Order descriptor values.
* `descriptors[descriptors][pseudo_aa_composition]` - calculate or import from descriptors csv the Pseudo Amino Acid Composition descriptor values.
* `descriptors[descriptors][amphiphilic_pseudo_aa_composition]` - calculate or import from descriptors csv the Amphiphilic Pseudo Amino Acid Composition descriptor values.


* `descriptor_parameters[normalized_moreaubroto_autocorrelation][lag]/descriptor_parameters[moran_autocorrelation][lag]/descriptor_parameters[geary_autocorrelation][lag]` - The maximum lag value for each of the autocorrelation descriptors. If invalid value input then a default of 30 is used.
* `descriptor_parameters[normalized_moreaubroto_autocorrelation][properties]/descriptor_parameters[moran_autocorrelation][properties]/descriptor_parameters[geary_autocorrelation][properties]` - List of protein physiochemical and structural descriptors used in the calculation of each of the autocorrelation descriptors, properties must be a lit of their AAIndex number/accession number. There must be a least 1 property value input.


**

* `pyDSP[use_dsp]` - 
* `pyDSP[spectrum]` - 
* `pyDSP[window]` - 
* `pyDSP[filter]` - 
* `pyDSP[convolution]` - convolution 

    "pyDSP":[
      {
        "use_dsp": 1,
        "spectrum": "power",
        "window": "hamming",
        "filter": null,
        "convolution": null
      }