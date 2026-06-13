# pySAR - Python Sequence Activity Relationship <a name="TOP"></a>

[![PyPI](https://img.shields.io/pypi/v/pySAR)](https://pypi.org/project/pySAR/)
[![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)](https://pypi.org/project/pySAR/)
[![PythonV](https://img.shields.io/pypi/pyversions/pySAR?logo=2)](https://pypi.org/project/pySAR/)
[![Documentation Status](https://readthedocs.org/projects/pysar/badge/?version=latest)](https://pysar.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

Usage
=====
### Config File
`pySAR` works mainly via JSON configuration files. There are many different customisable parameters for the functionalities in `pySAR` including the metaparameters of some of the available protein descriptors, all Digital Signal Processing (DSP) parameters in the `pyDSP` module, the type of regression model to use and parameters specific to the dataset - a description of each parameter is available on the [CONFIG.md][config] file. 

These config files offer a more straightforward way of making any changes to the `pySAR` pipeline. The names of **All** the parameters as listed in the example config files must remain unchanged, only the value of each parameter should be changed, any parameters not being used can be set to <em>null</em>. Additionally, you can pass in the individual parameter names and values to the `pySAR` and `Encoding` classes when numerically encoding the protein sequences via **kwargs**. An example of the config file used in my research project ([thermostability.json](https://github.com/amckenna41/pySAR/blob/master/config/thermostability.json)), with most of the available parameters, can be seen below and in the example config file - [CONFIG.md][config].

```json
{
    "dataset": 
      {
        "dataset": "thermostability.txt",
        "sequence_col": "sequence",
        "activity": "T50"
      },
    "model": 
      {
        "algorithm": "plsregression",
        "parameters": "",
        "test_split": 0.2
      },
    "descriptors":
        {
          "descriptors_csv": "descriptors_thermostability.csv",
          "moreaubroto_autocorrelation":
            {
            "lag":30,
            "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
              "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
            "normalize": 1
            },
            ...
      },
    "pyDSP":
      {
        "use_dsp": 1,
        "spectrum": "power",
        "window": {
          "type": "hamming",
          ...
        },
        "filter": {
          "type": null,
          ...
        }
      }
  }
```

### Examples...

<details><summary><b>Encoding protein sequences using all 566 AAIndex indices:</b></summary><br>
Encoding protein sequences in dataset using all <b>566 indices</b> in the <b>AAI1 database</b>. Each sequence encoded via an index in the AAI can be passed through an additional step where its protein spectra can be generated following an <b>FFT</b>. pySAR supports generation of the power, imaginary, real or absolute spectra as well as other DSP functionalities including windowing and filter functions.

In the example below, the encoded sequences will be used to generate a **imaginary protein spectra** with a **blackman** window function applied. This will then be used as feature data to build a predictive regression ML model that can be used for accurate prediction of the sought activity value (**thermostability**) of unseen protein sequences. The encoding class also takes the **JSON config file** as input which will have all the required parameter values. The output results will show the calculated metric values for each index in the AAI when measuring predicted vs observed activity values for the unseen test sequences.<br>

```python
from pySAR.encoding import Encoding

'''thermostability.json
{
  "dataset": 
    {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
    }
  "model": 
    {
    "algorithm": "randomforest",
    ...
    }
  "pyDSP": 
    {
    "use_dsp": 1,
    "spectrum": "imaginary",
    "window": {
      "type": "blackman"
      }
    }
}
'''
#create instance of Encoding class, using RF algorithm with its default params
encoding = Encoding(config_file='thermostability.json')

#encode sequences using all indices in the AAI if input parameter "aai_indices" is empty/None
aai_encoding = encoding.aai_encoding()

```
Output results showing AAI index and its category as well as all the associated metric values for each predictive model. From the results below we can determine that the **CHOP780206** index in the AAI has the highest predictability (**R2 score**) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of new unseen sequences:

|    | Index      | Category   |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Variance |
|---:|:-----------|:-----------|---------:|--------:|--------:|--------:|--------:|---------------------:|
|  0 | CHOP780206 | secondary_struct | 0.62737  | 3.85619 | 14.8702 | 1.63818 | 3.16755 |             0.713467 |
|  1 | QIAN880131 | secondary_struct | 0.626689 | 3.90576 | 15.255  | 1.63668 | 3.09849 |             0.631582 |
|  2 | QIAN880118 | secondary_struct | 0.625156 | 3.99581 | 15.9665 | 1.63333 | 3.32038 |             0.625897 |
|  3 | PRAM900104 | secondary_struct | 0.615866 | 3.90389 | 15.2403 | 1.61346 | 3.24906 |             0.617799 |
| .. | .......... | .......... | ........ | ....... | ....... | ....... | ....... |          ........... |
</details>

<details><summary><b>Encoding using list of 4 AAI indices, with no DSP functionalities:</summary></b><br>
This method follows a similar procedure as the previous step, except <b>4 indices</b> from the AAI are being specifically input into the function, with the encoded sequence output being concatenated together and used as feature data to build the predictive <b>PLSRegression</b> model with its default parameters. The config parameter <em> use_dsp </em> tells the function to not generate the protein spectra or apply any additional DSP processing to the sequences.<br>

```python
from pySAR.encoding import Encoding

'''thermostability.json
{
  "dataset": 
    {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
    }
  "model": 
    {
    "algorithm": "plsreg",
    "parameters": null
    }
  "pyDSP": 
    {
    "use_dsp": 0,
    ...
    }
}
'''
#create instance of Encoding class, using PLS algorithm with its default params
encoding = Encoding(config_file='thermostability.json')

#encode sequences using 4 indices specified by user, use_dsp = False
aai_encoding = encoding.aai_encoding(aai_indices=["PONP800102","RICJ880102","ROBB760107","KARS160113"])

```
Output DataFrame showing the 4 predictive models built using the PLS algorithm, with the 4 indices from the AAI. From the results below we can determine that the **PONP800102** index in the AAI has the highest predictability (<b>R2 score</b>) for our chosen dataset (<b>thermostability</b>) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Index      | Category    |       R2 |    RMSE |      MSE |     RPD |     MAE |   Explained Variance |
|---:|:-----------|:------------|---------:|--------:|---------:|--------:|--------:|---------------------:|
|  0 | PONP800102 | hydrophobic | 0.74726  | 3.0817  |  9.49688 | 1.98913 | 2.63742 |             0.751032 |
|  1 | ROBB760107 | secondary_struct  | 0.666527 | 3.19801 | 10.2273  | 1.73169 | 2.50305 |             0.668255 |
|  2 | RICJ880102 | secondary_struct  | 0.568067 | 3.83976 | 14.7438  | 1.52157 | 3.01342 |             0.568274 |
|  3 | KARS160113 | meta        | 0.544129 | 4.04266 | 16.3431  | 1.48108 | 3.26047 |             0.544693 |

</details>

<details><summary><b>Encoding protein sequences using all available protein descriptors:</summary></b><br>
Calculate the protein descriptor values for a dataset of protein sequences from the 33 available descriptors in the <em>descriptors</em> module. Use each descriptor as a <b>feature set</b> in the building of the predictive ML models used to predict the <b>activity</b> value of unseen sequences. By default, the function will look for a csv file pointed to by the <em>"descriptors_csv"</em> parameter in the config file that contains the pre-calculated descriptor values for a dataset. If file is not found then all descriptor values will be calculated for the dataset using the <em>descriptors</em> module and custom-built <i>protpy</i> package.

```python
from pySAR.encoding import Encoding

'''thermostability.json
{
  "dataset": 
    {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
    }
  "model": 
    {
    "algorithm": "adaboost",
    "parameters": [{
      "estimators": 100,
      "learning_rate": 1.5
      ...
    },
  "descriptors": 
  {
    "descriptors_csv": "descriptors_thermostability.csv",
    "moreaubroto_autocorrelation": {
      "lag": 30,
      "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
        "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
      "normalize": 1
    },
    ...
  }
}
'''
#create instance of Encoding class using AdaBoost algorithm, using 100 estimators & a learning rate of 1.5
encoding = Encoding(config_file='thermostability.json')

#building predictive models using all available descriptors, calculating evaluation metrics values for 
# models and storing into desc_results_df DataFrame
desc_results_df = encoding.descriptor_encoding()
```
Output results showing the protein descriptor and its group as well as all the associated metric values for each predictive model. From the results below we can determine that the **CTD Distribution** descriptor has the highest predictability (<b>R2 score</b>) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Descriptor              | Group           |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Variance |
|---:|:------------------------|:----------------|---------:|--------:|--------:|--------:|--------:|---------------------:|
|  0 | ctd_d           | CTD             | 0.721885 | 3.26159 | 10.638  | 1.89621 | 2.60679 |             0.727389 |
|  1 | geary_autocorrelation  | Autocorrelation | 0.648121 | 3.67418 | 13.4996 | 1.68579 | 2.82868 |             0.666745 |
|  2 | tripeptide_composition | Composition     | 0.616577 | 3.3979  | 11.5457 | 1.61496 | 2.53736 |             0.675571 |
|  3 | amino_acid_composition         | Composition     | 0.612824 | 3.37447 | 11.3871 | 1.60711 | 2.79698 |             0.643864 |
|  4 | ......                  | ......          | ......   | ......  | ......  | ......  | ......  |          ......      |
</details>

<details><summary><b>Encoding using AAI + protein descriptors:</summary></b><br>
Encoding protein sequences in the dataset using <b>ALL 566 indices</b> in the AAI database combined with <b>ALL available protein descriptors</b>. All 566 indices can be used in concatenation with 1, 2 or 3 descriptors. At each iteration the encoded sequences generated from the indices from the AAI will be combined with the <b>feature set</b> generated from the dataset's descriptor values and used to build a predictive regression ML model that can be used for the accurate prediction of the sought <b>activity/fitness</b> value of unseen protein sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.<br>

```python
from pySAR.encoding import Encoding

'''thermostability.json
{
  "dataset": 
  {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
  }
  "model": 
  {
    "algorithm": "randomforest",
    "parameters": 
      {
      "estimators": 100,
      "learning_rate": 1.5,
      ...
      }
  },
  "descriptors": 
  {
    "descriptors_csv": "descriptors_thermostability.csv",
    "moreaubroto_autocorrelation": {
      "lag": 30,
      "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
        "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
      "normalize": 1
    },
    ...
  },
  "pyDSP": 
  {
    "use_dsp": 0,
    "spectrum": "power",
    "window": ""
    ...
  }
}
'''
#create instance of Encoding class using RF algorithm, using 100 estimators with a learning rate of 1.5 - as listed in config
encoding = Encoding('thermostability.json')

#building predictive models using all available aa_indices + descriptors, calculating evaluation metric values for models and storing into aai_desc_results_df DataFrame
aai_desc_results_df = encoding.aai_descriptor_encoding()
```

Output results showing AAI index and its category, the protein descriptor and its group as well as all output metric values for each predictive model. From the results below we can determine that the **ARGP820103** index in concatenation with the **Conjoint Triad** descriptor has the highest predictability (<b>R2 score</b>) for our chosen dataset (<b>thermostability</b>) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Index      | Category    | Descriptor                 | Descriptor Group     |       R2 |    RMSE |
|---:|:-----------|:------------|:---------------------------|:---------------------|---------:|--------:|
|  0 | ARGP820103 | composition | _conjoint_triad            | Conjoint Triad       | 0.72754  | 3.22135 |
|  1 | ARGP820101 | hydrophobic | _quasi_seq_order           | Quasi-Sequence-Order | 0.722284 | 3.30995 |
|  2 | ARGP820101 | hydrophobic | _seq_order_coupling_number | Quasi-Sequence-Order | 0.722158 | 3.34926 |
|  3 | ANDN920101 | observable  | _seq_order_coupling_number | Quasi-Sequence-Order | 0.70826  | 3.25232 |
|  4 | .....      | .....       | .....                      | .....                | .....    | .....   |
</details>


<details><summary><b>Building predictive model from subset of AAI and protein descriptors:</summary></b><br>
The below code will build a <b>PLSRegression model</b> using the AAI index <b>CIDH920105</b> and the <b>amino acid composition</b> descriptor. The index is passed through a DSP pipeline and is transformed into its <b>informational protein spectra</b> using the <b>power spectra</b>, with a <b>hamming window</b> function applied to the output of the FFT. The concatenated features from the AAI index and the descriptor will be used as the feature data in building the PLS ML model. This model is then used to access its <b>predictability</b> by testing on test unseen sequences. The output results will show the calculated metric values when measuring <b>predicted vs observed activity values</b> for the test sequences.<br>

```python
from pySAR.pySAR import PySAR

'''thermostability.json
{
  "dataset": 
  {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
  },
  "model": 
  {
    "algorithm": "plsregression",
    "parameters": "",
    ...
  },
  "descriptors": 
  {
    "descriptors_csv": "descriptors_thermostability.csv",
    "moreaubroto_autocorrelation": {
      "lag": 30,
      "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
        "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
      "normalize": 1
    },
    ...
  },
  "pyDSP": 
  {
    "use_dsp": 1,
    "spectrum": "power",
    "window": "hamming",
    ...
  }
}
'''
#create instance of PySAR class, inputting path to configuration file
pySAR = PySAR(config_file="thermostability.json")

#encode protein sequences using both the CIDH920105 index + aa_composition descriptor
results_df = pySAR.encode_aai_descriptor(aai_indices="CIDH920105", descriptors="amino_acid_composition")
```

Output results showing AAI index and its category, the protein descriptor and its group as well as the metric values for the generated predictive model. From the results below we can determine that the **CIDH920105** index in concatenation with the **Amino Acid Composition** descriptor has medium predictability (R2 score) but a high error rate (MSE/RMSE) for our chosen dataset (thermostability) and this feature set combination is not that effective for predicting the thermostability of unseen sequences:

```python 
##########################################################################################
###################################### Parameters ########################################

# AAI Indices: CIDH920105
# Descriptors: amino_acid_composition
# Configuration File: thermostability_config.json
# Dataset: thermostability.txt
# Number of Sequences/Sequence Length: 261 x 466             
# Target Activity: T50
# Algorithm: PLSRegression
# Model Parameters: {'copy': True, 'max_iter': 500, 'n_components': 2, 'scale': True,
#'tol': 1e-06}
# Test Split: 0.2
# Feature Space: (261, 486)

##########################################################################################
######################################## Results #########################################

# R2: 0.6720111107323943
# RMSE: 3.7522525079464457 
# MSE: 14.079398883390391 
# MAE: 3.0713217158459805
# RPD 1.7461053136208489
# Explained Variance 0.6721157080699659

##########################################################################################
```
</details>

<details><summary><b>Calculate individual descriptor values, e.g Tripeptide Composition and Geary Autocorrelation:</summary></b><br>
The individual protein descriptor values for the dataset of protein sequences can be calculated using the custom-built <b>protpy</b> package via the <i>descriptor</i> module. The full list of descriptors can be seen via the function <i>all_descriptors_list()</i> as well as on the <b>protpy</b> repo homepage. 

```python
from pySAR.descriptors import Descriptors

#create instance of descriptors class
desc = Descriptors(config_file="thermostability.json")

#calculate tripeptide composition descriptor
tripeptide_composition = desc.get_tripeptide_composition()

#calculate geary autocorrelation descriptor
geary_autocorrelation = desc.get_geary_autocorrelation()
```
</details>

<details><summary><b>Parallel descriptor computation using n_jobs:</b></summary><br>
The <code>Descriptors</code> class accepts an <code>n_jobs</code> parameter that controls the number of worker threads used during computation. Setting it above 1 enables two levels of parallelism: descriptor groups are dispatched concurrently when calling <code>get_all_descriptors</code>, and individual sequences within each descriptor are processed in parallel threads. Values of 0 or below are silently clamped to 1 (sequential). The results are numerically identical to sequential computation.<br>

```python
from pySAR.descriptors import Descriptors

# sequential computation (default) — n_jobs=1
desc_seq = Descriptors(config_file="thermostability.json", n_jobs=1)

# parallel computation using 4 threads
desc_par = Descriptors(config_file="thermostability.json", n_jobs=4)

# both paths produce identical results
aa_comp = desc_par.get_amino_acid_composition()    # shape: (N, 20)

# parallel is most effective when computing all descriptors at once
all_desc = desc_par.get_all_descriptors()          # shape: (N, 10572+)
```
</details>

<details><summary><b>Calculate and export all protein descriptors:</summary></b><br>
Prior to evaluating the various available properties and features at which to encode a set of protein sequences, it is reccomened that you pre-calculate all the available descriptors in one go, saving them to a csv for later that <i>pySAR</i> will then import from. Output values are stored in a csv set by the <i>descriptors_csv</i> config parameter (the name of the exported csv via the <i>descriptors_export_filename</i> parameter can also be passed into the function). Output will be of the shape N x M, where N is the number of protein sequences in the dataset and M is the total number of features calculated from all 33 descriptors which varies depending on some descriptor-specific metaparameters. For example, using the thermostability dataset, the output will be 261 x 10572. <br>

```python
'''thermostability.json
{
  "dataset": 
  {
    "dataset": "thermostability.txt",
    "activity": "T50"
    ...
  },
  "model": 
  {
    ...
  }
  "descriptors": 
  {
    "descriptors_csv": "descriptors_thermostability.csv",
    "moreaubroto_autocorrelation": {
      "lag": 30,
      "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
        "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
      "normalize": 1
    },
    ...
  },
  "pyDSP": 
  {
    ...
  }
}
'''
#import descriptors class
from pySAR.descriptors import Descriptors

#create instance of descriptors class
desc = Descriptors(config_file="thermostability.json")

#export all descriptors to csv using parameters in config, export=True will export to csv
desc.get_all_descriptors(export=True, descriptors_export_filename="descriptors_thermostability.csv")
```
</details>

<details><summary><b>Get record from AAIndex database:</summary></b><br>
A custom-built package called <b>aaindex</b> was created for this project to work with all the data in the AAIndex databases, primarily the <b>aaindex1</b>. The AAIndex library offers diverse functionalities for obtaining all data from all records within the <b>aaindex1</b>. Each record is stored in json format and can be retrieved via its accession number, and can also be searched via its name/description. Each record contains the following attributes: description, references, category, notes, correlation coefficient, pmid and values.<br>

```python
from aaindex import aaindex1 

record = aaindex1['CHOP780206'] #get full record
description = aaindex1['CHOP780206'].description #get record's description
refs = aaindex1['CHOP780206'].references     #get record's references 
category = aaindex1['CHOP780206'].category #get record's category
notes = aaindex1['CHOP780206'].notes #get record's notes
correlation_coefficients = aaindex1['CHOP780206'].correlation_coefficients #get record's correlation coefficients
pmid = aaindex1['CHOP780206'].pmid #get record's pmid
values = aaindex1['CHOP780206'].values  #get amino acid values from record

num_record = aaindex1.num_records()  #get total number of records
record_names = aaindex1.record_names() #get list of all record names
amino_acids = aaindex1.amino_acids() #get list of all canonical amino acids
records = aaindex1.search("hydrophobicity") #get all records with hydrophobicity in their title/description
```
</details>

<details><summary><b>Parallel encoding across all AAI indices using n_jobs:</b></summary><br>
Setting <code>n_jobs</code> to a value greater than 1 distributes model-building across multiple CPU cores.  Pass <code>n_jobs=-1</code> to use all available cores. This applies to all three encoding methods and can significantly reduce wall-clock time when evaluating hundreds of indices or descriptor combinations.<br>

```python
from pySAR.encoding import Encoding, SortKey

encoding = Encoding(config_file='thermostability.json')

# build 566 AAI models in parallel using all CPU cores, sorted by RMSE
aai_results = encoding.aai_encoding(n_jobs=-1, sort_by=SortKey.RMSE)

# build all descriptor models in parallel using 4 workers
desc_results = encoding.descriptor_encoding(n_jobs=4)

# build AAI + descriptor models in parallel - can be many thousands of models
aai_desc_results = encoding.aai_descriptor_encoding(n_jobs=-1, max_models=1000)
```
For reproducible parallel runs, pass `random_state` to seed the ML models:

```python
aai_results = encoding.aai_encoding(n_jobs=-1, random_state=42)
```
</details>

<details><summary><b>Resuming a partially-completed encoding run:</b></summary><br>
Long encoding jobs (e.g. all 566 AAI indices or thousands of AAI+descriptor combinations) can be interrupted and resumed without re-running completed models. Enable checkpointing by passing <code>resume=True</code> and a path for the checkpoint file via <code>resume_file</code>. On the first run a checkpoint CSV is written after each batch; subsequent runs with the same file skip already-completed keys and append the new results.<br>

```python
from pySAR.encoding import Encoding

encoding = Encoding(config_file='thermostability.json')

# first run - starts from scratch and saves progress to checkpoint.csv after each index
aai_results = encoding.aai_encoding(
    resume=True,
    resume_file='aai_checkpoint.csv',
    n_jobs=4
)

# later run (e.g. after an interruption) - skips completed indices automatically
aai_results = encoding.aai_encoding(
    resume=True,
    resume_file='aai_checkpoint.csv',
    n_jobs=4
)
```
The same pattern works for `descriptor_encoding` and `aai_descriptor_encoding`:

```python
aai_desc_results = encoding.aai_descriptor_encoding(
    resume=True,
    resume_file='aai_desc_checkpoint.csv',
    n_jobs=-1
)
```
</details>

<details><summary><b>Using descriptor validation and utility methods:</b></summary><br>
The <code>Descriptors</code> class exposes several utility methods for validating inputs, inspecting descriptor metadata and managing internal state. These can be used independently of the main encoding workflow.<br>

```python
from pySAR.descriptors import Descriptors, DescriptorType
from pySAR.descriptors import InvalidDescriptorError, InvalidSequenceError

desc = Descriptors(config_file='thermostability.json')

# validate a list of descriptor names - raises InvalidDescriptorError for unknown names
valid_descs = desc.validate_descriptors(['amino_acid_composition', 'dipeptide_composition'])

# validate sequences in the loaded dataset - raises InvalidSequenceError for non-canonical amino acids
desc.validate_sequences()

# retrieve metadata (feature count, group, and parameters) for a descriptor
info = desc.get_descriptor_info('amino_acid_composition')
print(info)
# {'name': 'amino_acid_composition', 'group': 'Composition', 'feature_count': 20, 'parameters': {}}

# get the total number of features produced by the current descriptor configuration (per descriptor)
total_features = desc.descriptor_feature_count  # cached property; returns dict of {name: count}

# get the list of output column names for a specific descriptor (must be calculated first)
desc.get_amino_acid_composition()  # calculate it first
cols = desc.get_descriptor_columns('amino_acid_composition')

# reset all descriptor DataFrames back to empty (useful before re-calculation workflows)
desc.reset_descriptors()

# clear the internal feature-count cache (e.g. after changing descriptor metaparameters)
desc.clear_cache()
```

Use the `DescriptorType` enum to filter or identify descriptor families:

```python
from pySAR.descriptors import DescriptorType

# enum members: COMPOSITION, AUTOCORRELATION, SEQUENCE_ORDER, PSEUDO_AA, CTD, CONJOINT_TRIAD
print(DescriptorType.COMPOSITION.value)   # 'composition'
```
</details>


[Back to top](#TOP)