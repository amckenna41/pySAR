<p align="center">
<img src="https://raw.githubusercontent.com/amckenna41/pySAR/master/images/pySAR.png" alt="pySARLogo" height="300" width="400"/>
</p>

# pySAR - Python Sequence Activity Relationship #
[![PyPI](https://img.shields.io/pypi/v/pySAR)](https://pypi.org/project/pySAR/)
[![pytest](https://github.com/amckenna41/pySAR/workflows/Building%20and%20Testing/badge.svg)](https://github.com/amckenna41/pySAR/actions?query=workflowBuilding%20and%20Testing)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/amckenna41/pySAR/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/amckenna41/pySAR/tree/master)
[![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)](https://pypi.org/project/pySAR/)
[![PythonV](https://img.shields.io/pypi/pyversions/pySAR?logo=2)](https://pypi.org/project/pySAR/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/amckenna41/pySAR)](https://github.com/amckenna41/pySAR/issues)
[![codecov](https://codecov.io/gh/amckenna41/pySAR/branch/master/graph/badge.svg?token=4PQDVGKGYN)](https://codecov.io/gh/amckenna41/pySAR)
<!-- [![Commits](https://img.shields.io/github/commit-activity/w/amckenna41/pySAR)](https://github.com/amckenna41/pySAR) -->
<!-- [![Size](https://img.shields.io/github/repo-size/amckenna41/pySAR)](https://github.com/amckenna41/pySAR) -->
<!-- [![Build](https://img.shields.io/github/workflow/status/amckenna41/pySAR/Deploy%20to%20PyPI%20%F0%9F%93%A6)](https://github.com/amckenna41/pySAR/actions) -->
<!-- [![Build Status](https://travis-ci.com/amckenna41/pySAR.svg?branch=main)](https://travis-ci.com/amckenna41/pySAR) -->
<!-- [![DOI](https://zenodo.org/badge/344290370.svg)](https://zenodo.org/badge/latestdoi/344290370) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

Table of Contents
=================
  * [Introduction](#Introduction)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Directories](#directories)
  * [Tests](#tests)
  * [Issues](#Issues)
  * [Contact](#contact)
  * [License](#license)
  * [References](#references)


Research Article
================
The research article that accompanied this software is titled: "Machine Learning Based Predictive Model for the Analysis of Sequence Activity Relationships Using Protein Spectra and Protein Descriptors" and was published in the Journal of Biomedical Informatics and is available [here][article] [[1]](#references). There is also a quick <b>Colab notebook demo</b> of `pySAR` available [here][demo].

How to cite
===========
> Mckenna, A., & Dubey, S. (2022). Machine learning based predictive model for the analysis of sequence activity relationships using protein spectra and protein descriptors. Journal of Biomedical Informatics, 128(104016), 104016. https://doi.org/10.1016/j.jbi.2022.104016

Introduction
============
`pySAR` is a Python library for analysing Sequence Activity Relationships (SARs)/Sequence Function Relationships (SFRs) of protein sequences. `pySAR` offers extensive and verbose functionalities that allow you to numerically encode a dataset of protein sequences using a large abundance of available methodologies and features. The software uses physiochemical and biochemical features from the Amino Acid Index (AAI) database [[2]](#references), as well as allowing for the calculation of a range of structural, physiochemical and biochemical protein descriptors, via the custom-built [`protpy`][protpy] package.

After finding the optimal technique and feature set at which to numerically encode your dataset of sequences, `pySAR` can then be used to build a predictive regression ML model with the training data being that of the encoded protein sequences, and training labels being the in vitro experimentally pre-calculated activity values for each protein sequence. This model maps a set of protein sequences to the sought-after activity value, being able to accurately predict the activity/fitness value of new unseen sequences. The use-case for the software is within the field of Protein Engineering, Directed Evolution and or Drug Discovery, where a user has a set of in vitro experimentally determined activity/fitness values for a library of mutant protein sequences and wants to computationally predict the sought activity value for a selection of mutated unseen sequences, in the aim of finding the best sequence that minimises/maximises their activity value. <br>

In the published [research][article], the sought activity/fitness characterisitc is the thermostability of proteins from a recombination library designed from parental cytochrome P450's. This thermostability is measured using the T50 metric (temperature at which 50% of a protein is irreversibly denatured after 10 mins of incubation, ranging from 39.2 to 64.4 degrees C), which we want to maximise [[1]](#references).

Two additional <strong>custom-built</strong> softwares were created alongside `pySAR` - [`aaindex`][aaindex] and [`protpy`][protpy]. The `aaindex` software package is used for parsing the amino acid index which is a database of numerical indices representing various physicochemical and biochemical properties of amino acids and pairs of amino acids [[2]](#references). `protpy` is used for calculating a series of protein physiochemical, biochemical and structural protein descriptors. Both of these software packages are integrated into `pySAR` but can also be used individually for their respective purposes. 

Requirements
============
* [Python][python] >= 3.8
* [aaindex][aaindex] >= 1.1.1
* [protpy][protpy] >= 1.1.10
* [numpy][numpy] >= 1.24.2
* [pandas][pandas] >= 1.5.3
* [scikit-learn][sklearn] >= 1.2.1
* [scipy][scipy] >= 1.10.1
* [tqdm][tqdm] >= 4.65.0
* [matplotlib][matplotlib] >= 3.6.2
* [seaborn][seaborn] >= 0.12.2

Installation
============
Install the latest version of `pySAR` via [PyPi][PyPi] using pip:

```bash
pip3 install pysar --upgrade
```

Installation from source:
```bash
git clone -b master https://github.com/amckenna41/pySAR.git
python3 setup.py install
cd pySAR
```

Usage
=====
### Confile File
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
### Examples

<details><summary><b>Encoding protein sequences using all 566 AAIndex indices:</b></summary><br>
Encoding protein sequences in dataset using all 566 indices in the AAI1 database. Each sequence encoded via an index in the AAI can be passed through an additional step where its protein spectra can be generated following an FFT. pySAR supports generation of the power, imaginary, real or absolute spectra as well as other DSP functionalities including windowing and filter functions. <br>

In the example below, the encoded sequences will be used to generate a imaginary protein spectra with a blackman window function applied. This will then be used as feature data to build a predictive regression ML model that can be used for accurate prediction of the sought activity value (thermostability) of unseen protein sequences. The encoding class also takes the JSON config file as input which will have all the required parameter values. The output results will show the calculated metric values for each index in the AAI when measuring predicted vs observed activity values for the unseen test sequences.<br>

```python
#import encoding module
from pySAR.encoding import *

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
Output results showing AAI index and its category as well as all the associated metric values for each predictive model. From the results below we can determine that the **CHOP780206** index in the AAI has the highest predictability (R2 score) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of new unseen sequences:

|    | Index      | Category   |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:-----------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | CHOP780206 | secondary_struct | 0.62737  | 3.85619 | 14.8702 | 1.63818 | 3.16755 |        0.713467 |
|  1 | QIAN880131 | secondary_struct | 0.626689 | 3.90576 | 15.255  | 1.63668 | 3.09849 |        0.631582 |
|  2 | QIAN880118 | secondary_struct | 0.625156 | 3.99581 | 15.9665 | 1.63333 | 3.32038 |        0.625897 |
|  3 | PRAM900104 | secondary_struct | 0.615866 | 3.90389 | 15.2403 | 1.61346 | 3.24906 |        0.617799 |
| .. | .......... | .......... | ........ | ....... | ....... | ....... | ....... | ............... |
</details>

<details><summary><b>Encoding using list of 4 AAI indices, with no DSP functionalities:</summary></b><br>
This method follows a similar procedure as the previous step, except 4 indices from the AAI are being specifically input into the function, with the encoded sequence output being concatenated together and used as feature data to build the predictive PLSRegression model with its default parameters. The config parameter <em> use_dsp </em> tells the function to not generate the protein spectra or apply any additional DSP processing to the sequences.<br>

```python
#import encoding module
from pySAR.encoding import *

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
Output DataFrame showing the 4 predictive models built using the PLS algorithm, with the 4 indices from the AAI. From the results below we can determine that the **PONP800102** index in the AAI has the highest predictability (R2 score) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Index      | Category    |       R2 |    RMSE |      MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:------------|---------:|--------:|---------:|--------:|--------:|----------------:|
|  0 | PONP800102 | hydrophobic | 0.74726  | 3.0817  |  9.49688 | 1.98913 | 2.63742 |        0.751032 |
|  1 | ROBB760107 | secondary_struct  | 0.666527 | 3.19801 | 10.2273  | 1.73169 | 2.50305 |        0.668255 |
|  2 | RICJ880102 | secondary_struct  | 0.568067 | 3.83976 | 14.7438  | 1.52157 | 3.01342 |        0.568274 |
|  3 | KARS160113 | meta        | 0.544129 | 4.04266 | 16.3431  | 1.48108 | 3.26047 |        0.544693 |

</details>

<details><summary><b>Encoding protein sequences using all available protein descriptors:</summary></b><br>
Calculate the protein descriptor values for a dataset of protein sequences from the 15 available descriptors in the <em>descriptors</em> module. Use each descriptor as a feature set in the building of the predictive ML models used to predict the activity value of unseen sequences. By default, the function will look for a csv file pointed to by the <em>"descriptors_csv"</em> parameter in the config file that contains the pre-calculated descriptor values for a dataset. If file is not found then all descriptor values will be calculated for the dataset using the <em>descriptors</em> module and custom-built <i>protpy</i> package.

```python
#import encoding module
from pySAR.encoding import *

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
Output results showing the protein descriptor and its group as well as all the associated metric values for each predictive model. From the results below we can determine that the **CTD Distribution** descriptor has the highest predictability (R2 score) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Descriptor              | Group           |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:------------------------|:----------------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | ctd_d           | CTD             | 0.721885 | 3.26159 | 10.638  | 1.89621 | 2.60679 |        0.727389 |
|  1 | geary_autocorrelation  | Autocorrelation | 0.648121 | 3.67418 | 13.4996 | 1.68579 | 2.82868 |        0.666745 |
|  2 | tripeptide_composition | Composition     | 0.616577 | 3.3979  | 11.5457 | 1.61496 | 2.53736 |        0.675571 |
|  3 | amino_acid_composition         | Composition     | 0.612824 | 3.37447 | 11.3871 | 1.60711 | 2.79698 |        0.643864 |
|  4 | ......                  | ......          | ......   | ......  | ......  | ......  | ......  |        ......   |
</details>

<details><summary><b>Encoding using AAI + protein descriptors:</summary></b><br>
Encoding protein sequences in the dataset using ALL 566 indices in the AAI database combined with ALL available protein descriptors. All 566 indices can be used in concatenation with 1, 2 or 3 descriptors. At each iteration the encoded sequences generated from the indices from the AAI will be combined with the feature set generated from the dataset's descriptor values and used to build a predictive regression ML model that can be used for the accurate prediction of the sought activity/fitness value of unseen protein sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.<br>

```python
#import encoding module
from pySAR.encoding import *

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

Output results showing AAI index and its category, the protein descriptor and its group as well as all output metric values for each predictive model. From the results below we can determine that the **ARGP820103** index in concatenation with the **Conjoint Triad** descriptor has the highest predictability (R2 score) for our chosen dataset (thermostability) and this generated model can be used for predicting the thermostability of unseen sequences:

|    | Index      | Category    | Descriptor                 | Descriptor Group     |       R2 |    RMSE |
|---:|:-----------|:------------|:---------------------------|:---------------------|---------:|--------:|
|  0 | ARGP820103 | composition | _conjoint_triad            | Conjoint Triad       | 0.72754  | 3.22135 |
|  1 | ARGP820101 | hydrophobic | _quasi_seq_order           | Quasi-Sequence-Order | 0.722284 | 3.30995 |
|  2 | ARGP820101 | hydrophobic | _seq_order_coupling_number | Quasi-Sequence-Order | 0.722158 | 3.34926 |
|  3 | ANDN920101 | observable  | _seq_order_coupling_number | Quasi-Sequence-Order | 0.70826  | 3.25232 |
|  4 | .....      | .....       | .....                      | .....                | .....    | .....   |
</details>

<details><summary><b>Building predictive model from subset of AAI and protein descriptors:</summary></b><br>
The below code will build a PLSRegression model using the AAI index <b>CIDH920105</b> and the <b>amino acid composition</b> descriptor. The index is passed through a DSP pipeline and is transformed into its informational protein spectra using the <b>power spectra</b>, with a hamming window function applied to the output of the FFT. The concatenated features from the AAI index and the descriptor will be used as the feature data in building the PLS ML model. This model is then used to access its predictability by testing on test unseen sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.<br>

```python
#import pySAR module
from pySAR.pySAR import *

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
#import descriptors class
from pySAR.descriptors import *  

#create instance of descriptors class
desc = Descriptors(config_file="thermostability.json")

#calculate tripeptide composition descriptor
tripeptide_composition = desc.get_tripeptide_composition()

#calculate geary autocorrelation descriptor
geary_autocorrelation = desc.get_geary_autocorrelation()
```
</details>

<details><summary><b>Calculate and export all protein descriptors:</summary></b><br>
Prior to evaluating the various available properties and features at which to encode a set of protein sequences, it is reccomened that you pre-calculate all the available descriptors in one go, saving them to a csv for later that <i>pySAR</i> will then import from. Output values are stored in a csv set by the <i>descriptors_csv</i> config parameter (the name of the exported csv via the <i>descriptors_export_filename</i> parameter can also be passed into the function). Output will be of the shape N x M, where N is the number of protein sequences in the dataset and M is the total number of features calculated from all 15 descriptors which varies depending on some descriptor-specific metaparameters. For example, using the thermostability dataset, the output will be 261 x 9714. <br>

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
from pySAR.descriptors import *  

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
correllation_coefficient = aaindex1['CHOP780206'].correllation_coefficient #get record's correllation_coefficient
pmid = aaindex1['CHOP780206'].pmid #get record's pmid
values = aaindex1['CHOP780206'].values  #get amino acid values from record

num_record = aaindex1.num_records()  #get total number of records
record_names = aaindex1.record_names() #get list of all record names
amino_acids = aaindex1.amino_acids() #get list of all canonical amino acids
records = aaindex1.search("hydrophobicity") #get all records with hydrophobicity in their title/description
```
</details>

Directories and Files
=====================
* `/config` - configuration files for the example datasets that `pySAR` has been tested with, as well as the thermostability.json config file that was used in the research. These config files should be used as a template for future datasets used with `pySAR`.
* `/data` - data files used in the research proejct including the thermostability dataset, config file and pre-calculated protein descriptors.
* `/docs` - documentation for `pySAR` (pending).
* `/example_datasets` - example datasets used for the building and testing of `pySAR`, including the thermostability dataset used in the research. The format of these datasets should be used as a template for future datasets used with `pySAR`.
* `/images` - all images used throughout the repo.
* `/pySAR` - source code for `pySAR` software.
* `/tests` - unit and integration tests for `pySAR`.
* `pySAR_research.pdf` - published research article.
* `pySAR_research.pptx` - powerpoint demo of the software development process of pySAR.
* `CONFIG.md` - example markdown file describing each of the available parameters in the config files.

Issues
======
Any issues, errors or bugs can be raised via the [Issues](https://github.com/amckenna41/pySAR/issues) tab in the repository.

Tests
=====
To run all tests, from the main `pySAR` repo folder run:
```
python3 -m unittest discover tests
```

To run tests for specific module, from the main `pySAR` repo folder run:
```
python -m unittest tests.MODULE_NAME -v
```

Contact
=======
If you have any questions or comments, please contact amckenna41@qub.ac.uk or raise an issue on the [Issues][Issues] tab. <br><br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adam-mckenna-7a5b22151/)

License
=======
Distributed under the MIT License. See [`LICENSE`][license] for more details.  

References
==========
\[1\]: Mckenna, A., & Dubey, S. (2022). Machine learning based predictive model for the analysis of sequence activity relationships using protein spectra and protein descriptors. Journal of Biomedical Informatics, 128(104016), 104016. https://doi.org/10.1016/j.jbi.2022.104016<br><br>
\[2\]: Kawashima, S. and Kanehisa, M., 2000. AAindex: amino acid index database. Nucleic acids research, 28(1), pp.374-374. DOI: 10.1093/nar/27.1.368 <br><br>
\[3\]: Fontaine NT, Cadet XF, Vetrivel I. Novel Descriptors and Digital Signal Processing- Based Method for Protein Sequence Activity Relationship Study. Int J Mol Sci. 2019 Nov 11;20(22):5640. doi: 10.3390/ijms20225640. PMID: 31718061; PMCID: PMC6888668. <br><br>
\[4\]: Cadet, F., Fontaine, N., Li, G. et al. A machine learning approach for reliable prediction of amino acid interactions and its application in the directed evolution of enantioselective enzymes. Sci Rep 8, 16757 (2018).<br><br>
\[5\]: Lutz S. Beyond directed evolution--semi-rational protein engineering and design. Curr Opin Biotechnol. 2010 Dec;21(6):734-43. doi: 10.1016/j.copbio.2010.08.011. Epub 2010 Sep 24. PMID: 20869867; PMCID: PMC2982887. <br><br>
\[6\]: Yang, K.K., Wu, Z. & Arnold, F.H. Machine-learning-guided directed evolution for protein engineering. Nat Methods 16, 687–694 (2019). https://doi.org/10.1038/s41592-019-0496-6 <br><br>
\[7\]: Yuting Xu, Deeptak Verma, Robert P. Sheridan, Andy Liaw, Junshui Ma, Nicholas M. Marshall, John McIntosh, Edward C. Sherer, Vladimir Svetnik, and Jennifer M. Johnston
Journal of Chemical Information and Modeling 2020 60 (6), 2773-2790
DOI: 10.1021/acs.jcim.0c00073 <br><br>
\[8\]: Medina-Ortiz, D., Contreras, S., Amado-Hinojosa, J., Torres-Almonacid, J., Asenjo, J. A., Navarrete, M., & Olivera-Nappa, Á. (2020). Combination of digital signal processing and assembled predictive models facilitates the rational design of proteins. ArXiv [Cs.CE]. <br>

<a href="https://www.buymeacoffee.com/amckenna41" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

[Back to top](#TOP)

<!-- |Logo| image:: https://raw.githubusercontent.com/pySAR/pySAR/master/pySAR.png -->

[python]: https://www.python.org/downloads/release/python-360/
[aaindex]: https://github.com/amckenna41/aaindex
[protpy]: https://github.com/amckenna41/protpy
[numpy]: https://numpy.org/
[pandas]: https://pandas.pydata.org/
[sklearn]: https://scikit-learn.org/stable/
[scipy]: https://www.scipy.org/
[tqdm]: https://tqdm.github.io/
[seaborn]: https://seaborn.pydata.org/
[matplotlib]: https://matplotlib.org/
[PyPi]: https://pypi.org/project/pysar/
[article]: https://www.sciencedirect.com/science/article/abs/pii/S1532046422000326
[pdf]: https://github.com/amckenna41/pySAR/blob/master/pySAR_research.pdf
[ppt]: https://github.com/amckenna41/pySAR/blob/master/pySAR_demo.key
[demo]: https://colab.research.google.com/drive/1hxtnf8i4q13fB1_2TpJFimS5qfZi9RAo?usp=sharing
[Issues]: https://github.com/amckenna41/pySAR/issues
[license]: https://github.com/amckenna41/pySAR/blob/master/LICENSE
[config]: https://github.com/amckenna41/pySAR/blob/master/CONFIG.md