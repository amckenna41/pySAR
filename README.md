

<p align="center">
<img src="https://raw.githubusercontent.com/amckenna41/pySAR/master/images/pySAR.png" alt="pySARLogo" height="300" width="400"/>
</p>

# pySAR #
[![PyPI](https://img.shields.io/pypi/v/pySAR)](https://pypi.org/project/pySAR/)
[![pytest](https://github.com/amckenna41/pySAR/workflows/Building%20and%20Testing%20%F0%9F%90%8D/badge.svg)](https://github.com/amckenna41/pySAR/actions?query=workflowBuilding%20and%20Testing%20%F0%9F%90%8D)
[![CircleCI](https://circleci.com/gh/amckenna41/pySAR.svg?style=svg&circle-token=d860bb64668be19d44f106841b80eb47a8b7e7e8)](https://app.circleci.com/pipelines/github/amckenna41/pySAR)
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
-----------------

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
----------------
The research article that accompanied this software is titled: "Machine Learning Based Predictive Model for the Analysis of Sequence Activity Relationships Using Protein Spectra and Protein Descriptors". This research article is uploaded to the repository as [pySAR_research.pdf][pdf]. The article was published in the Journal of Biomedical Informatics and is available [here][article]. There is also a quick jupyter notebook demo of `pySAR` available [here][demo].

How to cite
-----------
> Mckenna, A., & Dubey, S. (2022). Machine learning based predictive model for the analysis of sequence activity relationships using protein spectra and protein descriptors. Journal of Biomedical Informatics, 128(104016), 104016. https://doi.org/10.1016/j.jbi.2022.104016

Introduction
------------
`pySAR` is a Python library for analysing Sequence Activity Relationships (SARs) of protein sequences. `pySAR` offers extensive and verbose functionalities that allow you to numerically encode a dataset of protein sequences using a large abundance of available methodologies and features. The software uses physiochemical and biochemical features from the Amino Acid Index (AAI) database [[1]](#references) as well as allowing for the calculation of a range of structural, physiochemical and biochemical protein descriptors.<br><br>
After finding the optimal technique and feature set at which to encode your dataset of sequences, `pySAR` can then be used to build a predictive regression model with the training data being that of the encoded sequences, and training labels being the experimentally pre-calculated activity values for each protein sequence. This model maps a set of protein sequences to the sought-after activity value, being able to accurately predict the activity/fitness value of new unseen sequences. The use-case for the software is within the field of protein engineering and Directed Evolution, where a user has a set of experimentally determined activity values for a library of mutant protein sequences and wants to computationally predict the sought activity value for a selection of mutated sequences, in the aim of finding the best sequence that minimises/maximises their activity value. <br>

Two additional <strong>custom-built</strong> softwares were created alongside pySAR - [aaindex][aaindex] and [protpy][protpy]. The aaindex software package is used for parsing the amino acid index which is a database of numerical indices representing various physicochemical and biochemical properties of amino acids and pairs of amino acids [[1]](#references). protpy is used for calculating a series of protein physiochemical, biochemical and structural protein descriptors. Both of these software packages are integrated into pySAR but can also be used individually for their respective purposes. 

Requirements
------------
* [Python][python] >= 3.7
* [aaindex][aaindex] >= 1.0.4
* [protpy][protpy] >= 1.0.7
* [numpy][numpy] >= 1.24.2
* [pandas][pandas] >= 1.5.3
* [scikit-learn][sklearn] >= 1.2.1
* [scipy][scipy] >= 1.10.1
* [tqdm][tqdm] >= 4.65.0
* [matplotlib][matplotlib] >= 3.6.2
* [seaborn][seaborn] >= 0.12.2

Installation
-----------------
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
-----
### Confile File
`pySAR` works through JSON configuration files. There are many different customisable parameters for the functionalities in `pySAR` including the metaparameters of each of the available protein descriptors, all Digital Signal Processing (DSP) parameters in the pyDSP module, the type of regression model to use and parameters specific to the dataset. These config files offer a more straightforward way of making any changes to the `pySAR` pipeline. The names of **All** the parameters as listed in the example config files must remain unchanged, only the value of each parameter should be changed, any parameters not being used can be set to <em>null</em>. An example of the config file used in my research project, with most of the available parameters, can be seen below and in <em>config/thermostability.json</em>.

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
          "descriptors_csv": "descriptors.csv",
          "descriptors": {
            "all_desc": 0,
            "amino_acid_composition": 1,
            "dipeptide_composition": 1,
            ...
        }
      },
    "descriptor_properties":
    {
      "normalized_moreaubroto_autocorrelation":
        {
        "lag":30,
        "properties":["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
        },
      ...
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
### Encoding using all 566 AAIndex indices
Encoding protein sequences in dataset using all 566 indices in the AAI database. Each sequence encoded via an index in the AAI can be passed through an additional step where its protein spectra can be generated following an FFT. `pySAR` supports generation of the power, imaginary, real or absolute spectra as well as other DSP functionalities including windowing, convolution and filter functions. In the example below, the encoded sequences will be used to generate a imaginary protein spectra with a blackman window function applied. This will then be used as feature data to build a predictive model that can be used for accurate prediction of the sought activity value of unseen protein sequences. The encoding class also takes only the JSON config file as input which will have all the required parameter values. The output results will show the calculated metric values for each index in the AAI when measuring predicted vs observed activity values for the unseen test sequences.

```python
from pySAR.encoding import *

'''test_config.json
{
  "dataset": 
    {
    "dataset": "test_dataset1.txt",
    "activity": "sought_activity"
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
    "window": "blackman"
    }
}
'''
#create instance of Encoding class, using RF algorithm with its default params
encoding = Encoding(config_file='test_config.json')

#encode sequences using all indices in the AAI if input parameter "aai_indices" is empty/None
aai_encoding = encoding.aai_encoding()

```
Output results showing AAI index and its category as well as all the associated metric values for each predictive model:
|    | Index      | Category   |       R2 |    RMSE |     MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:-----------|---------:|--------:|--------:|--------:|--------:|----------------:|
|  0 | CHOP780206 | secondary_struct | 0.62737  | 3.85619 | 14.8702 | 1.63818 | 3.16755 |        0.713467 |
|  1 | QIAN880131 | secondary_struct | 0.626689 | 3.90576 | 15.255  | 1.63668 | 3.09849 |        0.631582 |
|  2 | QIAN880118 | secondary_struct | 0.625156 | 3.99581 | 15.9665 | 1.63333 | 3.32038 |        0.625897 |
|  3 | PRAM900104 | secondary_struct | 0.615866 | 3.90389 | 15.2403 | 1.61346 | 3.24906 |        0.617799 |
| .. | .......... | .......... | ........ | ....... | ....... | ....... | ....... | ............... |

### Encoding using list of 4 AAI indices, with no DSP functionalities
Same procedure as prior, except 4 indices from the AAI are being specifically input into the function, with the encoded sequence output being concatenated together and used as feature data to build the predictive PlsRegression model with its default parameters. The config parameter <em> use_dsp </em> tells the function to not generate the protein spectra or apply any additional DSP processing to the sequences.

```python
from pySAR.encoding import *

'''test_config2.json
{
  "dataset": 
    {
    "dataset": "test_dataset2.txt",
    "activity": "sought_activity"
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
encoding = Encoding(config_file='test_config2.json')

#encode sequences using 4 indices specified by user, use_dsp = False
aai_encoding = encoding.aai_encoding(aai_list=["PONP800102","RICJ880102","ROBB760107","KARS160113"])

```
Output DataFrame showing the 4 predictive models built using the PLS algorithm, with the 4 indices from the AAI:
|    | Index      | Category    |       R2 |    RMSE |      MSE |     RPD |     MAE |   Explained Var |
|---:|:-----------|:------------|---------:|--------:|---------:|--------:|--------:|----------------:|
|  0 | PONP800102 | hydrophobic | 0.74726  | 3.0817  |  9.49688 | 1.98913 | 2.63742 |        0.751032 |
|  1 | ROBB760107 | secondary_struct  | 0.666527 | 3.19801 | 10.2273  | 1.73169 | 2.50305 |        0.668255 |
|  2 | RICJ880102 | secondary_struct  | 0.568067 | 3.83976 | 14.7438  | 1.52157 | 3.01342 |        0.568274 |
|  3 | KARS160113 | meta        | 0.544129 | 4.04266 | 16.3431  | 1.48108 | 3.26047 |        0.544693 |

### Encoding protein sequences using their calculated protein descriptors
Calculate the protein descriptor values for a dataset of protein sequences from the 15 available descriptors in the <em>descriptors</em> module. Use each descriptor as a feature set in the building of the predictive models used to predict the activity value of unseen sequences. By default, the function will look for a csv file pointed to by the <em>"descriptors_csv"</em> parameter in the config file that contains the pre-calculated descriptor values for a dataset. If file is not found then all descriptor values will be calculated for the dataset using the <em>descriptors_</em> module. If a descriptor in the config file is to be used in the feature data, its parameter is set to true/1. If <em>all_desc</em> is set to true/1 then all available descriptors are calculated using their respective functions.

```python
from pySAR.encoding import *

'''test_config3.json
{
  "dataset": 
    {
    "dataset": "test_dataset3.txt",
    "activity": "sought_activity"
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
    "descriptors_csv": "precalculated_descriptors.csv",
    "all_desc": 0,
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
encoding = Encoding(config_file='test_config3.json')

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
Encoding protein sequences in dataset using all 566 indices in the AAI database combined with protein descriptors. All 566 indices can be used in concatenation with 1, 2 or 3 descriptors. E.g: at each iteration the encoded sequences using the indices from the AAI will be used to generate a protein spectra using the power spectrum with no window function applied, this will then be combined with the feature set generated from the dataset's descriptor values and used to build a predictive model that can be used for accurate prediction of the sought activity value of unseen protein sequences. The output results will show the calculated metric values when measuring predicted vs observed activity values for the test sequences.

```python
from pySAR.encoding import *

'''test_config4.json
{
  "dataset": 
  {
    "dataset": "test_dataset4.txt",
    "activity": "sought_activity"
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
    "descriptors_csv": "precalculated_descriptors.csv",
    "all_desc": 0,
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
    "window": ""
    ...
  }
}
'''
#create instance of Encoding class using RF algorithm, using 100 estimators with a learning rate of 1.5
encoding = Encoding('test_config4.json')

#building predictive models using all available aa_indices + combination of 2 descriptors,
#   calculating evaluation metric values for models and storing into aai_desc_results_df DataFrame
aai_desc_results_df = encoding.aai_descriptor_encoding(desc_combo=2)

```
Output results showing AAI index and its category, the protein descriptor and its group as well as the R2 and RMSE values for each predictive model:

|    | Index      | Category    | Descriptor                 | Descriptor Group     |       R2 |    RMSE |
|---:|:-----------|:------------|:---------------------------|:---------------------|---------:|--------:|
|  0 | ARGP820103 | composition | _conjoint_triad            | Conjoint Triad       | 0.72754  | 3.22135 |
|  1 | ARGP820101 | hydrophobic | _quasi_seq_order           | Quasi-Sequence-Order | 0.722284 | 3.30995 |
|  2 | ARGP820101 | hydrophobic | _seq_order_coupling_number | Quasi-Sequence-Order | 0.722158 | 3.34926 |
|  3 | ANDN920101 | observable  | _seq_order_coupling_number | Quasi-Sequence-Order | 0.70826  | 3.25232 |
|  4 | .....      | .....       | .....                      | .....                | .....    | .....   |


### Building predictive model from AAI and protein descriptors:
e.g: the below code will build a PlsRegression model using the AAI index CIDH920105 and the 'amino acid composition' descriptor. The index is passed through a DSP pipeline and is transformed into its informational protein spectra using the power spectra, with a hamming window function applied to the output of the FFT. The concatenated features from the AAI index and the descriptor will be used as the feature data in building the PLS model.

```python
import pySAR as pysar   #import pySAR package

'''test_config5.json
{
  "dataset": 
  {
    "dataset": "test_dataset5.txt",
    "activity": "sought_activity"
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
    "descriptors_csv": "precalculated_descriptors.csv",
    "all_desc": 0,
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
#create instance of PySAR class
pySAR = pysar.PySAR(config_file="test_config5.json")
"""
PySAR parameters:

:config_file : str
    full path to config file containing all required pySAR parameters.

"""
#encode protein sequences using both the CIDH920105 index + aa_composition descriptor.
results_df = pySAR.encode_aai_desc(indices="CIDH920105", descriptors="amino_acid_composition")
```

### Generate all protein descriptors
Prior to evaluating the various available properties and features at which to encode a set of protein sequences, it is reccomened that you pre-calculate all the available descriptors in one go, saving them to a csv for later that `pySAR` will then import from. Output values are stored in csv set by <em>descriptors_csv</em> config parameter. Output will be of the shape N x 9920, using the default parameters, where N is the number of protein sequences in the dataset, but the size of the 2nd dimension (total number of features calculated from all 15 descriptors) may vary depending on some descriptor-specific metaparameters. Setting <em>all_desc</em> parameter to True means all descriptors will be calculated, by default this is False.
```python
from pySAR.descriptors_ import *

'''test_config6.json
{
  "dataset": 
  {
    "dataset": "test_dataset5.txt",
    "activity": "sought_activity"
    ...
  },
  "model": 
  {
    ...
  }
  "descriptors": 
  {
    "descriptors_csv": "precalculated_descriptors",
    "all_desc": 1,
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
#calculating all descriptor values and storing in file named by parameter descriptors_csv
desc = Descriptors("test_config6")

```
### Get record from AAIndex database
The AAIndex class offers diverse functionalities for obtaining any element from any record in the database. Each record is stored in json format in a class attribute called <em>aaindex_json</em>. You can search for a particular record by its index code, description or reference. You can also get the index category, and importantly its associated amino acid values.

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
```

Directories
-----------
* `/config` - configuration files for the example datasets that `pySAR` has been tested with, as well as the thermostability.json config file that was used in the research. These config files should be used as a template for future datasets used with `pySAR`.
* `/docs` - documentation for `pySAR` (pending).
* `/example_datasets` - example datasets used for the building and testing of `pySAR`, including the thermostability dataset used in the research. The format of these datasets shoould be used as a template for future datasets used with `pySAR`.
* `/images` - all images used throughout the repo.
* `/pySAR` - source code for `pySAR` software.
* `/tests` - unit and integration tests for `pySAR`.

Issues
-----
Any issues, errors or bugs can be raised via the [Issues](https://github.com/amckenna41/pySAR/issues) tab in the repository.

Tests
-----
To run all tests, from the main `pySAR` repo folder run:
```
python3 -m unittest discover tests
```

To run tests for specific module, from the main `pySAR` repo folder run:
```
python -m unittest tests.MODULE_NAME -v
```

Contact
-------
If you have any questions or comments, please contact amckenna41@qub.ac.uk or raise an issue on the [Issues][Issues] tab. <br><br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adam-mckenna-7a5b22151/)

License
-----------
Distributed under the MIT License. See `LICENSE` for more details.  

References
----------
\[1\]: Kawashima, S. and Kanehisa, M., 2000. AAindex: amino acid index database. Nucleic acids research, 28(1), pp.374-374. DOI: 10.1093/nar/27.1.368 <br><br>
\[2\]: Fontaine NT, Cadet XF, Vetrivel I. Novel Descriptors and Digital Signal Processing- Based Method for Protein Sequence Activity Relationship Study. Int J Mol Sci. 2019 Nov 11;20(22):5640. doi: 10.3390/ijms20225640. PMID: 31718061; PMCID: PMC6888668. <br><br>
\[3\]: Cadet, F., Fontaine, N., Li, G. et al. A machine learning approach for reliable prediction of amino acid interactions and its application in the directed evolution of enantioselective enzymes. Sci Rep 8, 16757 (2018).<br><br>
\[4\]: Lutz S. Beyond directed evolution--semi-rational protein engineering and design. Curr Opin Biotechnol. 2010 Dec;21(6):734-43. doi: 10.1016/j.copbio.2010.08.011. Epub 2010 Sep 24. PMID: 20869867; PMCID: PMC2982887. <br><br>
\[5\]: Yang, K.K., Wu, Z. & Arnold, F.H. Machine-learning-guided directed evolution for protein engineering. Nat Methods 16, 687–694 (2019). https://doi.org/10.1038/s41592-019-0496-6 <br><br>
\[6\]: Yuting Xu, Deeptak Verma, Robert P. Sheridan, Andy Liaw, Junshui Ma, Nicholas M. Marshall, John McIntosh, Edward C. Sherer, Vladimir Svetnik, and Jennifer M. Johnston
Journal of Chemical Information and Modeling 2020 60 (6), 2773-2790
DOI: 10.1021/acs.jcim.0c00073 <br><br>
\[7\]: Medina-Ortiz, D., Contreras, S., Amado-Hinojosa, J., Torres-Almonacid, J., Asenjo, J. A., Navarrete, M., & Olivera-Nappa, Á. (2020). Combination of digital signal processing and assembled predictive models facilitates the rational design of proteins. ArXiv [Cs.CE]. <br>

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
[demo]: https://github.com/amckenna41/pySAR/
[Issues]: https://github.com/amckenna41/pySAR/issues
