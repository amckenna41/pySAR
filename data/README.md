# Data used in pySAR research project 

Usage
-----
pySAR imports the dataset declared within the configuration file (thermostability.txt) from this data directory as well as the pre-calculated descriptor values csv (descriptors_thermostability.csv), if applicable, which it is also instantiated in the config file. An error will throw if the dataset and or descriptors csv is not found within this data directory. Please refer to the [CONFIG.md][config] example file for where to declare the two aforementioned parameters in the config file.

Data
----
* `thermostability.txt` - dataset studied in the associated research which consists of a dataset to measure the thermostability of various mutants from a recombination library designed from parental cytochrome P450's, measured using the T50 metric (temperature at which 50% of a protein is irreversibly denatured after 10 mins of incubation, ranging from 39.2 to 64.4 degrees C), which represents the protein activity of this dataset. [[1]](#references)
* `thermostability.json` - configuration file for using pySAR with the thermostability dataset studied in the research.
* `descriptors_thermostability.csv` - csv of all pre-calculated descriptors for thermostability dataset, calculated using the descriptors module and `protpy` package [[2]](#references). The dimensions for the csv file are 261 x 9714 (261 protein sequences with 9714 features), and it uses the default parameters for any descriptor that has metaparameters (autocorrelation, sequence order and pseudo composition). Calculating all available descriptors took about ~78 minutes. The columns and dimensions of each descriptor is outlined below:

* Amino Acid Composition - [0:20] (A,C,D,E...)
* Dipeptide Composition - [20:420] (AA,AC,AD,AE...)
* Tripeptide Composition - [420:8420] (AAA,AAC,AAD,AAE...)
* MoreauBroto Autocorrelation - [8420:8660] (MBAuto_CIDH920105_1,MBAuto_CIDH920105_2,MBAuto_CIDH920105_3,MBAuto_CIDH920105_4...)
* Moran Autocorrelation - [8660:8900] (MAuto_CIDH920105_1,MAuto_CIDH920105_2,MAuto_CIDH920105_3,MAuto_CIDH920105_4...)
* Geary Autocorrelation - [8900:9140] (GAuto_CIDH920105_1,GAuto_CIDH920105_2,GAuto_CIDH920105_3,GAuto_CIDH920105_4...)
* CTD - [9140:9161] (CTD_C_01_hydrophobicity,CTD_C_02_hydrophobicity,CTD_C_03_hydrophobicity,CTD_T_12_hydrophobicity...)
* Conjoint Triad - [9161:9504] (conj_triad_111,conj_triad_112,conj_triad_113,conj_triad_114...)
* Sequence Order Coupling Number - [9504:9534] (SOCN_SW1,SOCN_SW2,SOCN_SW3,SOCN_SW4...)
* Quasi Sequence Order - [9534:9584] (QSO_SW1,QSO_SW2,QSO_SW3,QSO_SW4...)
* Pseudo Amino Acid Composition - [9584:9634] (PAAC_1,PAAC_2,PAAC_3,PAAC_4...)
* Amphiphilic Pseudo Amino Acid Composition - [9634:9714] (APAAC_1,APAAC_2,APAAC_3,APAAC_4...)

References
----------
\[1\]: Li, Y., Drummond, D. A., Sawayama, A. M., Snow, C. D., Bloom, J. D., & Arnold, F. H. (2007). A diverse family of thermostable cytochrome P450s created by recombination of stabilizing fragments. Nature Biotechnology, 25(9), 1051–1056. https://doi.org/10.1038/nbt1333 <br>
\[2\]: https://github.com/amckenna41/protpy

[config]: https://github.com/amckenna41/pySAR/blob/master/CONFIG.md