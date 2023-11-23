# Example datasets for Sequence Activity Relationships (SAR)

Datasets
--------
* `thermostability.txt` - dataset studied in the associated work which consists of a dataset to measure the thermostability of various mutants
from a recombination library designed from parental cytochrome P450's, measured using the T50 metric (temperature at which 50% of a protein is
irreversibly denatured after 10 mins of incubation, ranging from 39.2 to 64.4 degrees C), which represents the protein activity of this dataset [[1]](#references). **Dataset dimensions: 261 x 5**.
* `absorption.txt` - dataset of 80 blue and red-shifted protein variants of the Gloeobacter Violaceus Rhodopsin (GR) protein that were mutated and substituted to tune its peak absorption wavelength. 1-5 mutations were generated in the course of tuning its absorption wavelength, for a total of 81 sequences, with the peak being captured as each sequence's activity ranging from values of 454 to 622 [[2]](#references). **Dataset dimensions: 81 x 5**.
* `enantioselectivity.txt` - dataset consisting of 37 mutants and one wild-type (WT) sequence from the Aspergillus Niger organism and their calculated enantioselectivity. Enantioselectivity refers to the selectivity of a reaction towards one enantiomer and is expressed by the E-value with a range between 0 and 115 [[3]](#references). **Dataset dimensions: 152 x 5**.
* `localization.txt` - dataset made up of 248 sequences made up of 2 separate, 10-block recombination libraries that were designed from 3 parental channelrhodopsin (ChRs). Each chimeric ChR variant in these libraries consist of blocks of sequences from parental ChRs. Genes for these sequences were synthesized and expressed in human embryonic kidney (HEK) cells, and their membrane localization was measured as log_GFP ranging from values of -9.513 to 105 [[4]](#references). **Dataset dimensions: 254 x 5**.
* `descriptors_absorption.csv` - pre-calculated protein descriptors using sequences from absorption test dataset. The dimensions for this csv are 81 x 9714 (81 protein sequences and 9714 features), when using default parameters as in the config file.
* `descriptors_enantioselectivity.csv` - pre-calculated protein descriptors using sequences from enantioselectivity test dataset. The dimensions for this csv are 152 x 9714 (152 protein sequences and 9714 features), when using default parameters as in the config file.
* `descriptors_localization.csv` - pre-calculated protein descriptors using sequences from localization test dataset. The dimensions for this csv are 254 x 9714 (254 protein sequences and 9714 features), when using default parameters as in the config file.

Each of the pre-calculated descriptor CSVs have 9714 total features (when using the default parameters), the columns and dimensions of each descriptor file is outlined below:

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

\[2\]: Engqvist, M. K. M., McIsaac, R. S., Dollinger, P., Flytzanis, N. C., Abrams, M., Schor, S., & Arnold, F. H. (2015). Directed evolution of Gloeobacter violaceus rhodopsin spectral properties. Journal of Molecular Biology, 427(1), 205–220. https://doi.org/10.1016/j.jmb.2014.06.015  <br>

\[3\]: Zaugg, J., Gumulya, Y., Malde, A. K., & Bodén, M. (2017). Learning epistatic interactions from sequence-activity data to predict enantioselectivity. Journal of Computer-Aided Molecular Design, 31(12), 1085–1096. https://doi.org/10.1007/s10822-017-0090-x <br>

\[4\]: Bedbrook, C. N., Rice, A. J., Yang, K. K., Ding, X., Chen, S., LeProust, E. M., Gradinaru, V., & Arnold, F. H. (2017). Structure-guided SCHEMA recombination generates diverse chimeric channelrhodopsins. Proceedings of the National Academy of Sciences of the United States of America, 114(13), E2624–E2633. https://doi.org/10.1073/pnas.1700269114
