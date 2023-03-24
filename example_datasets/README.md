# Example datasets for Sequence Activity Relationships (SAR)

Datasets
--------
* `thermostability.txt` - dataset studied in the associated work which consists of a dataset to measure the thermostability of various mutants
from a recombination library designed from parental cytochrome P450's, measured using the T50 metric (temperature at which 50% of a protein is
irreversibly denatured after 10 mins of incubation, ranging from 39.2 to 64.4 degrees C), which represents the protein activity of this dataset [[1]](#references).
* `absorption.txt` - dataset of 80 blue and red-shifted protein variants of the Gloeobacter Violaceus rhodopsin (GR) protein that were mutated and substituted to tune its peak absorption wavelength. 1-5 mutations were generated in the course of tuning its absorption wavelength, for a total of 81 sequences, with the peak being captured as each sequence's activity ranging from values of 454 to 622 [[2]](#references).
* `enantioselectivity.txt` - dataset consisting of 37 mutants and one WT (wild-type) sequence from the Aspergillus Niger organism and their calculated enantioselectivity. Enantioselectivity refers to the selectivity of a reaction towards one enantiomer and is expressed by the E-value with a range between 0 and 115 [[3]](#references).
* `localization.txt` - dataset made up of 248 sequences made up of 2 seperate, 10-block recombination libraries that were designed from 3 parental ChR's (channelrhodopsin). Each chimeric ChR variant in these libraries consist of blocks of sequences from parental ChRs. Genes for these sequences were synthesized and expressed in human embryonic kidney (HEK) cells, and their membrane localization was measured as log_GFP ranging from values of -9.513 to 105 [[4]](#references).

* `descriptors_absorption.csv` - pre-calculated protein descriptors using sequences from absorption test dataset.
* `descriptors_enantioselectivity.csv` - pre-calculated protein descriptors using sequences from enantioselectivity test dataset.
* `descriptors_localization.csv` - pre-calculated protein descriptors using sequences from localization test dataset.

References
----------
\[1\]: Li, Y., Drummond, D. A., Sawayama, A. M., Snow, C. D., Bloom, J. D., & Arnold, F. H. (2007). A diverse family of thermostable cytochrome P450s created by recombination of stabilizing fragments. Nature Biotechnology, 25(9), 1051–1056. https://doi.org/10.1038/nbt1333 <br>
\[2\]: Engqvist, M. K. M., McIsaac, R. S., Dollinger, P., Flytzanis, N. C., Abrams, M., Schor, S., & Arnold, F. H. (2015). Directed evolution of Gloeobacter violaceus rhodopsin spectral properties. Journal of Molecular Biology, 427(1), 205–220. https://doi.org/10.1016/j.jmb.2014.06.015  <br>
\[3\]: Zaugg, J., Gumulya, Y., Malde, A. K., & Bodén, M. (2017). Learning epistatic interactions from sequence-activity data to predict enantioselectivity. Journal of Computer-Aided Molecular Design, 31(12), 1085–1096. https://doi.org/10.1007/s10822-017-0090-x <br>
\[4\]: Bedbrook, C. N., Rice, A. J., Yang, K. K., Ding, X., Chen, S., LeProust, E. M., Gradinaru, V., & Arnold, F. H. (2017). Structure-guided SCHEMA recombination generates diverse chimeric channelrhodopsins. Proceedings of the National Academy of Sciences of the United States of America, 114(13), E2624–E2633. https://doi.org/10.1073/pnas.1700269114
