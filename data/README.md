# Data used in pySAR research project 

Usage
=====
pySAR imports the dataset declared within the configuration file (thermostability.txt) from this data directory as well as the pre-calculated descriptor values csv (descriptors_thermostability.csv), if applicable, which is also instantiated in the config file. An error will throw if the dataset and or descriptors csv is not found within this data directory. Please refer to the CONFIG.md file of where to declare the two aforementioned parameters in the config file.

Data
====
* `thermostability.txt` - dataset studied in the associated work which consists of a dataset to measure the thermostability of various mutants from a recombination library designed from parental cytochrome P450's, measured using the T50 metric (temperature at which 50% of a protein is irreversibly denatured after 10 mins of incubation, ranging from 39.2 to 64.4 degrees C), which represents the protein activity of this dataset. [[1]](#references)
* `thermostability.json` - configuration file for using pySAR with the thermostability dataset studied in the research.
* `descriptors_thermostability.csv` - csv of all pre-calculated descriptors for thermostability dataset. 

References
==========
\[1\]: Li, Y., Drummond, D. A., Sawayama, A. M., Snow, C. D., Bloom, J. D., & Arnold, F. H. (2007). A diverse family of thermostable cytochrome P450s created by recombination of stabilizing fragments. Nature Biotechnology, 25(9), 1051–1056. https://doi.org/10.1038/nbt1333 <br>