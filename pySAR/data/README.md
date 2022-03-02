# Data required for pySAR 

Data
----
* `aaindex1` - Amino Acid Index1 database [[1]](#references). 
* `aaindex2` - Amino Acid Index2 database [[1]](#references).
* `aaindex3` - Amino Acid Index3 database [[1]](#references).
* `aaindex1.json` - parsed verion of aaindex1 into JSON format to make it more manageable, readable and accessible for pySAR.
* `aaindex_to_category.txt` - mapping each aaindex feature/accession number to 1 of 9 categories. These categories are not strictly seperable but act to convey further info about each index.
* `aaindex_categories.txt` - parsed form of aaindex_to_category.txt so that accessing each indices category is much easier.
* `grantham-physiochemical-distance-matrix.json` - grantham distance score for each amino acid combination, used in the calculation of the sequence-order related protein descriptors.  [[2]](#references)
* `schneider-werde-physiochemical-distance-matrix.json` - schneider-werde distance score for each amino acid combination, used in the calculation of the sequence-order related protein descriptors.  [[3]](#references)

References
----------
\[1\]: Kawashima, S., Pokarowski, P., Pokarowska, M., Kolinski, A., Katayama, T., and Kanehisa, M.; AAindex: amino acid index database, progress report 2008. Nucleic Acids Res. 36, D202-D205 (2008). [PMID:17998252] <br>
\[2\]: Grantham R. Amino acid difference formula to help explain protein evolution. Science. 1974 Sep 6;185(4154):862-4. doi: 10.1126/science.185.4154.862. PMID: 4843792. <br>
\[3\]: Schneider G, Wrede P. The rational design of amino acid sequences by artificial neural networks and simulated molecular evolution: de novo design of an idealized leader peptidase cleavage site. Biophys J. 1994 Feb;66(2 Pt 1):335-44. doi: 10.1016/s0006-3495(94)80782-9. PMID: 8161687; PMCID: PMC1275700.