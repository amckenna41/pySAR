import pandas as pd
import os
# # from .descriptors import *
# import pySAR.descriptors.autocorrelation as auto
from pySAR.get_protein import *
from pySAR import *
from Bio import SeqIO



def main():

    # seq = SeqIO.read("test.fasta","fasta").seq
    # print(seq)
    # print (os.getcwd())
    # result = auto.moran_autocorrelation(seq)
    #

    # print(result)

    seq =  download_protein_from_uniprot("B6J853")

    # download_protein_from_ncbi("BCY15823", db="protein",email="email@example.com")

    download_protein_from_pdb("RC8")

    print(seq)
if __name__ == '__main__':
    main()


# get_protein_from_fasta
# download_protein_from_uniprot
# download_protein_from_ncbi
# download_protein_from_pdb
