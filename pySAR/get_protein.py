from .globals_ import DATA_DIR, OUTPUT_DIR, OUTPUT_FOLDER
import urllib.request as request
import requests
from contextlib import closing
import shutil
import os
from Bio import SeqIO
from Bio import Entrez

def get_protein_from_fasta(fasta_path):
    """
    Parse protein sequence from FASTA file.

    Parameters
    ----------
    :fasta_path : str
        path to FASTA file.

    Returns
    -------
    :sequence : str
        parsed protein sequence.
    """
    sequence = SeqIO.read(fasta_path, "fasta").seq
    return sequence

def download_protein_from_uniprot(prot_id, save_dir=DATA_DIR):
    """
    Download protein sequence from Uniprot database using its Uniprot ID.

    Parameters
    ----------
    :prot_id : str
        protein's Uniprot ID
    :save_dir : str (default="data")
        directory to save protein FASTA to.

    Returns
    -------
    None
    """
    #base URL for Uniprot
    url = "http://www.uniprot.org/uniprot/" + prot_id + ".fasta"

    #download protein using Uniprot URL, store in save_dir directory
    try:
        with closing(request.urlopen(url)) as r:
            with open((os.path.join('pySAR',save_dir,prot_id)), 'wb') as f:
                shutil.copyfileobj(r, f)
        print('Protein successfully downloaded.')
    except requests.exceptions.RequestException:
        print('Error downloading protein with ID {} from the url {}.'.format(prot_id, url))


def download_protein_from_ncbi(prot_id,db="protein",email="email@example.com", save_dir=DATA_DIR):
    """
    Download protein sequence from NCBI database using its protein ID.

    Parameters
    ----------
    :prot_id : str
        protein's NCBI ID
    :db : str
        target biological database for Entrez package to search for protein.
    :email : str
        email required for using Entrez package and searching through databases,
        used for logging, can be a bogus email if neccessary.
    :save_dir : str (default="DATA")
        directory to save protein to.

    Returns
    -------
    None
    """
    #error checking on db
    Entrez.email = email

    #create instance of Entrez object for pulling from DB
    net_handle = Entrez.efetch(
        db=db, id=prot_id, rettype="fasta", retmode="text"
    )

    #save protein to save_dir
    out_handle = open(os.path.join('pySAR',save_dir,prot_id + ".fasta"), "w")
    out_handle.write(net_handle.read())
    out_handle.close()
    net_handle.close()

def download_protein_from_pdb(pdb_id, save_dir=DATA_DIR):
    """
    Download protein from the PDB (Protein Data Bank) using its ID.

    Parameters
    ----------
    :pdb_id : str
        protein data bank protein ID.
    :save_dir : str (default="DATA")
        directory to save protein to.

    Returns
    -------
    None
    """
    #base URL for downloading from PDB
    url = 'http://files.rcsb.org/download/'+ pdb_id.lower()+'.pdb'

    #download protein using PDB URL, store in save_dir directory
    try:
        with closing(request.urlopen(url)) as r:
            with open((os.path.join('pySAR',save_dir,pdb_id+'.pdb')), 'wb') as f:
                shutil.copyfileobj(r, f)
        print('Protein successfully downloaded.')
    except requests.exceptions.RequestException:
        print('Error downloading protein with ID {} from the url {}.'.format(pdb_id, url))
