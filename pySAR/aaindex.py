
################################################################################
#################                    AAIndex                   #################
################################################################################

#importing required modules and dependencies
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sys, copy, re
import requests
import shutil
import csv
import urllib.request as request
from contextlib import closing
from collections import defaultdict
import itertools
from difflib import get_close_matches
from .globals_ import DATA_DIR, OUTPUT_DIR, OUTPUT_FOLDER

#look into and remove ('-') from get_encoding & get_amino_acids functions.
#fix     def get_record_from_name(self, name): <- not returning anything
class AAIndex():
    """
            Python parser for AAindex1: Amino Acid Index Database
                      (**abbreviated to AAI onwards**)

    The AAindex is a database of numerical indices representing various physicochemical
    and biochemical properties of amino acids and pairs of amino acids. The focus
    on this class is on the AAIndex 1 database which stores the amino acid index of
    20 numerical values for the 20 amino acids. http://www.genome.jp/aaindex/

    Some of the parsing functionality is inspired and based off AAIndex1 parser
    aaindex2json.py by harmslab:
    https://github.com/harmslab/hops/blob/master/hops/features/data/util/aaindex2json.py

    Data format of AAI1:
    ************************************************************************
    *                                                                      *
    * Each entry has the following format.                                 *
    *                                                                      *
    * H Accession number                                                   *
    * D Data description                                                   *
    * R PMID                                                               *
    * A Author(s)                                                          *
    * T Title of the article                                               *
    * J Journal reference                                                  *
    * * Comment or missing                                                 *
    * C Accession numbers of similar entries with the correlation          *
    *   coefficients of 0.8 (-0.8) or more (less).                         *
    *   Notice: The correlation coefficient is calculated with zeros       *
    *   filled for missing values.                                         *
    * I Amino acid index data in the following order                       *
    *   Ala    Arg    Asn    Asp    Cys    Gln    Glu    Gly    His    Ile *
    *   Leu    Lys    Met    Phe    Pro    Ser    Thr    Trp    Tyr    Val *
    * //                                                                   *
    ************************************************************************
    --------------------------------------------------------------------------------
    H ANDN920101
    D alpha-CH chemical shifts (Andersen et al., 1992)
    R PMID:1575719
    A Andersen, N.H., Cao, B. and Chen, C.
    T Peptide/protein structure analysis using the chemical shift index method:
      upfield alpha-CH values reveal dynamic helices and aL sites
    J Biochem. and Biophys. Res. Comm. 184, 1008-1014 (1992)
    C BUNA790102    0.949
    I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
        4.35    4.38    4.75    4.76    4.65    4.37    4.29    3.97    4.63    3.95
        4.17    4.36    4.52    4.66    4.44    4.50    4.35    4.70    4.60    3.95
    --------------------------------------------------------------------------------

    Attributes
    ----------
    aaindex_filename : str (default = 'aaindex1')
        local filename of aaindex1 file that class will import the database
        from in the data directory, default value of 'aaindex1' is reccomended.
    download_using : str (default = "ftp")
        decide to download AAI database using ftp or https, ftp used by default.

    Methods
    -------
    parse_aaindex_to_json():
    parse_aaindex_to_category(aaindex_category_file):
    download_aaindex():
    get_amino_acids():
    get_amino_acids_encoding():
    get_record_codes():
    get_num_records():
    get_record_names():
    get_record_from_code(code):
    get_record_from_name():
    get_values_from_record():
    get_ref_from_record():
    get_category_from_record():

    References
    ----------
    [1] S. Kawashima and M. Kanehisa, “AAindex: amino acid index database,”
        Nucleic Acids Res., vol. 28, no. 1, p. 374, 2000.
    [2] L. C. Wheeler, A. Perkins, C. E. Wong, and M. J. Harms, “Learning peptide
        recognition rules for a low-specificity protein,” Protein Sci., vol. 29,
        no. 11, pp. 2259–2273, 2020.
    """
    def __init__(self, aaindex_filename='aaindex1', download_using='ftp'):

        self.aaindex_filename = aaindex_filename
        self.download_using = download_using

        #download AAI database using ftp or https
        if self.download_using=='ftp':
            self.url = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"
        elif self.download_using=='http' or download_using=='https':
            self.url = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"
        else:
            self.url = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"

        #if AAIndex database not found in data directory then call download method
        if not (os.path.isfile(os.path.join(DATA_DIR,self.aaindex_filename+'.json'))):
            if not (os.path.isfile(os.path.join(DATA_DIR,self.aaindex_filename))):
                self.download_aaindex()

        #if parsed json of AAIndex already in file then read it and return  <- Not working at the moment
        # if (os.path.isfile(os.path.join(DATA_DIR, self.aaindex_filename +'.json'))):
        #     self.aaindex_json = json.load(os.path.join(DATA_DIR, self.aaindex_filename +'.json'))
        # else:
        #parse AAIndex database file into JSON format
        self.aaindex_json = self.parse_aaindex_to_json()

        self.categories = None

    def parse_aaindex_to_json(self):
        """
        Parse AAI database into JSON format. Each AAI record will be indexed by
        its feature code/index code, and will be in the format as shown in the
        example above. The file will be stored in a json file called according
        to (self.aaindex_filename+'.json') variable.

        Returns
        -------
        aaindex_json : dict
          parsed AAI database in dict form.
        """
        #initialise keys of AAI database
        template_dict = {"H":[],
                         "D":[],
                         "R":[],
                         "A":[],
                         "*":[],
                         "T":[],
                         "J":[],
                         "C":[],
                         "I":[]}

        #open AAI file for reading and parsing, by default it should be stored in DATA_DIR
        try:
            tmp_filepath = os.path.join(DATA_DIR,self.aaindex_filename)
            f = open(tmp_filepath,'r')
        except IOError:
            print('Error opening file, check filename = {} and is stored in \
                    {} directory.'.format(self.aaindex_filename, DATA_DIR))

        #read lines of file
        lines = f.readlines()
        f.close()

        clean_up_pattern = re.compile("\"")

        #initilaise parsed AAI database dictionary
        aaindex_json = {}
        current_dict = copy.deepcopy(template_dict)

        '''
        iterate through each line in the AAI database, parsing each record and its
        amino acid values into its own entry in the dictionary. Each index/record
        is seperated by a '//'. Remove any duplicate records, set any missing
        ('-') or NA amino acid values to 0. Store resulting dict into aaindex_json
        instance variable.
        '''
        for l in lines:

            if l.startswith("//"):

                # deal with meta data
                name = " ".join(current_dict["H"])
                name = clean_up_pattern.sub("'",name)

                description = " ".join(current_dict["D"])
                description = clean_up_pattern.sub("'",description)

                citation = "{} '{}' {}".format(" ".join(current_dict["A"]),
                                               " ".join(current_dict["T"]),
                                               " ".join(current_dict["J"]))

                citation = citation + "; Kawashima, S. and Kanehisa, M. \
                    'AAindex: amino acid index database.'  Nucleic Acids Res. 28, 374 (2000)."

                citation = clean_up_pattern.sub("'",citation)

                notes = " ".join(current_dict["*"])
                notes = clean_up_pattern.sub("'",notes)

                # parse amino acid data
                aa_lines = current_dict["I"]

                aa_names = aa_lines[0].split()
                row_0_names = [aa.split("/")[0] for aa in aa_names]
                row_1_names = [aa.split("/")[1] for aa in aa_names]

                row_0_values = aa_lines[1].split()
                row_1_values = aa_lines[2].split()

                values = {}
                for i in range(len(row_0_values)):
                    try:
                        values[row_0_names[i]] = float(row_0_values[i])
                    except ValueError:
                        values[row_0_names[i]] = "NA"

                    try:
                        values[row_1_names[i]] = float(row_1_values[i])
                    except ValueError:
                        values[row_1_names[i]] = "NA"

                # look for duplicate name entries
                try:
                    aaindex_json[name]
                    err = "duplicate value name ({})".format(name)
                    raise ValueError('Duplicate AAI Record name found')
                except KeyError:
                    pass

                aaindex_json[name] = {"description":description,
                                  "refs":citation,
                                  "notes":notes,
                                  "values":values}

                current_dict = copy.deepcopy(template_dict)
                continue

            this_entry = l[0]
            if l[0] != " ":
                current_entry = this_entry

            current_dict[current_entry].append(l[1:].strip())

        #append '-' to each aa index entry to account for missing AA in a protein sequence
        for index in aaindex_json:
            aaindex_json[index]['values']['-'] = 0

            #set any NA amino acid values to 0
            for val in aaindex_json[index]['values']:
                if aaindex_json[index]['values'][val] == 'NA':
                    aaindex_json[index]['values'][val] = 0

        #save parsed dictionary into JSON format to DATA_DIR
        with open((os.path.join(DATA_DIR, self.aaindex_filename + '.json')),'w') as output_F:
          json.dump(aaindex_json, output_F,indent=4, sort_keys=True)

        return aaindex_json

    def download_aaindex(self, save_dir=DATA_DIR):
        """
        If AAI database not found in DATA directory, then it will be downloaded from
        the dedicated FTP or HTTPS server from https://www.genome.jp/aaindex/.
        FTP is the default method used for downloading the database with the URL:
        "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1". If you want to
        download by https, set the 'download_using' instance variable to 'https'.

        Parameters
        ----------
        save_dir : str (default = DATA_DIR)
            Directory to save the AAI database to. By default it wil be stored in
            the global var DATA_DIR = 'data'.
        """
        #if directory doesnt exist then create it
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                raise OSError('Error creating save directory: {}'.format(save_dir))

        #fetch AAI database from URL if not present in DATA_DIR
        try:
            if not(os.path.isfile(os.path.join(save_dir, self.aaindex_filename))):
                try:
                    with closing(request.urlopen(self.url)) as r:
                        with open((os.path.join(save_dir, self.aaindex_filename)), 'wb') as f:
                            shutil.copyfileobj(r, f)
                    print('AAIndex1 successfully downloaded.')
                except requests.exceptions.RequestException:
                    print('Error downloading or exporting AAIndex from the url {}.'.format(self.url))
            else:
                pass    #AAIndex already in folder
                print('AAIndex already exists in folder.')
        except:
            raise OSError('Save Directory does not exist: {}.'.format(save_dir))

    def parse_aaindex_to_category(self, aaindex_category_file='aaindex-to-category.txt'):
        """
        Parse category file which maps each AAI record in the databse into 1 of 8 categories.
        Category file and parsing code inspired from:
        https://github.com/harmslab/hops

        Parameters
        ----------
        aaindex_category_file : str
            Name of category file to parse (default is "aaindex-to-category.txt").

        Returns
        -------
        aaindex_category : dict
            Dictionary that maps each AAI record into 1 of 8 categories.
        """
        #if input parameter is a full path, read it else read from default DATA_DIR
        if os.path.isfile(aaindex_category_file):
            f = open(aaindex_category_file,'r')
        else:
            try:
                f = open((os.path.join(DATA_DIR, aaindex_category_file)),'r')
            except IOError:
                print('Error opening AAIndex1 category file, check {} in {} directory'
                    .format(aaindex_category_file, DATA_DIR))

        #get total number of lines in file
        total_lines = len(f.readlines(  ))

        #open new file in data directory to store parsed category file
        f.seek(0)
        category_output_file = "aaindex-to-category-parsed.txt"
        f_out = open((os.path.join(DATA_DIR, category_output_file)), "w")

        #lines starting with '#' are file metadata so don't write these to parsed output file
        for line in f.readlines():
            if not (line.startswith('#')):
                f_out.write(line)
        f.close()
        f_out.close()

        #open parsed category file for reading
        with open(os.path.join(DATA_DIR, category_output_file)) as f_out:
            reader = csv.reader(f_out, delimiter="\t")
            d = list(reader)
        f_out.close()

        aaindex_category = {}

        #iterate through all lines in parsed file and store AAI indices as keys
        #   in the output dict and their respective categories as the values of dict
        for i in range(0,len(d)):
          for j in range(0,len(d[i])):
            category_substring = d[i][1].strip()
            category_substring = category_substring.split(" ",1)
            aaindex_category[d[i][0]] = category_substring[0]

        self.categories = aaindex_category

    def get_amino_acids(self):
        """
        Get all canonical amino acid letters. The '-' value will also be included
        in the list from this function as it accounts for the abcense of any
        amino acid or gaps in a AAI record.

        Returns
        -------
        amino_acids : list
            List of all 20 canoniocal amino acid letters as found in each record
            of the AAI database.
        """
        amino_acids = list(self.aaindex_json[list(self.aaindex_json.keys())[0]]["values"].keys())
        amino_acids.sort()

        return amino_acids

    def get_amino_acids_encoding(self):
      """
      Get one-hot encoding of amino acids.

      Returns
      -------
      onehot_encoded : np.ndarray
        one hot encoded array of the 20 canonical amino acids.
      """
      all_amino_acids = self.get_amino_acids()
      values = np.array(all_amino_acids[1:])    #convert amino acids to np array

      #encode amino acids with value between 0 and n_classes-1.
      label_encoder = LabelEncoder()
      integer_encoded = label_encoder.fit_transform(values)

      #encode amino acids as a one-hot numeric array.
      onehot_encoder = OneHotEncoder(sparse=False)
      integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
      onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

      return onehot_encoded

    def get_record_codes(self):
      """
      Get list of all AAI index codes for each record in the database.

      Returns
      -------
      records : list
        list of record names/index codes for all records in the AAI database.
      """
      records = list(self.aaindex_json.keys())
      records.sort()       #sort into alphabetical order

      return records

    def get_num_records(self):
      """
      Calculate total number of records/indices in the AAI database.

      Returns
      -------
      len(self.get_record_codes()) : int
        number of indices/records found in the AAI database.
      """
      return len(self.get_record_codes())

    # def get_feature_names(self):
    def get_record_names(self):
      """
      Return a list of all index descriptions for all records in the AAI database.

      Returns
      -------
      desc : list
         list of descriptions for all records in the AAI database.
      """
      desc = []

      #iterate through database, appending all descriptions to desc list
      for name in list(self.aaindex_json.values()):
        desc.append(name['description'])

      return desc

    def get_record_from_code(self, index_code):
      """
      Return full AAI database record details from its feature/index code.

      Parameters
      ----------
      index_code : str
         AAI database record feature code/index code.

      Returns
      -------
      record : dict
        dict of AAI database record.
      """
      #stripping input of whitespace
      try:
          index_code = index_code.strip()
      except:
        raise TypeError('Input parameter {} is not of correct datatype string, got {}' \
            .format(index_code, type(index_code)))

      #check that inputted index_code does exist in the AAI database
      if index_code not in (self.get_record_codes()):
          raise ValueError('Record Index ({}) not found in AAIndex'.format(index_code))

      record = (self.aaindex_json[index_code])

      return record

    def get_record_from_name(self, name):
        """
        Return full AAI database record details from its description. Search
        through the descriptions/names of all records in the database, returning
        record that matches name input parameter.

        Parameters
        ----------
        name : str
            AAI database record feature name/description.

        Returns
        -------
        feature : dict/None
            dict of full AAI database record or None if not found.
        """
        correct_index = 0
        for index, value in self.aaindex_json.items():
          if value['description'] != name:
           continue
          else:
           correct_index = index
           return self.aaindex_json[correct_index]

        return None

    def get_values_from_record(self, index_code):
      """
      Return amino acid values from database record of index in the AAI from
      its feature/index code.

      Parameters
      ----------
      index_code : str
          AAI database record feature code/index code.

      Returns
      -------
      values : dict
        amino acid values for specified AAI database record.
      """
      #stripping input of whitespace
      try:
          index_code.strip()
      except:
        raise TypeError('Input parameter {} is not of correct datatype string, got {}' \
            .format(index_code, type(index_code)))

      #check that inputted index_code does exist in the AAI database
      if index_code not in (self.get_record_codes()):
            raise ValueError('Record index {} not found in AAI'.format(index_code))

      #get amino acid values for specified index
      values = (self.aaindex_json[index_code]['values'])

      return values

    def get_ref_from_record(self, index_code):
      """
      Return reference details of a AAI database record from its feaure/index code.

      Parameters
      ----------
      index_code : str
        AAI database record feature code/index code.

      Returns
      -------
      refs : list
        reference details for specifed AAI database record.
      """
      #stripping input of whitespace
      try:
          index_code.strip()
      except:
        raise TypeError('Input parameter {} is not of correct datatype string, got {}' \
            .format(index_code, type(index_code)))

      #check that inputted index_code does exist in the AAI database
      if index_code not in (self.get_record_codes()):
            raise ValueError('Record index {} not found in AAI'.format(index_code))

      refs = (self.aaindex_json[index_code]['refs'])    #get refs from record

      return refs

    def get_category_from_record(self, index_code):
      """
      Return category of a AAI database record from its feaure/index code.

      Parameters
      ----------
      index_code : str
            AAI database record feature code/index code.

      Returns
      -------
      cat : str
        category of AAI database record according to parsed aaindex-to-category.txt file.
      """
      #stripping input of whitespace
      try:
          index_code.strip()
      except:
        raise TypeError('Input parameter {} is not of correct datatype string, got {}' \
            .format(index_code, type(index_code)))

      #check that inputted index_code does exist in the AAI database
      if index_code not in (self.get_record_codes()):
          raise ValueError('Record index {} not in AAI database'.format(index_code))

      if self.categories == None:
          self.parse_aaindex_to_category()

      cat = self.categories[index_code]

      return cat

######################          Getters & Setters          ######################

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def aaindex_filename(self):
        return self._aaindex_filename

    @aaindex_filename.setter
    def aaindex_filename(self, value):
        self._aaindex_filename = value

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

################################################################################

    def __str__(self):
        return "AAIndex1 Database of size {} - stored in {} ".format(
        self.get_num_records(),self.aaindex_filename
        )

    def __repr__(self):
        return (self.aaindex_json)

    def __sizeof__(self):
        """return size of AAI database file"""
        return os.path.getsize(os.path.isfile(os.path.join(DATA_DIR,self.aaindex_filename)))
