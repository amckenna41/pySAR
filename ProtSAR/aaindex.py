
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

from globals import OUTPUT_DIR, OUTPUT_FOLDER, DATA_DIR

class AAIndex():
    """
    Python parser for AAindex1: Amino Acid Index Database
              (**abbreviated to AAI1 onwards**)

    http://www.genome.jp/aaindex/

    AA1 parser based off aaindex2json.py by harmslab:
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
    aa_index_filename : str (default = 'aaindex1')
        local filenane of aaindex1 file that class will import the database
        from the data directory, default value of 'aaindex1' is reccomended.
    aa_index_json_filename: str (default = 'aaindex1.json')
        local filename of parsed aaindex1 in JSON form.
    url: str (default = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1")
        url where the AAIndex1 is stored and will be downloaded from
    aaindex_json: json
        parsed AAIndex1 JSON object
    aaindex_dict: dict
        parsed AAIndex1 from JSON into more readable and useable dictionary

    Returns
    -------

    """

    def __init__(self, aa_index_filename='aaindex1'):

        self._aa_index_filename = aa_index_filename
        self._url = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"
        self.aa_index_json_filename = self.aa_index_filename + '.json'

        #if AAIndex not found in data directory then download
        if not (os.path.isfile(os.path.join(DATA_DIR,self.aa_index_json_filename))):
            if not (os.path.isfile(os.path.join(DATA_DIR,self.aa_index_filename))):
                self.download_aaindex()

        #parse AAIndex1 file into JSON format
        self.aaindex_json = self.parse_aaindex_to_json()
        #parse AA1 file into more useable and readable dictionary format
        # self.out_dict = self.parse_aaindex_from_json(self.aaindex_json)

    def parse_aaindex_to_json(self):

        """
        Parse AAI1 database into JSON format. Each AAI1 record will be indexed by
        its feature code/index code, and will be in the format as shown in the
        example above. The file will be stored in a json file called according
        to the self.aa_index_json_filename variable.

        Returns
        -------
        out_dict : dict
          parsed AAI1 in dict form.

        """
        #initialise keys of AAI1
        template_dict = {"H":[],
                         "D":[],
                         "R":[],
                         "A":[],
                         "*":[],
                         "T":[],
                         "J":[],
                         "C":[],
                         "I":[]}

        #open AAI1 file for reading and parsing
        try:
            tmp_filepath = os.path.join(DATA_DIR,self.aa_index_filename)
            f = open(tmp_filepath,'r')
        except IOError:
            print('Error opening file, check filename = {} and {} stored in \
                    {} directory'.format(self.aa_index_filename, self.aa_index_filename,
                        DATA_DIR))

        #read lines of file
        lines = f.readlines()
        f.close()

        clean_up_pattern = re.compile("\"")

        #initilaise parsed AAI1 dictionary
        self.out_dict = {}
        current_dict = copy.deepcopy(template_dict)

        '''
        iterate through each line in the AAI1, parsing each record and its
        amino acid values into its own entry in the dictionary. Each index/record
        is seperated by a '//'. Remove any duplicate records, set any missing
        ('-') or NA amino acid values to 0. Store resulting dict as X .

        '''
        for l in lines:

            if l.startswith("//"):

                # Deal with meta data
                name = " ".join(current_dict["H"])
                name = clean_up_pattern.sub("'",name)

                description = " ".join(current_dict["D"])
                description = clean_up_pattern.sub("'",description)

                citation = "{} '{}' {}".format(" ".join(current_dict["A"]),
                                               " ".join(current_dict["T"]),
                                               " ".join(current_dict["J"]))

                citation = citation + "; Kawashima, S. and Kanehisa, M. 'AAindex: amino acid index database.'  Nucleic Acids Res. 28, 374 (2000)."

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
                    self.out_dict[name]
                    err = "duplicate value name ({})".format(name)
                    raise ValueError
                except KeyError:
                    pass

                self.out_dict[name] = {"description":description,
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
        for index in self.out_dict:
            self.out_dict[index]['values']['-'] = 0

            #set any NA amino acid values to 0
            for val in self.out_dict[index]['values']:
                if self.out_dict[index]['values'][val] == 'NA':
                    self.out_dict[index]['values'][val] = 0

        #save parsed dictionary into JSON format to DATA_DIR
        with open((os.path.join(DATA_DIR, self.aa_index_json_filename)),'w') as outputF:
          json.dump(self.out_dict, outputF,indent = 4, sort_keys=True)

        return self.out_dict

    def parse_aaindex_to_category(self, aaindex_category_file = 'aaindex-to-category.txt'):
        """
        Parse category file which maps each AAI1 index into 1 of 8 categories.
        Category file and parsing code inspired from:
        https://github.com/harmslab/hops

        Parameters
        ----------
        aaindex_category_file : str
            Name of category file to parse (default is "aaindex-to-category.txt")

        Returns
        -------
        aa_index_category : dict
            Dictionary that maps each AAI into 1 of 8 categories.

        """
        #open AAI1 category file for parsing
        try:
            f = open((os.path.join(DATA_DIR, aaindex_category_file)),'r')
        except IOError:
            print('Error opening AAIndex1 category file, check {} in {} directory'
                .format(aaindex_category_file, DATA_DIR))

        #get total number of lines in file
        totalLines = len(f.readlines(  ))

        #open new file in data directory to store parsed category file
        f.seek(0)
        categoryOutputFile = "aaindex-to-category-parsed.txt"
        f_out = open((os.path.join(DATA_DIR, categoryOutputFile)), "w")

        #lines starting with '#' are file metadata so don't write these to parsed output file
        for line in f.readlines():
            if not (line.startswith('#')):
                f_out.write(line)
        f.close()
        f_out.close()

        with open(os.path.join(DATA_DIR, categoryOutputFile)) as f_out:
            reader = csv.reader(f_out, delimiter="\t")
            d = list(reader)
        f_out.close()

        aa_index_category = {}

        #iterate through all lines in parsed file and store AAI indices as keys
        #in the output dict and their respective categories as the values of the dict
        for i in range(0,len(d)):
          for j in range(0,len(d[i])):
            category_substring = d[i][1].strip()
            category_substring = category_substring.split(" ",1)
            aa_index_category[d[i][0]] = category_substring[0]

        return aa_index_category

    def download_aaindex(self, save_dir=DATA_DIR):

        """
        If AAI1 not found in DATA directory, then it will be downloaded from
        the dedicated FTP or HTTPS server from https://www.genome.jp/aaindex/.
        FTP is the default method used for downloading the database with the URL:
        "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"

        Parameters
        ----------
        save_dir : str (deafult = DATA_DIR)
            Directory to save the AAI1 database to.

        """
        #if directory doesnt exist then create it
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                raise OSError('Error creating save directory: {}'.format(save_dir))

        try:
            if not(os.path.isfile(os.path.join(save_dir, self.aa_index_filename))):
                try:
                    with closing(request.urlopen(self._url)) as r:
                        with open((os.path.join(save_dir, self.aa_index_filename)), 'wb') as f:
                            shutil.copyfileobj(r, f)
                    print('AAIndex1 successfully downloaded')
                except requests.exceptions.RequestException:
                    print('Error downloading or exporting AAIndex from the url {}'.format(self.url))
            else:
                print('AAIndex1 already in dir: {} \n'.format(save_dir))
        except OSError:
            print('Save Directory does not exist: {}\n'.format(save_dir))
            return

    def get_amino_acids(self):

        """
        Get all canonical amino acid letters.

        Returns
        -------
        amino_acids : list
            List of all 20 canoniocal amino acid letters as found in each record
            of the AAI1.

        """
        amino_acids = list(self.aaindex_json[list(self.aaindex_json.keys())[0]]["values"].keys())
        amino_acids.sort()

        return amino_acids

    def get_amino_acids_encoding(self):

      """
      Get one hot encoding of amino acids.

      Returns
      -------
      onehot_encoded : np.ndarray
        one hot encoded array of the 20 canonical amino acids.

      """
      self.get_amino_acids()
      values = np.array(self.get_amino_acids())

      label_encoder = LabelEncoder()
      integer_encoded = label_encoder.fit_transform(values)

      onehot_encoder = OneHotEncoder(sparse=False)
      integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
      onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

      return onehot_encoded

    def get_feature_codes(self, combo=1):
      """
      Get AAI1 index codes for each record in the database. If combo parameter
      equal to 2 then all combinations of 2 indices will be returned.
      put equation to calc combos here:

      Parameters
      ----------
      combo: int (default = 1)
        select what combination of all indices to return, e.g. 1 will return
        a normal list of indices, 2 will return a list of all combinations of
        length 2 for all indices etc.

      Returns
      -------
      features : list
        list of feature names/index codes for all records in the AAI1.

      """
      features = list(self.aaindex_json.keys())
      features.sort()

      if combo == 2:
          return list(itertools.combinations(features,2))
      else:
          return features

    def get_num_features(self):

      """
      Calculate number of records/indices in the database.

      Returns
      -------
      len(self.get_feature_codes()) : int
        number of indices/records found in the database.

      """
      return len(self.get_feature_codes())

    def get_feature_names(self):

      """
      Return a list of all index descriptions for all records in the AAI1.

      Returns
      -------
      desc : list
         list of descriptions for all records in the AAI1.

      """
      aaindex_values = list(self.aaindex_json.values())
      desc = []

      for name in aaindex_values:
        desc.append(name['description'])

      return desc

    def get_feature_from_code(self, feature_code):

      """
      Return full AAI1 record details from its feature/index code..

      Parameters
      ----------
      feature_code: str
         AAI1 record feature code/index code.

      Returns
      -------
      values : dict
        dict of AAI1 record

      """
      #stripping input of whitespace
      feature_code.strip()
      assert feature_code in (self.get_feature_codes()), 'Feature Index ({}) not in AAIndex'.format(feature_code)
      values = (self.aaindex_json[feature_code])

      return values

    def get_feature_from_name(self, name):

        """
        Return full AAI1 record details from its description.

        Parameters
        ----------
        name: str
            AAI1 record feature name/description

        Returns
        -------
        feature : dict
            dict of AAI1 record

        """
        feature_names = self.get_feature_names()
        feature = ("\n".join(s for s in names if name.lower() in s.lower())).splitlines()

        return feature

    def get_values_from_code(self, feature_code):

      """
      Return amino acid values from record of index in the AAI1 from its feature/index code.

      Parameters
      ----------
      feature_code: str
          AAI1 record feature code/index code.

      Returns
      -------
      values : dict
        amino acid values for specified AAI1 record.

      """
      #stripping input of whitespace
      feature_code.strip()
      #check that inputted feature_code does exist in the AAI1
      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex'
      #get amino acid values for specified index
      values = (self.aaindex_json[feature_code]['values'])

      return values

    def get_ref_from_code(self, feature_code):

      """
      Return reference details of a record from its feaure/index code.

      Parameters
      ----------
      feature_code: str
        AAI1 record feature code/index code.

      Returns
      -------
      values : list
        reference details for specifed AAI1 record.

      """
      #stripping input of whitespace
      feature_code.strip()
      #check that inputted feature_code does exist in the AAI1
      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex1'
      values = (self.aaindex_json[feature_code]['refs'])

      return values

    def get_category(self, feature_code):

      """
      Return category of a record from its feaure/index code.

      Parameters
      ----------
      feature_code: str
            AAI1 record feature code/index code.

      Returns
      -------
      cat : str
        category of AAI1 index according to parsed aaindex-to-category.txt file

      """
      #stripping input of whitespace
      feature_code.strip()
      #check that inputted feature_code does exist in the AAI1
      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex1'

      categories = self.parse_aaindex_to_category()

      cat = categories[feature_code]

      return cat

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def aa_index_filename(self):
        return self._aa_index_filename

    @aa_index_filename.setter
    def aa_index_filename(self, value):
        self._aa_index_filename = value

    def __str__(self):

        return "AAIndex Features of size {} - stored in {} ".format(self.get_num_features(),self.aa_index_filename)

    #returns the number of indices found in the database
    def __len__(self):
        """      """
        return self.get_num_features()

    def __repr__(self):

        return 'Instance of {} class, Filename: {}, URL: {}'.format(self.__class__.__name__, self.aa_index_filename, self.url)

    #return size of AAI1 database file
    def __sizeof__(self):

        return os.path.getsize(os.path.isfile(os.path.join(DATA_DIR,self.aa_index_json_filename)))
