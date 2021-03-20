#similar search function that searches based on AAIndex name
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
from globals import *

'''
Python parser for AAindex: Amino Acid Index Database

http://www.genome.jp/aaindex/

AAIndex1 parser based off aaindex2json.py by harmslab:
https://github.com/harmslab/hops/blob/master/hops/features/data/util/aaindex2json.py

Data format of AAIndex1:
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

'''

'''

Data Format of AAindex2 and AAindex3
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
* M rows = ARNDCQEGHILKMFPSTWYV, cols = ARNDCQEGHILKMFPSTWYV           *
*   AA                                                                 *
*   AR RR                                                              *
*   AN RN NN                                                           *
*   AD RD ND DD                                                        *
*   AC RC NC DC CC                                                     *
*   AQ RQ NQ DQ CQ QQ                                                  *
*   AE RE NE DE CE QE EE                                               *
*   AG RG NG DG CG QG EG GG                                            *
*   AH RH NH DH CH QH EH GH HH                                         *
*   AI RI NI DI CI QI EI GI HI II                                      *
*   AL RL NL DL CL QL EL GL HL IL LL                                   *
*   AK RK NK DK CK QK EK GK HK IK LK KK                                *
*   AM RM NM DM CM QM EM GM HM IM LM KM MM                             *
*   AF RF NF DF CF QF EF GF HF IF LF KF MF FF                          *
*   AP RP NP DP CP QP EP GP HP IP LP KP MP FP PP                       *
*   AS RS NS DS CS QS ES GS HS IS LS KS MS FS PS SS                    *
*   AT RT NT DT CT QT ET GT HT IT LT KT MT FT PT ST TT                 *
*   AW RW NW DW CW QW EW GW HW IW LW KW MW FW PW SW TW WW              *
*   AY RY NY DY CY QY EY GY HY IY LY KY MY FY PY SY TY WY YY           *
*   AV RV NV DV CV QV EV GV HV IV LV KV MV FV PV SV TV WV YV VV        *
* //                                                                   *
************************************************************************

'''

## maybe split up the AAIndex object into individual Record objects,
#record obj contains stuff like the title, description etc of the AAIndex record, AAIndex obj made up of Records like in PyBioMed

class AAIndex():
    """
    Parameters:
    aa_index_filename:
    """
    AA_INDEX1_URL = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"
    aa_index_filename = 'aaindex1'
    aa_index_json_filename = 'aaindex1.json'

    def __init__(self):

        if not (os.path.isfile(os.path.join(DATA_DIR,self.aa_index_json_filename))):
            if not (os.path.isfile(os.path.join(DATA_DIR,self.aa_index_filename))):
                self.download_aaindex()
        self.aaindex_json = self.parse_aaindex_to_json()
        self.out_dict = self.parse_aaindex_from_json(self.aaindex_json)

    def parse_aaindex_to_json(self):

        template_dict = {"H":[],
                         "D":[],
                         "R":[],
                         "A":[],
                         "*":[],
                         "T":[],
                         "J":[],
                         "C":[],
                         "I":[]}
        try:
            tmp_filepath = os.path.join(DATA_DIR,self.aa_index_filename)
            f = open(tmp_filepath,'r')
        except IOError:
            print('Error opening file, check filename = aaindex1')
            return 0

        lines = f.readlines()
        f.close()

        clean_up_pattern = re.compile("\"")

        self.out_dict = {}
        current_dict = copy.deepcopy(template_dict)
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

            for val in self.out_dict[index]['values']:
                if self.out_dict[index]['values'][val] == 'NA':
                    self.out_dict[index]['values'][val] = 0

        with open((os.path.join(DATA_DIR, self.aa_index_json_filename)),'w') as outputF:
          json.dump(self.out_dict, outputF,indent = 4, sort_keys=True)

        return self.out_dict

    def parse_aaindex_from_json(self, aaindex_json):

        try:
           # json_data = json.load(open(self.aa_index_filename))
           json_data = aaindex_json
        except json.JSONDecodeError as jerr:
            print("Invalid JSON in file")

        # Figure out what features are in the array
        features = list(json_data.keys())
        features.sort()

        # Get all amino acids
        amino_acids = list(json_data[list(json_data.keys())[0]]["values"].keys())
        amino_acids.sort()

        # Data will bye a num_feature x num_aa array with standardized features
        data = np.zeros((len(features),len(amino_acids)),dtype=float)

        for i, f in enumerate(features):

            # Get values
            values = [json_data[f]["values"][aa] for aa in amino_acids]

            # Turn to float, converting "NA" to nan
            out = []
            for v in values:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(np.nan)

            # Replace nan with the mean value for the vector.
            out = np.array(out)

            na_mask = np.isnan(out)
            not_na_mask = np.array(1-na_mask,dtype=bool)

            mean_value = np.mean(out[not_na_mask])
            out[na_mask] = mean_value

            # # Standardize data so mean is zero and standard deviation is -1 to 1
            out = out - np.mean(out)
            out = out/np.std(out)

            # Record data
            data[i,:] = out
            self.feature_data = data
        # with open(os.path.join('features', self.aa_index_json_filename)) as output_json:
        #     json.dump(data, output_json,indent = 4, sort_keys=True)

        return self.feature_data

    def parse_aaindex_to_category(self, aaindex_category_file = 'aaindex-to-category.txt'):

        try:
            f = open((os.path.join(DATA_DIR, aaindex_category_file)),'r')
        except IOError:
            print('Error opening aaindex category file')
            return 0

        totalLines = len(f.readlines(  ))

        f.seek(0)
        categoryOutputFile = "aaindex-to-category-parsed.txt"
        f_out = open((os.path.join(DATA_DIR, categoryOutputFile)), "w")

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

        for i in range(0,len(d)):
          for j in range(0,len(d[i])):
            category_substring = d[i][1].strip()
            category_substring = category_substring.split(" ",1)
            aa_index_category[d[i][0]] = category_substring[0]

        return aa_index_category

    def download_aaindex(self):

        try:
            if not(os.path.isfile(os.path.join(DATA_DIR, self.aa_index_filename))):
                with closing(request.urlopen(self.AA_INDEX1_URL)) as r:
                    with open((os.path.join(DATA_DIR, self.aa_index_filename)), 'wb') as f:
                        shutil.copyfileobj(r, f)
                print('AAIndex1 successfully downloaded')

            else:
                print('AA Index File already in dir\n')
        except OSError:
            print('Error downloading and exporting AA index file\n')
            return

    def get_amino_acids(self):

        amino_acids = list(self.aaindex_json[list(self.aaindex_json.keys())[0]]["values"].keys())
        amino_acids.sort()

        return amino_acids

    def get_amino_acids_encoding(self):

      self.get_amino_acids()
      values = np.array(self.get_amino_acids())

      label_encoder = LabelEncoder()
      integer_encoded = label_encoder.fit_transform(values)

      onehot_encoder = OneHotEncoder(sparse=False)
      integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
      onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

      return onehot_encoded

    def get_feature_codes(self):

      features = list(self.aaindex_json.keys())
      features.sort()

      return features

    def get_num_features(self):

        return len(self.get_feature_codes())

    def get_feature_name(self):

      aaindex_values = list(self.aaindex_json.values())
      names = []

      for name in aaindex_values:
        names.append(name['description'])

      return names

    def get_feature_from_code(self, feature_code):

      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex'
      values = (self.aaindex_json[feature_code])

      return values

    def get_values_from_code(self, feature_code):

      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex'
      values = (self.aaindex_json[feature_code]['values'])

      return values

    def get_ref_from_code(self, feature_code):

      assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex'
      values = (self.aaindex_json[feature_code]['refs'])

      return values

    def get_category(self, feature_code):

        assert feature_code in (self.get_feature_codes()), 'Feature Index not in AAIndex'

        categories = self.parse_aaindex_to_category()

        cat = categories[feature_code]

        return cat

    def url(self):

        return self.url

    def plot_aaindex(self):
        pass

    def __str__(self):

        return "AAIndex Features of size {} - stored in {} ".format(self.get_num_features(),self.aa_index_filename)

    def __len__(self):
        """      """
        return get_num_features(self)

    def __doc__(self):
        pass

    def __repr__(self):
        pass

    def __sizeof__(self):
        pass

# https://github.com/speleo3/pymol-psico/blob/master/psico/aaindex.py
#https://gist.github.com/pansapiens/870908abc7d0292ace0654a036487303
#fix Input direcotry - change 'features' dir to DATA_DIR
class AAIndex2():
    """
    Parameters:
    aa_index_filename:
    """

    aa_index2_filename = 'aaindex2'
    AA_Index2_URL = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex2"

    def __init__(self, aaindex2_filename = 'aaindex2'):

        self.aaindex2_filename = aaindex2_filename
        self.aaindex2_json = aaindex2_filename + '.json'

        if not (os.path.isfile(os.path.join(DATA_DIR,self.aaindex2_json))):
            if not (os.path.isfile(os.path.join(DATA_DIR,self.aaindex2_filename))):
                self.download_aaindex2()
        self.aaindex2_json = self.parse_aaindex2_to_json()
        # self.out_dict = self.parse_aaindex2_from_json(self.aaindex2_json)

    def download_aaindex2(self):

        try:
            if not(os.path.isfile(os.path.join(DATA_DIR, self.aa_index2_filename))):
                with closing(request.urlopen(self.AA_Index2_URL)) as r:
                    with open((os.path.join(DATA_DIR, self.aaindex2_filename)), 'wb') as f:
                        shutil.copyfileobj(r, f)
                print('AAIndex2 successfully downloaded')

            else:
                print('AA Index2 File already in {} dir\n'.format(DATA_DIR))
        except OSError:
            print('Error downloading and exporting AA index2 file\n')
            return

    def parse_aaindex2_to_json(self):

        template_dict = {"H":[],
                         "D":[],
                         "R":[],
                         "A":[],
                         "*":[],
                         "T":[],
                         "J":[],
                         "M":[]}
        try:
            tmp_filepath = os.path.join(DATA_DIR,self.aaindex2_filename)
            f = open(tmp_filepath,'r')
        except IOError:
            print('Error opening file, check filename = aaindex2')
            return 0

        lines = f.readlines()
        f.close()

        aaindex = {}
        current_id = None
        record_type = None

        for ll in lines:
            l = ll.strip()
            if l == '':
                continue

            if ll[0] in ['H', 'D', 'R', 'A', 'T', '*', 'J', 'M']:
                record_type = l[0]

            if l == '//':
                record_type = '//'

            if record_type == 'H':
                current_id = l[2:]
                aaindex[current_id] = defaultdict(str)
                matrix = {}
                i_row = 0
            if record_type in ['D', 'R', 'A', 'T', '*', 'J']:
                aaindex[current_id][record_type] += l[2:]

            if record_type == 'M':
                if ll[0] == 'M':
                    s = l[2:].split(',')
                    rows = list(s[0].split('=')[1].strip())
                    cols = list(s[1].split('=')[1].strip())
                    aaindex[current_id]['row_index'] = rows
                    aaindex[current_id]['col_index'] = cols
                else:
                    values = []
                    for v in l.split():
                        if v != '-':
                            values.append(float(v))
                        else:
                            values.append(None)

                    for i_col, v in enumerate(values):
                        matrix[(cols[i_col], rows[i_row])] = v
                    i_row += 1

            if record_type == '//':
                aaindex[current_id]['matrix'] = matrix
                record_type = None

        # return aaindex
        tmp_df = pd.DataFrame(aaindex).to_csv('tmp_df.csv')
        tmp_csv = pd.read_csv('tmp_df.csv',index_col=0)
        # tmp_csv.drop(['0'],inplace=True,axis=0)
        os.remove('tmp_df.csv')

        tmp_json = tmp_csv.to_json()
        tmp_parsed_json = json.loads(tmp_json)

        with open((os.path.join(DATA_DIR, self.aaindex2_filename+'.json')),'w') as outputF:
            json.dump(tmp_parsed_json, outputF,indent = 4, sort_keys=True)

        return aaindex

    def __str__(self):

        return "AAIndex2 Features - stored in " + self.aa_index_filename

    def __len__(self):
        """      """
        return get_num_features(self)
