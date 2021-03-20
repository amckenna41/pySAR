from aaindex import *
from globals import *

aaindex = AAIndex()

print('testing')

def test_download():

    if (os.path.isfile(os.path.join(DATA_DIR, 'aaindex1'))):
        os.remove(os.path.join(DATA_DIR, 'aaindex1'))
    if (os.path.isfile(os.path.join(DATA_DIR, 'aaindex1.json'))):
        os.remove(os.path.join(DATA_DIR, 'aaindex1.json'))

    aa_index_ = AAIndex()

    assert(os.path.isfile(os.path.join(DATA_DIR, 'aaindex1')))
    assert(os.path.isfile(os.path.join(DATA_DIR, 'aaindex1.json')))

def test_url():

    AA_INDEX1_URL = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1"
    AA_INDEX2_URL = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex2"
    AA_INDEX3_URL = "ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex3"

    ##test URL endpoint for AAINDEX is active and not errorenous##
    r = requests.get(AA_INDEX1_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(AA_INDEX2_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(AA_INDEX3_URL, allow_redirects = True)
    assert(r.status_code == 200)

def test_get_amino_acids():

    valid_amino_acids = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acids = aaindex.get_amino_acids()

    for aa in amino_acids:
        assert(aa in valid_amino_acids)

def test_num_features():

    assert(aaindex.get_num_features() >= 566)

def test_feature_size():

    assert(len(aaindex.get_feature_codes()))

def test_get_feature_from_code():
    #**
    feature = ''
    feature_vals = aaindex.get_feature_from_code(feature)
    #get specific values for featire
    #comapre results from function to actual valeus from aaindex
    #repeat for several features

def test_feature_codes():

    feature1 = 'ABC'
    feature2 = 'ABC'
    feature3 = 'ABC'

    assert feature1 in aaindex.get_feature_codes()

    feature4 = 'Not in AAIndex'

    assert feature4 not in aaindex.get_feature_codes()

def test_category():

    feature1 = ''
    feature2 = ''
    feature3 = ''
    cat1 = aaindex.get_category(feature1)
    assert cat1 == 'hydrophobic etc'

def test_aaindex_names():

    feature1 = ''
    feature2 = ''
    feature3 = ''

    name1 = aaindex.get_feature_name(feature1)
    assert name1 == 'Feature name'

def test_aaindex_refs():

    feature1 = ''
    feature2 = ''
    feature3 = ''

    ref1 = aaindex.get_ref_from_code(feature1)
    assert ref1 == 'Reference name'
