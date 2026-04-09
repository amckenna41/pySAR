# Config file parameters <a name="TOP"></a>

`pySAR` works mainly via JSON configuration files. There are many different customisable parameters for the functionalities in `pySAR` including the metaparameters of some of the available protein descriptors, all Digital Signal Processing (DSP) parameters in the `pyDSP` module, the type of regression model to use and parameters specific to the dataset - a description of each parameter is available in the example below.

These config files offer a more straightforward way of making any changes to the `pySAR` pipeline. The names of **All** the parameters as listed in the example config files must remain unchanged, only the value of each parameter should be changed, any parameters not being used can be set to <em>null</em>. Additionally, you can pass in the individual parameter names and values to the `pySAR` and `Encoding` classes when numerically encoding the protein sequences via **kwargs**. An example of the config file used in my research project ([thermostability.json](https://github.com/amckenna41/pySAR/blob/master/config/thermostability.json)), with all of the available parameters, can be seen below.

```json
{
    "dataset": 
      {
        "dataset": "thermostability.txt",
        "sequence_col": "sequence",
        "activity": "t50"
      },
  
    "model": 
      {
        "algorithm": "plsregression",
        "parameters": "",
        "test_split": 0.2
      },
  
    "descriptors":
    {
      "descriptors_csv": "descriptors_thermostability.csv",
      "moreaubroto_autocorrelation":
        {
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
        },
      "moran_autocorrelation":
        {
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
        },
      "geary_autocorrelation":
        {
        "lag": 30,
        "properties": ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
          "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"],
        "normalize": 1
        },
      "ctd":
        {
        "property": "hydrophobicity",
        "all": 1
        },
      "sequence_order_coupling_number":
        {
        "lag": 30,
        "distance_matrix": "schneider-wrede.json"
        },
      "quasi_sequence_order":
        {
        "lag": 30,
        "weight": 0.1,
        "distance_matrix": "schneider-wrede.json"
        },
      "pseudo_amino_acid_composition":
        {
        "lambda": 30,
        "weight": 0.05,
        "properties": []
        },
      "amphiphilic_pseudo_amino_acid_composition":
        {
        "lambda": 30,
        "weight": 0.5
        },
      "charge_distribution":
        {
        "ph": 7.4
        },
      "kmer_composition":
        {
        "k": 2
        },
      "reduced_alphabet_composition":
        {
        "alphabet_size": 6
        },
      "motif_composition":
        {
        "motifs": null
        },
      "aggregation_propensity":
        {
        "window": 5,
        "hydrophobicity_threshold": 2.0,
        "charge_threshold": 1
        },
      "hydrophobic_moment":
        {
        "window": 11,
        "angle": 100
        }
  
      },

    "pyDSP":
      {
        "use_dsp": 0,
        "spectrum": "power",
        "window": {
          "type": "hamming",
          "sym": 1,
          "beta": null,
          "alpha": null,
          "nbar": null,
          "sll": null,
          "norm": null
        },
        "filter": {
          "type": null,
          "window_length": 5,
          "polyorder": 2,
          "deriv": 0,
          "delta": 1,
          "mode": "interp"
        }
      }  
  }
```


## Below is an explanation of each of the parameters within the JSON config files:

### Dataset Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `dataset[dataset]` | `str` | Path to the dataset file. | — |
| `dataset[sequence_col]` | `str` | Name of the column in the dataset holding protein sequences. | `"sequence"` |
| `dataset[activity]` | `str` | Name of the protein activity column in the dataset being studied. | — |

### Model Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `model[algorithm]` | `str` | Name of the ML regression algorithm to use. | — |
| `model[parameters]` | `dict` | Hyperparameters for the chosen ML algorithm. If left blank, the algorithm's default parameters are used. | Algorithm defaults |
| `model[test_split]` | `float` | Training/test split ratio, between 0 and 1. | `0.2` |

### Descriptor Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `descriptors[descriptors_csv]` | `str` | Path to a CSV file of pre-calculated descriptor values for the dataset, avoiding recalculation each run. | `null` |
| `descriptors[moreaubroto_autocorrelation][lag]` / `descriptors[moran_autocorrelation][lag]` / `descriptors[geary_autocorrelation][lag]` | `int` | Maximum lag value for each of the autocorrelation descriptors. | `30` |
| `descriptors[moreaubroto_autocorrelation][properties]` / `descriptors[moran_autocorrelation][properties]` / `descriptors[geary_autocorrelation][properties]` | `list[str]` | List of protein physicochemical/structural property AAIndex accession numbers used in autocorrelation calculation. At least 1 value required. | — |
| `descriptors[moreaubroto_autocorrelation][normalize]` / `descriptors[moran_autocorrelation][normalize]` / `descriptors[geary_autocorrelation][normalize]` | `bool` | Rescale/normalise autocorrelation values into the range 0–1. | `false` |
| `descriptors[ctd][property]` | `str` | Physicochemical property to use for CTD descriptors. Available: `hydrophobicity`, `normalized_vdwv`, `polarity`, `charge`, `secondary_struct`, `solvent_accessibility`, `polarizability`. | `"hydrophobicity"` |
| `descriptors[ctd][all]` | `bool` | If `true`, all 7 physicochemical properties are used for CTD (147 features total). | `false` |
| `descriptors[sequence_order_coupling_number][maxlag]` | `int` | Maximum lag for sequence order coupling number. Protein length must not be less than this value. | — |
| `descriptors[sequence_order_coupling_number][distance_matrix]` | `str` | Physicochemical distance matrix name for calculating sequence order coupling number. | — |
| `descriptors[quasi_sequence_order][maxlag]` | `int` | Maximum lag for quasi sequence order. Protein length must not be less than this value. | — |
| `descriptors[quasi_sequence_order][weight]` | `float` | Weighting factor for quasi sequence order calculation. | — |
| `descriptors[quasi_sequence_order][distance_matrix]` | `str` | Physicochemical distance matrix name for calculating quasi sequence order. | — |
| `descriptors[pseudo_amino_acid_composition][lambda]` | `int` | Rank correlation lambda parameter; must be a non-negative integer no larger than the protein sequence length. | — |
| `descriptors[pseudo_amino_acid_composition][weight]` | `float` | Weighting factor for pseudo amino acid composition calculation. | — |
| `descriptors[pseudo_amino_acid_composition][properties]` | `list[str]` | AAIndex property accession numbers used for sequence-order calculation. | `[]` |
| `descriptors[amphiphilic_pseudo_amino_acid_composition][lambda]` | `int` | Rank correlation lambda parameter; must be a non-negative integer no larger than the protein sequence length. | — |
| `descriptors[amphiphilic_pseudo_amino_acid_composition][weight]` | `float` | Weighting factor for amphiphilic pseudo amino acid composition calculation. | — |
| `descriptors[charge_distribution][ph]` | `float` | pH value used for computing positive, negative, and net charge via the Henderson-Hasselbalch equation. | `7.4` |
| `descriptors[kmer_composition][k]` | `int` | Length of the k-mer subsequences. `k=2` gives 400 dipeptide features; `k=3` gives 8000 tripeptide features. | `2` |
| `descriptors[reduced_alphabet_composition][alphabet_size]` | `int` | Number of physicochemical groupings to map the 20 canonical amino acids to. | `6` |
| `descriptors[motif_composition][motifs]` | `list[str]` \| `null` | List of regex motif patterns to search for in each sequence. If `null`, a built-in set of 8 biological motifs is used. | `null` |
| `descriptors[aggregation_propensity][window]` | `int` | Sliding-window size (in residues) used to evaluate aggregation-prone segments. | `5` |
| `descriptors[aggregation_propensity][hydrophobicity_threshold]` | `float` | Minimum mean Kyte-Doolittle hydrophobicity score for a window to be considered aggregation-prone. | `2.0` |
| `descriptors[aggregation_propensity][charge_threshold]` | `int` | Maximum absolute net charge allowed within a window for it to be flagged as aggregation-prone. | `1` |
| `descriptors[hydrophobic_moment][window]` | `int` | Sliding-window size (in residues) used for helical-wheel hydrophobic moment calculation. | `11` |
| `descriptors[hydrophobic_moment][angle]` | `int` | Helical-wheel rotation angle in degrees between consecutive residues. `100°` corresponds to an α-helix. | `100` |

### DSP Parameters

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `pyDSP[use_dsp]` | `bool` | Whether to apply Digital Signal Processing techniques to the encoded features before model training. | `false` |
| `pyDSP[spectrum]` | `str` | Informational spectrum type to use from the DSP output. Options: `power`, `absolute`, `imaginary`, `real`. | `"power"` |
| `pyDSP[window][type]` | `str` | Convolutional window to apply to the signal. Supported: `hamming`, `blackman`, `blackmanharris`, `gaussian`, `bartlett`, `kaiser`, `barthann`, `bohman`, `chebwin`, `cosine`, `exponential`, `flattop`, `hann`, `boxcar`, `hanning`, `nuttall`, `parzen`, `triang`, `tukey`. | `"hamming"` |
| `pyDSP[filter][type]` | `str` | Filter to apply to the signal output. Supported: `savgol`, `medfilt`, `symiirorder1`, `lfilter`, `hilbert`. | `null` |

[Back to top](#TOP)