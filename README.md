# ProAct # <a name="TOP"></a>

To DO:
https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names
Hanging Indents -
If statement correct indentation
Comments!
Use backslashes to seperate lines


<!-- # Correct:
# easy to match operators with operands
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest) -->

        Blank line
          <!-- Surround top-level function and class definitions with two blank lines.

          Method definitions inside a class are surrounded by a single blank line. -->

<!-- Imports:
# Wrong:
import sys, os
#correct
import os
import sys


Imports should be grouped in the following order:

Standard library imports.
Related third party imports.
Local application/library specific imports.
You should put a blank line between each group of imports.


Wildcard imports (from <module> import *) should be avoided, as they make it unclear which names are present in the namespace, confusing both readers and many automated tools

Module level "dunders" (i.e. names with two leading and two trailing underscores) such as __all__, __author__, __version__, etc. should be placed after the module docstring but before any import statements except from __future__ imports.


# Correct:
i = i + 1
# Wrong:
i=i+1

#Trailing Commas
# Correct:
FILES = [
    'setup.cfg',
    'tox.ini',
    ]
initialize(FILES,
           error=True,
           )

#Comments:
Comments that contradict the code are worse than no comments. Always make a priority of keeping the comments up-to-date when the code changes!

Comments should be complete sentences. The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).

#Naming conventions


Block comments generally consist of one or more paragraphs built out of complete sentences, with each sentence ending in a period.

Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability. Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.

Class names should normally use the CapWords convention.


Function names should be lowercase, with words separated by underscores as necessary to improve readability.

Variable names follow the same convention as function names.



Class names should follow the UpperCaseCamelCase convention
Python's built-in classes, however are typically lowercase words
Exception classes should end with (suffix) “Error”


Global variables should be all lowercase
Words in a global variable name should be separated by an underscore
It is preferable to use these variables inside one module only





Constants
Constants are usually defined on a module level and written in all capital letters with underscores separating words. Examples include MAX_OVERFLOW and TOTAL.


Comparisons to singletons like None should always be done with is or is not, never the equality operators.

# Correct:
if foo is not None:

Use exception chaining appropriately. In Python 3, "raise X from Y" should be used to indicate explicit replacement without losing the original traceback.

When catching exceptions, mention specific exceptions whenever possible instead of using a bare except: clause:



try:
    process_data()
except Exception as exc:
    raise DataProcessingFailedError(str(exc))

    Try ELSE !

    # Correct:
try:
    value = collection[key]
except KeyError:
    return key_not_found(key)
else:
    return handle_value(value)


    Be consistent in return statements. Either all return statements in a function should return an expression, or none of them should. If any return statement returns an expression, any return statements where no value is returned should explicitly state this as return None, and an explicit return statement should be present at the end of the function (if reachable):

    # Correct:

    def foo(x):
        if x >= 0:
            return math.sqrt(x)
        else:
            return None

use str.startswith and str.endswith instread of str[:3]
# Correct:
if foo.startswith('bar'):

USE# Correct:
if not seq:
if seq:

instead of

# Wrong:
if len(seq):
if not len(seq): -->
## status
> Development Stage

Type Hinting???
def hello_name(name: str) -> str:
    return(f"Hello {name}")

[![pytest](https://github.com/ray-project/tune-sklearn/workflows/Development/badge.svg)](https://github.com/ray-project/tune-sklearn/actions?query=workflow%3A%22Development%22)

![Platforms](https://img.shields.io/badge/platforms-linux%2C%20macOS%2C%20Windows-green)

## System Requirements ##

Python > 3.6
numpy >= 1.16.6
pandas >= 1.1.0
scikit-learn >= 0.24
scipy >= 1.4.1



## Installation ##
```
git clone ....
```

#Create local virtual env??


##Dockerfile???

## Directory folders:

* `/data` - stores all required feature data and datasets
* `/models` - stores model output
* `/tests` - tests for class methods and functions


## Usage ##


## Model Saving Structure ##

```
outputs
├── model_output_YYYY_MM_DD:HH:MM
│   └── plots         
│         └── figure1.png
│         └── figure2.png
│   └── results.csv
│   └── results.json
│   └── model.pkl
└-
```

## Run Tests
python -m unittest test_module1 test_module2


# Contact

If you have any questions or comments, please contact: amckenna41@qub.ac.uk @

[Back to top](#TOP)

<!--
#Put all data in data folder - each dataset has to be in the form:

Name: Sequence: Target

rename to

ProTSAR
ProtSAR  
ProPhySAR
ProDescSAR
ActSeqR


#downloading and importing PyBioMed:
 curl -LJO https://github.com/gadsbyfly/PyBioMed/archive/master.zip
 unzip PyBioMed-master.zip
 rm -r PyBioMed-master.zip
 mv PyBioMed-master PyBioMed



Import Dataset -> Calculate AAI for all sequences -> Calcualte descriptors from dataset ->
Build predictive modles -> Output Results


#check using standardscaler correctly;

from numpy import asarray
from sklearn.preprocessing import StandardScaler
# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
print(data)
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)
print(scaled) -->
