###############################################################################
#####   Setup.py - installs all the required packages and dependancies    #####
###############################################################################

import pathlib
from setuptools import setup, find_packages
import sys
import pySAR

#ensure python version is greater than 3
if (sys.version_info[0] < 3):
    sys.exit('Python 3 is the minimum version requirement.')

#get path to README file
HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(name='pySAR',
      version=pySAR.__version__,
      description='A Python package used to analysis Protein Sequence Activity Relationships',
      long_description = README,
      long_description_content_type = "text/markdown",
      author=pySAR.__license__,
      author_email=pySAR.__authorEmail__,
      license=pySAR.__license__,
      url=pySAR.__url__,
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],

      install_requires=[
          'numpy>=1.16.6',
          'pandas',
          'scipy',
          'delayed',
          'scikit-learn==0.24.1',
          'requests>=2.25',
          'urllib3>=1.26'
          'matplotlib',
          'seaborn',
          'tqdm',
          'pytest',
          'varname',
          'biopython'
      ],
     # packages=find_packages(), #create Manifest file to ignore results folder in dist
     packages=find_packages(exclude=["Results"]),
     include_package_data=True,
     zip_safe=False)
