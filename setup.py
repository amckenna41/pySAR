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

#parse README file
HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(name=pySAR.__name__,
      version=pySAR.__version__,
      description=pySAR.__description__,
      long_description = README,
      long_description_content_type = "text/markdown",
      author=pySAR.__license__,
      author_email=pySAR.__authorEmail__,
      maintainer=pySAR.__maintainer__,
      license=pySAR.__license__,
      url=pySAR.__url__,
      download_url=pySAR.__download_url__,
      keywords=pySAR.__keywords__,
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'delayed',
          #'scikit-learn==0.24.1',
          'scikit-learn',
          'matplotlib',
          'seaborn',
          'tqdm',
          'aaindex',
          'protpy'
      ],
     test_suite=pySAR.__test_suite__,
     packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "Results"]), 
     include_package_data=True,
     zip_safe=False)