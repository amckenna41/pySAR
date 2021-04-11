######################################################################
### Setup.py - installs all the required packages and dependancies ###
######################################################################
import pathlib
from setuptools import setup, find_packages
import sys

if sys.version_info[0] < 3:
    sys.exit('Python 3 is the minimum version requirement')

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setup(name='ProtSAR',
      version='1.0.1',
      description='Protein Sequence Activity Relationship',
      long_description = README,
      long_description_content_type = "text/markdown",
      url = "https://github.com/amckenna41/ProtSAR",
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='MIT',
      install_requires=[
          'numpy==1.16.6',
          'pandas',
          'scipy',
          'delayed',
          'scikit-learn==0.24.1',          
          'pyyaml',
          'requests',
          'matplotlib',
          'seaborn',
          'tqdm'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)
