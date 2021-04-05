######################################################################
### Setup.py - installs all the required packages and dependancies ###
######################################################################

from setuptools import setup, find_packages
import sys

if sys.version_info[0] < 3:
    sys.exit('Python 3 is the minimum version requirement')

setup(name='training',
      version='0.1',
      description='',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='',
      install_requires=[
          'numpy==1.16.6',
          'pandas',
          'scipy',
          'scikit-learn>=0.24',
          'pyyaml',
          'requests',
          'matplotlib',
          'seaborn',
          'tqdm',
          'google.cloud',
          'google-cloud-core==1.3.0',
          'google-api-core==1.16.0',
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)
