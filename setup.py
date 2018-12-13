#!/usr/bin/env python

from distutils.core import setup

setup(name='StruMs',
      version='0.2',
      description='Structural Motifs',
      author='Peter DeFord',
      #author_email='pdeford@jhu.edu',
      license = "MIT",
      packages=['strum'],
      package_data={'strum' : ['data/diprodb_2016.txt']},
      url='https://github.com/pdeford/strum',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'bx-python',
      ]
     )