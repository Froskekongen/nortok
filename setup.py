# -*- coding: utf-8 -*-
# setup.py template made by the 'datafolder' package

# If you need help about packaging, read
# https://python-packaging-user-guide.readthedocs.org/en/latest/distributing.html


import sys
import pkg_resources

from setuptools import setup


setup(name='nortok',
      version='0.1.4',
      description='Tokenization and parsing',
      url='http://github.com/Froskekongen/nortok',
      author='Erlend Aune',
      author_email='erlend.aune.1983@gmail.com',
      license='MIT',
      packages=['nortok'],
      package_dir={'nortok': 'nortok'},
      package_data={'nortok': ['data/*.*']},
      install_requires=[
          'lxml',
          'nltk',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
