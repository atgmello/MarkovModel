#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data
NAME = 'mm'
DESCRIPTION = 'Markov Model implementation with sklearn-like API.'
URL = 'https://github.com/atgmello/MarkovModel'
EMAIL = 'andre.tgmello@gmail.com'
AUTHOR = 'André Thomaz Gandolpho de Mello'
REQUIRES_PYTHON = '>=3.6.0'

# What packages are required for this module to be executed?


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# The rest you shouldn't have to touch too mcuh :)
# ------------------------------------------------
# Except, perhaps, the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description
# Note: this will only work if 'README.md' is present in your MANIFEST.in file
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'mm': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    licence='MIT',
    classifiers=[
        # Trove Classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)
