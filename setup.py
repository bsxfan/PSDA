#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022
# Author: Niko Brummer
# All Rights Reserved

from distutils.core import setup
from setuptools import find_packages


setup(
    name='PSDA',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/bsxfan/PSDA',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    license='MIT',
    author='Niko Brummer',
    author_email='niko.brummer@gmail.com',
    description='Python implementation of Probabilistic Spherical Discriminant Analysis.'
)
