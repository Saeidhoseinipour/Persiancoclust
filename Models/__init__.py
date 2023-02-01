#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Module `NMTFcoclust.Models` module gathers implementations of co-clustering
algorithms with Nonnegative Matrix Tri Factorization.
"""

from .NMTFcoclust_OPNMTF_alpha import OPNMTF
from .NMTFcoclust_ONMTF_alpha import CoclustELBMcem
from .NMTFcoclust_NMTF_alpha import CoclustSELBMcem



__all__ = ['OPNMTF',
		'CoclustELBMcem',
		'CoclustSELBMcem']

