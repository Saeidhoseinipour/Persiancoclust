#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Module `NMTFcoclust.Models` module gathers implementations of co-clustering
algorithms with Nonnegative Matrix Tri Factorization.
"""

from .NMTFcoclust_OPNMTF_alpha import OPNMTF
from .coclust_ELBMcem import CoclustELBMcem
from .coclust_SELBMcem import CoclustSELBMcem



__all__ = ['OPNMTF',
		'CoclustELBMcem',
		'CoclustSELBMcem']

