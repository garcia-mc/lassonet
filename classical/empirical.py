#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:44:19 2022

@author: duser1
"""

reference=data[(z==0).squeeze(),5]

treatment=data[(z==1).squeeze(),5]

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
ecdf0 = ECDF(reference)
ecdf1 = ECDF(treatment)




