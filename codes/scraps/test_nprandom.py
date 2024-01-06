#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:06:42 2023

@author: stevenjerjian
"""



import numpy as np


x = np.random.choice(range(0,10), size=5, replace=True, p=np.ones((10))*0.1)

print(x)
