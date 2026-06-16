"""
Created by Florian Le Guillou on June 2026.

Initializes the mapping source package.
"""

import os 
# Limit number of threads in numpy
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np