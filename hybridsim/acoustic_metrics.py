"""
functions to compute acoustic metrics for perceptual validation of hybrid model
"""

import numpy as np

def center_time(rir, label, fs = 44100):
    