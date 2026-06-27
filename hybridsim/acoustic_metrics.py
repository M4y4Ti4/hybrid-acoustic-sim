"""
functions to compute acoustic metrics for perceptual validation of hybrid model
"""

import numpy as np

def center_time(rir, label, fs = 44100):
    """
    calculate the center time of an RIR
    """

    t = np.arange(len(rir)) / fs

    rir2 = rir**2
    Ts = np.sum(t * rir2) / np.sum(rir2)

    return Ts


def clarity(rir, td, fs = 44100, ETL = 50):
    """
    calculate the clarity of an RIR
    ETL: early time limit (can be 50ms or 80ms)
    """

    te_samples = int(ETL * fs)

    early = np.sum(rir[td:te_samples] ** 2)
    late = np.sum(rir[te_samples:] ** 2)

    return 10 * np.log10(late / early)


