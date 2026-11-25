"""The constant-weight sweeps behind fig:tuned-w-metric, tab:tuned-w and fig:frontier-sd35m.

SD 3.5 Medium: the 42 weights the paper's per-metric search evaluated: w = 1 (the unguided arm) and 41
constants, from a 0.5 grid and then a 0.1 grid bracketing each metric's optimum. The search was adaptive; its
final set is fixed here so that a rebuild samples exactly the same arms. SD 1.5 and EDM2-S: plain 0.5 grids
from w = 1 (31 and 19 weights).
"""
import numpy as np

SD35M = [1.0, 1.1, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 3.0, 3.3, 3.4, 3.5, 3.6, 3.8,
         4.0, 4.1, 4.2, 4.5, 4.7, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.5, 11.0, 11.3, 11.4, 11.5, 11.6, 11.7, 12.0,
         13.0, 14.0]
SD15 = [float(w) for w in np.arange(1.0, 16.01, 0.5)]
EDM2S = [float(w) for w in np.arange(1.0, 10.01, 0.5)]


def arm(w):
    """The sample.py arm name of a constant weight."""
    return f"w{w:g}"
