from scipy.stats import kstest
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd


def get_metrics(a, a0, n_bins=30, logbin=False):

    # convert to series
    a = a.numpy()
    a0 = a0.numpy()
    a = pd.Series(a)
    a0 = pd.Series(a0)

    bmin, bmax = pd.concat([a, a0]).min(), pd.concat([a, a0]).max()

    if bmax > 0.0:
        brange = np.log10(bmax - bmin)
    else:
        brange = -1
    if (logbin or brange > 4) and bmin >= 0.0:
        bins = np.logspace(np.log10(bmin + 1e-1), np.log10(bmax), n_bins)
    else:
        bins = np.linspace(bmin, bmax, n_bins)
    bc0, be0 = np.histogram(a0, density=True, bins=bins)
    bc, be = np.histogram(a, density=True, bins=bins)

    ks = kstest(a.values, a0.values).statistic
    js_dist = jensenshannon(bc, bc0)
    metrics = {
        "ks_dist": ks,
        "js_dist": js_dist,
    }
    return metrics, bins
