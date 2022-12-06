import numpy as np
import pandas as pd

from .utils import median_heuristic, safe_rescale
from .kernel_classes import StringKernel
from gpflow.kernels import Matern32, RBF

import logging
logger = logging.getLogger(__name__)

def calc_mmd(K, g1_idxs, g2_idxs):
    """Calculate the maximum mean discrepancy given a kernel matrix
    and two sets of group indices.

    Uses the unbiased MMD estimator of Gretton et al. (2012).

    Args:
        K (np.ndarray): a kernel matrix
        g1_idxs (np.ndarray): indices for samples in group 1
        g2_idxs (np.ndarray): indices for samples in group 2

    Returns:
        An unbiased estimate of the maximum mean discrepancy
    """
    m = g1_idxs.shape[0]
    n = g2_idxs.shape[0]
    Kmm = K[np.ix_(g1_idxs,g1_idxs)]
    Knn = K[np.ix_(g2_idxs,g2_idxs)]
    Kmn = K[np.ix_(g1_idxs,g2_idxs)]
    return 1.0/m**2.0 * Kmm.sum() + 1.0/n**2.0 * Knn.sum() - (2.0/(m*n))*Kmn.sum()

def perm_mmd_test(X0, X1, kernel_maker, n_perms, rng):
    """The permutation test for the maxmimum mean discrepancy.

    This uses the positively-biased p-value estimator (see Phipson and Smyth, 2010)

    Args:
        X0 (np.ndarray): OTU table for group 0
        X1 (np.ndarray): OTU table for group 1
        kernel_maker (callable): returns another callable that computes a kernel matrix
        n_perms (int): number of permutations
        rng (np.random.Generator): a pRNG generator

    Returns:
        A 3-tuple containing
            1. the MMD values under permutation
            2. the observed MMD value
            3. the correspdonding p-value
    """
    logger.debug(f"MMD significance test with {n_perms} permutations")
    
    mmd_perm_vals = np.full(n_perms, np.nan)
    k_fn = kernel_maker(X0, X1)
    K_all = np.array(k_fn(np.concatenate([X0, X1], axis=0)))
    
    mmd_obs = calc_mmd(K_all, *np.split(np.arange(K_all.shape[0]), [X0.shape[0]]))
        
    # permute samples
    for i in range(n_perms):
        sample_perm_order = rng.permutation(np.arange(K_all.shape[0]))
        sample_perm_order = np.split(sample_perm_order, [X0.shape[0]])
        mmd_perm_vals[i] = calc_mmd(K_all, *sample_perm_order)

    p_value = (1.0 + np.nansum(mmd_perm_vals>=mmd_obs)) / (np.count_nonzero(~np.isnan(mmd_perm_vals)) + 1.0)
        
    return mmd_perm_vals, mmd_obs, p_value

# kernel factories
# useful for constructing a kernel that depends on the data
def make_rbf_kernel_fn(x0, x1, rescale):
    """Returns a callable that computes the
    RBF kernel with median heuristic lengthscale

    Args:
        x0 (np.ndarray): OTU table for group 0
        x1 (np.ndarray): OTU table for group 1
        rescale (bool): whether to standardise the variables

    Returns:
        A callable that computes the a kernel matrix from two arrays
    """
    x0x1 = np.concatenate([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)

    def _fn(x0, x1=None):
        return RBF(lengthscales=h_sq**0.5).K(x0, x1).numpy()

    return _fn # RBF(lengthscales=h_sq**0.5).K

def make_matern32_kernel_fn(x0, x1, rescale):
    """Returns a callable that computes the
    Matern32 kernel with median heuristic lengthscale

    Args:
        x0 (np.ndarray): OTU table for group 0
        x1 (np.ndarray): OTU table for group 1
        rescale (bool): whether to standardise the variables

    Returns:
        A callable that computes the a kernel matrix from two arrays
    """
    x0x1 = np.concatenate([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)

    def _fn(x0, x1=None):
        return Matern32(lengthscales=h_sq**0.5).K(x0, x1).numpy()

    return _fn # Matern32(lengthscales=h_sq**0.5).K

def string_kernel_fn_factory(Q, variance, forceQPD):
    """Returns a callable that computes the
    string kernel

    Args:
        Q (np.ndarray): matrix of string similarities
        variance (float): signal varance
        forceQPD (bool): whether to force Q to be positive semi-definite

    Returns:
        A callable that computes the a kernel matrix from two arrays
    """
    
    def _fn(x0=None, x1=None):
        kernel = StringKernel(Q, variance, forceQPD)
        return kernel.K
        
    return _fn

def make_gram_kernel_fn(x0, x1):
    """Returns a callable that computes the
    gram (linear) kernel

    Args:
        x0 (np.ndarray): OTU table for group 0
        x1 (np.ndarray): OTU table for group 1

    Returns:
        A callable that computes the a kernel matrix from two arrays
    """
    def _fn(x, y=None):
        if y is None:
            y = x
        return np.dot(x,y.T)
    
    return _fn