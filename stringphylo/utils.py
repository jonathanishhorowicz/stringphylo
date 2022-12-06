import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

def median_heuristic(x):
    """Compute the median of pairwise Euclidean distances between variables. Used 
    for RBF and Matern32 kernel lengthscale parameter

    Args:
        x (np.ndarray): array with samples on rows and OTUs on columns

    Returns:
        The median heuristic lengthscale
    """
    return np.median(pdist(x, metric="sqeuclidean"))

def safe_rescale(train_arr, test_arr=None):
    """Rescale variables in training samples train_arr to mean zero and variance one. 
    Also rescales test samples with the same mean/variance if provided

    Args:
        train_arr (np.ndarray): array with samples on rows and OTUs on columns
        test_arr (np.ndarray): array with samples on rows and OTUs on columns

    Returns:
        A 2-tuple containing
            1. the rescaled training array
            2. None if test_arr is None, otherwise the rescaled test array
    """
    
    if train_arr.ndim==1:
        train_arr = train_arr[:,np.newaxis]
    
    if test_arr is not None:
        if test_arr.ndim==1:
            test_arr = test_arr[:,np.newaxis]
        assert train_arr.shape[1]==test_arr.shape[1]
    ss = StandardScaler()
    train_arr_ = ss.fit_transform(train_arr)
    test_arr_ = ss.transform(test_arr) if test_arr is not None else None
    
    if isinstance(train_arr, pd.DataFrame):
        train_arr_ = pd.DataFrame(train_arr_, columns=train_arr.columns, index=train_arr.index)
    
    if isinstance(test_arr, pd.DataFrame):
        test_arr_ = pd.DataFrame(test_arr_, columns=test_arr.columns, index=test_arr.index)
    
    return train_arr_, test_arr_