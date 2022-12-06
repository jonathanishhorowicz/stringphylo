import numpy as np

import gpflow as gpf
from gpflow.kernels import Kernel
from gpflow.utilities import positive

import tensorflow as tf

from skbio.diversity import beta_diversity
from skbio import TreeNode

from .kernel_linalg import isPD, nearestPD

import logging
logger = logging.getLogger(__name__)

class StringKernel(Kernel):
    """A kernel matrix defined by K = XQX' where Q is a matrix of OTU-wise string similarities

    Inherits from GPflow.kernels.Kernel

    Attributes:
        _Q (np.ndarray): matrix of OTU-wise string similarities
        forceQPD (bool): whether to force Q to be positive semi-definite
        variance (float): signal variance
    """
    def __init__(self, Q, variance=1e-2, variance_lowerlim=0.0, forceQPD=False):
        logger.debug(f"Initialising SpectrumKern with Q shape {Q.shape}")
        super().__init__()
        self._Q = Q.copy() # OTU-wise similarity, p x p
        self.forceQPD = forceQPD
        if not isPD(self._Q):
            # logger.warning("StringKernel Q is not PD")
            if self.forceQPD:
                logger.warning("Forcing String Q to be PD")
                self._Q = nearestPD(self._Q)
            
        self.variance = gpf.Parameter(variance, transform=positive(variance_lowerlim))
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        logger.debug(f"X shape is {X.shape}, X2 shape is {X2.shape}, _Q shape is {self._Q.shape}")
        out = tf.linalg.matmul(tf.linalg.matmul(X, self._Q), tf.transpose(X2))
        logger.debug(f"output has shape {out.shape}")
        return out * self.variance
    
    def K_diag(self, X):
        return np.diag(self.K(X))

class UniFracKernel(Kernel):
    def __init__(self, tree, otu_names, weighted, variance, validate=False):
        """A kernel matrix derived from the UniFrac distance between samples

        Attributes:
            tree (skbio.TreeNode): phylogenetic tree
            otu_names (np.ndarray): OTU names
            weighted (bool): whether to use weighted (True) or unweighted UniFrac distance
            variance (float): signal variance
            validate (bool): whether to validate skbio inputs

        """
        super().__init__()
        
        self.tree = tree.copy()
        self.weighted = weighted
        self.otu_names = otu_names.copy() # [t.name for t in tree.tips()]
        self.validate = validate
        self._fn = self._init_k_fn()
        self.variance = gpf.Parameter(variance, transform=positive())
            
    def K(self, X, X2=None):
                
        if X2 is not None:
            XX = np.concatenate([X, X2], axis=0)
        else:
            XX = X
        logger.debug(f"XX shape is {XX.shape}")
        
        K = self._fn(XX)
        logger.debug(f"K has shape {K.shape}")
        
        if X2 is not None:
            K = K[np.ix_(np.arange(X.shape[0]), np.arange(X2.shape[0]))]
        
        return self.variance * self._centreK(K)
    
    def K_diag(self, X):
        return np.diag(self.K(X))
    
    def _centreK(self, K):
        n, m = K.shape
        J0 = np.eye(n) - (1.0/n) * np.matmul(np.ones((n,1)), np.transpose(np.ones((n,1))))
        J1 = np.eye(m) - (1.0/m) * np.matmul(np.ones((m,1)), np.transpose(np.ones((m,1))))
        logger.debug(f"J0 has shape {J0.shape}, J1 has shape {J1.shape}, K has shape {K.shape}")
        return -0.5 * np.matmul(J0, np.matmul(K, J1))
    
    def _init_k_fn(self):
        
        def _fn(x):
            return beta_diversity(
                metric="weighted_unifrac" if self.weighted else "unweighted_unifrac",
                counts=x,
                otu_ids=self.otu_names,
                tree=self.tree,
                validate=self.validate
            ).data
        
        return _fn