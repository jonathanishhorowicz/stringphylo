import numpy as np
import gpflow as gpf
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from .kernel_classes import StringKernel

import logging
logger = logging.getLogger(__name__)

def optimise_gpmod(model, **kwargs):
    """Minimises the training objective (either log-marginal likelihood or evidence lower bound)
    of a Gaussian process model. The optimisation is done in-place. Uses the default Scipy 
    optimiser (BFGS)

    Args:
        model (GPflow.GPModel): GPflow model
        **kwargs: optimisation arguments passed to gpflow.optimizers.Scipy
    """
    logger.debug("Optimising model")
    gpf.optimizers.Scipy(**kwargs).minimize(model.training_loss, variables=model.trainable_variables)

def evaluate_model(mod, X_test, y_test):
    """Compute the log-posterior density (on training data) and log-predictive density on held-
    out data.

    Args:
        mod (GPflow.GPModel): GPflow model
        X_test (np.ndarray): OTU abundances for the log-predictive density calculation
        y_test (np.ndarray): phenotypes for the log-predictive density calculation

    Returns:
        A dict containing the log-marginal likelihood and log-predictive density
    """
    lml = mod.log_posterior_density().numpy()
    lpd = np.nansum(mod.predict_log_density((X_test, y_test)).numpy())
    return {'lml' : lml, 'lpd' : lpd}

def fit_generic_gpmod(X, y, kernel_maker, noise_variance, opt):
    """Fit a GP model. The type of model (is inferred from the type of the labels)

    Args:
        X (np.ndarray): OTU abundances
        y (np.ndarray): phenotypes
        kernel_maker (callable): should return a GPflow kernel
        noise_variance (float): likelihood variance (GP regression only)
        opt (bool): whether to optimise the model hyperparameters

    Returns:
        A 3-tuple containing
            1. the fitted model
            2. None
            3. None
    """
    target_type = type_of_target(y)
    # logger.debug(f"y has type {target_type}")
    
    if target_type=='continuous':
    
        m = gpf.models.GPR(
            data=(X, y),
            kernel=kernel_maker(),
            noise_variance=noise_variance
        )
        
    elif target_type=='binary':
    
        m = gpf.models.VGP(
            data=(X, y),
            likelihood=gpf.likelihoods.Bernoulli(),
            kernel=kernel_maker()
        )
        
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    
    # ML-II optimisation
    if opt:
        optimise_gpmod(m)
        
    return m, None, None

def fit_string_gpmod(
    X, y,
    variance, noise_variance, opt,
    kernel_df, # included_otus,
    forceQPD=False):

    """Fit a GP model using the string kernel hyperparameters that lead to the 
    lowest training objective. The type of model (is inferred from the type of the labels)

    Args:
        X (np.ndarray): OTU abundances
        y (np.ndarray): phenotypes
        kernel_maker (callable): should return a GPflow kernel
        variance (float): kernel signal variance
        noise_variance (float): likelihood variance (GP regression only)
        opt (bool): whether to optimise the model hyperparameters
        kernel_df (pd.DataFrame): a dataframe with a column named Q and at least one other column of hyperparameters
        forceQPD (bool): whether to force Q matrices to be postiive semi-definite

    Returns:
        A 3-tuple containing
            1. the fitted model for the optimal string kernel hyperparameters
            2. the training objectives for each string kernel hyperparameter
            3. the optimal string kernel hyperparameters
    """
    
    logger.debug(f"String kernel model selection with {kernel_df.shape[0]} candidate kernels")
    
    log_marg_liks = np.full(kernel_df.shape[0], np.nan)
    
    # check each string hyperparameter value
    for i, (row_idx, row) in enumerate(kernel_df.iterrows()):
        
        try:
    
            mm, _, _ = fit_generic_gpmod(
                X, y,
                kernel_maker=lambda: StringKernel(
                    row.Q.to_numpy(), #row.Q.loc[included_otus,included_otus].to_numpy(),
                    variance=variance,
                    forceQPD=forceQPD),
                noise_variance=noise_variance,
                opt=opt
            )

            log_marg_liks[i] = mm.log_posterior_density().numpy()
            
        except Exception as e:
            logger.warning(e)
            log_marg_liks[i] = np.nan # failed fit
        
    best_model_idx = np.nanargmax(log_marg_liks)
    logger.debug(f"{best_model_idx} is the best model")
    best_row = kernel_df.iloc[best_model_idx,:]
    
    # re-fit model with best hyperparameters (in order to avoid saving all the models)
    m, _, _ = fit_generic_gpmod(
        X, y,
        kernel_maker=lambda: StringKernel(
            variance=variance,
            Q=best_row.Q.to_numpy(), # best_row.Q.loc[included_otus,included_otus].to_numpy(),
            forceQPD=forceQPD
        ),
        noise_variance=noise_variance,
        opt=opt
    )
    
    # dataframe of log-marginal likelihoods
    lml_df = pd.concat(
        [kernel_df.reset_index(drop=True), pd.Series(log_marg_liks, name="lml").reset_index(drop=True)],
        axis=1
    )
        
    return m, lml_df.drop(columns="Q"), best_row.drop(columns="Q")