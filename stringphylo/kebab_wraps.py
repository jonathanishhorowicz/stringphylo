import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

# check and see if we can load the R pacakge kebabs, which is
# required for string kernel computations
try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    from rpy2.robjects.vectors import StrVector
    found_rpy2 = True
except ImportError:
    found_rpy2 = False
    raise ImportError('Could not load rpy2 - string kernel computations will not be available')

# try and load kebabs
if found_rpy2:
    try:
        kebabs = importr('kebabs')
        _successful_kebab_init = True
    except PackageNotInstalledError:
        _successful_kebab_init = False
        raise PackageNotInstalledError('Could not load the R package `kebabs` - string kernel computations will not be available')

# try and load Biostrings
if _successful_kebab_init:
    try:
        biostrings = importr('Biostrings')
        _string_kernel_comp_avail = True
        logger.debug("Kebabs and Biostrings loaded successfully - String kernel computation available!")
    except PackageNotInstalledError:
        _string_kernel_comp_avail = False
        raise PackageNotInstalledError('Could not load the R package `Biostrings` - string kernel computations will not be available')

def requires_kebabs(func):
    """Decorator that checks all R packages have been loaded successfully before trying
    any string kernel computations
    """
    if not _string_kernel_comp_avail:
        raise ImportError("String kernel computation is not available")

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

@requires_kebabs
def spectrum_kernel(k=3, normalized=True, exact=True, **kwargs):
    """Initalise a spectrum kernel with a given k-mer length.

    The key arguments are exposed. This function returns an R object. See
    https://rdrr.io/bioc/kebabs/man/gappyPairKernel.html for the
    relevant documentation and description of the remaining arguments

    Args:
        k (int): k-mer length (maximum allowed value is 30)
        normalized (bool): whether to normalise the S matrix to ensure the diagonal 
                           is equal to 1
        exact (bool): whether to assume the representative sequences are exact

    Returns:
        A kernel object (of type py2.robjects.functions.SignatureTranslatedFunction). To be 
        passed to compute_Smat to compute OTU-wise similarities.

    """
    return kebabs.spectrumKernel(k=k, normalized=normalized, exact=exact, **kwargs)

@requires_kebabs
def gappy_pair_kernel(k=3, m=1, normalized=True, exact=True, **kwargs):
    """Initalise a gappy pair kernel with a given k-mer length and number of gaps.

    The key arguments are exposed. This function returns an R object. See
    https://rdrr.io/bioc/kebabs/man/gappyPairKernel.html for the
    relevant documentation and description of the remaining arguments

    Args:
        k (int): k-mer length (maximum allowed value is 15)
        g (int): number of gaps 
        normalized (bool): whether to normalise the S matrix to ensure the diagonal 
                           is equal to 1
        exact (bool): whether to assume the representative sequences are exact

    Returns:
        A kernel object (of type py2.robjects.functions.SignatureTranslatedFunction). To be 
        passed to compute_Smat to compute OTU-wise similarities.

    """
    return kebabs.gappyPairKernel(k=k, m=m, normalized=normalized, exact=exact, **kwargs)

@requires_kebabs
def mismatch_kernel(k=3, m=1, normalized=True, exact=True, **kwargs):
    """Initalise a mismatch pair kernel with a given k-mer length and number of mismatches.

    The key arguments are exposed. This function returns an R object. See
    https://rdrr.io/bioc/kebabs/man/gappyPairKernel.html for the
    relevant documentation and description of the remaining arguments

    Args:
        k (int): k-mer length (maximum allowed value is 15)
        m (int): number of mismatches. Increasing this above 3 will increase the computation
                 time significantly.
        normalized (bool): whether to normalise the S matrix to ensure the diagonal 
                           is equal to 1
        exact (bool): whether to assume the representative sequences are exact

    Returns:
        A kernel object (of type py2.robjects.functions.SignatureTranslatedFunction). To be 
        passed to compute_Smat to compute OTU-wise similarities.

    """
    return kebabs.mismatchKernel(k=k, m=m, normalized=normalized, exact=exact, **kwargs)

@requires_kebabs
def compute_Smat(kern_obj, repr_strings):
    """Compute the string kernel S (or Q) matrix on the representative strings
    
    Args:
        kern_obj (rpy2.robjects.functions.SignatureTranslatedFunction): an R object representing
            a kernel from the kebabs package (created using spectrum_kernel, gappy_pair_kernel
            or mismatch_kernel)
        repr_strings (list, np.ndarray or pd.Series): the representative sequences

    Returns:
        A DataFrame containing the S (or Q) matrix (size p x p for p OTUs). The OTU names are taken 
        from the index of repr_strings if it is a pd.Series or given as OTU0, OTU1, ... otherwise
    """
    if isinstance(repr_strings, pd.Series):
        otu_names = repr_strings.index.to_numpy()
    else:
        otu_names = [f"OTU{x}" for x in range(len(repr_strings))]
    
    Qmat = np.array(
        kern_obj(
            biostrings.DNAStringSet(
                StrVector(repr_strings)
            )
        )
    )
    
    return pd.DataFrame(Qmat, columns=otu_names, index=otu_names)
