# stringphylo - Modelling phylogenetic relationships between microbes using string kernels

# Installation

Clone this directory and use the command `python setup.py install`.

# Requirements

The code requires Python 3.7 or above, as well as:

* gpflow
* numpy
* pandas
* scikit_bio
* scikit_learn
* scipy
* setuptools
* tensorflow

In addition, the string kernels computations themselves rely on the [kebabs R package](https://bioconductor.org/packages/release/bioc/html/kebabs.html), which is accessed through `rpy2`. So unless you have pre-computed the OTU-wise similarity matrices you also require

* rpy2
* kebabs (Biocondutor)
* Biostrings (Bioconductor)

The code is test aginst the package versions in [requirements](requirements.txt).

# Functionality

Please see the [examples](examples) for tutorials on

* [Gaussian process regression](examples/Supervised_learning_with_GPs.ipynb) using string kernels, including hyperparameter selection
* [The kernel two-sample test](examples/MMD_two-sample_testing.ipynb) using string kernels
* [Computing string kernels](examples/Computing_String_Kernels.ipynb) in `StringPhylo`.

# Relevant Citations

J. Ish-Horowicz and S.L. Filippi (2022). Modelling phylogeny in 16S rRNA gene sequencing datasets using string kernels. [arXiv:2210.07696](https://arxiv.org/abs/2210.07696)


