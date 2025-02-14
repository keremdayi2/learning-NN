# learning-NN
Repository where I rigorously investigate learning in neural networks


# Datasets

## Real World Datasets

## Synthetic Datasets

To rigorously probe the training dynamics of NNs, we consider some toy settings $x\sim \mathcal{N}(0, I_d)$ and $x \sim \mathrm{Unif}\{\pm 1\}^d$. For both of these cases, we provide tools for generating fresh samples from $(x, y)$ where $y = f(x)$ for a user specified function. In particular,
* `synthetic/func.py` contains different function classes (e.g. parity, hermite) implemented in `pytorch`.
* `synthetic/dataset.py` contains the `Dataset` functionality to be compatible with the standard `pytorch` training loop.
