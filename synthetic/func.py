# python stuff
import argparse
import dataclasses
import typing
from typing import Union, Optional

import math

# numerics imports
import torch

"""
        Coefficients to specify a hermite polynomial should be of the following
        form for R^d
        (i1, i2, dots, id) : float
"""

class Hermite:
    def __init__(self, coefs : dict[tuple[int, ], float]):
        self.funcs = []
        raise NotImplementedError

    def monomial(self, x):
        raise NotImplementedError

def forward(self, x):
    raise NotImplementedError

def __call__(self, x):
    self.forward(x)


class ParityFunction:
    """
    ParityFunction
    a parity function
    """
    def __init__(self, 
            d : int, 
            idxs : list[list[int]], 
            device : str = 'cpu',
                batch_size : int = None):
        self.idxs = idxs
        self.k = len(idxs)
        self.device = device
        self.d = d
    
    def forward(self, x : torch.tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        y = torch.zeros(x.shape[0], dtype = x.dtype).to(self.device)

        for idx in self.idxs:
            # (B, len(idx))
            y += x[:, idx].prod(dim=1)

        return y

    def __call__(self, x):
        return self.forward(x)


    def example(device : str):
        k = 8
        idxs = [list(range(i+1)) for i in range(k)]
        return ParityFunction(d=16, idxs=idxs, device = device)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "try parity functions")

    parser.add_argument("-k", "--num_parities", type = int, required = True, help = "Number of parities that are summed in your fn")
    parser.add_argument("--device", type = str, help = "Device for pytorch computations")
    parser.add_argument("-b", "--batch_size", type = int, help = "The batch size of your inputs")
    parser.add_argument("-d", "--dimension", type = int, help = "The dimension of the data")

    args = parser.parse_args()

    device = args.device

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA device found. Using GPU for computations.")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS device found. Using GPU for computations.")
        else:
            device = torch.device("cpu")
            print("GPU device not found. Using CPU for computations.")

    B = args.batch_size
    d = args.dimension
    k = args.num_parities
    
    initialize_batch = not B is None

    # parity
    # x_1 + x_1 * x_2 + ... + x_1 * x_2 * ... * x_n
    idxs = [list(range(i+1)) for i in range(k)]

    parity_fn : ParityFunction

    if initialize_batch:
        parity_fn = ParityFunction(d, idxs, device=device, batch_size=B)
    else:
        parity_fn = ParityFunction(d, idxs, device=device)

    x = torch.where(torch.randn(B, d) > 0, torch.ones(B,d), -torch.ones(B,d)).to(device)

    print(x)
    print(parity_fn(x))
