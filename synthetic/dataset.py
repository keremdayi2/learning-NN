# python stuff
import argparse
import typing 
from typing import Union, Optional
from enum import Enum

import math
import time

# pytorch et. al
import torch
from torch.utils.data import Dataset, DataLoader


# ours

from .func import *

class Distribution(Enum):
    Gaussian = 1
    Boolean = 2

class FunctionType(Enum):
    Monomial = 1 
    Staircase = 2

class GaussianFunctionDataset(Dataset):
    def __init__(self, d : int, fn, device : str = 'cpu'):
        self.fn = fn
        self.d = d
        self.device = device
    
    def __len__(self):
        return int(1e15)

    def __getitem__(self, idx):
        with torch.no_grad():
            x = torch.randn(self.d).to(self.device)
            y = self.fn(x)
            return x, y

class BooleanFunctionDataset(Dataset):
    def __init__(self, d : int, fn, device : str = 'cpu'):
        self.fn = fn
        self.d = d
        self.device = device

    def __len__(self):
        return int(1e15)

    def __getitem__(self, idx):
        x = torch.randn(self.d).to(self.device)

        x = torch.where(x > 0, 1., -1.)

        y = self.fn(x)
        return x, y

class MultipleBooleanFunctionDataset(Dataset):
    def __init__(self, d : int, p: int, fns: list, device : str = 'cpu'):
        self.device = device
        self.fns = fns
        self.d = d
        self.p = p
        assert p == len(fns)

    def __len__(self):
        return int(1e15)

    def __getitem__(self, idx):
        x = torch.randn(self.d).to(self.device)

        x = torch.where(x > 0, 1., -1.)

        y = torch.tensor([fn(x) for fn in self.fns])

        return x, y

'''
    Generate instance of a synthetic problem given distribution, dimension, func_type, and device
        dist: Distributon 
        d: data dimension
        k: problem dimension (e.g. degree of poly, subspace dimension)
        fn_type: function type (see above)
        device: "cpu" or "cuda" or etc.
'''
def generate_synthetic_dataset(dist: Distribution, d : int, k : int, fn_type : FunctionType, device : str):
    # first initialize function
    idxs = []
    # then k is the degree of the function
    if fn_type == FunctionType.Monomial:
        idxs.append([i for i in range(k)])
    elif fn_type == FunctionType.Staircase:
        idxs = [list(range(i+1)) for i in range(k)]
    else:
        raise RuntimeError("You did not specify a valid function type")

    fn = None
    dataset = None

    if dist == Distribution.Boolean:
        fn = ParityFunction(d, idxs, device = device)
        dataset = BooleanFunctionDataset(d, fn, device = device)
    elif dist == Distribution.Gaussian:
        raise NotImplementedError("Gaussian data not implemented yet")
    else:
        raise RuntimeError("You did not specify a valid distribution type")

    return dataset
    

if __name__ == '__main__':
    functypes = ['parity']

    parser = argparse.ArgumentParser("The test code for synthetic data generation")

    parser.add_argument("--function", required = True, help = f"Specify function type in {functypes}")

    parser.add_argument("--device", type = str, help = "Device for pytorch computations")

    parser.add_argument("-b", "--batch_size", type = int, default = 4, help = "Batch size for the dataloader")

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

    func_type = args.function
    if func_type not in functypes:
        raise RuntimeError(f"Invalid function type {func_type}. Valid types are {functypes}")

    print(f"Generating dataset from function class: {func_type}")
    
    if func_type == "parity":
        fn = ParityFunction.example(device)
        d = fn.d
        dataset = BooleanFunctionDataset(d, fn, device = device)

    dataloader = DataLoader(dataset, batch_size = args.batch_size)

    for (x, y) in dataloader:
        print(f"x.shape: {x.shape}")
        print(f"y.shape: {y.shape}")
        break
