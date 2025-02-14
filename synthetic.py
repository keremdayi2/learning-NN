# python stuff
import argparse
import dataclasses
import typing 
from typing import Union, Optional

# pytorch et. al
import torch
from torch.utils.data import Dataset, DataLoader
from func import *

class GaussianFunctionDataset(Dataset):
    def __init__(self, d : int, fn, device : str = 'cpu'):
        self.fn = fn
        self.d = d
        self.device = device
    
    def __len__(self):
        return int(1e15)

    def __getitem__(self, idx):
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
        x = torch.where(torch.randn(self.d) > 0, torch.ones(self.d), -torch.ones(self.d)).to(self.device)
        y = self.fn(x)
        return x, y


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
