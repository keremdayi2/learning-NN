import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import dataloader

import synthetic.models as syn_models
import synthetic.dataset as syn_data
import synthetic.func as syn_func

import json
import functools
import os
import uuid
import train

def extract_evals(results):
    evals = results['evals']
    eval_dict = {}
    
    eval_dict['k'] = []
    
    for data_keys in evals[0][1].keys():
        eval_dict[data_keys] = []
    
    for fn_keys in evals[0][2].keys():
        eval_dict[fn_keys] = []
    
    for (k, datasets, eval_fns) in evals:
        eval_dict['k'].append(k)

        for key, val in datasets.items():
            eval_dict[key].append(val.item())
        
        for key, val in eval_fns.items():
            eval_dict[key].append(val.item())
        
    for key, val in eval_dict.items():
        eval_dict[key] = torch.tensor(val)
    
    return eval_dict

def estimate_inner_product(model, dataloader):
    with torch.no_grad():
        for (x,y) in dataloader:
            device = next(model.parameters()).device
            x,y = x.to(device), y.to(device)
            y_hat = model(x).flatten()
            y = y.flatten()
            return (y * y_hat).mean()

if __name__ == "__main__":
    device = 'cpu'
    d = 128
    k = 128
    P = 3
    lr = 0.25
    T = 2.5e3
    batch_size = 128

    test_batch_size= 128
    num_iterations = int(T * batch_size / lr)

    # do multiple runs
    runs = 1

    parities = [list(range(i+1)) for i in range(P)]

    target_fn = syn_func.ParityFunction(d, parities)
    
    # create datasets for each parity to estimate fourier coefficients
    parity_fns = [syn_func.ParityFunction(d, [p]) for p in parities ]
    print([fn.idxs for fn in parity_fns])

    parity_datasets = [syn_data.BooleanFunctionDataset(d, parity_fn) for parity_fn in parity_fns]
    parity_dataloaders = [torch.utils.data.DataLoader(parity_dataset, test_batch_size) for parity_dataset in parity_datasets] 

    # for fn, dataloader in zip(parity_fns, parity_dataloaders):
    #     print(estimate_inner_product(fn, dataloader))

    fourier_fns = [functools.partial(estimate_inner_product, dataloader=dataloader) for dataloader in parity_dataloaders]

    # for m, fn in zip(parity_fns, fourier_fns):
    #     print(fn(m))

    eval_fns = {}
    for p, fn in zip(parities, fourier_fns):
        eval_fns[f'x{p}'] = fn

    # initialize the dataset from the teacher model
    dataset = syn_data.BooleanFunctionDataset(d, target_fn)

    # initialize dataset
    model = syn_models.MeanField(d, k).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr/2.0)

    results = train.Trainer(model, dataset, 
                        loss_fn, optimizer,
                        batch_size, num_iterations,
                            {"test" : torch.utils.data.DataLoader(dataset, batch_size = test_batch_size)},
                        num_workers = 4,
                        eval_fns = eval_fns,
                        eval_freq=int(batch_size/lr)).train()
    
    dir = 'out/merged_staircase'
    
    # create unique id to save the file
    id = uuid.uuid1() 
    os.mkdir(f'{dir}/{id}')
    dir = f'{dir}/{id}'

    torch.save(results['losses'], f'{dir}/losses.pt')
    torch.save(extract_evals(results), f'{dir}/evals.pt')

    with open(f"{dir}/params.json", "w") as json_file:
        simulation_params = {
            'dimension' : d,
            'hidden_size' : k,
            'batch_size' : batch_size,
            'T' : T,
            'P' : P,
            'lr' : lr,
            'test_batch_size' : test_batch_size
        }
        json.dump(simulation_params, json_file) 
