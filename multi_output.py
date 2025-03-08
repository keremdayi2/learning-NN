# python stuff
import argparse
import time
import uuid
import os
import json

# pytorch etc.
import torch
import torch.nn as nn

# our own files
import train

# our own files
import synthetic.func as syn_fn
import synthetic.dataset as syn_dataset
import synthetic.models as syn_models

# default arguments used in training
DEFAULT_DIMENSION = 512
DEFAULT_BATCH_SZ = 64
DEFAULT_HIDDEN_SZ = 2048
DEFAULT_OPTIMIZER = "SGD"

data_device = 'cpu'

if __name__ == '__main__':
    program_start_time = time.time()

    parser = argparse.ArgumentParser("Training script")

    # specify required arguments
    parser.add_argument("--seed", type = int, required = True, help = "Specify seed")
    parser.add_argument("--num_layers", type = int, required = True, help = "Specify number of layers used in training")
    parser.add_argument("--num_outputs", type = int, required= True, help = "Specify the dimension of the outputs")
    parser.add_argument("--lr", type = float, required=True, help = "Specify the learning rate of the optimizer")
    parser.add_argument("--leap", type= int, required =True, help = "Specify the leap of the target function")

    # specify optional arguments
    parser.add_argument("--optimizer", type = str, default=DEFAULT_OPTIMIZER, help = "Specify the optimizer to be used")
    parser.add_argument("--batch_size", type = int, default=DEFAULT_BATCH_SZ)
    parser.add_argument("--hidden_size", type = int, default=DEFAULT_HIDDEN_SZ)
    parser.add_argument("--dimension", type = int, default=DEFAULT_DIMENSION)
    parser.add_argument("--num_iterations", type = int, default= int(1e4), help = "Specify the number of iterations")
    parser.add_argument("--num_workers", type = int, default = 0, help =
    "specify the number of workers to use in data loading")

    args = parser.parse_args()

    print("Running multi_output.py with configuration")
    for key, val in vars(args).items():
        print(f"{key}: {val}")

    # specify the device to run the computations on
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found. Using GPU for computations.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using GPU for computations.")
    else:
        device = torch.device("cpu")
        print("GPU device not found. Using CPU for computations.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # create the functions to be learned
    fns = []
    if args.num_outputs > 1:
        for i in range(args.num_outputs):
            rng = args.leap * i + 1
            fns.append(syn_fn.ParityFunction(args.dimension, [list(range(rng))], device = data_device))
    else:
        fns.append(syn_fn.ParityFunction(args.dimension, [list(range(args.leap))], device = data_device))

    dataset = syn_dataset.MultipleBooleanFunctionDataset(args.dimension, args.num_outputs, fns, device = data_device)

    # create the model to be used in training    
    model = syn_models.NLayer([args.dimension] + [args.hidden_size for i in range(args.num_layers - 1)] + [args.num_outputs], [nn.ReLU() for i in range(args.num_layers-1)]).to(device)

    print(f'Number of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    # specify loss
    loss_fn = nn.MSELoss()

    # specify optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    else:
        raise RuntimeError(f"Invalid optimizer {args.optimizer}")

    results = train.Trainer(model, dataset, loss_fn, optimizer,
            args.batch_size, args.num_iterations, {"test" : dataset},
            num_workers = args.num_workers).train()

        

    time_consumption = results['time_consumption']

    print(time_consumption)

    # save results to a json

    RUN_ID = str(uuid.uuid4())

    directory = "out/multi_output"

    print(f"Saving results for run {RUN_ID}")

    # create directory to save files in
    os.makedirs(f"{directory}/{RUN_ID}", exist_ok=True)
    
    directory = f"{directory}/{RUN_ID}"

    variables = vars(args)
    variables['run_id'] = RUN_ID

    with open(f"{directory}/params.json", "w") as file:
        json.dump(variables, file)

    # save loss tensors
    torch.save(results['losses'], f'{directory}/losses.pt')
    torch.save(results['multi_losses'], f'{directory}/multi_losses.pt')

    

    program_end_time = time.time()
    print(f"Program took: {program_end_time-program_start_time:0.1f}s to run")
    # print how long program took
     
