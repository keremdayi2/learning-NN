# python stuff
import argparse
import time
import sys

# pytorch etc.
import torch
import torch.nn as nn

# our own files
import synthetic.func as syn_fn
import synthetic.dataset as syn_dataset
import synthetic.models as syn_models

# tools
from tools import TimeLogger


# printing variables
ITERATION_PRINT_FREQUENCY = 2000
TEST_DATASET_SIZE = 64
EVAL_FREQUENCY = 5000

# problem variables
dimension = 512
rank = 3
fn_type = syn_dataset.FunctionType.Monomial

# model variables
hidden_size = 2048

# train variables
step_size = 1e-4
num_iterations = 5000
batch_size = 64

# model params 
num_layers = 2


optimizer = "Adam"

class Trainer:
    def __init__(self, 
        model : nn.Module, 
        dataset, 
        loss_fn,
        optimizer,
        batch_size : int, 
        num_iterations : int,
        eval_dataloaders : dict, # list of datasets to evaluate the model on 
        num_workers : int = 1,
        eval_fns : dict = {}, # list of eval functions to run
        eval_freq : int | None = None,
        print_freq :int | None = None
                 ):
        self.model = model
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, num_workers = num_workers)
        self.num_workers = num_workers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.eval_dataloaders = eval_dataloaders
        self.eval_fns = eval_fns
        self.eval_freq = EVAL_FREQUENCY if eval_freq is None else eval_freq
        self.print_freq = ITERATION_PRINT_FREQUENCY if print_freq is None else print_freq

        self.device = next(self.model.parameters()).device

    # return dataset_results, fn_results
    def eval(self):
        dataset_results = {}
        
        # run dataset evals
        with torch.no_grad():
            for label, dataloader in self.eval_dataloaders.items():

                n_outputs = 0
                for x,y in dataloader:
                    n_outputs = y.shape[1]
                    break

                avg_losses = torch.zeros(n_outputs)
                for i, (x,y) in enumerate(dataloader):
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)

                    for i in range(n_outputs):
                        avg_losses[i] += self.loss_fn(output[:, i], y[:, i]).item()
                    
                    break
                # add average loss on this dataset to the results
                dataset_results[label] = avg_losses

        fn_results = {}

        # run function evals
        with torch.no_grad():
            for label, fn in self.eval_fns.items():
                fn_results[label] = fn(self.model)

        return dataset_results, fn_results

    def train(self):
        time_logger = TimeLogger()

        eval_results = []

        losses = []
        multi_losses = []
        for i, (x,y) in enumerate(self.dataloader):
            # eval flag
            print_flag = i % self.print_freq == 0 or i == 0
            
            eval_flag = i % self.eval_freq == 0 or i == 0

            if i == self.num_iterations:
                break

            time_logger.log("data_gen")
            x, y = x.to(self.device), y.to(self.device)
            time_logger.log("data_2gpu")

            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.loss_fn(output, y)

            n_outputs = y.shape[1]

            losses.append(loss.item())

            # add multi losses for learning functions to R^p for p > 1
            l = []
            for j in range(n_outputs):
                l.append(self.loss_fn(output[:, j], y[:, j]))

            multi_losses.append(l)


            time_logger.log("output")

            loss.backward()

            time_logger.log("grad")

            # evaluate model given the datasets and functions and print results
            # if print_flag and not eval_flag:
            #     print(20 * '-' + f"Iteration {i+1}" + 20 * '-')
            #     print(f'Last loss {losses[-1]}')

            if eval_flag:
                a, b = self.eval()
                
                # how many lines to erase
                erase_count = 2 + len(a) + len(b)
                
                if i > 0: # means we have printed before, so erase those
                    sys.stdout.write("\033[F"*erase_count)
                print(20 * '-' + f"Iteration {i+1}/{self.num_iterations}" + 20 * '-')
                time_elapsed = time_logger.checkpoints[-1][1] - time_logger.checkpoints[0][1]
                unit_time = time_elapsed/(i+1)
                mins = int(time_elapsed) // 60
                secs = int(time_elapsed) % 60
                total_estimate = int(unit_time * self.num_iterations)
                mins_total = total_estimate//60
                secs_total = total_estimate % 60
                print(f"Time elapsed {mins}min {secs}s. Total estimate {mins_total}min {secs_total}s")
                for dataset, loss in a.items():
                    print(f"{dataset} loss: {loss}")

                for fn, val in b.items():
                    print(f"{fn} value {val}")

                eval_results.append((i, a, b))

                time_logger.log("eval") 
            
            self.optimizer.step()

            time_logger.log("optim")

        losses = torch.tensor(losses)
        multi_losses = torch.tensor(multi_losses)

        results = {
            'losses' : losses,
            'multi_losses' : multi_losses,
            'evals' : eval_results,
            'time_consumption' : time_logger.get_results()
        }

        return results

if __name__ == '__main__':
    program_start_time = time.time()

    parser = argparse.ArgumentParser("Training script")

    parser.add_argument("--device", type = str, required = True, help = "Specify device used for torch training")
    parser.add_argument("--seed", type = int, required = True, help = "Specify seed")
    parser.add_argument("--num_layers", type = int, required = True, help = "Specify number of layers used in training")

    args = parser.parse_args()

    device = args.device
    seed = args.seed
    num_layers = args.num_layers

    torch.manual_seed(seed)

    # set up dataset
    dataset = syn_dataset.generate_synthetic_dataset(syn_dataset.Distribution.Boolean, dimension, rank, fn_type, 'cpu')
    
    # fns = [syn_fn.ParityFunction(dimension, [[1]]), syn_fn.ParityFunction(dimension, [[1, 2, 3]])]
    # dataset = syn_dataset.MultipleBooleanFunctionDataset(d=dimension, p=2, fns = fns)
    
    # model used for training
    # model = syn_models.TwoLayer(dimension, hidden_size, nn.ReLU()).to(device)
    
    model = syn_models.NLayer([dimension] + [hidden_size for i in range(num_layers - 1)] + [1], [nn.ReLU() for i in range(num_layers-1)]).to(device)

    print(f'Number of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    # specify loss
    loss_fn = nn.MSELoss()

    # specify optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = step_size)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = step_size)
    else:
        raise RuntimeError("Invalid optimizer")

    results = Trainer(model, dataset, loss_fn, optimizer, batch_size, num_iterations, {"test" : dataset}).train()

    evals = results['evals']
    time_consumption = results['time_consumption']

    print(time_consumption)

    RUN_NAME = f"{num_layers}_layer_r2"

    torch.save(results['losses'], f'out/losses_{RUN_NAME}.pt')
    torch.save(results['multi_losses'], f'out/multi_losses_{RUN_NAME}.pt')


    program_end_time = time.time()
    print(f"Program took: {program_end_time-program_start_time:0.1f}s to run")
