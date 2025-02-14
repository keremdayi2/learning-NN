# python stuff
import argparse
import time

# pytorch etc.
import torch
import torch.nn as nn

# our own files
import synthetic.func as syn_fn
import synthetic.dataset as syn_dataset
import synthetic.models as syn_models

# tools
import tools.timers as timers


# printing variables
ITERATION_PRINT_FREQUENCY = 500
TEST_DATASET_SIZE = 16
EVAL_FREQUENCY = 500

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
        eval_datasets : dict, # list of datasets to evaluate the model on 
        eval_fns : dict = {} # list of eval functions to run
        ):
        self.model = model
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.eval_datasets = eval_datasets
        self.eval_fns = eval_fns

        self.device = next(self.model.parameters()).device

    # return dataset_results, fn_results
    def eval(self):
        dataset_results = {}
        
        # run dataset evals
        with torch.no_grad():
            for label, dataset in self.eval_datasets.items():
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

                n_outputs = 0
                for x,y in dataloader:
                    n_outputs = y.shape[1]
                    break

                avg_losses = torch.zeros(n_outputs)
                for i, (x,y) in enumerate(dataloader):
                    if i == TEST_DATASET_SIZE:
                        break

                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)

                    for i in range(n_outputs):
                        avg_losses[i] += self.loss_fn(output[:, i], y[:, i]).item() / TEST_DATASET_SIZE

                # add average loss on this dataset to the results
                dataset_results[label] = avg_losses

        fn_results = {}

        # run function evals
        with torch.no_grad():
            for label, fn in self.eval_fns.items():
                fn_results[label] = fn(self.model)

        return dataset_results, fn_results

    def train(self):
        time_logger = timers.TimeLogger()

        eval_results = []

        losses = []
        multi_losses = []
        for i, (x,y) in enumerate(self.dataloader):
            # eval flag
            eval_flag = i % ITERATION_PRINT_FREQUENCY == 0 or i == 0

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
            if eval_flag:
                print(20*'-' + f"Iteration {i+1}" + 20 * '-')
                a, b = self.eval()

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
