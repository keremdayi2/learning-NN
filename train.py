# python stuff
import argparse

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

# problem variables
dimension = 512
rank = 2
fn_type = syn_dataset.FunctionType.Monomial

# model variables
hidden_size = 2048

# train variables
step_size = 1e-4
num_iterations = 500
batch_size = 64

optimizer = "Adam"

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training script")

    parser.add_argument("--device", type = str, required = True, help = "Specify device used for torch training")
    parser.add_argument("--seed", type = int, required = True, help = "Specify seed")

    args = parser.parse_args()

    device = args.device
    seed = args.seed

    torch.manual_seed(seed)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("CUDA device found. Using GPU for computations.")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("MPS device found. Using GPU for computations.")
    # else:
    #     device = torch.device("cpu")
    #     print("GPU device not found. Using CPU for computations.")


    # set up dataset
    dataset = syn_dataset.generate_synthetic_dataset(syn_dataset.Distribution.Boolean, dimension, rank, fn_type, 'cpu')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    
    # model used for training
    # model = syn_models.TwoLayer(dimension, hidden_size, nn.ReLU()).to(device)
    num_layers = 10

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

    # training loop
    losses = []
    time_logger = timers.TimeLogger()

    for  i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        time_logger.log("data")

        print_flag = (i+1) % ITERATION_PRINT_FREQUENCY == 0 or i == 0
        if print_flag:
            print(20*'-' + f"Iteration {i+1}" + 20 * '-')

        output = model(x)
        loss = loss_fn(output, y)
        losses.append(loss.item())

        time_logger.log("output")

        loss.backward()

        time_logger.log("grad")

        if print_flag:
            with torch.no_grad():
                avg_loss = 0.
                for i, (x,y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)
                    avg_loss += loss_fn(model(x), y).item() / TEST_DATASET_SIZE

                    if i + 1 == TEST_DATASET_SIZE:
                        break

                print(f"Current average loss: {avg_loss}")

                # compute the norm of trainable parameters
                grad_norm = torch.norm(
                    torch.stack(
                        [torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
                        )
                    )

                print(f"Gradient norm: {grad_norm}")

            time_logger.log("test")

        optimizer.step()
        optimizer.zero_grad()

        time_logger.log("optim")

        if i + 1 >= num_iterations:
            break

    losses = torch.tensor(losses)
    torch.save(losses, f"out/losses-{seed}.pt")

    print(time_logger.get_results())
