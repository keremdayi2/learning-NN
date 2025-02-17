# learning-NN
Repository where I rigorously investigate learning in neural networks

# Usage

This library provides datasets to be used for experimentation. Mainly, you use it to generate a torch dataset object, which can be plugged into any training loop with a torch dataloader.

```python
import synthetic.function as syn_function
import synthetic.dataset as syn_dataset

idxs = [list(range(degree))]
fn = syn_fn.ParityFunction(dimension, idxs)

dataset = syn_dataset.BooleanFunctionDataset(dimension, fn)

model = # model of your choice (synthetic.models has some default models)

loss_fn = nn.MSELoss() # your favorite loss function here

optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = lr) # favorite optimizer

# optionally use the existing trainer provided, if not see below
results = train.Trainer(model, dataset, loss_fn, optimizer,
            batch_size, num_iterations, {"test" : dataset},
            num_workers = num_workers).train()
```

## Training loop

Because the library is designed to be compatible with existing pytorch functionality, the training loops is identical to a basic training loop (which is provided by train.Trainer). The only difference between this and a standard training loop is the `break` condition (since the dataloader can generate infinite data, it has no end).

```python
losses = []
for i, (x,y) in enumerate(dataloader):
    if i == num_iterations:
        break
    x,y = x.to(device), y.to(device)
    
    optimizer.zero_grad()

    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

    optimizer.step()

    
```

# Datasets

## Real World Datasets

## Synthetic Datasets

To rigorously probe the training dynamics of NNs, we consider some toy settings $x\sim \mathcal{N}(0, I_d)$ and $x \sim \mathrm{Unif}\{\pm 1\}^d$. For both of these cases, we provide tools for generating fresh samples from $(x, y)$ where $y = f(x)$ for a user specified function. In particular,
* `synthetic/func.py` contains different function classes (e.g. parity, hermite) implemented in `pytorch`.
* `synthetic/dataset.py` contains the `Dataset` functionality to be compatible with the standard `pytorch` training loop.
