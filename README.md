# DA6401 Assignment 1

**Name:** Babbadi Mukesh Chandu
**Roll Number:** EE23B093

## WandB Report

[View the full experiment report here](https://wandb.ai/ee23b093-indian-institute-of-technology-madras/da6401-assignment-1/reports/Deep-Learning-Assignment-1--VmlldzoxNjEzNDQyOA?accessToken=a4ewg1yco2gyl5njjp8dz5ac8dxegcag1w12la72mmibxtg8lcv7bf1fl2bnxewu)

- - -

## Git Hub Repo
[link to repo](https://github.com/mukeshchandu/DA6401_Assignment_1_2026)
## Overview

This assignment implements a fully connected neural network (MLP) from scratch using only NumPy — no PyTorch, no TensorFlow. The network is trained on MNIST and Fashion-MNIST for image classification.

Everything is hand-coded: forward pass, backpropagation, weight updates, loss functions, and all optimizers.

- - -

## Project Structure

```
da6401_assignment_1/
└── src/
    ├── train.py               # main training script
    ├── inference.py           # load model and evaluate on test set
    ├── explore.py             # logs sample images to wandb
    ├── sweep.py               # bayesian hyperparameter sweep via wandb
    ├── error_analysis.py      # confusion matrix + failure visualisation
    ├── best_model.npy         # saved weights of the best trained model
    ├── best_config.json       # config used to train the best model
    ├── ann/
    │   ├── neural_network.py  # NeuralNetwork class — train/forward/backward
    │   ├── neural_layer.py    # single layer — forward + gradient computation
    │   ├── activations.py     # relu, sigmoid, tanh, softmax + their derivatives
    │   ├── optimizers.py      # sgd, momentum, nag, rmsprop
    │   └── objective_functions.py  # cross entropy and mse loss + backward
    └── utils/
        └── data_loader.py     # loads and preprocesses mnist/fashion_mnist
```

- - -

## How to Install

``` bash
pip install numpy scikit-learn keras wandb matplotlib seaborn
```

- - -

## How to Train

``` bash
cd src
python train.py -d mnist -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -e 20 -b 32 -w_i xavier -w_p da6401-assignment-1
```

**All arguments:**

| Flag | What it does | Example |
| ---- | ------------ | ------- |
| `-d` | dataset | `mnist` or `fashion_mnist` |
| `-o` | optimizer | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-lr` | learning rate | `0.001` |
| `-nhl` | number of hidden layers | `3` |
| `-sz` | size of each hidden layer | `128 128 128` |
| `-a` | activation function | `relu`, `tanh`, `sigmoid` |
| `-e` | epochs | `20` |
| `-b` | batch size | `32` |
| `-w_i` | weight initialisation | `xavier`, `random`, `zeros` |
| `-wd` | weight decay (L2) | `0.0005` |
| `-l` | loss function | `cross_entropy`, `mean_squared_error` |
| `-w_p` | wandb project name | `da6401-assignment-1` |
| `--model_save_path` | where to save weights | `best_model.npy` |

- - -

## How to Run Inference

``` bash
cd src
python inference.py --model_save_path best_model.npy -d mnist
```

This automatically reads `best_config.json` to load the correct architecture and activation — no need to pass them manually.

- - -

## What Each Script Does

### `train.py`

The main entry point. Parses CLI args, loads data, builds the network, runs training epoch by epoch, and saves the best model (by val F1) to `best_model.npy`. All metrics are logged to wandb automatically — train/val loss, accuracy, F1, gradient norms, dead neuron fractions.

### `inference.py`

Loads a saved model and evaluates it on the test set. Prints loss, accuracy, precision, recall and F1. Reads `best_config.json` automatically so the architecture matches the saved weights exactly.

### `explore.py`

Logs one sample image per class from both MNIST and Fashion-MNIST to wandb as tables. Run this once to populate Section 1 of the report.

``` bash
python explore.py
```

### `sweep.py`

Runs a Bayesian hyperparameter search over 100 configurations using wandb sweeps. Searches over learning rate, optimizer, architecture depth, hidden size, activation, batch size, weight decay, and initialisation.

``` bash
python sweep.py
```

### `error_analysis.py`

Loads the best saved model, runs it on the test set, and generates two plots:

* `confusion_matrix.png` — shows which classes the model confuses most
* `creative_failures.png` — shows the 32 predictions where the model was most confident but still wrong

``` bash
python error_analysis.py
```

- - -

## Implementation Details

### Forward Pass

Each layer computes `z = X @ W + b` then applies the activation. The output layer uses a linear activation — softmax is only applied inside the loss function, not stored in the forward pass. This keeps the forward pass clean and matches standard practice.

### Backward Pass

Standard backpropagation. The loss function computes the initial delta, then each layer computes its weight gradient and passes the delta backwards. Gradients are returned as object arrays where index 0 is the output layer.

### Weight Initialisation

Xavier initialisation samples weights from `U[-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out))]`. This keeps the variance of activations roughly constant across layers, which helps avoid vanishing gradients at the start of training.

### Optimizers

All four optimizers are implemented from scratch:

* **SGD** — plain gradient descent
* **Momentum** — accumulates velocity across steps, overshoots less
* **NAG** — looks ahead before computing gradient, more stable than momentum
* **RMSProp** — adapts learning rate per parameter using running mean of squared gradients

### Loss Functions

* **Cross-Entropy** — combined with softmax, gradient simplifies to `(softmax(z) - y) / N`
* **MSE** — full Jacobian through softmax, slower to converge for classification

- - -

<br>
<br>
