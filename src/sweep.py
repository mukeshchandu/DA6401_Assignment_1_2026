import wandb
import argparse
import sys
sys.path.insert(0, ".")
from utils.data_loader import load as load_data
from ann.neural_network import NeuralNetwork
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [0.1, 0.01, 0.001, 0.0001]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "num_layers":    {"values": [1, 2, 3, 4, 5]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["relu", "tanh", "sigmoid"]},
        "weight_decay":  {"values": [0.0, 0.0005, 0.001, 0.005]},
        "batch_size":    {"values": [16, 32, 64, 128]},
        "weight_init":   {"values": ["random", "xavier"]},
    }
}

def train_sweep():
    run = wandb.init()
    c = run.config
    args = argparse.Namespace(
        dataset="mnist",
        epochs=15,
        loss="cross_entropy",
        optimizer=c.optimizer,
        learning_rate=c.learning_rate,
        weight_decay=c.weight_decay,
        num_layers=c.num_layers,
        hidden_size=[c.hidden_size] * c.num_layers,
        activation=c.activation,
        weight_init=c.weight_init,
        batch_size=c.batch_size,
        wandb_project="da6401-assignment-1",
        wandb_entity=None,
        model_save_path="sweep_model.npy",
    )
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)
    model.train(X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
sweep_id = wandb.sweep(sweep_config, project="da6401-assignment-1")
wandb.agent(sweep_id, function=train_sweep, count=100)