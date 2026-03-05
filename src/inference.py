"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load as load_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import loss as Lossfunc


def parse_arguments():
    """
    Parse command-line arguments for inference.

    - model_save_path: Path to saved model weights (relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_size: List of hidden layer sizes
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    # 1. Data
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Mini-batch size")

    # 2. Loss & Optimizer
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop"], default="rmsprop", help="Optimizer choice")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay for L2 regularization")

    # 3. Network Architecture
    parser.add_argument("-nhl", "--num_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs='+', default=[64, 64], help="List of hidden layer sizes (e.g., -sz 64 64)")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "relu"], default="relu", help="Hidden layer activation")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier", "zeros"], default="xavier", help="Weight initialization method")

    # 4. Logging and Saving
    parser.add_argument("-w_p", "--wandb_project", type=str, default="da6401-assignment-1", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username)")
    parser.add_argument("--model_save_path", type=str, default="best_model.npy", help="Relative path to the saved model weights")
    parser.add_argument("--config", type=str, default=None, help="Path to a config JSON file to override args")

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model weights from disk.
    Returns raw weight dictionary.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test, loss_type="cross_entropy"):
    """
    Evaluate model on test data.
    Returns dictionary with logits, loss, accuracy, f1, precision, recall.
    """
    loss_fn = Lossfunc(loss_type)
    logits = model.forward(X_test)
    loss_val = loss_fn.forward(y_test, logits)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "logits": logits,
        "loss": loss_val,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    """
    Main inference function.
    Returns dictionary with logits, loss, accuracy, f1, precision, recall.
    """
    args = parse_arguments()

    # Override args from config file if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        args.hidden_size = config.get("hidden_size", args.hidden_size)
        args.activation = config.get("activation", args.activation)
        args.loss = config.get("loss", args.loss)
        args.weight_init = config.get("weight_init", args.weight_init)
        args.optimizer = config.get("optimizer", args.optimizer)
        args.learning_rate = config.get("learning_rate", args.learning_rate)
        args.weight_decay = config.get("weight_decay", args.weight_decay)
        print(f"Architecture loaded from config: {args.config}")

    # Load test data
    _, _, _, _, X_test, y_test = load_data(args.dataset)
    print(f"Test samples: {X_test.shape[0]}")

    # Load weights and build model  ← FIX: was passing args into load_model()
    weights = load_model(args.model_save_path)
    model = NeuralNetwork(args)
    model.set_weights(weights)

    # Evaluate
    results = evaluate_model(model, X_test, y_test, loss_type=args.loss)

    print("\n--- Evaluation Results ---")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.2f}%")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")
    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()
