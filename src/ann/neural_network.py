"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from sklearn.metrics import f1_score
from ann.neural_layer import Neurallayer
from ann.optimizers import Optimizer
from ann.objective_functions import loss as Lossfunc

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _log(data):
    """Safe wandb logging - only logs if wandb run is active."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(data)


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments (or Namespace) for configuring the network.
                      Supports attribute names: hidden_size, activation, weight_init,
                      optimizer, learning_rate, weight_decay, loss.
        """
        self.cli = cli_args

        in_size = 784
        out_size = 10

        # Support both 'hidden_size' (list/tuple) and fallbacks
        hidden_size = getattr(cli_args, 'hidden_size', None)
        if hidden_size is None:
            # fallback: build uniform hidden layers from num_layers + hidden_size as int
            num_layers = getattr(cli_args, 'num_layers', 2)
            sz = getattr(cli_args, 'sz', 64)
            hidden_size = [sz] * num_layers
        hidden_size = list(hidden_size)

        activation   = getattr(cli_args, 'activation',    'relu')
        weight_init  = getattr(cli_args, 'weight_init',   'xavier')
        optimizer    = getattr(cli_args, 'optimizer',     'sgd')
        learning_rate= getattr(cli_args, 'learning_rate', 0.01)
        weight_decay = getattr(cli_args, 'weight_decay',  0.0)
        loss_type    = getattr(cli_args, 'loss',          'cross_entropy')

        layers_tot = [in_size] + hidden_size + [out_size]
        self.layers = []
        for i in range(len(layers_tot) - 1):
            act = activation if i < len(layers_tot) - 2 else "linear"
            layer = Neurallayer(layers_tot[i], layers_tot[i + 1], act, weight_init)
            self.layers.append(layer)

        self.optimizer = Optimizer(optimizer, learning_rate, weight_decay)
        self.loss_fn = Lossfunc(loss_type)
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input data of shape (batch, 784)

        Returns:
            Output logits of shape (batch, 10) - no softmax applied.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.

        Args:
            y_true: True one-hot labels
            y_pred: Predicted logits

        Returns:
            grad_W, grad_b - object arrays in FORWARD order:
            grad_W[0] = input layer gradient, grad_W[-1] = output layer gradient.
        """
        da = self.loss_fn.backward(y_true, y_pred)
        gradw = []
        gradb = []

        # Backprop in reverse order (output to input)
        for layer in self.layers[-1::-1]:
            da = layer.backward(da)
            gradw.append(layer.grad_W)
            gradb.append(layer.grad_b)

        # Reverse to return in forward order (input to output)
        gradw = gradw[::-1]
        gradb = gradb[::-1]

        self.grad_W = np.empty(len(gradw), dtype=object)
        self.grad_b = np.empty(len(gradb), dtype=object)
        for i, (w, b) in enumerate(zip(gradw, gradb)):
            self.grad_W[i] = w
            self.grad_b[i] = b

        return self.grad_W, self.grad_b

    def get_weights(self):
        """Return all layer weights as a dictionary."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Load weights from a dictionary into all layers."""
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32):
        """
        Train the network for the specified number of epochs.
        """
        n_neurons_to_log = min(5, self.layers[0].W.shape[1])
        iteration = 0
        nosamps = X_train.shape[0]
        best_val_f1 = 0

        for i in range(epochs):
            epoch_loss = 0
            ind = np.random.permutation(nosamps)
            x_new = X_train[ind]
            y_new = y_train[ind]

            for j in range(0, nosamps, batch_size):
                xb = x_new[j:j + batch_size]
                yb = y_new[j:j + batch_size]

                y_pred = self.forward(xb)
                batch_loss = self.loss_fn.forward(yb, y_pred)
                epoch_loss += batch_loss * xb.shape[0]

                self.backward(yb, y_pred)
                self.optimizer.step(self.layers)

                # Log per-neuron gradient norms for first 50 iterations
                if iteration < 50:
                    for neuron_idx in range(n_neurons_to_log):
                        g = float(np.linalg.norm(self.layers[0].grad_W[:, neuron_idx]))
                        _log({f"neuron{neuron_idx}_grad": g, "iteration": iteration})

                # Log gradient norm for first batch of each epoch
                if j == 0:
                    grad_norm = float(np.linalg.norm(self.layers[0].grad_W))
                    _log({"grad_norm_layer0": grad_norm, "epoch": i + 1})

                iteration += 1

            # --- Epoch-level metrics ---
            avg_train_loss = epoch_loss / nosamps
            train_preds = self.forward(X_train)
            train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)) * 100
            train_f1 = f1_score(np.argmax(y_train, axis=1), np.argmax(train_preds, axis=1), average="macro")

            val_preds = self.forward(X_val)
            val_loss = self.loss_fn.forward(y_val, val_preds)
            val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)) * 100
            val_f1 = f1_score(np.argmax(y_val, axis=1), np.argmax(val_preds, axis=1), average="macro")

            _log({
                "epoch": i + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            })

            print(
                f"Epoch {i+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
            )

            # Save best model based on validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_path = getattr(self.cli, 'model_save_path', 'best_model.npy')
                self.save_weights(save_path)
                print(f"  -> New best model saved (val F1: {val_f1:.4f})")

            # Dead neuron fraction per hidden layer
            sample = X_train[:500]
            out = sample
            for k, layer in enumerate(self.layers[:-1]):
                out = layer.forward(out)
                dead_frac = float(np.mean(out <= 0))
                _log({f"dead_frac_layer{k}": dead_frac, "epoch": i + 1})

        print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.

        Returns:
            (loss, accuracy)
        """
        preds = self.forward(X)
        loss = self.loss_fn.forward(y, preds)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, acc

    def save_weights(self, filepath):
        """Save model weights to a .npy file."""
        np.save(filepath, self.get_weights())

    def load_weights(self, filepath):
        """Load weights from a .npy file saved by save_weights()."""
        weights_dict = np.load(filepath, allow_pickle=True).item()
        self.set_weights(weights_dict)
