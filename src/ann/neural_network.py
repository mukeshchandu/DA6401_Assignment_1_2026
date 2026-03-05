"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import wandb
from ann.neural_layer import Neurallayer
from ann.optimizers import Optimizer
from ann.objective_functions import loss as Lossfunc
import numpy as np
from sklearn.metrics import f1_score
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        self.cli=cli_args
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        in_size=784
        out_size=10
        layers_tot=[in_size]+list(cli_args.hidden_size)+[out_size]
        self.layers=[]
        for i in range(len(layers_tot)-1):
            present_layer=Neurallayer(layers_tot[i],layers_tot[i+1],cli_args.activation if i<len(layers_tot)-2 else "linear",cli_args.weight_init)
            self.layers.append(present_layer)
        self.optimizer=Optimizer(cli_args.optimizer,self.cli.learning_rate,self.cli.weight_decay)
        self.loss_fn=Lossfunc(cli_args.loss)
        self.grad_W=None
        self.grad_b=None
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        out=X
        for layer in self.layers:
            out=layer.forward(out)
        return out
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        da=self.loss_fn.backward(y_true,y_pred)
        gradw=[]
        gradb=[]
        for layer in self.layers[-1::-1]:
            da=layer.backward(da)
            gradw.append(layer.grad_W)
            gradb.append(layer.grad_b)
        self.grad_W,self.grad_b=np.empty(len(gradw),dtype=object),np.empty(len(gradb),dtype=object)
        for i,(w,b) in enumerate(zip(gradw,gradb)):
            self.grad_W[i]=w
            self.grad_b[i]=b
        return self.grad_W,self.grad_b
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d
    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
    
    def train(self, X_train, y_train, X_val,y_val,epochs, batch_size=32):
        """
        Train the network for specified epochs.
        """
        n_neurons_to_log = min(5, self.layers[0].W.shape[1])
        iteration=0
        nosamps=X_train.shape[0]
        batchloss=0
        best_val_f1=0
        for i in range(epochs):
            epochloss=0
            ind=np.random.permutation(nosamps)
            x_new=X_train[ind]
            y_new=y_train[ind]
            for j in range(0,nosamps,batch_size):
                y_pred=self.forward(x_new[j:j+batch_size])
                batchloss=self.loss_fn.forward(y_new[j:j+batch_size],y_pred)
                epochloss+=batchloss*x_new[j:j+batch_size].shape[0]
                self.backward(y_new[j:j+batch_size],y_pred)
                self.optimizer.step(self.layers)
                if iteration < 50:
                    for neuron_idx in range(n_neurons_to_log):
                        g = float(np.linalg.norm(self.layers[0].grad_W[:, neuron_idx]))
                        wandb.log({f"neuron{neuron_idx}_grad": g, "iteration": iteration})
                iteration += 1
                if j==0:
                    grad_norm=float(np.linalg.norm(self.layers[0].grad_W))
                    wandb.log({"grad_norm_layer0":grad_norm,"epoch":i+1})
            train_preds = self.forward(X_train)
            train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)) * 100
            avg_train_loss = epochloss / nosamps
            train_f1=f1_score(np.argmax(y_train,axis=1),np.argmax(train_preds,axis=1),average="macro")
            
            # Validation metrics
            val_preds = self.forward(X_val)
            val_loss = self.loss_fn.forward(y_val, val_preds)
            val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)) * 100
            val_f1=f1_score(np.argmax(y_val,axis=1),np.argmax(val_preds,axis=1),average="macro")
            wandb.log({
                "epoch": i + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "train_f1":train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1":val_f1,
            })
            print(f"Epoch {i+1}/{epochs} | "  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} | " f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            # --- SAVE BEST MODEL ---
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_weights(self.cli.model_save_path)
                print(f"New model - val F1: {val_f1:.2f}")
            sample=X_train[:500]
            out=sample
            for i,layer in enumerate(self.layers[:-1]):
                out=layer.forward(out)
                dead_frac=float(np.mean(out<=0))
                wandb.log({f"dead_frac_layer{i}":dead_frac,"epoch":i+1})
        print(f"\nTraining complete. Best val F1: {best_val_f1:.2f}")
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        preds = self.forward(X)
        loss = self.loss_fn.forward(y, preds)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, acc
    def save_weights(self, filepath):
        np.save(filepath,self.get_weights())
    def load_weights(self, filepath):
        """Load weights from a .npy file saved by save_weights()."""
        weights_dict = np.load(filepath, allow_pickle=True).item()
        self.set_weights(weights_dict)
