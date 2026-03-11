"""
Main Neural Network Model class
Handles forward and backward propagation loops
""""""
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
    def __init__(self, cli_args):
        self.cli=cli_args
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
    def _to_onehot(self,y):
        y=np.asarray(y)
        if y.ndim==2 and y.shape[1]>1:
            return y.astype(np.float64)
        n_classes=self.layers[-1].out_dim
        flat=y.astype(int).flatten()
        oh=np.zeros((flat.shape[0],n_classes),dtype=np.float64)
        oh[np.arange(flat.shape[0]),flat]=1.0
        return oh
    def forward(self, X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim==3:
            X=X.reshape(X.shape[0],-1)
        if X.max()>1.0:
            X=X/255.0
        out=X
        for layer in self.layers:
            out=layer.forward(out)
        return out
    def backward(self, y_true, y_pred):
        y_true=self._to_onehot(y_true)
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
        if isinstance(weight_dict,np.ndarray) and weight_dict.shape==():
            weight_dict=weight_dict.item()
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
    def evaluate(self, X, y):
        y=self._to_onehot(y)
        preds = self.forward(X)
        loss = self.loss_fn.forward(y, preds)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) * 100
        return loss, acc
    def save_weights(self, filepath):
        np.save(filepath,self.get_weights())
    def load_weights(self, filepath):
        weights_dict = np.load(filepath, allow_pickle=True).item()
        self.set_weights(weights_dict)
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32):
        y_train=self._to_onehot(y_train)
        y_val=self._to_onehot(y_val)
        n_log = min(5, self.layers[0].W.shape[1])
        iteration, nosamps, best_f1 = 0, X_train.shape[0], 0
        for i in range(epochs):
            epochloss = 0
            idx = np.random.permutation(nosamps)
            xb, yb = X_train[idx], y_train[idx]
            for j in range(0, nosamps, batch_size):
                xbatch, ybatch = xb[j:j+batch_size], yb[j:j+batch_size]
                pred = self.forward(xbatch)
                epochloss += self.loss_fn.forward(ybatch, pred) * xbatch.shape[0]
                self.backward(ybatch, pred)
                self.optimizer.step(self.layers)
                if iteration < 50:
                    for ni in range(n_log):
                        wandb.log({f"neuron{ni}_grad": float(np.linalg.norm(self.layers[0].grad_W[:, ni])), "iteration": iteration})
                if j == 0:
                    wandb.log({"grad_norm_layer0": float(np.linalg.norm(self.layers[0].grad_W)), "epoch": i+1})
                iteration += 1
            tr_pred = self.forward(X_train)
            tr_acc = np.mean(np.argmax(tr_pred,1)==np.argmax(y_train,1))*100
            tr_f1 = f1_score(np.argmax(y_train,1), np.argmax(tr_pred,1), average="macro")
            tr_loss = epochloss / nosamps
            vp = self.forward(X_val)
            vl = self.loss_fn.forward(y_val, vp)
            va = np.mean(np.argmax(vp,1)==np.argmax(y_val,1))*100
            vf = f1_score(np.argmax(y_val,1), np.argmax(vp,1), average="macro")
            wandb.log({"epoch":i+1,"train_loss":tr_loss,"train_accuracy":tr_acc,"train_f1":tr_f1,
                       "val_loss":vl,"val_accuracy":va,"val_f1":vf})
            print(f"Epoch {i+1}/{epochs} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Train F1: {tr_f1:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.2f}% | Val F1: {vf:.4f}")
            if vf > best_f1:
                best_f1 = vf
                self.save_weights(getattr(self.cli, 'model_save_path', 'best_model.npy'))
                print(f"New model - val F1: {vf:.4f}")
            out = X_train[:500]
            for k, layer in enumerate(self.layers[:-1]):
                out = layer.forward(out)
                wandb.log({f"dead_frac_layer{k}": float(np.mean(out<=0)), "epoch": i+1})
        print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
