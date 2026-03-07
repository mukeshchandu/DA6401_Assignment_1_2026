import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json, sys, argparse
sys.path.insert(0, ".")
from utils.data_loader import load as load_data
from ann.neural_network import NeuralNetwork

# load best config
with open("best_config.json") as f:
    config = json.load(f)
args = argparse.Namespace(**config)

# load data and model
_, _, _, _, X_test, y_test = load_data(args.dataset)
model = NeuralNetwork(args)
model.load_weights(args.model_save_path)

wandb.init(project="da6401-assignment-1", name="error-analysis")

# predictions
logits = model.forward(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(logits, axis=1)
print(f"Test Accuracy: {np.mean(y_pred == y_true)*100:.2f}%")

# softmax for confidence
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)

# confusion matrix
class_names = [str(i) for i in range(10)]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix — Test Accuracy: {np.mean(y_pred==y_true)*100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

# 32 most confident wrong predictions
wrong = y_pred != y_true
wrong_imgs  = X_test[wrong].reshape(-1, 28, 28)
wrong_true  = y_true[wrong]
wrong_pred  = y_pred[wrong]
wrong_conf  = probs[wrong].max(axis=1)
top32 = np.argsort(wrong_conf)[::-1][:32]

fig, axes = plt.subplots(4, 8, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    idx = top32[i]
    ax.imshow(wrong_imgs[idx], cmap='gray')
    ax.set_title(f"T:{wrong_true[idx]} P:{wrong_pred[idx]}\n{wrong_conf[idx]:.2f}", fontsize=7, color='red')
    ax.axis('off')
plt.suptitle("32 Most Confident Misclassifications", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("creative_failures.png", dpi=150)
wandb.log({"creative_failures": wandb.Image("creative_failures.png")})

wandb.finish()
print("Done.")
