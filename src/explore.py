import wandb
import numpy as np
from keras.datasets import mnist, fashion_mnist
wandb.init(project="da6401-assignment-1", name="data-exploration")
(X, y), _ = mnist.load_data()
class_names_mnist = [str(i) for i in range(10)]
table = wandb.Table(columns=["Class ID", "Class Name", "Image"])
for cls in range(10):
    idxs = np.where(y == cls)[0][:5]
    for idx in idxs:
        table.add_data(cls, class_names_mnist[cls], wandb.Image(X[idx]))
wandb.log({"mnist_samples": table})
(Xf,yf),l= fashion_mnist.load_data()
class_names_fashion = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
table2 = wandb.Table(columns=["Class ID", "Class Name", "Image"])
for cls in range(10):
    idxs = np.where(yf == cls)[0][:5]
    for idx in idxs:
        table2.add_data(cls, class_names_fashion[cls], wandb.Image(Xf[idx]))
wandb.log({"fashion_samples": table2})
wandb.finish()
print("Done! Check your W&B project.")
