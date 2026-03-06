
"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import argparse
import wandb
import json
from utils.data_loader import load as load_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments.

    args: optional list of strings, if None reads from sys.argv
    """
    parser=argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("-d","--dataset",type=str,choices=["mnist","fashion_mnist"],default="mnist",help="Dataset to use")
    parser.add_argument("-e","--epochs",type=int,default=10,help="Number of training epochs")
    parser.add_argument("-b","--batch_size",type=int,default=32,help="Mini-batch size")
    # Loss & Optimizer
    parser.add_argument("-l","--loss",type=str,choices=["mean_squared_error","cross_entropy"],default="cross_entropy",help="Loss function")
    parser.add_argument("-o","--optimizer",type=str,choices=["sgd","momentum","nag","rmsprop"],default="rmsprop",help="Optimizer choice")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="Initial learning rate")
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0,help="Weight decay for L2 regularization")
    # Network Architecture
    parser.add_argument("-nhl","--num_layers",type=int,default=2,help="Number of hidden layers")
    parser.add_argument("-sz","--hidden_size",type=int,nargs='+',default=[64,64],help="List of hidden layer sizes (e.g., -sz 64 64)")
    parser.add_argument("-a","--activation",type=str,choices=["sigmoid","tanh","relu"],default="relu",help="Hidden layer activation")
    parser.add_argument("-w_i","--weight_init",type=str,choices=["random","xavier","zeros"],default="xavier",help="Weight initialization method")
    # Logging and Saving
    parser.add_argument("-w_p","--wandb_project",type=str,default="da6401-assignment-1",help="Weights & Biases project name")
    parser.add_argument("--wandb_entity",type=str,default=None,help="Weights & Biases entity (username)")
    parser.add_argument("--model_save_path",type=str,default="best_model.npy",help="Relative path to save the best model")
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args=parse_arguments()
    print(f"--- Starting Training Run ---")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {args.num_layers} hidden layers of sizes {args.hidden_size}")
    print(f"Optimizer: {args.optimizer} (LR: {args.learning_rate})")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"{args.dataset}_{args.optimizer}_lr{args.learning_rate}_hl{args.num_layers}_act{args.activation}"
    )
    X_train,y_train,X_val,y_val,X_test,y_test=load_data(args.dataset)
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]} , test samples:{X_test.shape[0]}")
    print("\nInitializing Neural Network...")
    model=NeuralNetwork(args)
    model.train(X_train,y_train,X_val,y_val,args.epochs,args.batch_size)
    test_loss,test_acc=model.evaluate(X_test,y_test)
    wandb.log({"test_loss":test_loss,"test_accuracy":test_acc})
    with open("best_config.json","w") as f:
        json.dump(vars(args),f,indent=4)
    print("best_config.json saved.")
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()
