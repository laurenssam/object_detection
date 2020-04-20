import argparse
import sys
from pathlib import Path


def parse_train_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help="Batch size for training")
    parser.add_argument('--continue_training', action='store_true', help="start training from given experiment")
    parser.add_argument('--exp_name', type=Path, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--print_freq', type=int, default=50, help="Print every n iterations")
    parser.add_argument('--run_colab', action='store_true', help='run locally or on colab')
    arguments = parser.parse_args(arguments)
    for k, v in vars(arguments).items():
        print("{0}: {1}".format(k, v))
    return vars(arguments)

def parse_adv_train_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help="Batch size for training")
    parser.add_argument('--continue_training', action='store_true', help="start training from given experiment")
    parser.add_argument('--exp_name', type=Path, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for training")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of tarining epochs")
    parser.add_argument('--print_freq', type=int, default=50, help="Print every n iterations")
    parser.add_argument('--run_colab', action='store_true', help='run locally or on colab')
    arguments = parser.parse_args(arguments)
    for k, v in vars(arguments).items():
        print("{0}: {1}".format(k, v))
    return vars(arguments)

def parse_val_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help="Batch size for evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, help="What model to load for evaluation")
    parser.add_argument('--run_colab', action='store_true', help='run locally or on colab')
    parser.add_argument('--set', type=str, default="test", help="Run evaluation on train or test set")
    parser.add_argument('--subset', type=int, default=0, help="Number of images to do evaluation on")
    arguments = parser.parse_args(arguments)
    for k, v in vars(arguments).items():
        print("{0}: {1}".format(k, v))
    return vars(arguments)

if __name__ == "__main__":
    result = parse_train_arguments(sys.argv[1:])
