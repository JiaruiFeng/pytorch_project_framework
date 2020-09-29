"""Arguments definition file, used to save and modify all the arguments in the project.
Author:
    Chris Chute (chute@stanford.edu)
Edior:
    Jiarui Feng
"""

import argparse
PROJECT_NAME="Enter the name of your project"

def get_data_process_args():
    """Get arguments needed in downloading and processing data"""
    parser = argparse.ArgumentParser('Data download and pre-process ')

    add_common_args(parser)
    # add arguments if needed
    """
        parser.add_argument('--train_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/train-v2.0.json')
    """
    args = parser.parse_args()

    return args


def get_train_args():
    """Get arguments needed in training the model."""
    parser = argparse.ArgumentParser(f'Train a model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)
    add_train_args(parser)
    #Add other arguments if needed

    args = parser.parse_args()

    return args

def get_cv_args():
    """Get arguments for Cross validation"""
    parser = argparse.ArgumentParser(f'Cross validate a model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)
    add_train_args(parser)
    #Add other arguments if needed

    args = parser.parse_args()

    return args

def get_test_args():
    """Get arguments needed in testing the model."""
    parser = argparse.ArgumentParser(f'Test a trained model on {PROJECT_NAME}')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')


    # Require mdoel load_path for testing
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):
    """Add arguments common in all files, typically file directory"""

    parser.add_argument('--data_dir',
                    type=str,
                    default='../data')

    parser.add_argument('--train_file',
                    type=str,
                    default='../data/train.npz')

    parser.add_argument('--dev_file',
                    type=str,
                    default='../data/dev.npz')

    parser.add_argument('--test_dir',
                    type=str,
                    default='../data/test.npz')




def add_train_test_args(parser):
    """Add arguments common to training and testing"""

    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')


    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Number of features in encoder hidden layers.')

    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')


    parser.add_argument('--model_name',
                        type=str,
                        default="model",
                        help='Which model to train or test')

def add_train_args(parser):
    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-7,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')