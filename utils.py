import json
import sys
from collections import OrderedDict
from argparse import ArgumentParser


def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, value in kwargs.items():
        message += f', {key}:{value}'
    message += ']'
    sys.stdout.write(message)
    sys.stdout.flush()


def save_args(args, filename):
    args = OrderedDict(vars(args))
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4)


def prepare_parser():
    parser = ArgumentParser(description='Network training configs')
    # Dataset configs
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Target dataset name (required)')
    parser.add_argument('-dd', '--dataset_dir', type=str, required=True,
                        help='Target dataset directory (required)')
    # Iteration config
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs [100]')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size [64]')
    parser.add_argument('-v', '--validation_split', type=float, default=0.1,
                        help='Validtion split ratio [0.1]')
    # Data pipeline configs
    parser.add_argument('--resize_shape', nargs=2, type=int, default=[32, 32],
                        help='Resize shape [32, 32]')
    parser.add_argument('--crop_shape', nargs=2, type=int, default=None,
                        help='Crop shape for images. [None]')
    parser.add_argument('--rotate', action='store_true',
                        help='Enable rotation in preprocessing')
    parser.add_argument('-flip-lr', '--flip_left_right', action='store_true',
                        help='Enable left-right flip in preprocessing')
    parser.add_argument('-flip-ud', '--flip_up_down', action='store_true',
                        help='Enable up-down flip in preprocessing')
    # Model config
    parser.add_argument('--filters', type=int, default=64,
                        help='Base filter number [64]')
    # Learning configs
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate [0.001]')
    return parser
