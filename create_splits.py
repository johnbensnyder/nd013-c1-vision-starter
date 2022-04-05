import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # Randomly split files so 60% training 20% val and 20% test
    train_dir = os.path.join(destination, 'train')
    val_dir = os.path.join(destination, 'val')
    test_dir = os.path.join(destination, 'test')
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    files = glob.glob(os.path.join(source, "*.tfrecord"))
    for file in files:
        random_draw = random.random()
        file_name = file.split('/')[-1]
        if random_draw<.6:
            os.symlink(file, os.path.join(train_dir, file_name))
        elif random_draw<.8:
            os.symlink(file, os.path.join(val_dir, file_name))
        else:
            os.symlink(file, os.path.join(test_dir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)