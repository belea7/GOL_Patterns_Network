"""
Contains methods to create data for training and testing the ANN
"""
from seagull.lifeforms.wiki import parse_rle
import network_constants as const
import numpy as np
from random import shuffle
import os


def get_inputs_from_dir(directory, expected, train=True):
    """
    Get all inputs from a directory containing RLE files.

    :param directory: path of directory
    :param expected: expected output
    :param train: is created for training or not
    :return: list of inputs
    """
    # Get all binary grid representations
    files = os.listdir(directory)
    grid = np.zeros((10, 10), dtype=int)
    matrix = []
    for file in files:
        with open(directory + '\\' + file, 'r') as f:
            data = f.read()
            new = grid.copy()

            # Get life form and place in 10x10 grid
            life_form = parse_rle(data)
            height, width = life_form.size
            new[0: 0 + height, 0: 0 + width] = life_form.layout
            matrix.append(new)

            # If training set - add also matrix after 1,2,3 rotations
            if train:
                rotate1 = new.copy()
                matrix.append(np.rot90(rotate1, 1))
                rotate2 = new.copy()
                matrix.append(np.rot90(rotate2, 2))
                rotate3 = new.copy()
                matrix.append(np.rot90(rotate3, 3))

    # Convert all matrix representation vectors
    examples = []
    for mat in matrix:
        vector = []
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                vector.append(mat[i][j])
        vector.append(expected)
        examples.append(vector)
    return examples


def create_train_set():
    """
    Create the train set

    :return: list of examples
    """
    # Get cyclic examples
    train_set = []
    directory = const.CYCLIC_EXAMPLES_DIR
    train_set.extend(get_inputs_from_dir(directory, 1))

    # Get not cyclic examples
    directory = const.NONCYCLIC_EXAMPLES_DIR
    train_set.extend(get_inputs_from_dir(directory, 0))

    shuffle(train_set)
    return train_set


def create_test_set():
    """
    Create the test set

    :return: list of examples
    """
    # Get cyclic examples
    test_set = []
    directory = const.CYCLIC_TEST_DIR
    names = [f for f in os.listdir(directory)]
    patterns = get_inputs_from_dir(directory, 1, False)
    for i in range(len(names)):
        test_set.append((names[i], patterns[i]))

    # Get not cyclic examples
    directory = const.NONCYCLIC_TEST_DIR
    names = [f for f in os.listdir(directory)]
    patterns = get_inputs_from_dir(directory, 0, False)
    for i in range(len(names)):
        test_set.append((names[i], patterns[i]))

    shuffle(test_set)
    return test_set
