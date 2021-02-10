# ANN constants
GRID_SIZE = 10
INPUT_SIZE = GRID_SIZE ** 2
INPUT_NEURONS = INPUT_SIZE
HIDDEN_NEURONS = 5
OUTPUT_NEURONS = 2
LEARNING_RATE = 0.5
EPOCHS = 150

# Data constants
EXAMPLES_DIR = 'train_set'
TEST_DIR = 'test_set'
CYCLIC_DIR = '/cyclic'
NOT_CYCLIC_DIR = '/not_cyclic'
CYCLIC_EXAMPLES_DIR = EXAMPLES_DIR + CYCLIC_DIR
NONCYCLIC_EXAMPLES_DIR = EXAMPLES_DIR + NOT_CYCLIC_DIR
CYCLIC_TEST_DIR = TEST_DIR + CYCLIC_DIR
NONCYCLIC_TEST_DIR = TEST_DIR + NOT_CYCLIC_DIR
