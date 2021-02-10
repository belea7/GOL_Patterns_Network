"""
Contains Neural Network and Neurons classes
"""
from random import random
import matplotlib.pyplot as plt
from data_generator import *


class Network:
    """
    Class representing neural network
    """
    def __init__(self):
        self.layers = self._init_layers()           # list of layer in network
        self.data = []

    @staticmethod
    def _init_layers():
        """
        Creates the networks layers

        :return: layers (list)
        """
        layers = []
        # Create hidden layer
        hidden_layer = []
        for _ in range(const.HIDDEN_NEURONS):
            hidden_layer.append(Neuron(const.INPUT_SIZE + 1))
        layers.append(hidden_layer)

        # Create output layer
        output_layer = []
        for _ in range(const.OUTPUT_NEURONS):
            output_layer.append(Neuron(const.HIDDEN_NEURONS + 1))
        layers.append(output_layer)
        return layers

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function

        :param x: input
        :return: output
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Sigmoid derivative

        :param x: input
        :return: output
        """
        return x * (1.0 - x)

    @staticmethod
    def step_function(x):
        """
        Step binary function

        :param x: input
        :return: output
        """
        return 1 if x > 0 else 0

    @staticmethod
    def step_function_derivative(x):
        """
        Step binary derivative

        :param x: input
        :return: output
        """
        return 0 if x != 0 else None

    @staticmethod
    def linear_combination(weights, inputs):
        """
        Linear combination between two vectors

        :param weights: first vector
        :param inputs: second vector
        :return: result
        """
        sum = weights[-1]
        for i in range(len(weights) - 1):
            sum += weights[i] * inputs[i]
        return sum

    def feed(self, example):
        """
        Feed an input to the network

        :param example: input
        :return: the output of the network
        """
        inputs = example
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                activation = self.linear_combination(neuron.weights, inputs)
                neuron.output = self.sigmoid(activation)
                outputs.append(neuron.output)
            inputs = outputs
        return inputs

    def update_weights(self, example, expected):
        """
        Update weights by performing backpropagation.

        :param example: example to fix the weights according
        :param expected: expected output for example
        :returns: None
        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            # If last layer - calculate the error value
            if i == len(self.layers) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron.output)
            else:
                # Backprop the error value form next layer
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)

            # Calculate delta for each neuron
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = errors[j] * self.sigmoid_derivative(neuron.output)

        # Update the weights
        for i in range(len(self.layers)):
            inputs = example[:-1]
            if i != 0:
                inputs = [neuron.output for neuron in self.layers[i - 1]]
            # Loop through layers and update weights
            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] += const.LEARNING_RATE * neuron.delta * inputs[j]
                neuron.weights[-1] += const.LEARNING_RATE * neuron.delta

    def train(self, train_set):
        """
        Train network

        :param train_set: list of examples
        :returns: None
        """
        errors = []
        for epoch in range(const.EPOCHS):
            sum_error = 0
            for example in train_set:
                outputs = self.feed(example)
                expected = [0 for _ in range(const.OUTPUT_NEURONS)]
                expected[example[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.update_weights(example, expected)
            print("Epoch:{0}, Error:{1}".format(epoch, round(sum_error, 3)))
            errors.append(sum_error)
        self.display_graph(errors)

    def test(self, test_set):
        """
        Tests network with new examples.

        :param test_set: new examples
        :return:
        """
        # Calculate error
        sum_error = 0
        for example in test_set:
            name, pattern = example
            outputs = self.feed(pattern)
            expected = [0 for _ in range(const.OUTPUT_NEURONS)]
            expected[pattern[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

        # Print results
        total = len(test_set)
        correct = 0
        wrong = 0
        misclassified = []
        for example in test_set:
            name, pattern = example
            outputs = network.feed(pattern)
            output = 1 if outputs[1] > outputs[0] else 0
            if output == pattern[-1]:
                correct += 1
            else:
                wrong += 1
                misclassified.append(name + "({})".format("cyclic" if pattern[-1] == 1 else "not cyclic"))
        print("Total:{0}, Correct:{1}, Wrong:{2}, Success rate:{3}%, Error:{4}"
              .format(total, correct, wrong, round(correct * 100 / total, 3), round(sum_error, 3)))
        print("Misclassified the following examples: {}".format(" , ".join(misclassified)))

    @staticmethod
    def display_graph(data):
        """
        Displays graph with error rates over epochs

        :param data: error rates
        :returns: None
        """
        epochs = range(const.EPOCHS)
        plt.plot(epochs, data)
        plt.grid(True)
        plt.title("Error rate over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("error rate")
        plt.show()


class Neuron:
    """
    Class representing neuron in the network
    """
    def __init__(self, weights_num):
        self.output = 0                                     # Last output produced
        self.weights = self._init_weights(weights_num)      # Neuron input weights
        self.delta = 0                                      # Neuron delta value for backpropagation

    @staticmethod
    def _init_weights(num):
        """
        Creates random input weights for the neuron

        :param num: number of connections
        :return: wights (list)
        """
        weights = []
        for _ in range(num):
            weights.append(random())
        return weights


if __name__ == '__main__':
    # Test training backprop algorithm
    train_set = create_train_set()
    test_set = create_test_set()

    print("Creating neural network...")
    network = Network()

    print("Training the network")
    network.train(train_set)

    print("network is trained. Testing on new examples")
    network.test(test_set)
