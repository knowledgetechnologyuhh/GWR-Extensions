"""
gwr-tb :: Gamma-GWR based on Marsland et al. (2002)'s Grow-When-Required network
@last-modified: 01 October 2020
@author: German I. Parisi (german.parisi@gmail.com)
@contributor: Nicolas Duczek (duczek@informatik.uni-hamburg.de)
"""

import numpy as np
import math
from heapq import nsmallest


class GammaGWR:

    def __init__(self):
        self.iterations = 0

    def compute_alphas(self, num_coeff):
        alpha_w = np.zeros(num_coeff)
        alpha_w[0] = self.alpha

        for h in range(1, len(alpha_w)):
            alpha_w[h] = np.exp(-h)

        alpha_w[1:] = (1-self.alpha) * (alpha_w[1:] / sum(alpha_w[1:]))
        return alpha_w

    def compute_distance(self, x, y):
        return np.linalg.norm(np.dot(self.alphas.T, (x-y)))

    def compute_context_distance(self, context, input_vector):
        return np.linalg.norm(context - input_vector)

    def predict_number_of_nodes(self, current_weight, steps):
        self.predictions = [current_weight]
        for i in range(0, steps):
            context_distances = np.zeros(self.num_nodes)
            for j in range(0, self.num_nodes):
                context_distances[j] = self.compute_context_distance(self.weights[j][1], self.predictions[-1])
            self.predictions.append(self.weights[context_distances.argmin()][0])

    def predict_next_node(self):
        self.predictions = self.predictions[1:]
        context_distances = np.zeros(self.num_nodes)
        for j in range(0, self.num_nodes):
            context_distances[j] = self.compute_context_distance(self.weights[j][1], self.predictions[-1])
        self.predictions.append(self.weights[context_distances.argmin()])

    def compute_prediction_error(self, weight):
        return np.linalg.norm(weight - self.predictions[0])

    def find_bmus(self, g_context):
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(self.weights[i], g_context)

        # Compute the best and second-best matching units
        bs = nsmallest(2, ((k, i) for i, k in enumerate(distances)))
        return bs[0][1], bs[0][0], bs[1][1]  # BMU/SBMU:  [0/1][0] - distance, [0/1][1] - index

    def expand_matrix(self, matrix):
        ext_matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1))))
        ext_matrix = np.vstack((ext_matrix, np.zeros((1, ext_matrix.shape[1]))))
        return ext_matrix

    def init_network(self, dataset, init_random, alpha=0.7, num_context=0):

        assert self.iterations < 1, "Can't initialize a trained network"
        assert dataset is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        self.dimension = dataset.vectors.shape[1]
        self.num_context = num_context
        self.depth = self.num_context + 1
        self.weights = [np.zeros((self.depth, self.dimension)),
                        np.zeros((self.depth, self.dimension))]
        self.alpha = alpha
        self.predictions = 0
        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.zeros((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # Label histograms
        # self.labels = [-np.ones(dataset.num_classes), -np.ones(dataset.num_classes)]

        # Initialize weights
        if init_random:
            init_index = np.random.randint(0, dataset.vectors.shape[0], 2)
        else:
            init_index = list(range(0, self.num_nodes))
        for i in range(0, len(init_index)):
            self.weights[i][0] = dataset.vectors[init_index[i]]
            # self.labels[i][int(dataset.labels[i])] = 1
            # print(self.weights[i])
        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)

    def update_weight(self, index, epsilon, new_node=False):
        if not new_node:
            delta = np.dot((self.g_context - self.weights[index]), (epsilon * self.habn[index]))
            self.weights[index] = self.weights[index] + delta
        else:
            new_weight = (self.weights[index] + self.g_context) / 2.0
            self.weights.append(new_weight)

    def habituate_node(self, index, tau, new_node=False):
        self.habn[index] += tau * 1.05 * (1 - self.habn[index]) - tau
        if new_node:
            self.habn.append(1)

    def update_neighbors(self, index, epsilon, new_node=False):
        if not new_node:
            b_neighbors = np.nonzero(self.edges[index])
            for z in range(0, len(b_neighbors[0])):
                nextIndex = b_neighbors[0][z]
                self.update_weight(nextIndex, epsilon)
                self.habituate_node(nextIndex, self.tau_n)

    # def update_labels(self, bmu, label, new_node=False):
    #     if not new_node:
    #         for a in range(0, self.num_classes):
    #             if a == label:
    #                 self.labels[bmu][a] += self.a_inc
    #             else:
    #                 if label != -1:
    #                     self.labels[bmu][a] -= self.a_dec
    #                     if (self.labels[bmu][a] < 0):
    #                         self.labels[bmu][a] = 0
    #     else:
    #         new_label = np.zeros(self.num_classes)
    #         if label != -1:
    #             new_label[int(label)] = self.a_inc
    #         self.labels.append(new_label)

    def update_edges(self, b_index, s_index, n_index=None):
        if n_index is None:
            self.edges[b_index, s_index] = 1
            self.edges[s_index, b_index] = 1
            self.ages[b_index, s_index] = 0
            self.ages[s_index, b_index] = 0
        else:
            self.edges = self.expand_matrix(self.edges)
            self.ages = self.expand_matrix(self.ages)
            self.edges[b_index, s_index] = 0
            self.edges[s_index, b_index] = 0
            self.ages[b_index, s_index] = 0
            self.ages[s_index, b_index] = 0
            self.edges[b_index, n_index] = 1
            self.edges[n_index, b_index] = 1
            self.edges[s_index, n_index] = 1
            self.edges[n_index, s_index] = 1
        self.ages[:, b_index] += 1

    def remove_old_edges(self):
        for i in range(0, self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            for j in neighbours[0]:
                if self.ages[i, j] > self.max_age:
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0
                    self.ages[i, j] = 0
                    self.ages[j, i] = 0

    def remove_isolated_nodes(self):
        index = 0
        removed = 0
        while (index < self.num_nodes):
            neighbours = np.nonzero(self.edges[index])
            if len(neighbours[0]) < 1:
                self.weights.pop(index)
                # self.labels.pop(index)
                self.habn.pop(index)
                self.edges = np.delete(self.edges, index, axis=0)
                self.edges = np.delete(self.edges, index, axis=1)
                self.ages = np.delete(self.ages, index, axis=0)
                self.ages = np.delete(self.ages, index, axis=1)
                self.num_nodes -= 1
                removed += 1
            else:
                index += 1
        if removed > 0:
            print ("(-- Removed %s neuron(s))" % removed)

    def train_ggwr(self, dataset, epochs, a_threshold, beta, l_rates):

        assert not self.locked, "Network is locked. Unlock to train."
        assert dataset.vectors.shape[1] == self.dimension, "Wrong dimensionality"

        self.samples = dataset.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples  # OK for batch, bad for incremental
        self.max_neighbors = 6
        self.max_age = 20
        self.new_node = 0.5
        self.num_classes = dataset.num_classes
        self.a_inc = 1
        self.a_dec = 0.1

        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))

        for epoch in range(0, self.max_epochs):

            for iteration in range(0, self.samples):

                # Generate input sample
                self.g_context[0] = dataset.vectors[iteration]
                # label = dataset.labels[iteration]

                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (self.beta * previous_bmu[z-1]) + ((1-self.beta) * previous_bmu[z])

                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(self.g_context)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                activation = np.exp(-b_distance)
                new_node = False

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (activation < self.a_threshold
                    and self.habn[b_index] < self.hab_threshold
                    and self.num_nodes < self.max_nodes):

                    new_node = True
                    n_index = self.num_nodes
                    self.num_nodes += 1

                # Update BMU's edges // Remove BMU's oldest ones
                self.update_edges(b_index, s_index, None if not new_node else n_index)

                # Update BMU's weight vector
                self.update_weight(b_index, self.epsilon_b, new_node)

                # Update BMU's neighbors
                self.update_neighbors(b_index, self.epsilon_n, new_node)

                # Habituate BMU
                self.habituate_node(b_index, self.tau_b, new_node)

                # Update BMU's label histogram
                # self.update_labels(b_index if not new_node else n_index, label, new_node)

                # Remove old edges
                self.remove_old_edges()

                # Remove isolated neurons
                self.remove_isolated_nodes()

                self.iterations += 1

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples

            print ("(Epoch: %s, NN: %s, ATQE: %s)" %
                   (epoch + 1, self.num_nodes, error_counter[epoch]))



    def test_gammagwr(self, test_dataset, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', None)
        self.bmus_index = -np.ones(self.samples)
        self.bmus_label = -np.ones(self.samples)
        self.bmus_activation = np.zeros(self.samples)

        input_context = np.zeros((self.depth, self.dimension))

        if test_accuracy: acc_counter = 0

        for i in range(0, test_dataset.vectors.shape[0]):
            input_context[0] = test_dataset.vectors[i]

            # Find the BMU
            b_index, b_distance, s_index = self.find_bmus(input_context)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            # self.bmus_label[i] = np.argmax(self.labels[b_index])

            for j in range(1, self.depth):
                input_context[j] = input_context[j-1]

            if test_accuracy:
                # if self.bmus_label[i] == test_dataset.labels[i]:
                    acc_counter += 1.0

        if test_accuracy:
            self.test_accuracy = acc_counter / test_dataset.vectors.shape[0]