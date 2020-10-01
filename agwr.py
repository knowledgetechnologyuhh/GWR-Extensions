"""
gwr-tb :: Associative GWR based on Marsland et al. (2002)'s Grow-When-Required network
@last-modified: 01 October 2020
@author: German I. Parisi (german.parisi@gmail.com)
@contributor: Nicolas Duczek (duczek@informatik.uni-hamburg.de)
"""

import scipy.spatial
import math
import numpy as np
from heapq import nsmallest
from typing import Tuple, Union, Callable, Any

class AssociativeGWR:

    def __init__(self):
        self.iterations = 0

    def compute_distance(self, x, y, m):
        return np.linalg.norm(x-y) if m else scipy.spatial.distance.cosine(x, y)

    def find_bmus(self, sample):
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(self.weights[i], sample, self.dis_metric)

        # Compute the best and second-best matching units
        bs = nsmallest(2, ((k, i) for i, k in enumerate(distances)))
        return bs[0][1], bs[0][0], bs[1][1]  # BMU/SBMU:  [0/1][0] - distance, [0/1][1] - index

    def expand_matrix(self, matrix):
        ext_matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1))))
        ext_matrix = np.vstack((ext_matrix, np.zeros((1, ext_matrix.shape[1]))))
        return ext_matrix

    def init_network(self, dataset, init_random):

        assert self.iterations < 1, "Can't initialize a trained network"
        assert dataset is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons with dimensionality given by dataset
        self.num_nodes = 2
        self.dimension = dataset.vectors.shape[1]
        self.weights = [np.zeros(self.dimension), np.zeros(self.dimension)]

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
            self.weights[i] = dataset.vectors[init_index[i]]
            # self.labels[i][int(dataset.labels[i])] = 1
            # print(self.weights[i])

    def update_weight(self, input_vector, index, epsilon, new_node=False):
        if not new_node:
            delta = np.dot((input_vector - self.weights[index]), (epsilon * self.habn[index]))
            self.weights[index] = self.weights[index] + delta
        else:
            new_weight = (self.weights[index] + input_vector) / 2.0
            self.weights.append(new_weight)

    def habituate_node(self, index, tau, new_node=False):
        self.habn[index] += tau * 1.05 * (1 - self.habn[index]) - tau
        if new_node:
            self.habn.append(1)

    def update_neighbors(self, input_vector, index, epsilon, new_node=False):
        if not new_node:
            b_neighbors = np.nonzero(self.edges[index])
            for z in range(0, len(b_neighbors[0])):
                nextIndex = b_neighbors[0][z]
                self.update_weight(input_vector, nextIndex, epsilon)
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

    def train_agwr(self, dataset, epochs, a_threshold, l_rates):

        assert not self.locked, "Network is locked. Unlock to train."
        assert dataset.vectors.shape[1] == self.dimension, "Wrong data dimensionality"

        self.samples = dataset.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples # OK for batch, bad for incremental
        self.dis_metric = 1 # 1 = Euclidean, 0 = Cosine
        self.max_neighbors = 6
        self.max_age = 20
        self.num_classes = dataset.num_classes
        self.a_inc = 1
        self.a_dec = 0.1

        # Start training
        error_counter = np.zeros(self.max_epochs)

        for epoch in range(0, self.max_epochs):

            for iteration in range(0, self.samples):

                # Generate input sample
                sample = dataset.vectors[iteration]
                # label = dataset.labels[iteration]

                # Find best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(sample)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                activation = np.exp(-b_distance)
                new_node = False

                if (activation < self.a_threshold
                        and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):

                    new_node = True
                    n_index = self.num_nodes
                    self.num_nodes += 1

                # Update BMU's edges and ages
                self.update_edges(b_index, s_index, None if not new_node else n_index)

                # Update BMU's weight vector
                self.update_weight(sample, b_index, self.epsilon_b, new_node)

                # Update BMU's neighbors
                self.update_neighbors(sample, b_index, self.epsilon_n, new_node)

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

            print ("(Epoch: %s, NN: %s, AQE: %s)" %
                   (epoch + 1, self.num_nodes, error_counter[epoch]))

    def test_agwr(self, test_dataset):
        self.bmus_index = -np.ones(self.samples)
        # self.bmus_label = -np.ones(self.samples)
        self.bmus_activation = np.zeros(self.samples)
        acc_counter = 0
        for i in range(0, test_dataset.vectors.shape[0]):
            sample = test_dataset.vectors[i]
            b_index, b_distance, s_index = self.find_bmus(sample)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            # self.bmus_label[i] = np.argmax(self.labels[b_index])

            # if self.bmus_label[i] == test_dataset.labels[i]:
            #     acc_counter += 1.0

        # self.test_accuracy = acc_counter / test_dataset.vectors.shape[0]