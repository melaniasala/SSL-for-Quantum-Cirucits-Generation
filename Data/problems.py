"""
    File with all the classes related to the optimization problems based on graphs.
    Notice that Problem and Graph Problem classes, in this framework, work as interfaces:
    each method inside them is, in fact, overridden by the children classes (MaxCut, MVC, ...)
"""

import numpy as np
from .qaoa import QUBO


# Generic Problem class representing an optimization problem
class Problem:

    def __init__(self, n_variables):
        self.n = n_variables


# Generic optimization problem having a related graph.
class GraphProblem(Problem):

    def compute_qubo(self, adj):
        """
        Compute the QUBO matrix related to the graph with adjacency matrix adj

        :param adj: symmetric adjacency matrix of the graph
        :return: matrix Q
        """
        pass

    def get_name(self):
        """
        Get the name of the problem kind (ex: maxCut, mvc, ...)

        :return: the string with the name of the problem
        """
        pass


class MaxCut(GraphProblem):
    """
    Implementation of the MaxCut problem
    """

    def compute_qubo(self, adj):
        q = np.zeros([self.n, self.n])
        for i in range(0, len(adj)):
            for j in range(0, len(adj)):
                if i == j:
                    q[i, j] = -np.sum(adj[i])
                else:
                    try:
                        q[i, j] = adj[i, j]
                    except IndexError:
                        print('Oooops...')
        return QUBO(q)

    def get_name(self):
        return 'maxCut'


class MVC(GraphProblem):
    """
        Implementation of the Minimum Vertex Cover (MVC) problem
    """

    def compute_qubo(self, adj):

        q = np.zeros([self.n, self.n])
        for i in range(0, len(adj)):
            for j in range(0, len(adj)):
                if i == j:
                    q[i, j] = 1 - self.n * np.sum(adj[i])
                else:
                    q[i, j] = 1/2 * self.n * adj[i, j]
        return QUBO(q)

    def get_name(self):
        return 'mvc'


class MIS(GraphProblem):
    """
        Implementation of the Maximum Independent Set (MIS) problem
    """

    def compute_qubo(self, adj):
        q = np.zeros([self.n, self.n])
        for i in range(0, len(adj)):
            for j in range(0, len(adj)):
                if i == j:
                    q[i, j] = - 1
                else:
                    q[i, j] = adj[i, j]
        return QUBO(q)

    def get_name(self):
        return 'mis'


class MaxClique(GraphProblem):
    """
        Implementation of the Max-Clique problem
    """

    def compute_qubo(self, adj):
        # Equivalent to MIS on the complement graph
        new_adj = np.ones([len(adj), len(adj)]) - adj
        q = np.zeros([self.n, self.n])
        for i in range(0, len(new_adj)):
            for j in range(0, len(new_adj)):
                if i == j:
                    q[i, j] = - 1
                else:
                    q[i, j] = new_adj[i, j]
        return QUBO(q)

    def get_name(self):
        return 'maxClique'


class CommunityDetection(GraphProblem):
    """
        Implementation of the Community Detection problem
    """

    def compute_qubo(self, adj):
        d = np.sum(adj, axis=0)
        transposed_d = d.reshape((len(d), 1))
        d = d.reshape((1, len(d)))
        squared_d = np.dot(transposed_d, d)

        B = (adj - squared_d / np.sum(d))

        Q = - 2 * B / np.sum(d)

        q = np.zeros([self.n, self.n])
        for i in range(0, len(adj)):
            for j in range(0, len(adj)):
                if i == j:
                    q[i, j] = Q[i, j]
                else:
                    q[i, j] = Q[i, j]
        return QUBO(q)

    def get_name(self):
        return 'commDet'