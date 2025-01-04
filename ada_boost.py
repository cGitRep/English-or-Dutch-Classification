import json
import math

import decision_tree

"""
saves or loads a hypothesis into a file passed as an arg
"""


def save_hypothesis(hypothesis, file):
    with open(file, 'w') as f:
        json.dump(hypothesis, f)


def load_hypothesis(file):
    with open(file, 'r') as f:
        return json.load(f)


"""
adaboost!
K - number of iterations
returns tuple: hypothesis, hypothesis_weight
"""


def adaboost(data, K):
    # set array of weights
    weights = [1 / len(data)] * len(data)
    hypothesis = []
    hypothesis_weight = []
    e = 1e-10

    for k in range(K):
        L = decision_tree.decisionTree(data, 1, weights)
        error = 0

        for i in range(len(data)):
            # compute error
            if decision_tree.traverse(L, data[i]) != data[i][-1]:
                error += weights[i]

        error = max(error, e)
        weight_change = error / (1 - error)

        for i in range(len(data)):
            # update weights for correct examples
            if decision_tree.traverse(L, data[i]) == data[i][-1]:
                weights[i] = weights[i] * weight_change

        # normalize
        weight_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i] / weight_sum

        hypothesis_weight.append(.5 * math.log((1 - error) / error))
        hypothesis.append(L)

    return hypothesis, hypothesis_weight
