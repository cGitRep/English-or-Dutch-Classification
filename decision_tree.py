import math

"""
Loops through a decision tree for a given sentence 
and returns the predicted label
"""


def traverse(tree, sentence):
    # reached decision/label
    if isinstance(tree, str):
        return tree

    if tree is None:
        return None

    for feature_index, subtree in tree.items():
        feature_bool = sentence[int(feature_index) - 1]

        if feature_bool == "True":
            return traverse(subtree["True"], sentence)
        if feature_bool == "False":
            return traverse(subtree["False"], sentence)


def accuracy(data, hypothesis, labels):
    correct = 0
    i = 0

    for sentence in data:
        label = labels[i]
        prediction = traverse(hypothesis, sentence)

        if prediction == label:
            correct += 1
        i += 1

    return correct / len(data)


# calc weighted entropy
def getWeightedEntropy(data, weights):
    weight_sum = sum(weights)
    en_weight = 0
    nl_weight = 0

    for val, weight in zip(data, weights):
        if val[-1] == "en":
            en_weight += weight
        if val[-1] == "nl":
            nl_weight += weight

    entropy = 0
    pe = 0
    pn = 0
    if weight_sum > 0:
        pe = en_weight / weight_sum
        pn = nl_weight / weight_sum

    if pe > 0:
        entropy = -pe * math.log(pe, 2)
    if pn > 0:
        entropy += -pn * math.log(pn, 2)

    return entropy


# calc entropy
def getEntropy(data):
    if len(data) == 0:
        return 0

    entropy = 0
    numE = 0
    numN = 0
    for val in data:
        if val[-1] == "en":
            numE += 1
        if val[-1] == "nl":
            numN += 1

    pe = numE / len(data)
    pn = numN / len(data)

    if pe > 0:
        entropy = -pe * math.log(pe, 2)
    if pn > 0:
        entropy += -pn * math.log(pn, 2)

    return entropy


# weighted change in entropy
def getWeightedGain(data, i, weights):
    weightSum = sum(weights)
    entropy = getWeightedEntropy(data, weights)
    trueData = []
    falseData = []
    trueWeights = []
    falseWeights = []

    for val, weight in zip(data, weights):
        if val[i] == "True":
            trueData.append(val)
            trueWeights.append(weight)
        if val[i] == "False":
            falseData.append(val)
            falseWeights.append(weight)

    trueEntropy = getWeightedEntropy(trueData, trueWeights)
    falseEntropy = getWeightedEntropy(falseData, falseWeights)

    trueWeightSum = sum(trueWeights)
    falseWeightSum = sum(falseWeights)

    gainEntropy = (trueWeightSum / weightSum) * trueEntropy
    gainEntropy += (falseWeightSum / weightSum) * falseEntropy

    return entropy - gainEntropy


# change in entropy
def getGain(data, i):
    entropy = getEntropy(data)
    trueData = []
    falseData = []

    for val in data:
        if val[i] == "True":
            trueData.append(val)
        if val[i] == "False":
            falseData.append(val)

    # get entropy for each feature
    trueEntropy = getEntropy(trueData)
    falseEntropy = getEntropy(falseData)

    # entropy(i) - ((weighted avg * entropy) of true and false)
    gainEntropy = len(trueData) / len(data) * trueEntropy
    gainEntropy += len(falseData) / len(data) * falseEntropy

    return entropy - gainEntropy


def getBestSplit(data, weights):
    bestSplit = -1
    maxGain = -1

    for i in range(len(data[0]) - 1):
        gain = getWeightedGain(data, i, weights)
        if gain > maxGain:
            maxGain = gain
            bestSplit = i

    return bestSplit


def decisionTree(data, depth, weights):
    if len(data) == 0:
        return None

    if depth < 1:
        classes = [val[-1] for val in data]
        return max(classes, key=classes.count)

    bestSplit = getBestSplit(data, weights)
    trueData = []
    falseData = []
    trueWeight = []
    falseWeight = []

    for val, weight in zip(data, weights):
        if val[bestSplit] == "True":
            trueData.append(val)
            trueWeight.append(weight)
        if val[bestSplit] == "False":
            falseData.append(val)
            falseWeight.append(weight)

    tree = {bestSplit + 1: {"True": decisionTree(trueData, depth - 1, trueWeight), "False": decisionTree(falseData, depth - 1, falseWeight)}}

    return tree
