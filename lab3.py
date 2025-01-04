import string
import sys

import ada_boost
import decision_tree
import results

"""
CONSTANTS
"""
# ada boost iterations
K = 100
# max depth for decision trees
tree_depth = 10

"""
reads the feature file
returns array of features
"""


def read_feature(file):
    with open(file) as f:
        features = f.read().strip().splitlines()
    return features


"""
reads the train and test files
parses and removes punctuation
returns array of labels, and array of sentences that each contain an array of words in the sentence
"""


def read_data(file):
    labels = []
    sentences = []

    with open(file) as f:
        for line in f:
            part = line.strip().split('|')

            # train data
            if len(part) == 2:
                label = part[0]
                labels.append(label)
                words = part[1].split()
                sentences.append(words)
            # test data
            else:
                words = part[0].split()
                sentences.append(words)

    return labels, sentences


"""
coverts all words in a sentence to lowercase
& removes punctuation 
arg: array of sentences
returns: array of updated sentences
"""


def convert_data(sentences):
    newSentences = []
    for sentence in sentences:
        newWords = []
        for word in sentence:
            exclude = set(string.punctuation)
            newWord = word.lower()
            newWord = ''.join(ch for ch in newWord if ch not in exclude)
            newWords.append(newWord)
        newSentences.append(newWords)
    return newSentences


"""
makes the feature/attribute table to be used by the decision tree
arg: sentences array, features array, labels array
returns: array for each sentence that contains if they have a feature or not and what language it is

i.e. 3 features, 2 sentences
[False, True, True, en],
[True, False, False, nl]
"""


def make_data(sentences, features, labels):
    data = []
    i = 0

    # make data for predictions
    if not labels:
        # check if a sentence contains a feature
        for sentence in sentences:
            featureArray = []
            for feature in features:
                featureFound = False
                for word in sentence:
                    if word == feature:
                        featureFound = True
                        break
                if featureFound:
                    featureArray.append("True")
                else:
                    featureArray.append("False")
            # add what language the features belong to
            data.append(featureArray)

    # make data to train
    else:
        # check if a sentence contains a feature
        for sentence in sentences:
            featureArray = []
            for feature in features:
                featureFound = False
                for word in sentence:
                    if word == feature:
                        featureFound = True
                        break
                if featureFound:
                    featureArray.append("True")
                else:
                    featureArray.append("False")
            # add what language the features belong to
            featureArray.append(labels[i])
            data.append(featureArray)
            i += 1

    return data


def train(examples, features, hypothesisOut, learningType):
    features = read_feature(features)
    labels, sentences = read_data(examples)
    data = make_data(sentences, features, labels)

    if learningType == "dt":
        weights = [1 / len(data)] * len(data)
        dt = decision_tree.decisionTree(data, tree_depth, weights)
        ada_boost.save_hypothesis(dt, hypothesisOut)

    if learningType == "ada":
        hypothesis, hypothesis_weight = ada_boost.adaboost(data, K)
        ada_boost.save_hypothesis((hypothesis, hypothesis_weight), hypothesisOut)


def predict(examples, features, hypothesis):
    features = read_feature(features)
    labels, sentences = read_data(examples)
    data = make_data(sentences, features, labels)

    # ada ensemble
    if isinstance(hypothesis, list):
        for sentence in data:
            print(results.predict(sentence, hypothesis))

    # decision tree
    else:
        for sentence in data:
            print(decision_tree.traverse(hypothesis, sentence))


def main():
    # train
    if len(sys.argv) == 6:
        examples = sys.argv[2]
        features = sys.argv[3]
        hypothesisOut = sys.argv[4]
        learningType = sys.argv[5]

        train(examples, features, hypothesisOut, learningType)

    # predict
    if len(sys.argv) == 5:
        examples = sys.argv[2]
        features = sys.argv[3]
        hypothesis_file = sys.argv[4]
        hypothesis = ada_boost.load_hypothesis(hypothesis_file)
        predict(examples, features, hypothesis)

    # used for debugging
    # train("train_medium.dat", "features.txt", "hypothesis.txt", "dt")
    # hypothesis = ada_boost.load_hypothesis("hypothesis.txt")
    # predict("large_test", "features.txt", hypothesis)


if __name__ == "__main__":
    main()
