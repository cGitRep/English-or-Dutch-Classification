import decision_tree

"""
predicts label for given sentence using the hypothesis given by ada boost
returns either en or nl
"""


def predict(sentence, hypothesis):
    num_en = 0
    num_nl = 0
    hypo = hypothesis[0]
    weights = hypothesis[1]

    for i in range(len(hypo)):
        label = decision_tree.traverse(hypo[i], sentence)

        if label == "en":
            num_en += weights[i]
        if label == "nl":
            num_nl += weights[i]

    if num_en > num_nl:
        return "en"
    else:
        return "nl"


"""
returns accuracy of an ada boost hypothesis
"""


def accuracy(data, hypothesis, labels):
    correct = 0
    i = 0
    for sentence in data:
        label = labels[i]
        prediction = predict(sentence, hypothesis)

        if prediction == label:
            correct += 1
        i += 1

    return correct / len(data)
