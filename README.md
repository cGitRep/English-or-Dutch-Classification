# Language-Classification-Program

The program reads a number of Dutch and English sentences (located in the training folder) and trains a hypothesis (saved in hypothesis.txt) using given features (located in the features folder). Users can choose to train the hypothesis using a decision tree or AdaBoost. The program can then use this hypothesis to predict what language a given sentence is.


# 2 types of command line arguments

**train "train-examples" "features" "hypothesisOut" "learning-type"**
- train-examples: file that contains a number of sentences and whether they are English or Dutch
- features: file that contains the features that will be used to train the hypothesis
- hypothesisOut: name of the file that the program will write the hypothesis to
- learning-type: either "dt" (decision tree) or "ada" (AdaBoost)
- ex: "python lab3.py train train_medium.dat features.txt hypothesis.txt dt"

**predict "test-examples" "features" "hypothesis"**
- test-examples: file that contains sentences that are either in English or Dutch
- features: file that contains the features used to create the hypothesis
- hypothesis: the trained decision tree/ensemble created by the program
- for each sentence, the program returns either "en" for English or "nl" for Dutch
- ex: "python lab3.py predict large_test features.txt hypothesis.txt"


# Other info
-> The given hypothesis in "best.model" is currently the best decision tree I've come up with and has around a 97.8% success rate 

-> I use "hypothesis.txt" as the file to write to when creating a hypothesis, but it can be any file

-> All feature, testing, and training files must follow the same formating as the ones given
