# Language-Classification-Lab

The program reads a number of Dutch and English sentences (located in train.dat files) and trains a hypothesis (saved in hypothesis.txt) using given features (located in features.txt files). Users can choose to train the hypothesis using a decision tree or AdaBoost. The program can then use this hypothesis to predict what language a givene sentence is. Written for the Intro to AI lab 3 assignment.


# Command Line Arguments

train "examples" "features" "hypothesisOut" "learning-type"
- examples: file that contains a number of sentences and whether they are english or dutch (must follow the same formating as the train.dat files)
- features: file that contains the features (separated by newlines) used to test the hypothesis
- hypothesisOut: name of the file that the program will write the hypothesis to
- learning-type: either "dt" (decision tree) or "ada" (AdaBoost)

predict "examples" "features" "hypothesis"
- examples: file that contains sentences (separated by newlines) that are either in English or Dutch
- features: file that contains the features used to create the hypothesis
- hypothesis: the trained decision tree/ensemble


-> The given hypothesis in "best.model" is currently the best decision tree I have come up with and has around a 97.8% success rate 
-> I use "hypothesis.txt" as the file to write to when creating a hypothesis
