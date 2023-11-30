# Implement the structured perceptron algorithm for Part-of-Speech (POS) tagging.


Task
=============================================================================================================
To implement the perceptron training algorithm using HMM-based features and the train and test data.
How does the accuracy on test data accuracy compare to HW 1?


Instructions for running "Perceptron.py"
=============================================================================================================
To run the script "Perceptron.py" change the values of "train_file", "test_file" and "tag_file" variables in the script.
We also have to specify the no. of iterations.


Description
=============================================================================================================
The structured perceptron algorithm is used to find the weights for all the features using the train file.
Then, these weights are used to find the best tag sequence for a given sentence in the test file using the Viterbi algorithm.


Accuracy:
=============================================================================================================
Correct tags: 51073 (8 iterations)
Total tags: 56684
Accuracy: 90.1%

The structured perceptron seems to give a lower accuracy (90.1%) than the HMM (91.75%).
One reason that I can think of is the no. of iterations (8), if the no. of iterations are increased then the structured perceptron may give a better accuracy.


References
=============================================================================================================
This was done as a homework problem in the Statistical Speech and Language Processing class (CSC 448, Fall 2017) by [Prof. Daniel Gildea](https://www.cs.rochester.edu/~gildea/) at the University of Rochester, New York. <br />
Have questions? Shoot me an [email](https://sites.google.com/view/deepakpandita/contact).
