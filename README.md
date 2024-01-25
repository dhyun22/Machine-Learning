# Assignment 4 : Machine Learning
http://34.64.101.25/assign4/#table-of-contents

## Question 1 (6 points): Perceptron
Your tasks are to:

- Implement the run(self, x) method. This should compute the dot product of the stored weight vector and the given input, returning an nn.DotProduct object.
- Implement get_prediction(self, x), which should return 1 if the dot product is non-negative or −1 otherwise. You should use nn.as_scalar to convert a scalar Node into a Python floating-point number.
- Write the train(self) method. This should repeatedly loop over the data set and make updates on examples that are misclassified. Use the update method of the nn.Parameter class to update the weights. When an entire pass over the data set is completed without making any mistakes, 100% training accuracy has been achieved, and training can terminate.
- In this assignment, the only way to change the value of a parameter is by calling parameter.update(direction, multiplier), which will perform the update to the weights:

## Question 2 (6 points): Non-linear Regression
Your tasks are to:

- Implement RegressionModel.__init__ with any needed initialization.
- Implement RegressionModel.run to return a batch_size by 1 node that represents your model’s prediction.
- Implement RegressionModel.get_loss to return a loss for given inputs and target outputs.
- Implement RegressionModel.train, which should train your model using gradient-based updates.

## Question 3 (6 points): Digit Classification
For this question, you will train a network to classify handwritten digits from the MNIST dataset.
</br>
Each digit is of size 28 by 28 pixels, the values of which are stored in a 784-dimensional vector of floating point numbers. Each output we provide is a 10-dimensional vector which has zeros in all positions, except for a one in the position corresponding to the correct class of the digit.
</br>
Complete the implementation of the DigitClassificationModel class in models.py. The return value from DigitClassificationModel.run() should be a batch_size by 10 node containing scores, where higher scores indicate a higher probability of a digit belonging to a particular class (0-9). You should use nn.SoftmaxLoss as your loss. Do not put a ReLU activation in the last linear layer of the network.

## Question 4 (7 points): Language Identification
Complete the implementation of the LanguageIDModel class.
To receive full points on this problem, your architecture should be able to achieve an accuracy of at least 81% on the test set.

