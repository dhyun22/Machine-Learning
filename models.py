import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.batch_size = 1
        self.weight = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.weight

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.weight, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        learning_status = True
        while learning_status == True:
            learning_status = False
            for x, y in dataset.iterate_once(self.batch_size):
                result = self.get_prediction(x)
                if result != nn.as_scalar(y):
                    self.weight.update(nn.Constant(nn.as_scalar(y)*x.data), 1)
                    learning_status = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 10
        self.learning_rate = -0.001
        self.weight1 = nn.Parameter(1, 15)
        self.batch1 = nn.Parameter(1, 15)
        self.weight2 = nn.Parameter(15, 10)
        self.batch2 = nn.Parameter(1, 10)
        self.weight3 = nn.Parameter(10, 1)
        self.batch3 = nn.Parameter(1, 1)
        self.parameters = [self.weight1, self.batch1,
                           self.weight2, self.batch2, self.weight3, self.batch3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        layer1 = nn.AddBias(nn.Linear(x, self.weight1), self.batch1)
        layer2 = nn.AddBias(
            nn.Linear(nn.ReLU(layer1), self.weight2), self.batch2)
        layer3 = nn.AddBias(nn.Linear(
            nn.ReLU(layer2), self.weight3), self.batch3)
        return layer3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_status = True
        loss_var = float('inf')
        last_loss = None
        while learning_status == True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                learning_status = False
                if last_loss:
                    loss_var = abs(nn.as_scalar(loss)-nn.as_scalar(last_loss))
                last_loss = loss
                if loss_var > 0.00001:
                    learning_status = True
                    gradients = nn.gradients(loss, self.parameters)
                    for i in range(len(self.parameters)):
                        self.parameters[i].update(
                            gradients[i], self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 100
        self.learning_rate = 0.1
        self.weight1 = nn.Parameter(784, 256)
        self.batch1 = nn.Parameter(1, 256)
        self.weight2 = nn.Parameter(256, 128)
        self.batch2 = nn.Parameter(1, 128)
        self.weight3 = nn.Parameter(128, 64)
        self.batch3 = nn.Parameter(1, 64)
        self.weight4 = nn.Parameter(64, 10)
        self.batch4 = nn.Parameter(1, 10)
        self.params = [self.weight1, self.batch1, self.weight2, self.batch2,
                       self.weight3, self.batch3, self.weight4, self.batch4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.batch1))
        layer2 = nn.ReLU(nn.AddBias(
            nn.Linear(layer1, self.weight2), self.batch2))
        layer3 = nn.ReLU(nn.AddBias(
            nn.Linear(layer2, self.weight3), self.batch3))
        layer4 = nn.AddBias(nn.Linear(layer3, self.weight4), self.batch4)
        return layer4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        valid_acc = 0
        while valid_acc < 0.98:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.params)
                for param, grad in zip(self.params, gradients):
                    param.update(grad, -self.learning_rate)
                loss = nn.as_scalar(loss)
            valid_acc = dataset.get_validation_accuracy()


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.learning_rate = 0.1
        self.initial_weight = nn.Parameter(self.num_chars, 256)
        self.initial_bias = nn.Parameter(1, 256)
        self.input_to_hidden_weights = nn.Parameter(self.num_chars, 256)
        self.hidden_to_hidden_weights = nn.Parameter(256, 256)
        self.hidden_bias = nn.Parameter(1, 256)
        self.hidden_to_output_weights = nn.Parameter(256, len(self.languages))
        self.output_bias = nn.Parameter(1, len(self.languages))
        self.params = [self.initial_weight, self.initial_bias, self.input_to_hidden_weights, self.hidden_to_hidden_weights,
                       self.hidden_bias, self.hidden_to_output_weights, self.output_bias]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden_state = nn.ReLU(nn.AddBias(
            nn.Linear(xs[0], self.initial_weight), self.initial_bias))
        for char in xs[1:]:
            hidden_state = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.input_to_hidden_weights),
                                                     nn.Linear(hidden_state, self.hidden_to_hidden_weights)), self.hidden_bias))
        output = nn.AddBias(
            nn.Linear(hidden_state, self.hidden_to_output_weights), self.output_bias)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(xs)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        valid_acc = 0
        while valid_acc < 0.85:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.params)
                for param, grad in zip(self.params, gradients):
                    param.update(grad, -self.learning_rate)
                loss = nn.as_scalar(loss)
            valid_acc = dataset.get_validation_accuracy()
