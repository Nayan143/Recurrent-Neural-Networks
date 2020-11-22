import numpy as np
import common as cm


class BasicRNNCell:
    """Class RNN cell. This class contains the implementation of a basic RNN cell."""

    def __init__(self, n_in, n_hidden=100):
        """
        Initialize all the necessary variables for a basic RNN cell

        :param n_in: dimension of input samples
        :param n_hidden: dimension of hidden units. default=100
        :rtype: None
        """
        # Glorot initialization variables for use in relu elements
        glorot_hx = np.sqrt(2/(n_in + n_hidden))
        glorot_hh = np.sqrt(2/(n_hidden + n_hidden))

        # Member variables that need to be a part of the RNN
        # Weights
        self.W_hh = np.random.normal(0, glorot_hh, size=(n_hidden, n_hidden))  # He initialization

        # Initialize to identity matrix
        self.W_hh = np.eye(n_hidden)  # Initialize the hidden state weight matrix to identity

        self.W_hx = np.random.normal(0, glorot_hx, size=(n_hidden, n_in))  # He initialization

        # Weight derivatives
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hx = np.zeros_like(self.W_hx)

        # Biases
        self.bh = np.zeros(shape=(n_hidden, 1))
        # Bias derivatives
        self.dbh = np.zeros_like(self.bh)

        # Hidden states and hidden state derivative
        self.hs = [np.zeros(shape=(n_hidden, 1))]

        # storing of input
        self.xs = []
        self.dxs = []

        # Stored ReLU positions
        self.relu_idx = []

    def __str__(self):
        """
        Function for basic string representation of the class

        :return: class name as string
        """
        return "BasicRNNCell"

    def fprop(self, data):
        """
        Forward propagation using a relu nonlinearity.

        :param data: dims = (nSamples, nIn) encoded input as a numpy array. The dimensions are expanded along axis 1.
        :return: dims = (nSamples, nHidden) the output as generated by the network
        """
        n_samples, n_in = data.shape


        for i in range(n_samples):
            # Append the input vector for later reusability
            self.xs.append(data[i:i+1].T)

            h = np.dot(self.W_hx, self.xs[-1]) + np.dot(self.W_hh, self.hs[-1]) + self.bh  # Comp of linear components
            h = np.clip(h, 0, None)  # ReLU non-linearity
            self.relu_idx.append(h == 0)
            self.hs.append(h)  # Storing of hidden layer values



        # Ignore the initial hidden state in the output
        return np.squeeze(np.array(self.hs[1:]))

    def bprop(self, dys):
        """
        Backpropagation routine which accumulates the derivatives of the weights.
        Additionally it should be called once for every time that fprop has been called

        :param dys: dims = nSamples x (nVocab, 1) output gradient along which to compute the weight gradients
        :return: dxs: dims = nSamples x (nInp, 1) input gradient which is passed back to previous layers
        """
        dh = None
        for dy in reversed(dys):
            if dh is None:
                dh = dy.copy()
            else:
                dh = np.dot(self.W_hh.T, dh) + dy
            dh[self.relu_idx[-1]] = 0  # ReLU derivative

            # Remove old index matrix and hidden state
            self.relu_idx.pop(-1)
            self.hs.pop(-1)

            # Backpropagate
            # In forward pass: h = np.dot(self.W_hx, self.xs[-1]) + np.dot(self.W_hh, self.hs[-1]) + self.bh
            self.dW_hh += np.dot(dh, self.hs[-1].T)
            self.dW_hx += np.dot(dh, self.xs[-1].T)  # Bprop into h_raw
            self.dbh += dh
            self.dxs.insert(0, np.dot(self.W_hx.T, dh))

            # Remove the old x values
            self.xs.pop(-1)
            pass

        return self.dxs

    def clip_gradients(self):
        """
        Clip the gradients so that the norm remains smaller than 5

        :return: None
        """
        for grad in [self.dW_hx, self.dbh, self.dW_hh]:
            threshhold = 5
            cm.clip_gradient(grad, threshhold)

    def update_weights(self, update_fun=None, eta=0.001):
        """
        Update the weights according to some update rule. If no update rule is supplied, simple gradient descent is used

        :param update_fun: function which takes two parameters (weight, derivative) and computes the updated weight
        as return value.
        :param eta: learning rate
        :return: None
        """

        if update_fun is None:
            def update_fun(x, y):
                return -eta*y
        for weight, der in zip([self.W_hh, self.W_hx, self.bh],
                               [self.dW_hh, self.dW_hx, self.dbh]):
            weight += update_fun(weight, der)
            der *= 0

        self.dxs = []

    def get_params(self):
        """
        Generator for returning parameter and parameter derivative pair

        :return: tuple in the form (param, dparam)
        """
        for weight, der, name in zip([self.W_hh, self.W_hx, self.bh],
                                     [self.dW_hh, self.dW_hx, self.dbh],
                                     ["W_hh", "W_hx", "bh"]):
            yield (weight, der, name)

    def clear_stored_states(self):
        """
        Function that removes all previously stored hidden states and accumulated derivatives

        :return: None
        """
        # Reset hidden state, stored inputs, stored output grads
        self.hs = [self.hs[0]]
        self.xs = []
        self.dxs = []

    def clear_stored_derivs(self):
        """
        Function that removes all previously stored derivatives

        :return: None
        """
        for _, der, _ in self.get_params():
            der *= 0
