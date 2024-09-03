import numpy as np
import scipy
from scipy.special import erf


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):

        dLdZ = dLdA * self.A * (1 - self.A)

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

        return self.A
    def backward(self, dLdA):

        dLdZ = dLdA * (1 - self.A**2)

        return dLdZ

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):

        dLdZ = np.where(self.A > 0, dLdA, 0)

        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """

    def forward(self, Z):

        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))

        return self.A

    def backward(self, dLdA):

        dAdZ = (0.5 * (1 + erf(self.Z / np.sqrt(2))) +
                 self.Z * (1 / np.sqrt(2 * np.pi)) * np.exp(-self.Z ** 2 / 2))

        dLdZ = dLdA * dAdZ

        return dLdZ




class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        # Z_shifted for numerical stability. Ensure max is subtracted row-wise
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        self.A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=1, keepdims=True) # TODO

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = dLdA.shape[0] # TODO
        C = dLdA.shape[1] # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros(C, C) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    J[m,n] = np.where(m == n,
                                      self.A[i, m] * (1 - self.A[i, m]),
                                      -self.A[i, m] * self.A[i, n]) # TODO

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i,:] @ J # TODO

        return dLdZ