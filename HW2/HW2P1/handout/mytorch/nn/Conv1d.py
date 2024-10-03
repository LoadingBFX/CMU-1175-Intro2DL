# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(output_size):
                    Z[i, j, k] = np.sum(A[i, :, k:k + self.kernel_size] * self.W[j, :, :]) + self.b[j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        _, in_channels, input_size = self.A.shape

        # Initialize gradients
        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)
        dLdA = np.zeros_like(self.A)

        # Compute gradients
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_size):
                    self.dLdW[j, :, :] += dLdZ[i, j, k] * self.A[i, :, k:k + self.kernel_size]
                    self.dLdb[j] += dLdZ[i, j, k]
                    dLdA[i, :, k:k + self.kernel_size] += dLdZ[i, j, k] * self.W[j, :, :]

        return dLdA




class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Line 1: Pad with zeros
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), 'constant')

        # Line 2: Conv1d forward
        Z = self.conv1d_stride1.forward(A_padded)

        # Line 3: Downsample1d forward
        Z = self.downsample1d.forward(Z)

        return Z


    def backward(self, dLdZ):
            """
            Argument:
                dLdZ (np.array): (batch_size, out_channels, output_size)
            Return:
                dLdA (np.array): (batch_size, in_channels, input_size)
            """
            # Line 1: Downsample1d backward
            dLdZ = self.downsample1d.backward(dLdZ)

            # Line 2: Conv1d backward
            dLdA = self.conv1d_stride1.backward(dLdZ)

            # Line 3: Unpad
            dLdA = dLdA[:, :, self.pad:dLdA.shape[2]-self.pad]

            return dLdA

