import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_size = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_size, output_size))
        max_indices = np.zeros((batch_size, in_channels, output_size, output_size))
        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_size):
                    for l in range(output_size):
                        region = A[i, j, k:k + self.kernel, l:l + self.kernel]
                        Z[i, j, k, l] = np.max(region)
                        max_indices[i, j, k, l] = np.argmax(region)

        self.max_indices = max_indices
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        _, _, input_width, input_height = self.A.shape

        dLdA = np.zeros_like(self.A)

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        idx = int(self.max_indices[i, j, k, l])
                        h_idx, w_idx = np.unravel_index(idx, (self.kernel, self.kernel))
                        dLdA[i, j, k + h_idx, l + w_idx] += dLdZ[i, j, k, l]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_size = input_width - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_size, output_size))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_size):
                    for l in range(output_size):
                        region = A[i, j, k:k + self.kernel, l:l + self.kernel]
                        Z[i, j, k, l] = np.mean(region)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        _, _, input_width, input_height = self.A.shape

        dLdA = np.zeros_like(self.A)
        scaling_factor = 1 / (self.kernel * self.kernel)

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        dLdA[i, j, k:k + self.kernel, l:l + self.kernel] += dLdZ[i, j, k, l] * scaling_factor

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
