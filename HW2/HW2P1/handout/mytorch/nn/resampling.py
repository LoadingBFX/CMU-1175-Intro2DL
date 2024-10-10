import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = (input_width - 1) * self.upsampling_factor + 1
        Z = np.zeros((batch_size, in_channels, output_width))

        for i in range(input_width):
            Z[:, :, i * self.upsampling_factor] = A[:, :, i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width - 1) // self.upsampling_factor + 1
        dLdA = np.zeros((batch_size, in_channels, input_width))

        for i in range(input_width):
            dLdA[:, :, i] = dLdZ[:, :, i * self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        self.input_width = input_width
        Z = A[:, :, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width =  self.input_width
        dLdA = np.zeros((batch_size, in_channels, input_width))
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA




class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = (input_height - 1) * self.upsampling_factor + 1
        output_width = (input_width - 1) * self.upsampling_factor + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # batch_size, in_channels, output_height, output_width = dLdZ.shape
        # input_height = (output_height - 1) // self.upsampling_factor + 1
        # input_width = (output_width - 1) // self.upsampling_factor + 1
        # dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        #
        # for i in range(input_height):
        #     for j in range(input_width):
        #         dLdA[:, :, i, j] = dLdZ[:, :, i * self.upsampling_factor, j * self.upsampling_factor]

        return dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.input_height = None
        self.input_width = None
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape
        # output_height = input_height // self.downsampling_factor
        # output_width = input_width // self.downsampling_factor
        self.input_width = input_width
        self.input_height = input_height
        # Z = A[:, :, :output_height * self.downsampling_factor:self.downsampling_factor,
        #     :output_width * self.downsampling_factor:self.downsampling_factor]

        return A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape

        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))

        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
