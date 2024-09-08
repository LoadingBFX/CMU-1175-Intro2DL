import torch
import torch.nn
import torch.nn as nn

class Network(torch.nn.Module):

    def __init__(self, input_size, output_size):

        super(Network, self).__init__()
        print("input_size:", input_size)
        print("output_size:", output_size)


        # version-0
        # self.model = torch.nn.Sequential(
        #         torch.nn.Linear(input_size, 2048),
        #         torch.nn.GELU(),
        #         torch.nn.BatchNorm1d(2048),
        #         torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, output_size)
        # )

        self.model = torch.nn.Sequential(

            torch.nn.Linear(input_size, 2048),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(2048, 2048),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(2048, 2048),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(2048, 1024),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.18),

            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.18),

            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(256),
            # torch.nn.Dropout(0.18),

            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(256),

            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(128),
            # torch.nn.Dropout(0.18),

            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(128),

            torch.nn.Linear(128, output_size)
        )


        # version-1
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.Dropout(0.5),
        #
        #
        #
        #     torch.nn.Linear(2048, output_size)
        # )

        # version-2
        # self.layers = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 4096),  # 1988 -> 4096
        #     torch.nn.BatchNorm1d(4096),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(4096, 8192),  # 4096 -> 8192
        #     torch.nn.BatchNorm1d(8192),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(8192, 4096),  # 8192 -> 4096
        #     torch.nn.BatchNorm1d(4096),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(4096, 2048),  # 4096 -> 2048
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(2048, 1024),  # 2048 -> 1024
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(1024, 512),  # 1024 -> 512
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(512, output_size)  # 512 -> 42
        # )

    def forward(self, x):

        # #version -0
        out = self.model(x)


        return out