import torch
import torch.nn

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
        #  version-1 35 CONTEXT 基线 baseline
        #  epoch115 val loss 0.39923486094606125 Val Acc: 86.7901% kaggle 86.71%
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(128, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 3072),
        #     torch.nn.BatchNorm1d(3072),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(3072, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, output_size)
        # )
#换 激活函数
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.ReLU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ReLU(),
        #
        #     torch.nn.Linear(128, output_size)
        # )
#  加drop out
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 2048),
#             torch.nn.BatchNorm1d(2048),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(2048, 2048),
#             torch.nn.BatchNorm1d(2048),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(2048, 2048),
#             torch.nn.BatchNorm1d(2048),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(2048, 1024),
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(1024, 1024),
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(1024, 1024),
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(1024, 1024),
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(1024, 512),
#             torch.nn.BatchNorm1d(512),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(512, 512),
#             torch.nn.BatchNorm1d(512),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(512, 256),
#             torch.nn.BatchNorm1d(256),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(256, 256),
#             torch.nn.BatchNorm1d(256),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(256, 128),
#             torch.nn.BatchNorm1d(128),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(128, 128),
#             torch.nn.BatchNorm1d(128),
#             torch.nn.GELU(),
#             torch.nn.Dropout(0.3),
#
#             torch.nn.Linear(128, output_size)
#         )




        # version1.1  add more layers
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #
        #     torch.nn.Linear(64, output_size)
        # )

        #version 1.2 wider
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 3072),
        #     torch.nn.BatchNorm1d(3072),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(3072, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     # torch.nn.Linear(1024, 1024),
        #     # torch.nn.BatchNorm1d(1024),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.2),
        #     #
        #     # torch.nn.Linear(1024, 1024),
        #     # torch.nn.BatchNorm1d(1024),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # wider 2
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 3072),
        #     torch.nn.BatchNorm1d(3072),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     # torch.nn.Linear(3072, 3072),
        #     # torch.nn.BatchNorm1d(3072),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(3072, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     # torch.nn.Linear(1024, 1024),
        #     # torch.nn.BatchNorm1d(1024),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.2),
        #     #
        #     # torch.nn.Linear(1024, 1024),
        #     # torch.nn.BatchNorm1d(1024),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     # torch.nn.Linear(512, 512),
        #     # torch.nn.BatchNorm1d(512),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.2),
        #     #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #
        #     # torch.nn.Linear(256, 256),
        #     # torch.nn.BatchNorm1d(256),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #     #
        #     # torch.nn.Linear(128, 128),
        #     # torch.nn.BatchNorm1d(128),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.1),
        #     #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.1),
        #     #
        #     # torch.nn.Linear(64, 64),
        #     # torch.nn.BatchNorm1d(64),
        #     # torch.nn.GELU(),
        #     # torch.nn.Dropout(0.1),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(64, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(128, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 4096),
        #     torch.nn.BatchNorm1d(4096),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(4096, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, output_size)
        # )


        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.35),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.35),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.35),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.35),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.35),
        #
        #     torch.nn.Linear(2048, output_size)
        # )

        # deeper
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.4),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, output_size)
        # )

        # test1
        # self.model = torch.nn.Sequential(
        #
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #
        #     torch.nn.Linear(128, output_size)
        # )

        # test2
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(128, output_size)
        # )

        # test3 2222
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # test4
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        #
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, output_size)
        # )
        #

    # 111111  86.83
    #     self.model = torch.nn.Sequential(
    #         torch.nn.Linear(input_size, 2048),
    #         torch.nn.BatchNorm1d(2048),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.3),
    #
    #         torch.nn.Linear(2048, 2048),
    #         torch.nn.BatchNorm1d(2048),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.3),
    #
    #         torch.nn.Linear(2048, 2048),
    #         torch.nn.BatchNorm1d(2048),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.3),
    #
    #         torch.nn.Linear(2048, 1024),
    #         torch.nn.BatchNorm1d(1024),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.3),
    #
    #         torch.nn.Linear(1024, 1024),
    #         torch.nn.BatchNorm1d(1024),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.2),
    #
    #         torch.nn.Linear(1024, 1024),
    #         torch.nn.BatchNorm1d(1024),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.2),
    #
    #         torch.nn.Linear(1024, 1024),
    #         torch.nn.BatchNorm1d(1024),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.2),
    #
    #         torch.nn.Linear(1024, 512),
    #         torch.nn.BatchNorm1d(512),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.2),
    #
    #         torch.nn.Linear(512, 512),
    #         torch.nn.BatchNorm1d(512),
    #         torch.nn.GELU(),
    #         torch.nn.Dropout(0.2),
    #
    #         torch.nn.Linear(512, 256),
    #         torch.nn.BatchNorm1d(256),
    #         torch.nn.GELU(),
    #         # torch.nn.Dropout(0.18),
    #
    #         torch.nn.Linear(256, 256),
    #         torch.nn.BatchNorm1d(256),
    #         torch.nn.GELU(),
    #
    #         torch.nn.Linear(256, 128),
    #         torch.nn.BatchNorm1d(128),
    #         torch.nn.GELU(),
    #
    #         torch.nn.Linear(128, 128),
    #         torch.nn.BatchNorm1d(128),
    #         torch.nn.GELU(),
    #
    #         torch.nn.Linear(128, 64),
    #         torch.nn.BatchNorm1d(64),
    #         torch.nn.GELU(),
    #
    #         torch.nn.Linear(64, 64),
    #         torch.nn.BatchNorm1d(64),
    #         torch.nn.GELU(),
    #
    #         torch.nn.Linear(64, output_size)
    #     )


        #
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 4096),
        #     torch.nn.BatchNorm1d(4096),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(4096, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     # torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(128, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(64, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #
        #     torch.nn.Linear(64, output_size)
        # )

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.3),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, 128),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(128, output_size)
        # )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.22),

            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.GELU(),
            torch.nn.Dropout(0.24),

            torch.nn.Linear(2048, 3000),
            torch.nn.BatchNorm1d(3000),
            torch.nn.GELU(),
            torch.nn.Dropout(0.28),

            torch.nn.Linear(3000, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.GELU(),
            torch.nn.Dropout(0.24),

            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),


            torch.nn.Linear(256, output_size)
        )



        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(2048, 3072),
        #     torch.nn.BatchNorm1d(3072),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.25),
        #
        #     torch.nn.Linear(3072, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.22),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 924),
        #     torch.nn.BatchNorm1d(924),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.18),
        #
        #     torch.nn.Linear(924, output_size)
        # )
        # version-2
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(256, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #
        #     torch.nn.Linear(64, output_size)
        # )
        # Params:    19,990.00K
        # self.model = torch.nn.Sequential(
        #
        #     torch.nn.Linear(input_size, 4096),
        #     torch.nn.BatchNorm1d(4096),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(4096, 2048),
        #     torch.nn.BatchNorm1d(2048),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(2048, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(1024, 256),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(256, 84),
        #     torch.nn.BatchNorm1d(84),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.5),
        #
        #     torch.nn.Linear(84, output_size)
        # )
        #




    def forward(self, x):

        # #version -0
        out = self.model(x)


        return out