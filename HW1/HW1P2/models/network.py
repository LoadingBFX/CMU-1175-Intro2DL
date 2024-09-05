import torch
import torch.nn
import torch.nn as nn

class Network(torch.nn.Module):

    def __init__(self, input_size, output_size):

        super(Network, self).__init__()
        # version-0
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )
        # self.model = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 1D卷积层
        #     nn.ReLU(),  # 激活函数
        #     nn.MaxPool1d(kernel_size=2, stride=2),  # 池化层
        #     nn.Flatten(),  # 展平为全连接层的输入
        #     nn.Linear(16 * (input_size // 2), 256),  # 全连接层1
        #     nn.ReLU(),  # 激活函数
        #     nn.Linear(256, output_size)  # 全连接层2 (输出层)
        # )

        # version-1
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, output_size)
        # )

        # version-2
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(1024),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(512, 256),
        #     torch.nn.GELU(),
        #     torch.nn.BatchNorm1d(256),
        #     torch.nn.Dropout(0.2),
        #
        #     torch.nn.Linear(256, output_size)
        # )

    def forward(self, x):
        # version -0
        out = self.model(x)

        # x = x.unsqueeze(1)
        # out = self.model(x)

        return out