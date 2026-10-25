import torch
import torch.nn as nn

class GenomicCNN1D(nn.Module):
    def __init__(self, conv_act, dense_act, init_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv_act = conv_act
        self.dense = nn.Linear(256 * 32, 512)
        self.dense_act = dense_act
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        x = self.pool(self.conv_act(self.conv1(x)))
        x = self.pool(self.conv_act(self.conv2(x)))
        x = self.pool(self.conv_act(self.conv3(x))).flatten(1)
        return torch.sigmoid(self.out(self.dropout(self.dense_act(self.dense(x)))))

class GenomicCNN2D(nn.Module):
    def __init__(self, conv_act, dense_act, init_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_act = conv_act
        self.dense = nn.Linear(256 * 4 * 12, 512)
        self.dense_act = dense_act
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        x = self.pool(self.conv_act(self.conv1(x)))
        x = self.pool(self.conv_act(self.conv2(x)))
        x = self.pool(self.conv_act(self.conv3(x))).flatten(1)
        return torch.sigmoid(self.out(self.dropout(self.dense_act(self.dense(x)))))
