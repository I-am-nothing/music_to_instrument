import torch
from torch import nn
from torch.nn import functional as fn


class Annotater(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["train"]["num_layers"]
        self.hidden_size = config["train"]["hidden_size"]
        self.dropout = config["train"]["dropout"]
        self.n_feats = config["dataset"]["n_feats"]
        self.n_fft = config["dataset"]["n_fft"]

        self.original_sec = config["generate"]["seconds_per_segment"]
        self.changed_sec = config["dataset"]["seconds_per_item"]
        self.train_sample_rate = config["dataset"]["train_sample_rate"]

        self.conv1 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.dense1 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.LazyLinear(int((self.original_sec + self.changed_sec) * 768))
        )

        self.lstm1 = nn.LSTM(
            input_size=int((self.original_sec + self.changed_sec) * 768), hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
            bidirectional=True
        )

        self.dense_position = nn.Linear(self.hidden_size * 2, 2)
        self.dense_status = nn.Linear(self.hidden_size * 2, 1)

    def init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 2, batch_size, hs), torch.zeros(n * 2, batch_size, hs)

    def forward(self, x1, x2, hidden):
        x = torch.cat((x1, x2), dim=3)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = x.unsqueeze(0)
        x, (hn, cn) = self.lstm1(x, hidden)
        x = x.squeeze(0)
        x_pos = self.dense_position(x)
        x_sta = self.dense_status(x)

        return torch.sigmoid(x_pos), torch.sigmoid(x_sta), (hn, cn)
