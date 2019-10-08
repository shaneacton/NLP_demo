import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # text = [sent len, batch size]
        input = self.dropout(input)
        output, (hidden, cell) = self.rnn(input)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))


        return self.fc(hidden)