import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        if bidirectional:
            #twice as many out features in bidir due to cat of hidden states
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

        self.sig = nn.Sigmoid()

    def forward(self, input):
        # text = [sent len, batch size]
        input = self.dropout(input)
        output, (hidden, cell) = self.rnn(input)

        if self.bidirectional:
            # must concat the hidden states from each directions LSTM together in case of bidir
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.sig(self.fc(hidden))
