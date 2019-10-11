import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        #standard practice for non contextual word embeddings is to use an embedding module
        #it takes in a matrix of word ids and outputs the associated vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.bidirectional = bidirectional

        if bidirectional:
            #twice as many out features in bidir due to cat of hidden states
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        # pack sequence, this allows the LSTM to ignore the <PAD> token
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        if self.bidirectional:
            # must concat the hidden states from each directions LSTM together in case of bidir
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)
