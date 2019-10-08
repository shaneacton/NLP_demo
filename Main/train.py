from DataProcessor import IMDB
from LSTM.LSTM import LSTM
import torch
import torch.optim as optim

from DataProcessor import ELMo, GloVe

model = LSTM(300, 200, 2, True, 0.05)

train, test, vocab = IMDB.load_tokenised_imdb()
vocab = vocab.vocab

for i, val in enumerate(train):
    # for each batch
    batch, label = val
    input = []
    for current_words in batch:
        # for each word batch group
        # each current words is a 1*batch vec

        print("current words size:", current_words.size())

        words = [vocab.itos[w.item()].lower() for w in current_words]

        print(words)
        vectors = [GloVe.get_glove_embeddings(word) for word in words]

        try:
            vectors = torch.stack(vectors, dim=0)
            out = model(vectors)
        except:
            for vec in vectors:
                print("crash:", (vec.size()))
            for current_words in words:
                print("crash:", current_words)

        # print(vectors)

    break
