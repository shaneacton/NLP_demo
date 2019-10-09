import numpy as np

from DataProcessor import BERT
from DataProcessor.BERT import BertEmbedder
from LSTM.LSTM import LSTM
from LSTM.Emb_LSTM import Emb_LSTM

import torch
import torch.optim as optim
import torch.nn as nn

# from DataProcessor import ELMo, GloVe
from torchtext import data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCEWithLogitsLoss()

# train, test, vocab = IMDB.load_tokenised_imdb()
# vocab = vocab.vocab

# print("loaded imdb")
max_batches = 15000
batch_size = 2

# epoch: 0 loss: 0.7446197867393494
# epoch: 1 loss: 0.7090064883232117
# epoch: 2 loss: 0.7069557905197144
# epoch: 3 loss: 0.6838464736938477

def train_model(embedding_dims = 768):
    model = LSTM(embedding_dims, 150, 1, 1, False, 0).cuda()
    optimizer = optim.Adam(model.parameters())
    model.train()

    train, test = BERT.prepare_data_bert(batch_size=batch_size)

    print("loaded imdb")

    bert_embedder = BertEmbedder().cuda()

    for e in range(15):
        epoch_loss = 0

        for i, val in enumerate(train):
            # for each batch
            optimizer.zero_grad()
            # print(val)
            token_ids, masks, labels = tuple(t.to(device) for t in val)
            # print("tokens:", token_ids.shape)
            embedded_batch = bert_embedder(token_ids.to(torch.long), masks)
            # print(" before embedding shape:",embedded_batch.shape)
            embedded_batch = embedded_batch.transpose(0,1)
            # print(" after embedding shape:",embedded_batch.shape)

            # for current_words in batch:
            #     # for each word batch group
            #     # each current words is a 1*batch vec
            #
            #     # print("current words size:", current_words.size())
            #
            #     words = [vocab.itos[w.item()].lower() for w in current_words]
            #
            #     # print(words)
            #     vectors = []
            #     for word in words:
            #         # word = word.replace("'","")
            #         if(len(word) == 0):
            #             word = "<unk>"
            #         #
            #         # print(word)
            #         # print("glove size: ", GloVe.get_glove_embeddings(word).shape)
            #
            #         vectors.append(GloVe.get_glove_embeddings(word))
            #     batch_size = len(vectors)
            #     vectors = torch.cat(vectors, dim=0).reshape(1 ,batch_size,embedding_dims)
            #     inputs.append(vectors)
                # print("input:" , vectors.shape)
            # inputs = torch.cat(inputs, dim=0).reshape(-1, batch_size, embedding_dims)
            output, hid = model(embedded_batch)
            loss = criterion(output.squeeze(), labels.float().squeeze())
            # print("out:",out)
            # print("label:",label.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i + 1 >= max_batches:
                break

        epoch_loss = epoch_loss/ (min(max_batches,len(train)))
        print("epoch:" , e , "loss:",epoch_loss)


#
# def train2():
#     TEXT = data.Field(tokenize='spacy', include_lengths=True)
#     LABEL = data.LabelField(dtype=torch.float)
#
#     TEXT.build_vocab(train,
#                      vectors="glove.6B.100d",
#                      unk_init=torch.Tensor.normal_)
#
#     LABEL.build_vocab(train)
#
#     print("len vocab:",len(vocab))
#     model = Emb_LSTM(len(vocab),300, 150, 1, 1, False, 0, pad_idx=1)
#     optimizer = optim.Adam(model.parameters())
#     model.train()
#
#     print(vocab.vectors)
#
#     pretrained_embeddings = vocab.vocab.vectors
#     print(pretrained_embeddings.shape)
#
#     model.embedding.weight.data.copy_(pretrained_embeddings)


