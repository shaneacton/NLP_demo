
from DataProcessor import BERT
from DataProcessor.BERT import BertEmbedder
from LSTM.LSTM import LSTM

import torch
import torch.optim as optim
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()

max_batches = 15000
batch_size = 1


def train_model(embedding_dims = 768):
    model = LSTM(embedding_dims, 150, 1, 1, False, 0).cuda()
    optimizer = optim.Adam(model.parameters())

    bert_embedder = BertEmbedder().cuda()

    train, test = BERT.prepare_data_bert(batch_size=batch_size)

    print("loaded imdb")


    for e in range(15):
        epoch_loss = 0
        model.train()

        for i, val in enumerate(train):
            # for each batch
            optimizer.zero_grad()
            # print(val)
            token_ids, masks, labels = tuple(t.to(device) for t in val)
            # print("tokens:", token_ids.shape)
            embedded_batch = bert_embedder(token_ids.to(torch.long), masks)
            # print(" before embedding shape:",embedded_batch.shape)
            embedded_batch = embedded_batch.transpose(0,1)

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
        print("test acc:" , evaluate(model, test, ))


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    # print("label:",y,"\npreds:",rounded_preds)
    correct = (rounded_preds == y).float() #convert into float for division
    # print(correct)
    acc = correct.sum() / correct.numel()
    return acc

def evaluate(model, test):
    epoch_acc = 0

    model.eval()
    bert_embedder = BertEmbedder().cuda()

    with torch.no_grad():
        for i, val in enumerate(test):
            token_ids, masks, labels = tuple(t.to(device) for t in val)
            # print("tokens:", token_ids.shape)
            embedded_batch = bert_embedder(token_ids.to(torch.long), masks)
            # print(" before embedding shape:",embedded_batch.shape)
            embedded_batch = embedded_batch.transpose(0, 1)

            predictions, hid = model(embedded_batch)

            acc = binary_accuracy(predictions.squeeze(), labels.squeeze())

            epoch_acc += acc.item()

    return  epoch_acc / len(test)

