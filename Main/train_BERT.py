from DataProcessor import BERT
from DataProcessor.BERT import BertEmbedder
from LSTM.BERT_LSTM import LSTM

import torch
import torch.optim as optim
import torch.nn as nn

#main file in charge of training and evaluating the BERT LSTM
#also in charge of loading data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()

batch_size = 4

train_data, test_data = BERT.prepare_data_bert(batch_size=batch_size)
print("loaded BERT imdb")


def train(num_epochs, embedding_dims=768, bidirectional=True, max_batches=-1):
    model = LSTM(embedding_dims, 150, 1, 1, bidirectional, 0)
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    bert_embedder = BertEmbedder().cuda()

    for e in range(num_epochs):
        epoch_loss = 0
        model.train()

        for i, val in enumerate(train_data):
            # for each batch
            optimizer.zero_grad()
            # print(val)
            token_ids, masks, labels = tuple(t.to(device) for t in val)
            # print("tokens:", token_ids.shape)
            embedded_batch = bert_embedder(token_ids.to(torch.long), masks)
            # print(" before embedding shape:",embedded_batch.shape)
            embedded_batch = embedded_batch.transpose(0, 1)

            output = model(embedded_batch)
            loss = criterion(output.squeeze(), labels.float().squeeze())
            # print("out:",out)
            # print("label:",label.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if max_batches != -1 and i + 1 >= max_batches:
                break

        epoch_loss /= (min(max_batches, len(train_data))) if max_batches != -1 else len(train_data)
        print("epoch:", e, "loss:", epoch_loss)
        print("test acc:", evaluate(model, test_data))


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    # print('preds', preds)
    # print('y', y)
    rounded_preds = torch.round(preds)
    # print('rounded', rounded_preds)
    correct = (rounded_preds == y).float()  # convert into float for division
    # print('correct', correct)
    acc = correct.sum() / correct.numel()
    # print('acc', acc)
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

            predictions = model(embedded_batch)

            acc = binary_accuracy(predictions.squeeze(), labels.squeeze())

            epoch_acc += acc.item()

    return epoch_acc / len(test)


if __name__ == "__main__":
    train_data()
