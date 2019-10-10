import torch
from torchtext import data
import torch.nn as nn

from LSTM.GloVe_LSTM import RNN
import torch.optim as optim

from torchtext import datasets
import random

import time


BATCH_SIZE = 4


SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)


train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print("loaded and tokenised glove imdb")



train_data, valid_data = train_data.split(random_state = random.seed(SEED))



MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)


criterion = nn.BCEWithLogitsLoss()

criterion = criterion.to(device)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train_epoch(model, iterator, optimizer, criterion, max_batches = -1):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    i=0
    for batch in iterator:
        i+=1
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze()

        loss = criterion(predictions, batch.label.squeeze())

        acc = binary_accuracy(predictions, batch.label.squeeze())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if max_batches!= -1 and i>=max_batches:
            break

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze()

            loss = criterion(predictions, batch.label.squeeze())

            acc = binary_accuracy(predictions, batch.label.squeeze())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def init_model(bidirection):
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    OUTPUT_DIM = 1
    N_LAYERS = 1
    DROPOUT = 0.1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                bidirection,
                DROPOUT,
                PAD_IDX)

    # print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors

    # print("embeddings shape: ", pretrained_embeddings.shape)

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # print("embeddings weight shape:", model.embedding.weight.data.shape)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    return model, optimizer


def train(num_epochs,bidirectional = True, max_batches = -1):

    model, optimizer = init_model(bidirectional)


    best_valid_loss = float('inf')

    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion, max_batches=max_batches)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == "__main__":
    train()