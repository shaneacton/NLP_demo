# https://towardsdatascience.com/bert-to-the-rescue-17671379687f

import numpy as np
import random as rn
import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torchnlp.datasets import imdb_dataset
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Bert Model
class BertEmbedder(nn.Module):
    def __init__(self):
        super(BertEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, tokens, masks=None):
        embedding, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        return embedding


def prepare_data_bert(batch_size):
    """:returns train and test loader for the IMDB dataset formatted correctly for BERT, each item in the dataset is in
    the form (token_ids, masks, labels)"""
    print('Loading IMDB data...')

    train_data, test_data = imdb_dataset(train=True, test=True)
    rn.shuffle(train_data)
    rn.shuffle(test_data)
    train_data = train_data[:1000]
    test_data = test_data[:100]

    train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
    test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))

    print('Tokenizing for BERT')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], train_texts))
    test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], test_texts))
    print(list(map(tokenizer.convert_tokens_to_ids, train_tokens))[0])
    train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=512,
                                     truncating='post', padding='post', dtype='int')
    # print(train_tokens_ids[0])
    test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)), maxlen=512,
                                    truncating='post',
                                    padding='post', dtype='int')

    train_y = np.array(np.array(train_labels) == 'pos', dtype=np.uint8)
    test_y = np.array(np.array(test_labels) == 'pos', dtype=np.uint8)
    train_y.shape, test_y.shape, np.mean(train_y), np.mean(test_y)

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

    train_tokens_tensor = torch.tensor(train_tokens_ids)
    train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()

    test_tokens_tensor = torch.tensor(test_tokens_ids)
    test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()

    train_masks_tensor = torch.tensor(train_masks)
    test_masks_tensor = torch.tensor(test_masks)

    train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader


def prep_sentence_bert(s, tokenizer):
    tokens = list(['[CLS]'] + tokenizer.tokenize(s)[:510] + ['[SEP]'])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor(token_ids)


if __name__ == '__main__':
    print(prep_sentence_bert('test sentence hello hello hi',
                             BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)))
