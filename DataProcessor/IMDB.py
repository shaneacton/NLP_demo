import torch
from torchtext import data
from torchtext import datasets
import json


path = "../Data/IMDB/"

def save_tokenised_imdb():
    """
        tokenising with spacy is very slow, but neccesary for edge cases
        tokenise once and save speeds up development cycle
    """

    TEXT = data.Field(tokenize='spacy', include_lengths=True, )
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    import random
    train_data, valid_data = train_data.split(random_state = random.seed(0))

    train_examples = [vars(t) for t in train_data]
    test_examples = [vars(t) for t in test_data]

    with open(path + 'train.json', 'w+') as f:
        for example in train_examples:
            json.dump(example, f)
            f.write('\n')

    with open(path + 'test.json', 'w+') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')


def load_tokenised_imdb():
    """load the presaved tokenised IMDB dataset"""

    TEXT = data.Field()
    LABEL = data.LabelField()

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

    train_data, test_data = data.TabularDataset.splits(
        path='.',
        train= path + 'train.json',
        test= path + 'test.json',
        format='json',
        fields=fields
    )

    train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), batch_sizes=(16, 256),
        sort_key=lambda x: len(x.text), device=0)

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    # print(TEXT.vocab.itos[3])

    return train_iter, test_iter, TEXT

def get_imdb_vocab():
    """
    :return: a set of all the words found in the imdb dataset
    """
    #used to narrow down 840 B glove words
    words = set()
    with open(path + 'imdb.vocab', 'r+', encoding="utf8") as f:
        for line in f:
            word = line.replace("\n","")
            # print(word)
            words.add(word)

    return words

if __name__ == "__main__":
    save_tokenised_imdb()
    #get_imdb_vocab()

