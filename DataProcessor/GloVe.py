from DataProcessor import IMDB
import numpy as np
import torch
import spacy
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'GloVe') + '/'
nlp = spacy.load("en_core_web_sm")
vectors = {}


def get_glove_embeddings(text):
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        # print(token.text, token.pos_, token.dep_)
        tokens.append(token.text)

    word2vec = get_glove_word2vec()

    embeddings = []
    # print("word2vec:", word2vec.keys())

    for word in tokens:
        if word in word2vec.keys():
            # print("word '",word,"' in word2vec", sep = '')
            embeddings.append(word2vec[word])
        else:
            # print("word '",word,"' not in word2vec", sep = '')
            embeddings.append(np.random.rand(300))

    # print(embeddings)

    return torch.from_numpy(np.array(embeddings)).float()

def get_glove_word2vec():

    if(len(vectors) > 0):
        return vectors

    with open(path + f'glove.imdb_vocab.300d.txt', 'rb') as f:
        for l in f:
            line = l.split()
            word = str(line[0])
            word = word[2:]
            word = word[:-1]
            try:
                vect = np.array(line[1:]).astype(np.float)
                # print(vect)
            except:
                print("error with val", line[1:], "\nfull line:", line)
                continue

            vectors[word] = vect

    return vectors

def save_glove_imdb_vocab():
    lines = []

    imdb_vocab = IMDB.get_imdb_vocab()

    with open(path + f'glove.840B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if(word not in imdb_vocab):
                continue

            lines.append(l.decode())

    with open(path + f'glove.imdb_vocab.300d.txt', 'w+') as f:
        for line in lines:
            f.write(line)



if __name__ == "__main__":
    save_glove_imdb_vocab()
