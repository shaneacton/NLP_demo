# NLP Demo

This repo contains our word vectorization demo for our AI course.

## Development

This demo requires Python 3.

```
$ virtualenv pyenv
$ pyenv/bin/pip install -r requirements.txt
$ mkdir -p Data/IMDB
$ mkdir -p Data/GloVe
```

Then download `test.json`, `train.json` and `imdb.vocab` into `Data/IMDB` and
`glove.imdb_vocab.300d.txt` `Data/GloVe`.

It can then be run by

```
$ pyenv/bin/python -m Main.main
```
