from Main import train_BERT
from Main import train_GloVe

# runs BERT and GLOVE eval for both uni and bi directional LSTMS

max_batches = -1
num_epochs = 15

print("BERT:")

print("bidirectional:")
train_BERT.train(num_epochs, max_batches=max_batches, bidirectional=True)

print("unidirectional:")
train_BERT.train(num_epochs, max_batches=max_batches, bidirectional=False)

print("Glove:")
#
print("bidirectional:")
train_GloVe.train(num_epochs, max_batches=max_batches, bidirectional=True)
#
print("unidirectional:")
train_GloVe.train(num_epochs, max_batches=max_batches, bidirectional=False)
