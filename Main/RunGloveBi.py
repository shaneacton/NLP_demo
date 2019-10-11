from Main import train_GloVe
max_batches = -1
num_epochs = 15

print("Glove:")

print("bidirectional:")
train_GloVe.train(num_epochs, max_batches=max_batches, bidirectional=True)