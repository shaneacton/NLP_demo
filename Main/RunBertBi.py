from Main import train_BERT

max_batches = -1
num_epochs = 15

print("BERT:")

print("bidirectional:")
train_BERT.train(num_epochs, max_batches=max_batches, bidirectional=True)
