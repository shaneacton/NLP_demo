from Main import train_BERT, train_GloVe

max_batches = 1
num_epochs = 1

print("BERT:")

print("bidirectional:")
train_BERT.train(num_epochs, max_batches = max_batches, bidirectional= True)

print("unidirectional:")
train_BERT.train(num_epochs, max_batches = max_batches, bidirectional= False)

print("Glove:")

print("bidirectional:")
train_GloVe.train(num_epochs,max_batches = max_batches, bidirectional= True)

print("unidirectional:")
train_GloVe.train(num_epochs,max_batches = max_batches, bidirectional= False)