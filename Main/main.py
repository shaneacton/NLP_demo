from DataProcessor import ELMo, GloVe
import time

text = "is the only show preserved from the experimental theatre movement " \
       "in New York in the 1960s (the origins of Off Off Broadway)."

print("Elmo:\t\n",ELMo.get_elmo_embeddings(text))
print("GloVe:\t\n",GloVe.get_glove_embeddings(text))

average_over = 10

start = time.time()
for i in range(average_over):
    ELMo.get_elmo_embeddings(text)
print("elmo takes:", ((time.time()-start)/average_over) , "s")

start = time.time()
for i in range(average_over):
    GloVe.get_glove_embeddings(text)
print("glove takes:", ((time.time()-start)/average_over) , "s")


