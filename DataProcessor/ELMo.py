import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
# print(tf.__version__)
# print(hub.__version__)

elmo = hub.Module("../Data/ELMo", trainable=False)
session = tf.compat.v1.Session()
session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

def get_elmo_embeddings(text):
    text = text.lower()
    sentences = text.split(".")
    embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]

    # print(embeddings)


    # message_embeddings = session.run(embeddings)
    #
    # embeddings = []
    # print("message embeddings:\n" , message_embeddings)
    #
    # for sentence_embedding in message_embeddings:
    #     sentence_embedding = [sentence_embedding[i] for i in range(len(sentence_embedding))
    #                           if (i == 0 or sentence_embedding[i][0] != sentence_embedding[i-1][0] )]
    #     # print("sentence embeddings:\n", sentence_embedding)
    #
    #     embeddings.extend(sentence_embedding)

    return np.array(embeddings)
