import sys
sys.path.insert(0, ".")  # nopep8
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# Path to preprocessed data
clean_data_dir = "tests/webhose_50k/data/clean_data/v3"

#Model Path
model_dir = "tests/webhose_50k/model/v3"

#Pre-Trained Glove Embedding Location
glove_path = "tests/webhose_50k/glove_embeddings/glove.6B.300d.txt"

#Load word to ID mapping
with open("{}/word_to_idx.pickle".format(clean_data_dir), "rb") as w2i_in:
    word_to_idx = pickle.load(w2i_in)

#Load Initial Embedding matrix
#embed_matrix = np.load("{}/embedding_matrix.npy".format(clean_data_dir))

# Word vectors after training
npz = np.load(open("{}/model_params.npz".format(model_dir), 'rb'), allow_pickle=True)
trained_model_params = {k: v for (k, v) in npz.items()}

trained_word_vectors = trained_model_params["word_vectors"]
trained_topic_vectors = trained_model_params["topic_vectors"]
trained_doc_vectors = trained_model_params["doc_vectors"]
term_frequency = trained_model_params["term_frequency"]
vocab = trained_model_params["vocab"]
doc_lengths = trained_model_params["doc_lengths"]
topic_term_dists = trained_model_params["topic_term_dists"]
doc_topic_dists = trained_model_params["doc_topic_dists"]


print("Trained Word Vectors Shape: {}".format(trained_word_vectors.shape))
print("Trained Topic Vectors Shape: {}".format(trained_topic_vectors.shape))
print("Trained Document Vectors Shape: {}".format(trained_doc_vectors.shape))
print("Trained Term Freq Shape: {}".format(term_frequency.shape))
print("Trained Vocab Shape: {}".format(vocab.shape))
print("Trained Doc Lengths Shape: {}".format(doc_lengths.shape))
print("Trained Topic Term Dists Shape: {}".format(topic_term_dists.shape))
print("Trained Doc Topic Dists Shape: {}".format(doc_topic_dists.shape))

#print(doc_topic_dists[0])
#print(np.argmax(trained_word_vectors[0]))

dummy_article_word_list = ["dialing","definitions","glasses","redistribute","classifications","esencia","offeror","consenting","personality","representation"]

topic_vectors_norm = np.sqrt(np.sum(trained_topic_vectors ** 2, axis=1))
trained_topic_vectors = np.transpose(np.transpose(trained_topic_vectors) / topic_vectors_norm)

word_vectors_norm = np.sqrt(np.sum(trained_word_vectors ** 2, axis=1))
trained_word_vectors = np.transpose(np.transpose(trained_word_vectors) / word_vectors_norm)

topic = trained_topic_vectors[10]

dummy_document_vector = np.zeros(trained_topic_vectors.shape[1])
for word in dummy_article_word_list:
    dummy_document_vector = dummy_document_vector + trained_word_vectors[word_to_idx[word]]
#dummy_document_vector = dummy_document_vector/len(dummy_article_word_list)

#print(topic)
#print(word)
sim = np.dot(topic, dummy_document_vector)
print(sim)
cosine_sim = cosine_similarity(trained_topic_vectors, [dummy_document_vector])
print(cosine_sim)
print(np.argmax(cosine_sim))
softmax_cosine_sim = np.exp(cosine_sim)/sum(np.exp(cosine_sim))
print(softmax_cosine_sim)
print(np.argmax(softmax_cosine_sim))

within_topic_cosine_sim = cosine_similarity(trained_topic_vectors, trained_topic_vectors)
#print(within_topic_cosine_sim)
#print(np.argmax(within_topic_cosine_sim, axis=1))