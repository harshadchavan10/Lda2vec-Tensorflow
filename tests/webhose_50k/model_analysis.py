import sys
sys.path.insert(0, ".")  # nopep8
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# Path to preprocessed data
clean_data_dir = "tests/webhose_50k/data/clean_data"

#Model Path
model_dir = "tests/webhose_50k/model/v1"

#Pre-Trained Glove Embedding Location
glove_path = "tests/webhose_50k/glove_embeddings/glove.6B.300d.txt"

#Load word to ID mapping
with open("{}/word_to_idx.pickle".format(clean_data_dir), "rb") as w2i_in:
    word_to_idx = pickle.load(w2i_in)

print(word_to_idx["1.75"])

exit()

#Load Initial Embedding matrix
embed_matrix = np.load("{}/embedding_matrix.npy".format(clean_data_dir))

#norm = np.sqrt(np.sum(embed_matrix ** 2, axis=1))
#embed_matrix = np.transpose(np.transpose(embed_matrix) / norm)

# Word vectors after training
npz = np.load(open("{}/model_params.npz".format(model_dir), 'rb'), allow_pickle=True)
trained_model_params = {k: v for (k, v) in npz.items()}
trained_word_vectors = trained_model_params["word_vectors"]

#norm = np.sqrt(np.sum(trained_word_vectors ** 2, axis=1))
#trained_word_vectors = np.transpose(np.transpose(trained_word_vectors) / norm)

# Load Glove word vectors
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
glove_embeddings = dict(get_coefs(*o.split(" ")) for o in open(glove_path))

print(len(word_to_idx.keys()))
print(embed_matrix.shape)
print(trained_word_vectors.shape)
print(len(glove_embeddings.keys()))

sim1 = cosine_similarity([embed_matrix[word_to_idx["king"]]],
                        [trained_word_vectors[word_to_idx["king"]]])
print(sim1)

sim2 = cosine_similarity([embed_matrix[word_to_idx["king"]]],
                        [glove_embeddings["king"]])
print(sim2)

sim3 = cosine_similarity([glove_embeddings["king"]],
                        [trained_word_vectors[word_to_idx["king"]]])
print(sim3)

sim4 = cosine_similarity([trained_word_vectors[word_to_idx["country"]]],
                        [trained_word_vectors[word_to_idx["nation"]]])
print(sim4)

all_close_1 = np.allclose(embed_matrix[word_to_idx["king"]], glove_embeddings["king"])
print(all_close_1)

all_close_2 = np.allclose(embed_matrix[word_to_idx["man"]] - embed_matrix[word_to_idx["woman"]],
                          trained_word_vectors[word_to_idx["man"]] - trained_word_vectors[word_to_idx["woman"]])
print(all_close_2)

all_close_3 = np.allclose(trained_word_vectors[word_to_idx["boy"]] - trained_word_vectors[word_to_idx["girl"]],
                          trained_word_vectors[word_to_idx["man"]] - trained_word_vectors[word_to_idx["woman"]])
print(all_close_3)

sim5 = cosine_similarity([trained_word_vectors[word_to_idx["boy"]] - trained_word_vectors[word_to_idx["girl"]]],
                          [trained_word_vectors[word_to_idx["man"]] - trained_word_vectors[word_to_idx["woman"]]])
print(sim5)

all_close_4 = np.allclose(embed_matrix[word_to_idx["boy"]] - embed_matrix[word_to_idx["girl"]],
                          embed_matrix[word_to_idx["man"]] - embed_matrix[word_to_idx["woman"]])
print(all_close_4)

sim6 = cosine_similarity([embed_matrix[word_to_idx["boy"]] - embed_matrix[word_to_idx["girl"]]],
                          [embed_matrix[word_to_idx["man"]] - embed_matrix[word_to_idx["woman"]]])
print(sim6)

u = embed_matrix[word_to_idx["man"]] - embed_matrix[word_to_idx["woman"]]
v = trained_word_vectors[word_to_idx["man"]] - trained_word_vectors[word_to_idx["woman"]]
angle_cos = np.dot(u,v)/norm(u)/norm(v)
angle = np.arccos(np.clip(angle_cos, -1, 1))
print(angle)

del glove_embeddings
del trained_word_vectors
del embed_matrix