import sys
sys.path.insert(0, ".")  # nopep8
import os
from lda2vec import utils, model
import numpy as np

# Path to preprocessed data
clean_data_dir = "tests/webhose_50k/data/clean_data"

#Model Path
model_dir = "tests/webhose_50k/model/v1"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
(idx_to_word, word_to_idx, freqs, pivot_ids,
 target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(clean_data_dir, load_embed_matrix=load_embeds)

# Number of unique documents
num_docs = doc_ids.max() + 1
# Number of unique words in vocabulary (int)
vocab_size = len(freqs)
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = 20
# Amount of iterations over entire dataset
num_epochs = 10
# Batch size - Increase/decrease depending on memory usage
batch_size = 500
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 0
# Pretrained embeddings value
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True


# Initialize the model
m = model(num_docs,
          vocab_size,
          num_topics,
          embedding_size=embed_size,
          pretrained_embeddings=pretrained_embeddings,
          freqs=freqs,
          batch_size = batch_size,
          save_graph_def=save_graph,
          fixed_words=True,
          logdir=model_dir)

# Train the model
m.train(pivot_ids,
        target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        idx_to_word=idx_to_word,
        switch_loss_epoch=switch_loss_epoch)

# Visualize topics with pyldavis
trained_model_data = utils.generate_ldavis_data(clean_data_dir, m, idx_to_word, freqs, vocab_size)

np.savez("{}/model_params".format(model_dir), **trained_model_data)