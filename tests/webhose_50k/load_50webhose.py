import sys
sys.path.insert(0, ".")  # nopep8
import os
import pandas as pd
import json
from tqdm import tqdm
from random import shuffle

from lda2vec.nlppipe import Preprocessor

# Data directory
raw_data_dir ="tests/webhose_50k/data/raw_data"

#Pre-Trained Glove Embedding Location
glove_path = "tests/webhose_50k/glove_embeddings/glove.6B.300d.txt"

# Preprocessed data location
clean_data_dir = "tests/webhose_50k/data/clean_data/v3"

# Should we load pretrained embeddings from file
load_embeds = True

if not os.path.exists(clean_data_dir):
    os.makedirs(clean_data_dir)

if not os.path.exists("{}/train.csv".format(clean_data_dir)):
    clean_row_list = []
    for sub_dir in os.listdir(raw_data_dir):
        print("Processing Sub Dir: {}".format(sub_dir))
        for i, doc in tqdm(enumerate(os.listdir("{}/{}".format(raw_data_dir, sub_dir)))):
            with open("{}/{}/{}".format(raw_data_dir, sub_dir, doc)) as fp:
                item = json.load(fp)
                clean_row_list.append({
                    "texts": item["text"].replace("\n", " ")
                })

    shuffle(clean_row_list)

    train_set = clean_row_list[:5000]
    test_set = clean_row_list[45000:]

    # Create a df for Pre processor to consume
    train_df = pd.DataFrame(train_set)
    test_df = pd.DataFrame(test_set)

    train_df.to_csv("{}/train.csv".format(clean_data_dir), index=False)
    test_df.to_csv("{}/test.csv".format(clean_data_dir), index=False)
else:
    train_df = pd.read_csv("{}/train.csv".format(clean_data_dir))
    test_df = pd.read_csv("{}/test.csv".format(clean_data_dir))

print("Number of Articles in Training Set: {}".format(str(len(train_df.index))))
print("Number of Articles in Test Set: {}".format(str(len(test_df.index))))

# Initialize a preprocessor
P = Preprocessor(train_df, "texts", max_features=30000, maxlen=10000, min_count=20)

# Run the preprocessing on your dataframe
P.preprocess()

# Load embeddings from file if we choose to do so
if load_embeds:
    # Load embedding matrix from file path - change path to where you saved them
    embedding_matrix = P.load_glove(glove_path)
else:
    embedding_matrix = None

# Save data to data_dir
P.save_data(clean_data_dir, embedding_matrix=embedding_matrix)
