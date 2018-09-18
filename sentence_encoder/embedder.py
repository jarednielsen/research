# embedder.py
"""
Embeds Bednar's conference talk sentences using Google's USEv2.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.tensorboard.plugins import projector


"""
We should have a single file of plaintext sentences
and a single file saved numpy array of sentences
and a single file saved numpy array of embeddings.

This module assumes the data is preprocessed and embedded.
"""

# Change this to work with different datasets
PREFIX = 'data/bednar_sentences'
PLAINTEXT_FILE = '{}.txt'.format(PREFIX)
TEXTARRAY_FILE = '{}.npy'.format(PREFIX)
EMBED_FILE = '{}_embeddings.npy'.format(PREFIX)

# TensorBoard paths
PATH = os.getcwd()
LOG_DIR = PATH + '/tensorboard/bednar'
METADATA_FILE = '{}/bednar_sentences_labels.tsv'.format(LOG_DIR)


def preprocess():
    """
    Assumes `PLAINTEXT_FILE` has been populated, one sentence per line,
    and fills `TEXTARRAY_FILE` and `EMBED_FILE` as np.array.
    """
    with open(PLAINTEXT_FILE) as fp:
        sentences = fp.readlines()
        sentences = np.array(sentences)
    
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    with tf.Session() as sess:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings_op = embed(sentences)
        embeddings = session.run(embeddings_op)

    np.save(TEXTARRAY_FILE, sentences)
    np.save(EMBED_FILE, embeddings)

def load():
    """
    Helper method for visualizations.
    """
    sentences = np.load(TEXTARRAY_FILE)
    embeddings = np.load(EMBED_FILE)
    return sentences, embeddings

def display_embeddings_similarity_matrix():
    sentences, embeddings = load() # sentences: (92,), embeddings: (92,512)
    corr = np.inner(embeddings, embeddings) # corr: (92,92)
    g = sns.heatmap(corr, vmin=0, vmax=1)
    g.set_title("Bednar's Semantic Similarity")
    plt.show()

def display_embeddings_tsne():
    sentences, embeddings = load()
    embeddings_var = tf.Variable(embeddings)

    with tf.Session() as sess:
        saver = tf.train.Saver([embeddings_var])
        sess.run(embeddings_var.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'sentences.ckpt'))
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings_var.name
        embedding.metadata_path = METADATA_FILE # tsv file with the labels

        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)



if __name__ == "__main__":
    print("Running embedder.py")
    display_embeddings_tsne()
    print("Metadata file:", METADATA_FILE)