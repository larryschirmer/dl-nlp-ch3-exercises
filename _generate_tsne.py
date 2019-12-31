import numpy as np
from keras.layers import Embedding
from _sentiment_doc_2_vec_utils import process_data, tsne_plot


def load_embedding(filename, total_docs, embedding_dim):
    embedding_index = {}
    f = open(filename)
    n = 0

    for line in f:
        values = line.split()
        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        if n:  # skip embedding header line
            embedding_index[word] = coefs

        n += 1

    f.close()

    embedding_matrix = np.zeros((total_docs + 1, embedding_dim))

    for doc_index in range(total_docs):
        embedding_vector = embedding_index.get('doc_' + str(doc_index))

        if embedding_vector is not None:
            embedding_matrix[doc_index] = embedding_vector

    return embedding_matrix


input_filename = "yelp_labelled.csv"
window_size = 3
pretrained_embedding = '_sentiment_doc_2_vec_embed.txt'
legend_filename = "_sentiment_doc_2_vec_legend"
embedding_size = 100

contexts, docids, targets, vocab, labels = process_data(
    input_filename, window_size, output_legend_name=legend_filename)
total_docs = len(docids)
embedding = load_embedding(pretrained_embedding, total_docs, embedding_size)


print("generating tsne 0")
figure_filename0 = "_sentiment_doc_2_vec_tsne2"
tsne_plot(embedding, total_docs, labels,
          figure_name=figure_filename0, max_docs=1000, perplexity=2)

print("generating tsne 1")
figure_filename1 = "_sentiment_doc_2_vec_tsne5"
tsne_plot(embedding, total_docs, labels,
          figure_name=figure_filename1, max_docs=1000, perplexity=5)

print("generating tsne 2")
figure_filename2 = "_sentiment_doc_2_vec_tsne30"
tsne_plot(embedding, total_docs, labels,
          figure_name=figure_filename2, max_docs=1000, perplexity=30)

print("generating tsne 3")
figure_filename3 = "_sentiment_doc_2_vec_tsne50"
tsne_plot(embedding, total_docs, labels,
          figure_name=figure_filename3, max_docs=1000, perplexity=50)

print("generating tsne 4")
figure_filename4 = "_sentiment_doc_2_vec_tsne100"
tsne_plot(embedding, total_docs, labels,
          figure_name=figure_filename4, max_docs=1000, perplexity=100)
