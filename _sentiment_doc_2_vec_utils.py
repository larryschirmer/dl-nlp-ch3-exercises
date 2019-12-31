import re
import codecs
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
from sklearn.manifold import TSNE


def tsne_plot(embedding, total_docs, labels, max_docs=100, figure_name='tsne', perplexity=40):
    plot_labels = []
    tokens = []
    def figsize_scale(docs): return round((docs/100) * 8)

    n = 0
    for doc_index in range(total_docs):
        if n < max_docs:
            tokens.append(embedding[doc_index])
            plot_labels.append('doc_' + str(doc_index))
            n += 1

    tsne_model = TSNE(perplexity=perplexity, init='pca', n_iter=5000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    color = lambda label_val: 'red' if label_val == '0' else 'blue'
    figsize = figsize_scale(max_docs)
    plt.figure(figsize=(figsize, figsize))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=color(labels[i]))
        plt.annotate(plot_labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(figure_name+'.png', bbox_inches='tight')


def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines


def create_vocabulary(vocabulary, sentences):
    vocabulary["<unk>"] = 0

    for sentence in sentences[1:]: # skip headers line
        sentence = sentence.split("\t")[0]
        
        for word in sentence.strip().split():
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def save_embedding(outputFile, weights, total_docs):
    with codecs.open(outputFile, "w") as f:
        f.write(str(total_docs) + ' ' + str(weights.shape[1]) + "\n")

        for doc_index in range(total_docs):
            f.write("doc_" + str(doc_index) + " ")

            for weight_index in range(len(weights[doc_index])):
                f.write(str(weights[doc_index][weight_index]) + " ")

            f.write("\n")


def process_data(input_data, window_size, output_legend_name="docs_legenda"):
    docs = getLines(input_data)
    vocab = dict()
    create_vocabulary(vocab, docs)
    docid = 0
    contexts = []
    docids = []
    targets = []
    labels = []

    f = open(output_legend_name + ".txt", "w")

    for sentence in docs[1:]: # skip headers line
        [review, label] = sentence.split("\t")
        labels.append(label)
        f.write("%d %s\n" % (docid, review))
        docids.append(docid)

        review = review.lower()
        review = re.sub(r'[^a-zA-Z0-9\s]', ' ', review)
        tokens = [token for token in review.split(" ") if token in vocab]
        n_grams = list(ngrams(tokens, window_size))

        for ngram_index in range(len(n_grams) - 1):
            context = [docid]
            n_gram = n_grams[ngram_index]

            for word in n_gram:
                word = re.sub("[.,:;'\"!?()]+", "", word.lower())
                context.append(vocab[word])

            contexts.append(context)
            targets.append(label)

        docid += 1

    f.close()
    return np.array(contexts), np.array(docids), np.array(targets), vocab, labels


def batch_generator(contexts, targets, batch_size):
    w1 = np.zeros((batch_size, 1))
    w2 = np.zeros((batch_size, 1))
    w3 = np.zeros((batch_size, 1))
    docid = np.zeros((batch_size, 1))
    batch_targets = np.zeros((batch_size, 1))

    while True:
        for batch_index in range(batch_size):
            rand_item = random.randint(0, len(targets) - 1)

            w1[batch_index] = contexts[rand_item][1]
            w2[batch_index] = contexts[rand_item][2]
            w3[batch_index] = contexts[rand_item][3]
            docid[batch_index] = contexts[rand_item][0]
            batch_targets[batch_index] = targets[rand_item]

        yield [w1, w2, w3, docid], [batch_targets]
