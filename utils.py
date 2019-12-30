import re
import codecs
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.preprocessing.sequence import pad_sequences, skipgrams


def tsne_plot(model, max_words=100, figure_name='tsne'):
    labels = []
    tokens = []

    n = 0
    for word in model:
        if n < max_words:
            tokens.append(model[word])
            labels.append(word)
            n += 1

    tsne_model = TSNE(perplexity=40, init='pca', n_iter=10000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(figure_name+'.png', bbox_inches='tight')


def save_embedding(outputFile, weights, vocabulary):
    rev = {v: k for k, v in vocabulary.items()}

    with codecs.open(outputFile, "w") as f:
        f.write(str(len(vocabulary)) + ' ' + str(weights.shape[1]) + "\n")

        for word_index, word in sorted(iter(rev.items())):
            f.write(word + " ")

            for weight_index in range(len(weights[word_index])):
                f.write(str(weights[word_index][weight_index]) + " ")

            f.write("\n")


def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines


def create_vocabulary(vocabulary, sentences):
    vocabulary["<unk>"] = 0

    for sentence in sentences:
        for word in sentence.strip().split():
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def process_training_data(textFile, max_len):
    data = []
    sentences = getLines(textFile)
    vocab = dict()
    labels = []
    create_vocabulary(vocab, sentences)

    for s in sentences:
        words = []
        m = re.match("^([^\t]+)\t(.+)$", s.rstrip())

        if m:
            sentence = m.group(1)
            labels.append(int(m.group(2)))

        for word in sentence.split(" "):
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())

            if word != '':
                words.append(vocab[word])

        data.append(words)

    data = pad_sequences(data, maxlen=max_len, padding='post')

    return data, labels, vocab


def process_test_data(textFile, vocab, max_len):
    data = []
    sentences = getLines(textFile)
    labels = []
    create_vocabulary(vocab, sentences)

    for s in sentences:
        words = []
        m = re.match("^([^\t]+)\t(.+)$", s.rstrip())

        if m:
            sentence = m.group(1)
            labels.append(int(m.group(2)))

        for word in sentence.split(" "):
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())

            if word != '':
                if word in vocab:
                    words.append(vocab[word])
                else:
                    words.append(vocab["<unk>"])

        data.append(words)

    data = pad_sequences(data, maxlen=max_len, padding='post')

    return data, labels


def process_data(textFile, window_size):
    couples = []
    labels = []
    sentences = getLines(textFile)
    vocab = dict()
    create_vocabulary(vocab, sentences)
    vocab_size = len(vocab)

    for sentence in sentences:
        words = []

        for word in sentence.split(" "):
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())

            if word != '':
                words.append(vocab[word])

        couple, label = skipgrams(words, vocab_size, window_size=window_size)
        couples.extend(couple)
        labels.extend(label)

    return vocab, couples, labels


def batch_generator(target, context, labels, batch_size):
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for batch_index in range(batch_size):
            index = random.randint(0, len(target) - 1)
            batch_target[batch_index] = target[index]
            batch_context[batch_index] = context[index]
            batch_labels[batch_index] = labels[index]

        yield [batch_target, batch_context], [batch_labels]


def load_embedding(filename, vocab, embedding_dim):
    embedding_index = {}
    f = open(filename)
    n = 0

    for line in f:
        values = line.split()
        word = values[0]

        if word in vocab:
            coefs = np.asarray(values[1:], dtype='float32')

            if n:  # skip embedding header line
                embedding_index[word] = coefs

            n += 1

    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))

    for word, word_index in vocab.items():
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[word_index] = embedding_vector

    return embedding_matrix
