import re
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.preprocessing.sequence import pad_sequences


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
