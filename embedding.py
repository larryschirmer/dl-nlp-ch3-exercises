import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from utils import tsne_plot

docs = ["Chuck Berry rolled over everyone who came before him ? and turned up everyone who came after. We'll miss you",
        "Help protect the progress we've made in helping millions of Americans get covered.",
        "Let's leave our children and grandchildren a planet that's healthier than the one we have today.",
        "The American people are waiting for Senate leaders to do their jobs.",
        "We must take bold steps now ? climate change is already impacting millions of people.",
        "Don't forget to watch Larry King tonight",
        "Ivanka is now on Twitter - You can follow her",
        "Last night Melania and I attended the Skating with the Stars Gala at Wollman Rink in Central Park",
        "People who have the ability to work should. But with the government happy to send checks",
        "I will be signing copies of my new book"
        ]

docs = [d.lower() for d in docs]

count_vec = CountVectorizer().fit(docs)
tokenizer = count_vec.build_tokenizer()

input_array = []
for doc in docs:
    x = []
    for token in tokenizer(doc):
        x.append(count_vec.vocabulary_.get(token))

    input_array.append(x)

max_len = max([len(d) for d in input_array])
vocab_size = len(count_vec.vocabulary_)
input_array = pad_sequences(input_array, maxlen=max_len, padding='post')


model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_len))
model.compile('rmsprop', 'mse')

output_array = model.predict(input_array)

M = {}
for sec_index in range(len(input_array)):
    for word_index in range(len(input_array[sec_index])):
        token = input_array[sec_index][word_index]
        word = count_vec.get_feature_names()[token]
        M[word] = output_array[sec_index][word_index]

tsne_plot(M, max_words=vocab_size)
