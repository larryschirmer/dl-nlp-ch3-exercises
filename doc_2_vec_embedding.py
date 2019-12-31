from keras.models import Model
from keras.layers import Input, Dense, Reshape, Embedding, merge, concatenate, Flatten, average
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import Adam

from doc_2_vec_utils import process_data, batch_generator, save_embedding, tsne_plot

# constants
window_size = 3
embedding_size = 100
batch_size = 100
epochs = 20
input_filename = "yelp_reviews.csv"
legend_filename = "doc_2_vec_legend"
figure_filename = "doc_2_vec_tsne"
loss = 'sparse_categorical_crossentropy'
optimizer = Adam()
metrics = ['accuracy']

# pre-process data
contexts, docids, targets, vocab = process_data(
    input_filename, window_size, output_legend_name=legend_filename)
vocab_size = len(vocab)
total_docs = len(docids)

# generate model
input_w1 = Input((1,))
input_w2 = Input((1,))
input_w3 = Input((1,))
input_docid = Input((1,))

embedding = Embedding(vocab_size, embedding_size,
                      input_length=1, name="embedding")
embedding_doc = Embedding(total_docs + 1, embedding_size)

docid = embedding_doc(input_docid)
docid = Reshape((embedding_size, 1))(docid)

w1 = embedding(input_w1)
w1 = Reshape((embedding_size, 1))(w1)

w2 = embedding(input_w2)
w2 = Reshape((embedding_size, 1))(w2)

w3 = embedding(input_w3)
w3 = Reshape((embedding_size, 1))(w3)

context_docid = concatenate([w1, w2, w3, docid])
context_docid = Flatten()(context_docid)

output = Dense(vocab_size, activation='softmax')(context_docid)
model = Model(input=[input_w1, input_w2, input_w3, input_docid], output=output)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.summary()

model.fit_generator(batch_generator(contexts, targets, batch_size),
                    steps_per_epoch=batch_size, epochs=epochs)

save_embedding('doc_2_vec_embedding.txt',
               embedding.get_weights()[0], total_docs)
tsne_plot(embedding.get_weights()[0], total_docs,
          figure_name=figure_filename, max_docs=1000)
