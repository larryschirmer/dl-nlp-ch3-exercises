from keras.models import Model
from keras.layers import Embedding, Input, Dense, Reshape, dot
from utils import process_data, save_embedding, batch_generator

# constants
window_size = 3
epochs = 1000
embedding_size = 100
batch_size = 100
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
train_filename = 'yelp_reviews.csv'


# pre-process data
vocab, couples, labels = process_data(train_filename, window_size)


# generate model
vocab_size = len(vocab)
word_target, word_context = zip(*couples)

input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, embedding_size, input_length=1)
target = embedding(input_target)
target = Reshape((embedding_size,))(target)
context = embedding(input_context)
context = Reshape((embedding_size,))(context)

dot_product = dot([target, context], 1)
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.summary()
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

model.fit_generator(batch_generator(word_target, word_context, labels, batch_size),
                    steps_per_epoch=batch_size, epochs=epochs)

save_embedding('skipgram-embedding_labeled.txt',
               embedding.get_weights()[0], vocab)
