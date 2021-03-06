from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from utils import process_training_data, process_test_data, save_embedding

# constants
max_len = 100
embedding_size = 100
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
train_filename = 'yelp_reviews_train.tsv'
test_filename = 'yelp_reviews_test.tsv'

# pre-process data
data, labels, vocab = process_training_data(train_filename, max_len)
test_data, test_labels = process_test_data(test_filename, vocab, max_len)

# generate model
vocab_size = len(vocab)

model = Sequential()
embedding = Embedding(vocab_size, embedding_size, input_length=max_len)
model.add(embedding)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(data, labels, epochs=100, verbose=1, batch_size=32,
          shuffle=True, validation_data=(test_data, test_labels))

save_embedding('embedding_labeled.txt', embedding.get_weights()[0], vocab)
