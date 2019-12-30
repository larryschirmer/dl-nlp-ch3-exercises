from gensim.models import KeyedVectors

embedding_filepath = 'embedding_labeled.txt'

w2v = KeyedVectors.load_word2vec_format(
    embedding_filepath, unicode_errors='ignore')

for word in sorted(w2v.wv.vocab):
    print(word, w2v.most_similar(word, topn=3))
