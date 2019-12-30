from gensim.models import KeyedVectors

embedding_filepath = 'embedding_labeled.txt'

w2v = KeyedVectors.load_word2vec_format(
    embedding_filepath, unicode_errors='ignore')

# for word in sorted(w2v.wv.vocab)[:100]:
#     print(word, w2v.most_similar(word, topn=3))

print('waitress', w2v.most_similar('waitress', topn=3))
print('seasonal', w2v.most_similar('seasonal', topn=3))
print('indian', w2v.most_similar('indian', topn=3))
print('café', w2v.most_similar('café', topn=3))
print('baba', w2v.most_similar('baba', topn=3))
print('bone', w2v.most_similar('bone', topn=3))