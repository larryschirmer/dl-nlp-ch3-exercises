import re
import sys
from gensim.models import KeyedVectors


def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines


embedding_filepath = '_sentiment_doc_2_vec_embed.txt'
legend_filepath = '_sentiment_doc_2_vec_legend.txt'
doc = int(sys.argv[1])

legend = getLines(legend_filepath)
w2v = KeyedVectors.load_word2vec_format(
    embedding_filepath, unicode_errors='ignore')


matched_docs = w2v.most_similar('doc_' + str(doc), topn=3)
print(legend[doc])
for matched_doc, confidence in matched_docs:
    doc_num = int(re.sub('doc_', '', matched_doc))
    print('---')
    print(matched_doc + ' ' + str(round(confidence, 3)))
    print(legend[doc_num])
