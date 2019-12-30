Scenario: You are creating word embeddings that are optimal for a given task at hand. You happen to be working on a sentiment analysis task for restaurant reviews. The word embeddings learned should be optimized in the context of this specific task.

see `embedding.py` and `view_embedding.py`

Scenario: You want to establish meaningful, synonym-like relations between words, based on a large text corpus (a set of restaurant reviews) you have at hand. The relations you are after should be based on the statistics of words co-occurring with each other. Two words are more similar if they occur in the same contexts. You want to use these relations for implementing an editor tool that suggests synonyms for words while you are editing a document, and you obviously do not want to write out these relations by hand. How can Embeddings help you out?

Scenario: Rather than use an embedding that was trained for a certain task (like sentiment labeling), you want to use a pre-trained word2vec model which is based on other (better, more) training data. Hopefully, it improves the results on your specific task. How do you implement this?


yelp review data source:
https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

GloVe data source:
https://nlp.stanford.edu/projects/glove/

Install instructions:
- to use `pos` with `tsne_plot`, run `python -m spacy download en_core_web_sm`