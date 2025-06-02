from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.matutils import Sparse2Corpus


def vectorize(texts, method='count', vectorizer_params=None):
    if vectorizer_params is None:
        vectorizer_params = {}

    if method == 'count':
        vec = CountVectorizer(**vectorizer_params)
    elif method == 'tfidf':
        vec = TfidfVectorizer(**vectorizer_params)
    else:
        raise ValueError("Unsupported method")

    X = vec.fit_transform(texts)
    return X, vec

def convert_to_gensim(X, vec):
    corpus = Sparse2Corpus(X, documents_columns=False)
    id2word = dict((id, word) for word, id in vec.vocabulary_.items())
    return corpus, id2word