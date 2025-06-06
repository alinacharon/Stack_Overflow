from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
