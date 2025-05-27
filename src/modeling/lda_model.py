import os
import mlflow
from gensim import models

def train_lda_model(corpus, dictionary, num_topics, passes):
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes
    )

    mlflow.log_param('num_topics', num_topics)
    mlflow.log_param('passes', passes)

    model_dir = 'models/lda'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'lda_model.gensim')
    lda_model.save(model_path)

    mlflow.log_artifact(model_path, artifact_path='models')

    return lda_model