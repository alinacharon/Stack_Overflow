import numpy as np
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from scipy.stats import entropy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def calculate_coherence(model, corpus, dictionary, texts, coherence_type='c_v'):
    
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence_type
    )
    return coherence_model.get_coherence()


def calculate_topic_diversity(model, num_topics):
    
    topic_words = []
    for i in range(num_topics):
        topic_words.append([word for word, _ in model.show_topic(i, topn=20)])

    unique_words = set()
    for topic in topic_words:
        unique_words.update(topic)

    return len(unique_words) / (num_topics * 20)


def calculate_topic_balance(model, corpus):

    topic_distributions = []
    for doc in corpus:
        topic_dist = model.get_document_topics(doc)
        topic_distributions.append([prob for _, prob in topic_dist])

    avg_entropy = np.mean([entropy(dist) for dist in topic_distributions])
    return avg_entropy


def visualize_topics(model, corpus, dictionary, output_path='lda_visualization.html'):

    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_path)


def visualize_document_distribution(model, corpus, output_path='document_distribution.png'):

    num_topics = model.num_topics
    doc_topics = []
    for doc in corpus:
        topic_dist = model.get_document_topics(doc, minimum_probability=0)
        topic_vector = [prob for _, prob in sorted(
            topic_dist, key=lambda x: x[0])]
        doc_topics.append(topic_vector)
    doc_topics = np.array(doc_topics)

    tsne = TSNE(n_components=2, random_state=42)
    doc_topics_2d = tsne.fit_transform(doc_topics)

    plt.figure(figsize=(10, 8))
    plt.scatter(doc_topics_2d[:, 0], doc_topics_2d[:, 1], alpha=0.5)
    plt.title('Document distribution in topic space')
    plt.savefig(output_path)
    plt.close()


def evaluate_lda_model(model, corpus, dictionary, texts, num_topics):
   
    metrics = {
        'perplexity': model.log_perplexity(corpus),
        'coherence_cv': calculate_coherence(model, corpus, dictionary, texts, 'c_v'),
        'coherence_umass': calculate_coherence(model, corpus, dictionary, texts, 'u_mass'),
        'topic_diversity': calculate_topic_diversity(model, num_topics),
        'topic_balance': calculate_topic_balance(model, corpus)
    }

    return metrics
