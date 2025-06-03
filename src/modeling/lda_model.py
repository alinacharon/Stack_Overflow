import os
import numpy as np
from gensim import models
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from scipy.stats import entropy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def train_lda_model(corpus, dictionary, num_topics, passes):
    """
    Train LDA model with specified parameters
    """
    # Generate timestamp for unique model names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_run_name = f"lda_{num_topics}topics_{timestamp}"

    logger.info(f"Training LDA model with {num_topics} topics...")
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes
    )

    # Save model
    model_dir = 'models/lda'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_run_name}.gensim")
    lda_model.save(model_path)

    return lda_model, {
        'model_path': model_path,
        'model_name': 'lda',
        'timestamp': timestamp,
        'num_topics': num_topics,
        'passes': passes
    }


def calculate_coherence(model, corpus, dictionary, texts, coherence_type='c_v'):
    """
    Calculate topic coherence score
    """
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence_type
    )
    return coherence_model.get_coherence()


def calculate_topic_diversity(model, num_topics):
    """
    Calculate topic diversity score

    """
    topic_words = []
    for i in range(num_topics):
        topic_words.append([word for word, _ in model.show_topic(i, topn=20)])

    unique_words = set()
    for topic in topic_words:
        unique_words.update(topic)

    return len(unique_words) / (num_topics * 20)


def calculate_topic_balance(model, corpus):
    """
    Calculate topic balance score using entropy

    """
    topic_distributions = []
    for doc in corpus:
        topic_dist = model.get_document_topics(doc)
        topic_distributions.append([prob for _, prob in topic_dist])

    avg_entropy = np.mean([entropy(dist) for dist in topic_distributions])
    return avg_entropy


def visualize_topics(model, corpus, dictionary, output_path=None, config=None):
    """
    Create interactive visualization of topics using pyLDAvis
    """
    if output_path is None and config is not None:
        output_dir = config['lda']['visualization']['output_dir']
        output_path = os.path.join(
            output_dir, config['lda']['visualization']['topics_vis_path'])

    logger.info("Creating topic visualization...")
    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_path)
    return output_path


def visualize_document_distribution(model, corpus, output_path=None, config=None):
    """
    Create 2D visualization of document distribution in topic space
    """
    if output_path is None and config is not None:
        output_dir = config['lda']['visualization']['output_dir']
        output_path = os.path.join(
            output_dir, config['lda']['visualization']['distribution_vis_path'])

    logger.info("Creating document distribution visualization...")
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
    return output_path


def evaluate_lda_model(model, corpus, dictionary, texts, num_topics):
    """
    Evaluate LDA model using multiple metrics
    """
    logger.info("Evaluating LDA model...")
    metrics = {
        'perplexity': model.log_perplexity(corpus),
        'coherence_cv': calculate_coherence(model, corpus, dictionary, texts, 'c_v'),
        'coherence_umass': calculate_coherence(model, corpus, dictionary, texts, 'u_mass'),
        'topic_diversity': calculate_topic_diversity(model, num_topics),
        'topic_balance': calculate_topic_balance(model, corpus)
    }

    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    return metrics


def train_and_evaluate_lda(corpus, dictionary, texts, num_topics, passes, config=None):
    """
    Train and evaluate LDA model with visualization
    """
    logger.info(
        f"Starting LDA training and evaluation with {num_topics} topics...")

    # Train model
    model, model_info = train_lda_model(corpus, dictionary, num_topics, passes)

    # Evaluate model
    metrics = evaluate_lda_model(model, corpus, dictionary, texts, num_topics)

    # Create visualizations
    vis_path = visualize_topics(model, corpus, dictionary, config=config)
    dist_path = visualize_document_distribution(model, corpus, config=config)

    logger.info("LDA training and evaluation completed successfully")

    return model, {
        **model_info,
        'metrics': metrics,
        'visualization_path': vis_path,
        'distribution_path': dist_path
    }
