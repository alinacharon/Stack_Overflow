# Note technique – Généralisation de l’approche MLOps

## 🎯 Objectif

Ce document présente une étude de cas basée sur un projet réel de classification multilabel de questions Stack Overflow. Il vise à explorer les approches et outils permettant de généraliser une approche MLOps : automatisation du pipeline, traçabilité des expériences, et suivi des performances du modèle en production.

---

## 🧩 1. Contexte du projet

- **Projet** : Classification automatique des questions Stack Overflow à partir de leur contenu textuel.
- **Objectif** : Prédire les tags associés à chaque question.
- **Défis rencontrés** :
  - Classification multilabel avec forte déséquilibre des classes.
  - Plus de 1000 tags possibles.
  - Modèles lents à entraîner sur des données volumineuses.
  - Besoin de stabilité et de suivi du modèle dans le temps.

---

## ⚙️ 2. Pipeline MLOps proposé

### Étapes clefs :

1. **Collecte & Prétraitement**
   - Données via l’API Stack Exchange
   - Nettoyage du texte, vectorisation (TF-IDF ou embeddings)
   - Filtrage des tags (seulement ceux qui couvrent 90% des occurrences totales)
   - Encodage multilabel via `MultiLabelBinarizer`

2. **Modélisation**
   - Algorithmes testés : `LogisticRegression`, `RandomForest`, `LinearSVC`, etc.
   - Recherche d’hyperparamètres via `RandomizedSearchCV`
   - Scores utilisés : `f1_micro`, `hamming_loss`, `subset_accuracy`

3. **Déploiement local (prototype)**
   - API simple via Flask ou FastAPI
   - Modèle sauvegardé dans `models/supervised/`
   - Chargement du modèle + vectorizer pour la prédiction

4. **Monitoring & Suivi**
   - Suivi manuel mensuel sur 12 mois simulés
   - Calcul et visualisation des performances par mois
   - Préparation à l’intégration d’outils comme `EvidentlyAI`

---

## 🛠 3. Outils recommandés

| Étape                  | Outils recommandés                                     |
|------------------------|--------------------------------------------------------|
| Collecte               | Stack Exchange API, `pandas`, `json`                  |
| Preprocessing          | `scikit-learn`, `nltk`, `spacy`, `mlb`                |
| Modélisation           | `scikit-learn`, `xgboost`, `lightgbm`, `Optuna`       |
| Tracking des expériences | `MLflow`, `Weights & Biases`, `Neptune.ai`           |
| Déploiement            | `Flask`, `FastAPI`, `Docker`, `DVC`, `GitHub Actions` |
| Monitoring             | `EvidentlyAI`, `Prometheus`, `Grafana`                |
| CI/CD                  | `Git`, `GitHub Actions`, `DVC pipelines`              |

---

## 📊 4. Étude de cas : suivi de performance temporelle

En attendant l’intégration complète d’un outil de monitoring, nous avons simulé le suivi de la performance sur 12 mois :

```python
df['CreationDate'] = pd.to_datetime(df['CreationDate'])
df['month'] = df['CreationDate'].dt.to_period('M')

monthly_data = {
    str(month): df_month
    for month, df_month in df.groupby('month')
}

for month, df_month in monthly_data.items():
    X_month = vectorizer.transform(df_month['text'])
    y_true = mlb.transform(df_month['tags'])
    y_pred = model.predict(X_month)
    
    print(f"---- {month} ----")
    print("F1 score:", f1_score(y_true, y_pred, average='micro'))
    # Ajouter éventuellement plus de métriques
```

Cette analyse permet de :
- Suivre l’évolution de la qualité du modèle
- Détecter des dérives potentielles (data drift, concept drift)
- Identifier les mois où certains tags deviennent moins prédictibles

---

## 🧠 5. Proposition de généralisation MLOps

### Idée : Structurer un pipeline réutilisable avec :

- Un dossier `/pipeline/` contenant tous les scripts (ingestion, nettoyage, entraînement)
- Des paramètres centralisés dans un fichier `config.yaml`
- Suivi automatique avec `MLflow`
- Déploiement continu avec `GitHub Actions`
- Monitoring automatique avec `EvidentlyAI`

---

## ✅ 6. Bénéfices attendus

- **Reproductibilité** des résultats
- **Fiabilité** du modèle en production
- **Réduction de la dette technique**
- **Suivi en temps réel** des dérives
- **Collaboration facilitée** entre data scientists et développeurs

---

## 📌 Conclusion

Cette étude montre qu’il est possible de structurer une démarche MLOps à partir d’un projet réel, en mettant en œuvre des outils simples mais puissants comme `MLflow`, `EvidentlyAI`, `scikit-learn`, `FastAPI` et `GitHub Actions`. Ce pipeline peut ensuite être adapté et étendu à d’autres projets similaires.