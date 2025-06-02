import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def visualize_classification_report(file_path):
    # Read the report file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    metrics = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0] not in ['precision', 'recall', 'f1-score', 'support', 'accuracy', 'macro', 'weighted']:
            try:
                metrics.append({
                    'class': parts[0],
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3])
                })
            except ValueError:
                continue

    # Create DataFrame
    df = pd.DataFrame(metrics)

    if df.empty:
        print(f"Could not find metrics in file {file_path}")
        return

    # Set style for bar plots
    sns.set_style('darkgrid')
    default_color = '#825ea2'

    # Create graphs
    plt.figure(figsize=(15, 5))

    # 1. Precision graph
    plt.subplot(1, 3, 1)
    sns.barplot(x='class', y='precision', data=df, color=default_color)
    plt.title('Precision by class')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # 2. Recall graph
    plt.subplot(1, 3, 2)
    sns.barplot(x='class', y='recall', data=df, color=default_color)
    plt.title('Recall by class')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # 3. F1-score graph
    plt.subplot(1, 3, 3)
    sns.barplot(x='class', y='f1-score', data=df, color=default_color)
    plt.title('F1-score by class')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Create heatmap for all metrics
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars=['class'], value_vars=[
                        'precision', 'recall', 'f1-score'])
    sns.heatmap(df_melted.pivot(index='class', columns='variable', values='value'),
                annot=True,
                cmap='coolwarm',
                vmin=0,
                vmax=1)
    plt.title('Heatmap of metrics by class')
    plt.tight_layout()
    plt.show()

    # Print average values of metrics
    print("\nAverage values of metrics:")
    print(df[['precision', 'recall', 'f1-score']].mean())

    return df


def compare_models_metrics(model_dfs):
    all_metrics = []

    for model_name, df in model_dfs.items():
        all_metrics.append({
            'model': model_name,
            'avg_type': 'macro',
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1-score': df['f1-score'].mean()
        })

        all_metrics.append({
            'model': model_name,
            'avg_type': 'weighted',
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1-score': df['f1-score'].mean()
        })

    df = pd.DataFrame(all_metrics)


    # Set style
    sns.set_style('darkgrid')

    # Create comparison graphs
    metrics = ['precision', 'recall', 'f1-score']

    plt.figure(figsize=(12, 6))
    df_macro = df[df['avg_type'] == 'macro']
    df_macro_melted = df_macro.melt(id_vars=['model'],
                                    value_vars=metrics,
                                    var_name='metric',
                                    value_name='value')

    sns.barplot(x='model', y='value', hue='metric', data=df_macro_melted)
    plt.title('Comparison of models (macro avg)')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # weighted avg
    plt.figure(figsize=(12, 6))
    df_weighted = df[df['avg_type'] == 'weighted']
    df_weighted_melted = df_weighted.melt(id_vars=['model'],
                                          value_vars=metrics,
                                          var_name='metric',
                                          value_name='value')

    sns.barplot(x='model', y='value', hue='metric', data=df_weighted_melted)
    plt.title('Comparison of models (weighted avg)')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
