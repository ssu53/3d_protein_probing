"""Plot results using W&B output CSV file."""
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EMBEDDING_METHODS_ORDER = ['baseline', 'plm']
EMBEDDING_METHOD_TO_HATCH = {
    'baseline': '',
    'plm': '/'
}
EMBEDDING_METHOD_TO_UPPER = {
    'baseline': 'Baseline',
    'plm': 'PLM'
}
ENCODER_TYPES_ORDER = ['mlp', 'egnn', 'tfn']
ENCODER_TYPE_TO_COLOR = {
    'mlp': 'tab:blue',
    'egnn': 'tab:orange',
    'tfn': 'tab:red'
}
OFFSET = 1


def plot_wand_results(
        data_path: Path,
        concept: str,
        metric: str,
        save_path: Path
) -> None:
    """Plot results using W&B output CSV file."""
    # Load data
    data = pd.read_csv(data_path)

    # Extract run information
    runs = data.columns[1::3]
    run_names = [run.split('-')[0].strip().lstrip(f'{concept}_') for run in runs]
    embedding_methods = [run_name.split('_')[0] for run_name in run_names]
    encoder_types = [run_name.split('_')[1] for run_name in run_names]

    # Merge results across splits
    experiment_to_results = defaultdict(list)
    for run, embedding_method, encoder_type in zip(runs, embedding_methods, encoder_types):
        experiment_to_results[f'{embedding_method} {encoder_type}'].append(data[run].dropna().iloc[0])

    # Set up encoder type order
    assert set(encoder_types) <= set(ENCODER_TYPES_ORDER)
    encoder_types_order = [encoder_type for encoder_type in ENCODER_TYPES_ORDER if encoder_type in encoder_types]

    # Set up embedding method order
    assert set(embedding_methods) <= set(EMBEDDING_METHODS_ORDER)
    embedding_methods_order = [embedding_method for embedding_method in EMBEDDING_METHODS_ORDER if embedding_method in embedding_methods]

    # Plot results
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    xticks, xticklabels, legend_items = [], [], []
    for embedding_method_idx, embedding_method in enumerate(embedding_methods_order):
        legend_items.append(
            mpatches.Patch(
                facecolor='lightgray',
                hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method],
                label=EMBEDDING_METHOD_TO_UPPER[embedding_method]
            )
        )

        for encoder_type_idx, encoder_type in enumerate(encoder_types_order):
            results = experiment_to_results[f'{embedding_method} {encoder_type}']
            xtick = encoder_type_idx + embedding_method_idx * (len(encoder_types_order) + OFFSET)
            xticks.append(xtick)
            xticklabels.append(encoder_type.upper())
            ax.bar(
                xtick,
                np.mean(results),
                yerr=np.std(results),
                color=ENCODER_TYPE_TO_COLOR[encoder_type],
                capsize=5,
                hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method]
            )

    ax.set_xticks(xticks, rotation=45)
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel('Metric')
    ax.set_xlabel(f'{concept} {metric}')
    ax.legend(handles=legend_items)

    # Save plot
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_wand_results)
