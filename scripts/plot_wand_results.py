"""Plot results using W&B output CSV file."""
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


EMBEDDING_METHODS = ['baseline', 'plm']
ENCODER_TYPES = ['mlp', 'egnn', 'tfn']
EMBEDDING_METHOD_TO_HATCH = {
    'baseline': '',
    'plm': '/'
}
EMBEDDING_METHOD_TO_UPPER = {
    'baseline': 'Baseline',
    'plm': 'PLM'
}
ENCODER_TYPE_TO_COLOR = {
    'mlp': 'tab:blue',
    'egnn': 'tab:orange',
    'tfn': 'tab:red'
}
OFFSET = 1


def default_dict_to_regular(obj: Any) -> dict:
    if isinstance(obj, defaultdict):
        obj = {key: default_dict_to_regular(value) for key, value in obj.items()}

    return obj


def plot_wand_results(
        data_paths: list[Path],
        metrics: list[str],
        save_path: Path
) -> None:
    """Plot results using W&B output CSV file."""
    # Check input sizes
    assert len(data_paths) == len(metrics)

    # Extract results
    concept_to_embedding_to_encoder_to_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    concept_to_metric = {}

    for data_path, metric in zip(data_paths, metrics):
        # Load data
        data = pd.read_csv(data_path)

        # Extract run information
        runs = data.columns[1::3]
        concepts, run_names, embedding_methods, encoder_types = [], [], [], []

        for run in runs:
            # Get concept
            concept = None
            for embedding_method in EMBEDDING_METHODS:
                if embedding_method in run:
                    concept = run[:run.index(embedding_method)].strip('_')
                    break

            if concept is None:
                raise ValueError(f'Could not find concept in {run}')

            # Get run information
            run_name = run.split('-')[0].strip().replace(f'{concept}_', '')
            run_names.append(run_name)

            concept = concept.replace('_', ' ').title()
            concepts.append(concept)

            embedding_methods.append(run_name.split('_')[0])
            encoder_types.append(run_name.split('_')[1])

            concept_to_metric[concept] = metric

        # Merge results across splits
        for run, concept, embedding_method, encoder_type in zip(runs, concepts, embedding_methods, encoder_types):
            results = data[run].dropna()

            if len(results) > 1:
                raise ValueError(f'Found more than one result for {run}')

            concept_to_embedding_to_encoder_to_results[concept][embedding_method][encoder_type].append(results.iloc[0])

    # Convert defaultdict to regular dict
    concept_to_embedding_to_encoder_to_results = default_dict_to_regular(concept_to_embedding_to_encoder_to_results)

    # Set up subplots
    concepts = sorted(concept_to_embedding_to_encoder_to_results)
    num_plots = len(concepts)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, sharey=True, figsize=(4 * num_plots, 4))

    # Plot each set of results
    for ax, concept in tqdm(zip(axes, concepts), total=num_plots):
        xticks, xticklabels = [], []

        for embedding_method_idx, embedding_method in enumerate(EMBEDDING_METHODS):
            for encoder_type_idx, encoder_type in enumerate(ENCODER_TYPES):
                results = concept_to_embedding_to_encoder_to_results[concept][embedding_method][encoder_type]
                xtick = encoder_type_idx + embedding_method_idx * (len(ENCODER_TYPES) + OFFSET)
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

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(f'{concept}\n({concept_to_metric[concept]})', weight='bold')

    # Add plot-wide details
    fig.subplots_adjust(wspace=0)
    axes[0].set_ylabel('Metric')
    axes[0].legend(handles=[
        mpatches.Patch(
            facecolor='lightgray',
            hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method],
            label=EMBEDDING_METHOD_TO_UPPER[embedding_method]
        )
        for embedding_method in EMBEDDING_METHODS
    ], loc='upper left')

    for ax in axes[1:]:
        ax.tick_params(axis='y', which='both', length=0)

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_wand_results)
