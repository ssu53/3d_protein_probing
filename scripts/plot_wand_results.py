"""Plot results using W&B output CSV file."""
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

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
METRIC_SHORT_TO_LONG = {
    'ap': 'Average Precision',
    'accuracy': 'Accuracy',
    'r2': 'R^2'
}
OFFSET = 1
CONCEPT_TO_NAME = {
    'residue_sasa': 'SASA',
    'secondary_structure': 'Secondary Structure',
    'residue_locations': 'Locations',
    'residue_distances': 'Distances',
    'residue_distances_by_residue': 'Distances By Residue',
    'residue_contacts': 'Contacts',
    'residue_contacts_by_residue': 'Contacts By Residue',
    'bond_angles': 'Bond Angles',
    'dihedral_angles': 'Dihedral Angles',
    'solubility': 'Solubility'
}
CONCEPT_SUBSET_ORDER = {
    'geometry': [
        'SASA',
        'Secondary Structure',
        'Locations',
        'Distances',
        'Distances By Residue',
        'Contacts',
        'Contacts By Residue',
        'Bond Angles',
        'Dihedral Angles'
    ],
    'downstream': [
        'Solubility'
    ]
}


def default_dict_to_regular(obj: Any) -> dict:
    if isinstance(obj, defaultdict):
        obj = {key: default_dict_to_regular(value) for key, value in obj.items()}

    return obj


def plot_wand_results(
        data_path: Path,
        save_path: Path,
        metrics: tuple[str] = ('ap', 'accuracy', 'r2'),
        concept_subset: Literal['geometry', 'downstream'] | None = None
) -> None:
    """Plot results using W&B output CSV file."""
    # Load data
    data = pd.read_csv(data_path)

    # Extract results
    concept_to_embedding_to_encoder_to_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    concept_to_metric = {}

    for metric in metrics:
        metric_data = data[data[f'test_{metric}'].notna()]

        for concept, embedding_method, encoder_type, result in zip(
                metric_data['concept'],
                metric_data['embedding_method'],
                metric_data['encoder_type'],
                metric_data[f'test_{metric}']
        ):
            concept = CONCEPT_TO_NAME[concept]
            concept_to_embedding_to_encoder_to_results[concept][embedding_method][encoder_type].append(result)
            concept_to_metric[concept] = METRIC_SHORT_TO_LONG[metric]

    # Convert defaultdict to regular dict
    concept_to_embedding_to_encoder_to_results = default_dict_to_regular(concept_to_embedding_to_encoder_to_results)

    # Set up subplots
    concepts = set(concept_to_embedding_to_encoder_to_results)

    if concept_subset is not None:
        concepts = [concept for concept in CONCEPT_SUBSET_ORDER[concept_subset] if concept in concepts]
    else:
        concepts = sorted(concepts)

    num_plots = len(concepts)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, sharey=True, figsize=(4 * num_plots, 4))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

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
