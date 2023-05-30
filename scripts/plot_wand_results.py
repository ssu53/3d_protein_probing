"""Plot results using W&B output CSV file."""
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


EMBEDDING_METHODS = [
    'one',
    'baseline',
    'plm'
]
ENCODER_TYPES = [
    'mlp',
    'egnn',
    'tfn',
    'ipa'
]
EMBEDDING_METHOD_TO_HATCH = {
    'one': 'o',
    'baseline': '',
    'plm': '/'
}
EMBEDDING_METHOD_TO_UPPER = {
    'one': 'Constant',
    'baseline': 'Raw',
    'plm': 'PLM'
}
ENCODER_TYPE_TO_COLOR = {
    'mlp': 'tab:blue',
    'egnn': 'tab:orange',
    'tfn': 'tab:red',
    'ipa': 'tab:brown'
}
METRIC_SHORT_TO_LONG = {
    'ap': 'Average Precision',
    'macro_ap': 'Average Precision',
    'micro_ap': 'Average Precision',
    'micro_mean_ap': 'Average Precision',
    'accuracy': 'Accuracy',
    'macro_accuracy': 'Accuracy',
    'micro_accuracy': 'Accuracy',
    'r2': 'R^2',
    'macro_r2': r'R$^{\mathbf{2}}$',
    'micro_r2': r'R$^{\mathbf{2}}$'
}
OFFSET = 1
CONCEPT_TO_NAME = {
    'residue_sasa': 'SASA',
    'secondary_structure': 'Secondary Structure',
    # 'residue_locations': 'Locations',
    'residue_distances': 'Pair Distance',
    'residue_distances_by_residue': 'Average Pair Distance',
    'residue_contacts': 'Pair Contact',
    'residue_contacts_by_residue': 'Any Pair Contact',
    'bond_angles': 'Bond Angle',
    'dihedral_angles': 'Dihedral Angle',
    'solubility': 'Solubility',
    'enzyme_commission': 'Enzyme Commission',
    'gene_ontology': 'Gene Ontology'
}
CONCEPT_SUBSET_ORDER = {
    'geometry': [
        'SASA',
        'Secondary Structure',
        # 'Locations',
        'Pair Distance',
        'Average Pair Distance',
        'Pair Contact',
        'Any Pair Contact',
        'Bond Angle',
        'Dihedral Angle'
    ],
    'downstream': [
        'Solubility',
        'Enzyme Commission',
        'Gene Ontology'
    ]
}
ENCODER_TO_EMBEDDING_TO_X = {
    'mlp': {
        'one': 0,
        'baseline': 1,
        'plm': 3
    },
    'egnn': {
        'one': 5,
        'baseline': 9,
        'plm': 13
    },
    'tfn': {
        'one': 6,
        'baseline': 10,
        'plm': 14
    },
    'ipa': {
        'one': 7,
        'baseline': 11,
        'plm': 15
    }
}
XTICKS = [0.5, 3, 6, 10, 14]
XTICK_LABELS = ['Baseline', 'Seq', 'Coords', 'Struct', 'Seq & Struct']


def default_dict_to_regular(obj: Any) -> dict:
    if isinstance(obj, defaultdict):
        obj = {key: default_dict_to_regular(value) for key, value in obj.items()}

    return obj


def plot_wand_results(
        data_path: Path,
        save_path: Path,
        metrics: tuple[str, ...] = ('ap', 'accuracy', 'r2'),
        concept_subset: Literal['geometry', 'downstream'] | None = None,
        num_rows: int = 1
) -> None:
    """Plot results using W&B output CSV file.

    :param data_path: Path to W&B output CSV file.
    :param save_path: Path to save plot.
    :param metrics: Metrics to plot (each concept can have at most one metric).
    :param concept_subset: Subset of concepts to plot.
    :param num_rows: Number of rows of subplots.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Extract results
    concept_to_embedding_to_encoder_to_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    concept_to_metric = {}

    for metric in metrics:
        metric_data = data[data[f'test_{metric}'].notna()]
        metric_data = metric_data[metric_data['interaction_model'].isna()]
        metric_data = metric_data[metric_data['concept'] != 'residue_locations']

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
    num_cols = num_plots // num_rows + (num_plots % num_rows > 0)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharey=True, figsize=(num_cols * 4 * 1.2, num_rows * 3 * 1.2))

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # Plot each set of results
    for row_idx, axes_row in enumerate(tqdm(axes)):
        # Set up row of axes
        axes_row[0].set_ylabel('Metric')
        # axes_row[0].legend(handles=[
        #     mpatches.Patch(
        #         facecolor='lightgray',
        #         hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method],
        #         label=EMBEDDING_METHOD_TO_UPPER[embedding_method]
        #     )
        #     for embedding_method in EMBEDDING_METHODS
        # ], loc='upper left')

        for ax in axes_row[1:]:
            ax.tick_params(axis='y', which='both', length=0)

        # Plot results for each concept in subplots
        concepts_row = concepts[row_idx * axes.shape[1]: (row_idx + 1) * axes.shape[1]]
        for ax, concept in tqdm(zip(axes_row, concepts_row), total=num_plots, leave=False):
            # xticks, xticklabels = [], []

            for encoder_type_idx, encoder_type in enumerate(ENCODER_TYPES):
                for embedding_method_idx, embedding_method in enumerate(EMBEDDING_METHODS):
                    try:
                        results = concept_to_embedding_to_encoder_to_results[concept][embedding_method][encoder_type]
                    except KeyError:
                        results = [0]

                    x = ENCODER_TO_EMBEDDING_TO_X[encoder_type][embedding_method]
                    # xtick = embedding_method_idx + encoder_type_idx * (len(EMBEDDING_METHODS) + OFFSET)
                    # xticks.append(xtick)
                    # xticklabels.append(encoder_type.upper())
                    ax.bar(
                        x,
                        np.mean(results),
                        # yerr=np.std(results),
                        color=ENCODER_TYPE_TO_COLOR[encoder_type],
                        capsize=5,
                        hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method],
                        label=encoder_type.upper() if embedding_method == 'baseline' else None
                    )

            # xticks = np.array(xticks)
            ax.set_xticks(XTICKS)
            ax.set_xticklabels(XTICK_LABELS)
            ax.set_xlabel(f'{concept}\n({concept_to_metric[concept]})', weight='bold')
            ax.set_ylim(0, 1)

        handles, labels = axes_row[0].get_legend_handles_labels()
        axes_row[0].legend(loc='upper left', handles=[
            mpatches.Patch(
                facecolor='lightgray',
                hatch=EMBEDDING_METHOD_TO_HATCH[embedding_method],
                label=EMBEDDING_METHOD_TO_UPPER[embedding_method]
            )
            for embedding_method in EMBEDDING_METHODS
        ] + handles)

    # Add plot-wide details
    fig.subplots_adjust(wspace=0, hspace=0.5)

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_wand_results)
