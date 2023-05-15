"""Plot results using W&B output CSV file."""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENCODER_TYPES_ORDER = ['mlp', 'egnn', 'tfn']


def plot_wand_results(
        data_path: Path,
        concept: str,
        metric: str,
        save_dir: Path
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
        experiment_to_results[f'{encoder_type} {embedding_method}'].append(data[run].dropna().iloc[0])

    # Set up encoder type order
    assert set(encoder_types) <= set(ENCODER_TYPES_ORDER)
    encoder_types_order = [encoder_type for encoder_type in ENCODER_TYPES_ORDER if encoder_type in encoder_types]
    encoder_types_order_upper = [encoder_type.upper() for encoder_type in encoder_types_order]

    # Plot results
    save_dir.mkdir(parents=True, exist_ok=True)

    for embedding_method in embedding_methods:
        experiments = [f'{encoder_type} {embedding_method}' for encoder_type in encoder_types_order]

        plt.clf()
        for i, experiment in enumerate(experiments):
            results = experiment_to_results[experiment]
            plt.bar(i, np.mean(results), yerr=np.std(results), capsize=5)

        plt.xticks(range(len(encoder_types_order_upper)), encoder_types_order_upper, rotation=45)
        plt.ylabel(metric)
        plt.title(f'{embedding_method} {concept}')

        # Save plot
        plt.savefig(save_dir / f'{embedding_method}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_wand_results)
