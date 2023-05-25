"""Plot sequence vs structure for protein function prediction."""
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from plot_wand_results import EMBEDDING_METHOD_TO_UPPER, METRIC_SHORT_TO_LONG


def plot_function_results(
        data_path: Path,
        save_path: Path,
        model_1_embedding: Literal['one', 'baseline', 'plm'],
        model_1_encoder: Literal['mlp', 'egnn', 'tfn', 'ipa'],
        model_2_embedding: Literal['one', 'baseline', 'plm'],
        model_2_encoder: Literal['mlp', 'egnn', 'tfn', 'ipa'],
        metric: str = 'micro_ap'
) -> None:
    # Load results
    data = pd.read_csv(data_path)

    # Collect sequence vs structure results
    metric_columns = [column for column in data.columns if column.startswith(f'test_{metric}_')]

    constraint_1 = (data['embedding_method'] == model_1_embedding) & \
                   (data['encoder_type'] == model_1_encoder) & \
                   (data['interaction_model'].isna())
    results_1 = data[constraint_1][metric_columns]

    constraint_2 = (data['embedding_method'] == model_2_embedding) & \
                           (data['encoder_type'] == model_2_encoder) & \
                           (data['interaction_model'].isna())
    results_2 = data[constraint_2][metric_columns]

    assert len(results_1) == len(results_2) == 1

    results_1 = results_1.iloc[0].dropna().tolist()
    results_2 = results_2.iloc[0].dropna().tolist()

    print(f'Number of model 1 results: {len(results_1)}')
    print(f'Number of model 2 results: {len(results_2)}')

    # Plot sequence vs structure results
    plt.scatter(results_1, results_2, s=5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel(f'{EMBEDDING_METHOD_TO_UPPER[model_1_embedding]} {model_1_encoder.upper()} {METRIC_SHORT_TO_LONG[metric]}')
    plt.ylabel(f'{EMBEDDING_METHOD_TO_UPPER[model_2_embedding]} {model_2_encoder.upper()} {METRIC_SHORT_TO_LONG[metric]}')
    plt.title('Protein Function Prediction')

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from tap import tapify

    tapify(plot_function_results)
