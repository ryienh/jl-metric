import time
import pandas as pd
import numpy as np

from perturb_fidelity import get_datasets
from function_descriptors import (
    calc_static_descriptors,
    calc_dynamic_descriptors,
    calc_feature_probabilities,
)

N_MAX = 10


def average_benchmarks_with_error(results):
    avg_results = {}
    error_results = {}

    for metric in results[next(iter(results))].keys():
        values = [dataset[metric] for dataset in results.values()]

        avg_results[metric] = np.mean(values)
        error_results[metric] = np.std(values)

    avg_df = pd.DataFrame(
        {
            "Metric": avg_results.keys(),
            "Average": avg_results.values(),
            "Error (±)": error_results.values(),
        }
    )

    return avg_df


def comp_eff(dataset, num_events=100, seed=442):

    global_results = {}

    sample = dataset[:num_events]

    metadata = {"seed": seed}
    static_metrics = ["AND", "LCC", "NC", "PLE"]
    for metric in static_metrics:
        start_time = time.time()
        _ = calc_static_descriptors(
            data=sample, metadata=metadata, nyquist_interval=1, types=[metric]
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        global_results[metric] = elapsed_time

    dynamic_metrics = ["jl_embd", "activity_rate"]
    for metric in dynamic_metrics:
        start_time = time.time()
        _ = calc_dynamic_descriptors(
            data=sample,
            metadata=metadata,
            types=[metric],
            node_dim=1000,
            graph_dim=1000,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        global_results[metric] = elapsed_time

    feature_metrics = ["softmax"]
    for metric in feature_metrics:
        start_time = time.time()
        _ = calc_feature_probabilities(
            data=sample,
            metadata=metadata,
            types=[metric],
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        global_results[metric] = elapsed_time

    return global_results


if __name__ == "__main__":

    datasets = get_datasets(
        grid_nn=10000, grid_max_time=10000, grid_seed=101
    )  # grid info must match that in tgn_train_and_save.py
    datasets = datasets[0]

    dataset_names = ["Reddit", "Wikipedia", "MOOC", "LastFM", "Grid"]

    trimmed_datasets = []
    for dataset in datasets:
        trimmed_datasets.append(dataset[:N_MAX])  # type: ignore

    results = {}
    for dataset, name in zip(trimmed_datasets, dataset_names):
        print(f"Calculating efficiency for {name}")
        results[name] = comp_eff(dataset, num_events=100)

    to_print = average_benchmarks_with_error(results)
    print(to_print)
