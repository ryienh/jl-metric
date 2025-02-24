import os
import numpy as np
import torch
from scipy.stats import kstest, pearsonr, spearmanr
from collections import defaultdict
import torch.nn as nn
import pickle
import click

from mmd import mmd_linear
from distance_metrics import get_metrics


def save_progress(filepath, data):
    data = {k: dict(v) for k, v in data.items()}
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_progress(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return defaultdict(
        lambda: defaultdict(
            lambda: {
                "p_values": [],
                "ks_distances": [],
                "mmd_distances": [],
                "cosine_distances": [],
                "kl_distances": [],
                "js_distances": [],
            }
        )
    )


def initialize_distances_if_needed(distances, key):
    if key not in distances:
        distances[key] = {
            "p_values": [],
            "ks_distances": [],
            "mmd_distances": [],
            "cosine_distances": [],
            "kl_distances": [],
            "js_distances": [],
        }


cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)


# linear kernel
def kernel(x_a, x_b):

    if x_a.dim() == 0 and x_b.dim() == 0:
        return (x_a * x_b + 1) ** 3
    else:
        return (torch.dot(x_a, x_b) + 1) ** 3


def kernel_matrix(X, Y):

    if X.dim() == 1:
        X = X.unsqueeze(1)  # Shape (m, 1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)  # Shape (n, 1)
    K = torch.mm(X, Y.t())  # Shape (m, n)
    K = (K + 1) ** 3
    return K


def calculate_cosine_similarity(S_g, S_r):

    S_g = S_g.to(torch.float32)
    S_r = S_r.to(torch.float32)
    if S_g.dim() == 1:
        return -1 * cosine_similarity(S_g, S_r).item()
    else:
        similarities = []
        for i in range(S_g.size(0)):
            similarities.append(cosine_similarity(S_g[i], S_r[i]).item())
        return -1 * (sum(similarities) / len(similarities))


def calculate_mmd(S_g, S_r, batch_size=10000):

    S_g = S_g.to(torch.float32)
    S_r = S_r.to(torch.float32)

    m, n = S_g.size(0), S_r.size(0)

    term_1 = 0
    term_2 = 0
    term_3 = 0

    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        K_xx = kernel_matrix(S_g[i:end_i], S_g[i:end_i])
        term_1 += K_xx.sum()

    for j in range(0, n, batch_size):
        end_j = min(j + batch_size, n)
        K_yy = kernel_matrix(S_r[j:end_j], S_r[j:end_j])
        term_2 += K_yy.sum()

    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            K_xy = kernel_matrix(S_g[i:end_i], S_r[j:end_j])
            term_3 += K_xy.sum()

    term_1 = term_1 / (m * m)
    term_2 = term_2 / (n * n)
    term_3 = 2 * term_3 / (m * n)

    mmd = term_1 + term_2 - term_3
    return mmd


def calculate_distances_with_save(descriptor_root, progress_file, batch_size=10000):
    """Calculate distances between original and perturbed descriptors."""
    distances = load_progress(progress_file)
    processed_datasets = {key.split("_")[0] for key in distances.keys()}

    for dataset_name in os.listdir(descriptor_root):
        if dataset_name in processed_datasets:
            click.echo(f"Skipping {dataset_name} as it has already been processed.")
            continue

        dataset_dir = os.path.join(descriptor_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        click.echo(f"\nProcessing dataset: {dataset_name}")

        for perturb_type in os.listdir(dataset_dir):
            perturb_dir = os.path.join(dataset_dir, perturb_type)
            if not os.path.isdir(perturb_dir):
                continue

            click.echo(f"  Perturbation type: {perturb_type}")

            for seed_dir in os.listdir(perturb_dir):
                if seed_dir.startswith("seed_"):
                    seed = int(seed_dir.split("_")[1])
                    seed_path = os.path.join(perturb_dir, seed_dir)

                    p0_dir = os.path.join(seed_path, "p_0.00")
                    if not os.path.exists(p0_dir):
                        continue
                    p0_descriptors = load_descriptors(p0_dir)

                    for p_dir in os.listdir(seed_path):
                        p_value = float(p_dir.split("_")[1])
                        p_descriptor_path = os.path.join(seed_path, p_dir)
                        descriptors = load_descriptors(p_descriptor_path)

                        for metric in p0_descriptors.keys():
                            key = f"{dataset_name}_{perturb_type}_seed_{seed}_{metric}"
                            click.echo(f"    Calculating {key} (p={p_value})")

                            initialize_distances_if_needed(distances, key)
                            distances[key]["p_values"].append(p_value)

                            # Calculate distances based on metric type
                            if metric not in ["jl_embd", "softmax"]:
                                ks_stat, _ = kstest(
                                    p0_descriptors[metric].numpy(),
                                    descriptors[metric].numpy(),
                                )
                                distances[key]["ks_distances"].append(ks_stat)

                                mmd_dist = mmd_linear(
                                    p0_descriptors[metric]
                                    .to(torch.float32)
                                    .reshape(-1, 1)
                                    .numpy(),
                                    descriptors[metric]
                                    .to(torch.float32)
                                    .reshape(-1, 1)
                                    .numpy(),
                                )
                                distances[key]["mmd_distances"].append(mmd_dist.item())

                            if metric == "jl_embd":
                                # Calculate JL embedding specific distances
                                cosine_sim = calculate_cosine_similarity(
                                    p0_descriptors[metric], descriptors[metric]
                                )
                                distances[key]["cosine_distances"].append(cosine_sim)

                                ks_stat, _ = kstest(
                                    torch.mean(p0_descriptors[metric], dim=0).numpy(),
                                    torch.mean(descriptors[metric], dim=0).numpy(),
                                )
                                distances[key]["ks_distances"].append(ks_stat)

                                mmd_dist = mmd_linear(
                                    p0_descriptors[metric].numpy(),
                                    descriptors[metric].numpy(),
                                )
                                distances[key]["mmd_distances"].append(mmd_dist.item())

                            if metric == "softmax":
                                # Handle softmax metric
                                if dataset_name == "LastFM":
                                    distances[key]["mmd_distances"].append(0)
                                    distances[key]["js_distances"].append(0)
                                    distances[key]["ks_distances"].append(0)
                                else:
                                    js_distances = []
                                    ks_distances = []

                                    for i in range(p0_descriptors[metric].shape[1]):
                                        feature_metrics, _ = get_metrics(
                                            p0_descriptors[metric][:, i],
                                            descriptors[metric][:, i],
                                        )
                                        js_distances.append(feature_metrics["js_dist"])
                                        ks_distances.append(feature_metrics["ks_dist"])

                                    distances[key]["js_distances"].append(
                                        np.mean(js_distances)
                                    )
                                    distances[key]["ks_distances"].append(
                                        np.mean(ks_distances)
                                    )

                                    mmd_dist = mmd_linear(
                                        p0_descriptors[metric], descriptors[metric]
                                    )
                                    distances[key]["mmd_distances"].append(mmd_dist)

        save_progress(progress_file, distances)
        click.echo(f"Progress saved for {dataset_name}")

    return distances


# Function to calculate correlation between "p" and distances
def calculate_correlations(distances, correlation_type):

    assert correlation_type in [
        "pearson",
        "spearman",
    ], f"Correlation type must be one of pearson, spearman, not {correlation_type}"

    correlation_function = spearmanr if correlation_type == "spearman" else pearsonr

    correlations = {}

    for key, value in distances.items():

        p_values = torch.tensor(value["p_values"])
        if "ks_distances" in value and len(value["ks_distances"]) > 0:
            ks_distances = torch.tensor(value["ks_distances"])
            ks_corr, _ = correlation_function(p_values, ks_distances)
            if np.isnan(ks_corr):  # type: ignore
                ks_corr = 0
            correlations[f"{key}_ks"] = ks_corr
        if "mmd_distances" in value and len(value["mmd_distances"]) > 0:
            mmd_distances = torch.tensor(value["mmd_distances"])
            mmd_corr, _ = correlation_function(p_values, mmd_distances)
            if np.isnan(mmd_corr):  # type: ignore
                mmd_corr = 0
            correlations[f"{key}_mmd"] = mmd_corr
        if "cosine_distances" in value and len(value["cosine_distances"]) > 0:
            cosine_distances = torch.tensor(value["cosine_distances"])
            cosine_corr, _ = correlation_function(p_values, cosine_distances)
            if np.isnan(cosine_corr):  # type: ignore
                cosine_corr = 0
            correlations[f"{key}_cosine"] = cosine_corr

        if "kl_distances" in value and len(value["kl_distances"]) > 0:
            kl_distances = torch.tensor(value["kl_distances"])
            threshold = np.percentile(kl_distances[np.isfinite(kl_distances)], 99)
            kl_distances = np.where(np.isinf(kl_distances), threshold, kl_distances)
            kl_corr, _ = correlation_function(p_values, kl_distances)
            if np.isnan(kl_corr):  # type: ignore
                kl_corr = 0
            correlations[f"{key}_kl"] = kl_corr

        if "js_distances" in value and len(value["js_distances"]) > 0:
            js_distances = torch.tensor(value["js_distances"])
            js_distances = np.nan_to_num(js_distances, nan=0)
            threshold = np.percentile(js_distances[np.isfinite(js_distances)], 99)
            js_distances = np.where(np.isinf(js_distances), threshold, js_distances)
            js_distances = np.nan_to_num(js_distances, nan=0)
            js_corr, _ = correlation_function(p_values, js_distances)
            if np.isnan(js_corr):  # type: ignore
                js_corr = 0
            correlations[f"{key}_js"] = js_corr

    return correlations


def load_descriptors(descriptor_dir):

    descriptors = {}
    for file_name in os.listdir(descriptor_dir):
        if file_name.endswith(".pt"):
            descriptor_name = file_name.split(".pt")[0]
            descriptor_path = os.path.join(descriptor_dir, file_name)
            descriptors[descriptor_name] = torch.load(descriptor_path)
    return descriptors


def get_datasets_and_seeds(descriptor_root):
    datasets = {}
    for dataset_name in os.listdir(descriptor_root):
        dataset_dir = os.path.join(descriptor_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for perturb_type in os.listdir(dataset_dir):
            perturb_dir = os.path.join(dataset_dir, perturb_type)
            if not os.path.isdir(perturb_dir):
                continue
            seeds = []
            for seed_dir in os.listdir(perturb_dir):
                if seed_dir.startswith("seed_"):
                    seed = int(seed_dir.split("_")[1])
                    seeds.append(seed)
            if seeds:
                datasets[(dataset_name, perturb_type)] = seeds
    return datasets


def normalize_correlations(all_distances, correlations):

    normalized_correlations = {}

    for key, corr in correlations.items():
        key_parts = key.split("_")

        for idx, part in enumerate(key_parts):
            if part.startswith("seed"):
                perturb_type = "_".join(key_parts[1:idx])
                seed = key_parts[idx + 1]
                metric_name = "_".join(key_parts[idx + 2 :])
                break

        p0_key = f"{key_parts[0]}_{perturb_type}_seed_{seed}_{metric_name}"
        if p0_key in all_distances and 0.00 in all_distances[p0_key]["p_values"]:
            p0_index = all_distances[p0_key]["p_values"].index(0.00)
            p0_value = (
                all_distances[p0_key]["mmd_distances"][p0_index]
                if "mmd_distances" in all_distances[p0_key]
                else 0
            )

            normalized_corr = corr - p0_value
            normalized_correlations[key] = normalized_corr
        else:
            normalized_correlations[key] = corr

    return normalized_correlations


@click.command()
@click.option(
    "--descriptor-root",
    default="./descriptors",
    help="Directory containing computed descriptors",
)
@click.option(
    "--correlation-type",
    type=click.Choice(["pearson", "spearman"], case_sensitive=False),
    default="spearman",
    help="Type of correlation to compute",
)
@click.option(
    "--progress-file",
    default="progress.pkl",
    help="File to save/load intermediate progress",
)
@click.option(
    "--correlation-file",
    default="correlations.pt",
    help="File to save final correlations",
)
@click.option(
    "--unnormalized-correlation-file",
    default="correlations_unnormalized.pt",
    help="File to save unnormalized correlations",
)
@click.option("--batch-size", default=10000, help="Batch size for MMD calculations")
def main(
    descriptor_root,
    correlation_type,
    progress_file,
    correlation_file,
    unnormalized_correlation_file,
    batch_size,
):
    """Calculate distances and correlations between original and perturbed descriptors."""

    click.echo(f"Starting correlation analysis with:")
    click.echo(f"- Descriptor directory: {descriptor_root}")
    click.echo(f"- Correlation type: {correlation_type}")
    click.echo(f"- Progress file: {progress_file}")
    click.echo(f"- Output files: {correlation_file}, {unnormalized_correlation_file}")

    # Get datasets and seeds
    datasets_and_seeds = get_datasets_and_seeds(descriptor_root)
    click.echo(f"\nFound {len(datasets_and_seeds)} dataset-perturbation combinations")

    # Calculate distances
    click.echo("\nCalculating distances...")
    all_distances = calculate_distances_with_save(
        descriptor_root, progress_file=progress_file, batch_size=batch_size
    )

    # Calculate correlations
    click.echo("\nCalculating correlations...")
    unnormalized_correlations = calculate_correlations(
        all_distances, correlation_type=correlation_type
    )
    correlations = normalize_correlations(all_distances, unnormalized_correlations)

    # Save results
    torch.save(unnormalized_correlations, unnormalized_correlation_file)
    click.echo(f"Unnormalized correlations saved to {unnormalized_correlation_file}")

    torch.save(correlations, correlation_file)
    click.echo(f"Normalized correlations saved to {correlation_file}")

    # Print summary statistics
    click.echo("\nCorrelation Summary:")
    metric_correlations = defaultdict(list)
    for key, corr in correlations.items():
        key_parts = key.split("_")
        for idx, part in enumerate(key_parts):
            if part.startswith("seed"):
                metric_name = "_".join(key_parts[idx + 2 :])
                break
        metric_correlations[metric_name].append(corr)

    click.echo("\nMean Correlations (across all seeds):")
    for metric, corr_values in metric_correlations.items():
        avg_corr = sum(corr_values) / len(corr_values)
        click.echo(f"{metric}: {avg_corr:.4f}")


if __name__ == "__main__":
    main()
