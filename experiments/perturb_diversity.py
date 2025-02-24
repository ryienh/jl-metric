import numpy as np
import torch
import torch_geometric
import random
from sklearn.cluster import AffinityPropagation
import pickle
import os
import click
from torch_geometric.data import TemporalData

from perturb_fidelity import get_datasets, save_dataset


def trim_ds(dataset, node_to_mode, mode_clusters, num_events=2000):
    trimmed_dataset = TemporalData(
        src=dataset.src[:num_events],
        dst=dataset.dst[:num_events],
        t=dataset.t[:num_events],
        msg=dataset.msg[:num_events] if dataset.msg is not None else None,
    )

    unique_nodes = set(trimmed_dataset.src.tolist()) | set(trimmed_dataset.dst.tolist())  # type: ignore

    trimmed_node_to_mode = {
        node: mode for node, mode in node_to_mode.items() if node in unique_nodes
    }

    unique_modes = set(trimmed_node_to_mode.values())

    trimmed_mode_clusters = mode_clusters[list(unique_modes)]

    return trimmed_dataset, trimmed_node_to_mode, trimmed_mode_clusters


def get_modes(node_embeddings, dataset_name, cache_dir="modes_cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = os.path.join(cache_dir, f"{dataset_name}_modes.pkl")

    if os.path.exists(cache_file):
        click.echo(f"Loading cached modes from {cache_file}...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            return cached_data["node_to_mode"], cached_data["mode_clusters"]

    click.echo("Computing modes with Affinity Propagation clustering...")
    if isinstance(node_embeddings, torch.Tensor):
        node_embeddings = node_embeddings.cpu().detach().numpy()

    clustering = AffinityPropagation(random_state=5).fit(node_embeddings)
    cluster_labels = clustering.labels_

    node_to_mode = {
        node_id: cluster_labels[node_id] for node_id in range(len(cluster_labels))
    }

    mode_clusters = clustering.cluster_centers_

    with open(cache_file, "wb") as f:
        pickle.dump({"node_to_mode": node_to_mode, "mode_clusters": mode_clusters}, f)

    click.echo(f"Modes cached to {cache_file}")
    return node_to_mode, mode_clusters


def mode_dropping(data, node_modes, cluster_centers, p):
    num_modes = len(cluster_centers)
    modes_to_drop = [mode for mode in range(num_modes) if random.random() < p]

    safe_indices = []
    dropped_indices = []

    src = data.src.clone()
    dst = data.dst.clone()
    msg = data.msg.clone()

    for i in range(len(src)):
        src_node = src[i].item()
        dst_node = dst[i].item()

        src_mode = node_modes[src_node]
        dst_mode = node_modes[dst_node]

        if src_mode in modes_to_drop or dst_mode in modes_to_drop:
            dropped_indices.append(i)
        else:
            safe_indices.append(i)

    if not safe_indices:
        click.echo("Warning: No safe events found. Using random event replication.")
        safe_indices = list(range(len(src)))

    for idx in dropped_indices:
        replacement_idx = random.choice(safe_indices)
        src[idx] = src[replacement_idx]
        dst[idx] = dst[replacement_idx]
        msg[idx] = msg[replacement_idx]

    return TemporalData(src=src, dst=dst, t=data.t, msg=msg)


def mode_collapse(data, node_modes, cluster_centers, p):
    if p == 0.0:
        return data

    src = data.src.clone()
    dst = data.dst.clone()
    msg = data.msg.clone()

    num_modes = len(cluster_centers)
    mode_messages = {mode: [] for mode in range(num_modes)}

    for i in range(len(src)):
        src_node = src[i].item()
        dst_node = dst[i].item()

        src_mode = node_modes[src_node]
        dst_mode = node_modes[dst_node]

        mode_messages[src_mode].append(msg[i].cpu().numpy())
        mode_messages[dst_mode].append(msg[i].cpu().numpy())

    representative_messages = {}
    for mode, messages in mode_messages.items():
        if len(messages) > 0:
            representative_messages[mode] = np.mean(messages, axis=0)
        else:
            representative_messages[mode] = np.zeros(msg.shape[1])

    for mode in representative_messages:
        representative_messages[mode] = torch.tensor(
            representative_messages[mode], dtype=msg.dtype
        )

    for i in range(len(src)):
        if random.random() < p:
            src_node = src[i].item()
            dst_node = dst[i].item()

            src_mode = node_modes[src_node]
            dst_mode = node_modes[dst_node]

            src[i] = torch.tensor(
                np.argmax(
                    np.linalg.norm(cluster_centers - cluster_centers[src_mode], axis=1)
                )
            )
            dst[i] = torch.tensor(
                np.argmax(
                    np.linalg.norm(cluster_centers - cluster_centers[dst_mode], axis=1)
                )
            )

            msg[i] = representative_messages[src_mode]

    return TemporalData(src=src, dst=dst, t=data.t, msg=msg)


def apply_perturbations_diversity(
    data, dataset_name, node_modes, cluster_centers, seeds=10, ps=None, output_dir=None
):
    if ps is None:
        ps = np.linspace(0, 0.9, 10)

    for seed in range(seeds):
        random.seed(seed)
        torch_geometric.seed_everything(seed)
        for p in ps:
            click.echo(
                f"Applying perturbations for {dataset_name} (seed={seed}, p={p:.2f})"
            )

            perturbed_data = mode_dropping(data, node_modes, cluster_centers, p)
            save_dataset(
                perturbed_data, dataset_name, "mode_dropping", seed, p, output_dir
            )

            perturbed_data = mode_collapse(data, node_modes, cluster_centers, p)
            save_dataset(
                perturbed_data, dataset_name, "mode_collapse", seed, p, output_dir
            )


@click.command()
@click.option(
    "--output-dir",
    default="./perturbed_data",
    help="Directory to save perturbed datasets",
)
@click.option(
    "--model-dir", default="./tgn", help="Directory containing trained TGN models"
)
@click.option(
    "--num-events", default=10000, help="Number of events to use from each dataset"
)
@click.option(
    "--num-seeds", default=10, help="Number of random seeds for perturbations"
)
@click.option(
    "--cache-dir", default="modes_cache", help="Directory to cache computed modes"
)
@click.option(
    "--datasets",
    default=["all"],
    multiple=True,
    help='Datasets to process. Use "all" or specify individual ones: Reddit, Wikipedia, MOOC, LastFM, Grid',
)
@click.option(
    "--grid-nn", default=10000, help="Number of nodes for synthetic grid dataset"
)
@click.option(
    "--grid-max-time", default=10000, help="Maximum time for synthetic grid dataset"
)
@click.option("--grid-seed", default=101, help="Random seed for synthetic grid dataset")
def main(
    output_dir,
    model_dir,
    num_events,
    num_seeds,
    cache_dir,
    datasets,
    grid_nn,
    grid_max_time,
    grid_seed,
):
    """Generate diversity-based perturbations of temporal interaction datasets using trained TGN models."""

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output directory: {output_dir}")

    all_datasets = ["Reddit", "Wikipedia", "MOOC", "LastFM", "Grid"]
    if "all" in datasets:
        selected_datasets = all_datasets
    else:
        selected_datasets = [d for d in datasets if d in all_datasets]

    if not selected_datasets:
        click.echo("Error: No valid datasets selected")
        return

    click.echo(f"Processing datasets: {', '.join(selected_datasets)}")

    # Load all datasets - now properly unpacking the tuple
    datasets_list, dataset_names = get_datasets(
        grid_nn=grid_nn,
        grid_max_time=grid_max_time,
        grid_seed=grid_seed,
        selected_datasets=selected_datasets,  # Pass selected datasets to get_datasets
    )

    # Create dataset dictionary with the returned datasets and their names
    dataset_dict = dict(zip(dataset_names, datasets_list))

    # Process each selected dataset
    for name in selected_datasets:
        if name not in dataset_dict:
            click.echo(f"Warning: Dataset {name} not found. Skipping...")
            continue

        click.echo(f"\nProcessing {name} dataset...")

        # Load trained model
        model_path = os.path.join(model_dir, f"{name}.pt")
        if not os.path.exists(model_path):
            click.echo(f"Warning: Model not found at {model_path}. Skipping {name}.")
            continue

        memory = torch.load(model_path)
        node_embedding = memory.memory

        # Get modes
        modes, cluster_centers = get_modes(node_embedding, name, cache_dir)

        # Get and trim dataset
        dataset = dataset_dict[name]
        dataset.y = None  # type: ignore

        dataset, _, _ = trim_ds(
            dataset,
            modes,
            cluster_centers,
            num_events=num_events,
        )

        # Apply perturbations
        click.echo(f"Applying perturbations for dataset: {name}")
        apply_perturbations_diversity(
            dataset,
            name,
            modes,
            cluster_centers,
            seeds=num_seeds,
            output_dir=output_dir,
        )

    click.echo("\nAll perturbations completed!")


if __name__ == "__main__":
    main()
