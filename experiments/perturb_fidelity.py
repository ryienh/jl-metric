import click
import torch
import torch_geometric
import os
import random
import numpy as np

from torch_geometric.data import TemporalData
from torch_geometric.datasets import JODIEDataset

from grid import generate_synthetic_ctdg


def edge_rewiring(data, p):
    num_edges = data.src.size(0)
    perturbed_dst = data.dst.clone()

    for i in range(num_edges):
        if random.random() < p:
            perturbed_dst[i] = random.randint(0, data.src.max().item())

    perturbed_ctdg = TemporalData(
        src=data.src.clone(),
        dst=perturbed_dst,
        t=data.t.clone(),
        msg=data.msg.clone(),
    )

    return perturbed_ctdg


def event_permutation(data, p):
    num_edges = data.src.size(0)

    perturbed_src = data.src.clone()
    perturbed_dst = data.dst.clone()
    perturbed_t = data.t.clone()
    perturbed_msg = data.msg.clone()

    for i in range(num_edges):
        if random.random() < p:
            # Randomly select a different event to switch with
            random_index = random.randint(0, num_edges - 1)
            while random_index == i:
                random_index = random.randint(0, num_edges - 1)

            perturbed_msg[i], perturbed_msg[random_index] = (
                perturbed_msg[random_index].clone(),
                perturbed_msg[i].clone(),
            )

    perturbed_ctdg = TemporalData(
        src=perturbed_src,
        dst=perturbed_dst,
        t=perturbed_t,
        msg=perturbed_msg,
    )

    return perturbed_ctdg


def temporal_perturbation(data, p):
    perturbed_t = data.t.clone()
    num_edges = data.src.size(0)

    for i in range(num_edges):
        if random.random() < p:
            # Get the previous and next timestamps
            prev_t = data.t[i - 1] if i > 0 else data.t[i]
            next_t = data.t[i + 1] if i < num_edges - 1 else data.t[i]
            perturbed_t[i] = (
                data.t[i]
                if torch.abs(prev_t - next_t) < 2
                else random.randint(prev_t + 1, next_t - 1)
            )

    perturbed_ctdg = TemporalData(
        src=data.src.clone(),
        dst=data.dst.clone(),
        t=perturbed_t.clone(),
        msg=data.msg.clone(),
    )

    return perturbed_ctdg


def get_datasets(
    jodie_path="./jodie_data/",
    grid_nn=1024,
    grid_max_time=10000,
    grid_seed=101,
    subset_events=None,
    selected_datasets=None,
):
    available_datasets = {
        "Reddit": lambda: JODIEDataset(root=jodie_path, name="Reddit")[0],
        "Wikipedia": lambda: JODIEDataset(root=jodie_path, name="Wikipedia")[0],
        "MOOC": lambda: JODIEDataset(root=jodie_path, name="MOOC")[0],
        "LastFM": lambda: JODIEDataset(root=jodie_path, name="LastFM")[0],
        "Grid": lambda: generate_synthetic_ctdg(
            num_nodes=grid_nn, max_time=grid_max_time, random_seed=grid_seed
        ),
    }

    if selected_datasets is None or "all" in selected_datasets:
        selected_datasets = list(available_datasets.keys())

    datasets = []
    dataset_names = []

    for name in selected_datasets:
        if name in available_datasets:
            data = available_datasets[name]()
            if subset_events is not None:
                data = data[:subset_events]
            datasets.append(data)
            dataset_names.append(name)
        else:
            click.echo(f"Warning: Dataset {name} not found. Skipping...", err=True)

    return datasets, dataset_names


# Save function
def save_dataset(data, dataset_name, perturb_type, seed, p, output_dir):
    perturb_dir = os.path.join(
        output_dir, f"{dataset_name}/{perturb_type}/seed_{seed}/p_{p:.2f}"
    )
    os.makedirs(perturb_dir, exist_ok=True)
    torch.save(data, os.path.join(perturb_dir, "perturbation.pt"))


# Apply perturbations to datasets
def apply_perturbations(data, dataset_name, seeds=10, ps=None, output_dir=None):
    if ps is None:
        ps = np.linspace(0, 0.9, 10)

    for seed in range(seeds):
        random.seed(seed)
        torch_geometric.seed_everything(seed)
        for p in ps:
            perturbed_data = edge_rewiring(data, p)
            save_dataset(
                perturbed_data, dataset_name, "edge_rewiring", seed, p, output_dir
            )

            perturbed_data = event_permutation(data, p)
            save_dataset(
                perturbed_data, dataset_name, "event_permutation", seed, p, output_dir
            )

            perturbed_data = temporal_perturbation(data, p)
            save_dataset(
                perturbed_data,
                dataset_name,
                "temporal_perturbation",
                seed,
                p,
                output_dir,
            )


@click.command()
@click.option(
    "--output-dir",
    default="./perturbed_data",
    help="Directory to save the perturbed datasets",
)
@click.option(
    "--datasets",
    default=["all"],
    multiple=True,
    help='Datasets to process. Use "all" for all datasets or specify a subset: Reddit, Wikipedia, MOOC, LastFM, Grid',
)
@click.option("--num-seeds", default=10, help="Number of random seeds to use")
@click.option(
    "--subset-events",
    default=1000,
    help="Number of events to use from each dataset. Use -1 for all events",
)
@click.option("--jodie-path", default="./jodie_data/", help="Path to JODIE datasets")
@click.option(
    "--grid-nn", default=1024, help="Number of nodes for synthetic grid dataset"
)
@click.option(
    "--grid-max-time", default=10000, help="Maximum time for synthetic grid dataset"
)
@click.option("--grid-seed", default=101, help="Random seed for synthetic grid dataset")
def main(
    output_dir,
    datasets,
    num_seeds,
    subset_events,
    jodie_path,
    grid_nn,
    grid_max_time,
    grid_seed,
):
    """Generate perturbed versions of temporal interaction datasets."""

    # Convert subset_events None if -1
    if subset_events == -1:
        subset_events = None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get datasets
    datasets_list, dataset_names = get_datasets(
        jodie_path=jodie_path,
        grid_nn=grid_nn,
        grid_max_time=grid_max_time,
        grid_seed=grid_seed,
        subset_events=subset_events,
        selected_datasets=datasets,
    )

    # Process each dataset
    for dataset, name in zip(datasets_list, dataset_names):
        click.echo(f"Applying perturbations for dataset: {name}")
        apply_perturbations(dataset, name, seeds=num_seeds, output_dir=output_dir)

    click.echo("Processing complete!")


if __name__ == "__main__":
    main()
