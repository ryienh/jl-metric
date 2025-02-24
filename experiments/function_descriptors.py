import os
import torch
from pathlib import Path
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
import networkx as nx
import tqdm
from torch_geometric import seed_everything
from torch_geometric.utils import to_networkx, degree
import multiprocessing
from functools import partial
import click

from jl_metric import jl_metric


def remap_node_ids(static_graph):

    edge_index = static_graph.edge_index

    unique_nodes = torch.unique(edge_index)

    node_id_map = {old_id.item(): new_id for new_id, old_id in enumerate(unique_nodes)}

    remapped_edge_index = edge_index.clone()
    for old_id, new_id in node_id_map.items():
        remapped_edge_index[edge_index == old_id] = new_id

    static_graph.edge_index = remapped_edge_index

    return static_graph


def get_nyquist_interval(graphs):

    intervals = []
    for graph in graphs:
        nyquist_interval = np.min(np.diff(np.sort(graph.t.numpy())))
        intervals.append(nyquist_interval)
    return min(intervals)


def find_nearest_event(arr, x):
    idx = np.searchsorted(arr, x)

    if idx == 0:
        return None, None

    pos = idx - 1
    value = arr[pos]

    return pos, value


def calc_static_descriptors(
    data,
    metadata,
    nyquist_interval,
    types=["AND", "LCC", "NC", "PLE"],
    sampling_rates=[1.0],
):

    seed_everything(metadata["seed"])

    if types is None:
        types = ["AND", "LCC", "NC", "PLE"]

    descriptors = defaultdict(list)

    src = data.src.numpy()
    dst = data.dst.numpy()
    t = data.t.numpy()
    edge_attr = data.msg.numpy()

    if nyquist_interval == 0:
        print("Warning, nyquist_interval is 0, will sample at every integer.")
        nyquist_interval = 1  # assumes integer timestamps

    for sampling_rate in sampling_rates:

        sampling_interval = sampling_rate * nyquist_interval

        snapshot_edge_attrs = {}
        snapshot_edges = set()  # Track edges without duplicates

        last_sample_time = 0

        prev_current_event_idx = -1

        temp = int(np.max(t))

        for i in tqdm.tqdm(range(temp), ncols=50):
            current_time = i
            current_event_idx, current_event = find_nearest_event(t, current_time)

            if current_event_idx is None:
                continue

            edge = (src[current_event_idx], dst[current_event_idx])
            edge = tuple(sorted(edge))

            if current_time >= last_sample_time + sampling_interval:
                edge_index = (
                    torch.tensor(np.array(list(snapshot_edges)), dtype=torch.long)
                    .t()
                    .contiguous()
                )

                edge_attr_np_array = np.array(
                    [snapshot_edge_attrs[edge] for edge in snapshot_edges]
                )

                edge_attr_tensor = torch.tensor(edge_attr_np_array, dtype=torch.float)

                static_graph = Data(
                    edge_index=edge_index,
                    edge_attr=edge_attr_tensor,
                )

                static_graph = remap_node_ids(static_graph)
                static_graph.num_nodes = (
                    0
                    if static_graph.edge_index is None
                    or static_graph.edge_index.shape[0] == 0
                    else torch.unique(static_graph.edge_index.reshape(-1)).shape[0]
                )  # type: ignore

                # Calculate each descriptor for the current static graph snapshot
                if "AND" in types:
                    avg_degree = (
                        calc_average_node_degree(static_graph)
                        if static_graph.edge_index is not None
                        else 0.0
                    )
                    descriptors[f"AND_{sampling_rate}"].append(avg_degree)

                if "LCC" in types:
                    lcc = (
                        calc_largest_connected_component(static_graph)
                        if static_graph.edge_index is not None
                        else 0.0
                    )
                    descriptors[f"LCC_{sampling_rate}"].append(lcc)

                if "NC" in types:
                    nc = (
                        calc_number_of_components(static_graph)
                        if static_graph.edge_index is not None
                        else 0.0
                    )
                    descriptors[f"NC_{sampling_rate}"].append(nc)

                if "PLE" in types:
                    lcc = (
                        calc_power_law_exponent(static_graph)
                        if static_graph.edge_index is not None
                        else 0.0
                    )
                    descriptors[f"PLE_{sampling_rate}"].append(lcc)

                last_sample_time = current_time

            if current_event_idx != prev_current_event_idx:
                if edge in snapshot_edges:
                    snapshot_edge_attrs[edge] = edge_attr[current_event_idx]
                else:
                    snapshot_edges.add(edge)
                    snapshot_edge_attrs[edge] = edge_attr[current_event_idx]

            prev_current_event_idx = current_event_idx

    for key in descriptors:
        descriptors[key] = torch.tensor(descriptors[key])  # type: ignore

    return descriptors


def calc_average_node_degree(static_graph):

    if static_graph.edge_index.shape[0] == 0:
        return 0.0

    num_nodes = static_graph.num_nodes
    num_edges = static_graph.edge_index.size(1)
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
    return avg_degree


def calc_largest_connected_component(static_graph):

    if static_graph.edge_index.shape[0] == 0:
        print("warning: empty graph")
        return 0.0
    G = to_networkx(static_graph, to_undirected=True)

    connected_components = nx.connected_components(G)

    largest_component_size = max(len(component) for component in connected_components)

    return largest_component_size


def calc_power_law_exponent(static_graph):

    num_nodes = static_graph.num_nodes
    edge_index = static_graph.edge_index

    if edge_index.shape[0] == 0:
        return 0.0

    out_degrees = degree(static_graph.edge_index[0], num_nodes=static_graph.num_nodes)
    in_degrees = degree(static_graph.edge_index[1], num_nodes=static_graph.num_nodes)

    degrees = out_degrees + in_degrees

    if degrees.numel() == 0:
        print("warning: empty graph")
        return 0.0

    d_min = torch.min(degrees).item()

    if d_min == 0:
        print("warning: graph has isolated nodes")
        return 0.0

    log_ratio_sum = torch.sum(torch.log(degrees / d_min)).item()

    ple = 1 + num_nodes / log_ratio_sum if log_ratio_sum > 0 else 0.0

    return ple


def calc_number_of_components(static_graph):
    if static_graph.edge_index.shape[0] == 0:
        print("warning: empty graph")
        return 0.0

    G = to_networkx(static_graph, to_undirected=True)

    num_components = nx.number_connected_components(G)

    return float(num_components)


def calc_feature_probabilities(data, metadata, types):

    descriptors = defaultdict(torch.Tensor)

    if "softmax" in types:
        msg = data.msg
        descriptors["softmax"] = msg

    return descriptors


def calc_dynamic_descriptors(data, metadata, types, node_dim, graph_dim):

    if types is None:
        types = ["jl_embd", "activity_rate"]

    descriptors = defaultdict(torch.Tensor)

    if "jl_embd" in types:
        jl_embedding = jl_metric(
            data,
            node_proj_dim=node_dim,
            graph_proj_dim=graph_dim,
            seed=metadata["seed"],
        )
        descriptors["jl_embd"] = jl_embedding

    if "activity_rate" in types:
        activity_rate = calc_activity_rate(data)
        descriptors["activity_rate"] = activity_rate

    return descriptors


def calc_activity_rate(data):

    num_nodes = max(data.src.max().item(), data.dst.max().item()) + 1
    activity_counts = torch.zeros(num_nodes, dtype=torch.long)

    activity_counts.index_add_(0, data.src, torch.ones_like(data.src, dtype=torch.long))
    activity_counts.index_add_(0, data.dst, torch.ones_like(data.dst, dtype=torch.long))

    return activity_counts


def load_perturbed_datasets_iter(root_dir):

    for dataset_name in os.listdir(root_dir):
        dataset_dir = Path(root_dir) / dataset_name
        for perturb_type in os.listdir(dataset_dir):
            perturb_dir = dataset_dir / perturb_type
            for seed_dir in os.listdir(perturb_dir):
                seed = int(seed_dir.split("_")[1])
                seed_path = perturb_dir / seed_dir
                for p_dir in os.listdir(seed_path):
                    p_value = float(p_dir.split("_")[1])
                    p_path = seed_path / p_dir

                    dataset_file = p_path / "perturbation.pt"
                    data = torch.load(dataset_file)

                    yield {
                        "data": data,
                        "metadata": {
                            "dataset_name": dataset_name,
                            "perturb_type": perturb_type,
                            "seed": seed,
                            "p": p_value,
                        },
                    }


def save_descriptors(descriptors, metadata, root_dir="../jl-metric/descriptors"):

    save_dir = (
        Path(root_dir)
        / metadata["dataset_name"]
        / metadata["perturb_type"]
        / f"seed_{metadata['seed']}"
        / f"p_{metadata['p']:.2f}"
    )
    os.makedirs(save_dir, exist_ok=True)

    for descriptor_name, descriptor_tensor in descriptors.items():
        save_path = save_dir / f"{descriptor_name}.pt"
        torch.save(descriptor_tensor, save_path)


def descriptors_exist(metadata, descriptor_root):

    descriptor_dir = os.path.join(
        descriptor_root,
        metadata["dataset_name"],
        metadata["perturb_type"],
        f"seed_{metadata['seed']}",
        f"p_{metadata['p']:.2f}",
    )

    return os.path.exists(descriptor_dir) and len(os.listdir(descriptor_dir)) > 0


def process_dataset(
    dataset_info,
    descriptor_root,
    node_dim,
    graph_dim,
    event_subset,
    static_types=None,
    feature_types=None,
    dynamic_types=None,
):
    data = dataset_info["data"]
    metadata = dataset_info["metadata"]

    if descriptors_exist(metadata, descriptor_root):
        click.echo(f"Descriptors already exist for {metadata}. Skipping...")
        return

    if event_subset is not None:
        data = data[:event_subset]
    click.echo(f"Successfully loaded CTDG for {metadata['dataset_name']}")

    nyquist = 1
    static_descriptors = calc_static_descriptors(
        data,
        metadata,
        nyquist_interval=nyquist,
        types=static_types or ["AND", "LCC", "NC", "PLE"],
    )
    click.echo("Successfully calculated static descriptors")

    feature_descriptors = calc_feature_probabilities(
        data,
        metadata,
        types=feature_types or ["softmax"],
    )
    click.echo("Successfully calculated feature descriptors")

    dynamic_descriptors = calc_dynamic_descriptors(
        data,
        metadata,
        types=dynamic_types or ["jl_embd", "activity_rate"],
        node_dim=node_dim,
        graph_dim=graph_dim,
    )
    click.echo("Successfully calculated dynamic descriptors")

    all_descriptors = {
        **static_descriptors,
        **feature_descriptors,
        **dynamic_descriptors,
    }

    save_descriptors(all_descriptors, metadata, descriptor_root)
    click.echo(f"Saved descriptors for {metadata}")


@click.command()
@click.option(
    "--perturbed-data-dir",
    default="./perturbed_data",
    help="Directory containing perturbed datasets",
)
@click.option(
    "--descriptor-root",
    default="./descriptors",
    help="Directory to save computed descriptors",
)
@click.option(
    "--node-dim", default=100, help="Dimension for node embeddings in JL metric"
)
@click.option(
    "--graph-dim", default=100, help="Dimension for graph embeddings in JL metric"
)
@click.option(
    "--event-subset",
    default=1000,
    help="Number of events to process from each dataset. Use -1 for all events",
)
@click.option(
    "--num-processes", default=64, help="Number of parallel processes for computation"
)
@click.option(
    "--static-types",
    "-s",
    multiple=True,
    type=click.Choice(["AND", "LCC", "NC", "PLE"], case_sensitive=True),
    default=["AND", "LCC", "NC", "PLE"],
    help="Static descriptor types to compute",
)
@click.option(
    "--feature-types",
    "-f",
    multiple=True,
    type=click.Choice(["softmax"], case_sensitive=True),
    default=["softmax"],
    help="Feature descriptor types to compute",
)
@click.option(
    "--dynamic-types",
    "-d",
    multiple=True,
    type=click.Choice(["jl_embd", "activity_rate"], case_sensitive=True),
    default=["jl_embd", "activity_rate"],
    help="Dynamic descriptor types to compute",
)
def main(
    perturbed_data_dir,
    descriptor_root,
    node_dim,
    graph_dim,
    event_subset,
    num_processes,
    static_types,
    feature_types,
    dynamic_types,
):
    """Calculate various graph descriptors for perturbed temporal interaction datasets."""

    click.echo(f"Calculating descriptors with:")
    click.echo(f"- Node dimension: {node_dim}")
    click.echo(f"- Graph dimension: {graph_dim}")
    click.echo(f"- Events per dataset: {event_subset if event_subset != -1 else 'all'}")
    click.echo(f"- Using {num_processes} parallel processes")
    click.echo(f"- Static descriptors: {', '.join(static_types)}")
    click.echo(f"- Feature descriptors: {', '.join(feature_types)}")
    click.echo(f"- Dynamic descriptors: {', '.join(dynamic_types)}")

    # Convert event_subset to None if -1
    if event_subset == -1:
        event_subset = None

    # Load datasets
    datasets = list(load_perturbed_datasets_iter(perturbed_data_dir))
    click.echo(f"Found {len(datasets)} datasets to process")

    # Create partial function with configured parameters
    func = partial(
        process_dataset,
        descriptor_root=descriptor_root,
        node_dim=node_dim,
        graph_dim=graph_dim,
        event_subset=event_subset,
        static_types=static_types,
        feature_types=feature_types,
        dynamic_types=dynamic_types,
    )

    # Process datasets in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(func, datasets)

    click.echo("Finished calculating descriptors for all datasets.")


if __name__ == "__main__":
    main()
