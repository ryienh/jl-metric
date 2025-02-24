import torch
import torch_geometric
from torch_geometric.data import TemporalData
import random


def perturb_ctdg(
    original_ctdg, num_perturbations, random_seed, perturb_type="edge_rewire"
):
    torch_geometric.seed_everything(random_seed)

    perturbed_graphs = []
    num_edges = original_ctdg.src.size(0)

    # Calculate the perturbation probabilities, evenly spaced between 0 and 1
    perturbation_probs = torch.linspace(0, 1, steps=num_perturbations)

    for p in perturbation_probs:
        if perturb_type == "edge_rewire":
            perturbed_dst = original_ctdg.dst.clone()
            for i in range(num_edges):
                if random.random() < p:
                    perturbed_dst[i] = random.randint(0, original_ctdg.src.max().item())
            perturbed_ctdg = TemporalData(
                src=original_ctdg.src.clone(),
                dst=perturbed_dst,
                t=original_ctdg.t.clone(),
                msg=original_ctdg.msg.clone(),
            )

        elif perturb_type == "time":
            perturbed_t = original_ctdg.t.clone()
            for i in range(num_edges):
                if random.random() < p:
                    # Get the previous and next timestamps
                    prev_t = original_ctdg.t[i - 1] if i > 0 else original_ctdg.t[i]
                    next_t = (
                        original_ctdg.t[i + 1]
                        if i < num_edges - 1
                        else original_ctdg.t[i]
                    )

                    perturbed_t[i] = random.randint(prev_t + 1, next_t - 1)

            # new method does not change order/topology
            perturbed_ctdg = TemporalData(
                src=original_ctdg.src.clone(),
                dst=original_ctdg.dst.clone(),
                t=perturbed_t.clone(),
                msg=original_ctdg.msg.clone(),
            )

        # modified so events are swapped rather than copied, should be more percise perturbation
        elif perturb_type == "events":
            perturbed_src = original_ctdg.src.clone()
            perturbed_dst = original_ctdg.dst.clone()
            perturbed_t = original_ctdg.t.clone()
            perturbed_msg = original_ctdg.msg.clone()

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

            # Create the perturbed TemporalData
            perturbed_ctdg = TemporalData(
                src=perturbed_src,
                dst=perturbed_dst,
                t=perturbed_t,
                msg=perturbed_msg,
            )

        else:
            raise ValueError(
                "Unknown perturb_type. Choose from 'edge_rewire', 'time', or 'events'."
            )

        perturbed_graphs.append(perturbed_ctdg)

    return perturbed_graphs, perturbation_probs


def generate_synthetic_ctdg(num_nodes, max_time, random_seed):
    torch_geometric.seed_everything(random_seed)

    # Determine grid size
    grid_size = int(torch.sqrt(torch.tensor(num_nodes)).item())

    # Lists to hold event data
    src_nodes = []
    dst_nodes = []
    times = []
    msgs = []

    time_interval = max_time // (
        grid_size**2
    )  # Spread events uniformly across the timeline

    # Create events by connecting nodes in a grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j

            # Connect to the right neighbor
            if j < grid_size - 1:
                right_neighbor = node + 1
                src_nodes.append(node)
                dst_nodes.append(right_neighbor)
                times.append(len(src_nodes) * time_interval)

                t = len(src_nodes) * time_interval
                msgs.append([i * t, j + t])  # Example message for edge creation

            # Connect to the bottom neighbor
            if i < grid_size - 1:
                bottom_neighbor = node + grid_size
                src_nodes.append(node)
                dst_nodes.append(bottom_neighbor)
                times.append(len(src_nodes) * time_interval)

                t = len(src_nodes) * time_interval
                msgs.append([i * t, j + t])  # Another example

    # Convert lists to tensors
    src_nodes = torch.tensor(src_nodes, dtype=torch.long)
    dst_nodes = torch.tensor(dst_nodes, dtype=torch.long)
    times = torch.tensor(times, dtype=torch.float)
    msgs = torch.tensor(msgs, dtype=torch.float)

    # Create the TemporalData object
    ctdg = TemporalData(src=src_nodes, dst=dst_nodes, t=times, msg=msgs)

    return ctdg


if __name__ == "__main__":
    # Example usage
    num_nodes = 16
    max_time = 10000  # Max time for the event sequence
    random_seed = 42

    synthetic_ctdg = generate_synthetic_ctdg(num_nodes, max_time, random_seed)

    print(synthetic_ctdg)
