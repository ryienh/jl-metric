import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import TemporalData


def create_node_representations(events, embd):

    node_reps = {}

    # Min-max norm for messages
    msg_min = events.msg.min(dim=0)[0]
    msg_max = events.msg.max(dim=0)[0]
    normalized_msgs = (events.msg - msg_min) / (
        msg_max - msg_min + 1e-6
    )  # Avoid division by zero

    # Min-max norm for time
    time_min = events.t.min()
    time_max = events.t.max()
    normalized_times = (events.t - time_min) / (time_max - time_min + 1e-6)

    for i in range(events.src.size(0)):
        src, dst = int(events.src[i].item()), int(events.dst[i].item())
        msg = normalized_msgs[i]
        time_enc = normalized_times[i].reshape(1)

        combined_src = torch.cat([msg, time_enc, torch.tensor([0])], dim=0)
        combined_dst = torch.cat([msg, time_enc, torch.tensor([1])], dim=0)

        if src not in node_reps:
            node_reps[src] = []
        if dst not in node_reps:
            node_reps[dst] = []

        node_reps[src].append(combined_src)
        node_reps[dst].append(combined_dst)

    for node, reps in node_reps.items():
        node_reps[node] = torch.cat(reps) @ embd[: (reps[0].shape[0]) * len(reps)]

    return node_reps


def jl_project_graph_level(node_reps, proj_dim):
    node_dim = list(node_reps.values())[0].size(0)
    num_nodes = len(node_reps.keys())

    node_reps = {index + 1: feature for index, feature in enumerate(node_reps.values())}

    # Unstructured ROM-based here. For SRM, directly use: https://github.com/joneswack/dp-rfs
    orthonormal_matrix = torch.randn((num_nodes, proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / num_nodes)
    )

    node_matrix = torch.zeros((num_nodes, node_dim))

    for node_id, rep in node_reps.items():
        node_matrix[int(node_id) - 1] = rep

    graph_rep = node_matrix.T @ orthonormal_matrix
    return graph_rep


def jl_metric(events, node_proj_dim, graph_proj_dim, seed):

    torch_geometric.seed_everything(seed)

    MAX_EVENTS = 100000  # hardcode

    # Unstructured ROM-based here. For SRM, directly use: https://github.com/joneswack/dp-rfs
    embd = torch.randn((MAX_EVENTS, node_proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / MAX_EVENTS)
    )

    node_reps = create_node_representations(events, embd)
    graph_rep = jl_project_graph_level(node_reps, graph_proj_dim)
    return graph_rep


if __name__ == "__main__":

    # Example TemporalData
    events = TemporalData(
        src=Tensor([1, 2, 3, 4]),
        dst=Tensor([2, 3, 4, 5]),
        t=Tensor([1000, 1010, 1100, 2000]),
        msg=Tensor([[1, 0], [1, 0], [0, 1], [0, 1]]),  # Assuming msg has 2D features
    )

    # Set dimensions
    feature_dim = events.msg.size(1)  # type: ignore
    node_proj_dim = 64
    graph_proj_dim = 64
    seed = 442

    graph_rep = jl_metric(events, node_proj_dim, graph_proj_dim, seed)

    print("Graph Representation dimension: ", graph_rep)
    print(graph_rep.shape)
