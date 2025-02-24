"""
Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
"""

import os.path as osp
import os
import click
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from grid import generate_synthetic_ctdg


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


def load_dataset(dataset_name, data_path):
    """Load dataset based on name."""
    if dataset_name.lower() != "grid":
        dataset = JODIEDataset(data_path, name=dataset_name)
        data = dataset[0]
        click.echo(f"Loaded {dataset_name} dataset from JODIE")
    else:
        click.echo("Loading synthetic grid dataset")
        dataset = generate_synthetic_ctdg(
            num_nodes=10000, max_time=10000, random_seed=101
        )
        data = dataset
        data.t = data.t.to(torch.int64)

    return data


def train_epoch(
    train_loader,
    memory,
    gnn,
    link_pred,
    optimizer,
    criterion,
    device,
    data,
    neighbor_loader,
    assoc,
    train_data,
):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, memory, gnn, link_pred, device, data, neighbor_loader, assoc):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        )

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


@click.command()
@click.option(
    "--dataset",
    "dataset_name",
    type=click.Choice(
        ["Reddit", "Wikipedia", "MOOC", "LastFM", "Grid"], case_sensitive=False
    ),
    default="Grid",
    help="Dataset to train on",
)
@click.option("--model-dir", default="./tgn", help="Directory to save the model")
@click.option("--data-path", default="./jodie_data", help="Path to the data directory")
@click.option("--epochs", default=50, help="Number of epochs to train")
@click.option("--batch-size", default=200, help="Batch size for training")
def main(dataset_name, model_dir, data_path, epochs, batch_size):
    """Train a Temporal Graph Network (TGN) on temporal interaction data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    save_path = osp.join(model_dir, f"{dataset_name}.pt")

    # Load dataset
    data = load_dataset(dataset_name, data_path)
    data = data.to(device)

    # Split data
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15
    )

    # Initialize loaders
    train_loader = TemporalDataLoader(
        train_data,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )
    val_loader = TemporalDataLoader(
        val_data,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )
    test_loader = TemporalDataLoader(
        test_data,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

    # Initialize model components
    memory_dim = time_dim = embedding_dim = 100

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=0.0001,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # Training loop
    best_val_ap = 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            train_loader,
            memory,
            gnn,
            link_pred,
            optimizer,
            criterion,
            device,
            data,
            neighbor_loader,
            assoc,
            train_data,
        )

        click.echo(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

        val_ap, val_auc = test(
            val_loader, memory, gnn, link_pred, device, data, neighbor_loader, assoc
        )
        test_ap, test_auc = test(
            test_loader, memory, gnn, link_pred, device, data, neighbor_loader, assoc
        )

        click.echo(f"Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}")
        click.echo(f"Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}")

        if val_ap > best_val_ap:
            torch.save(memory, save_path)
            best_val_ap = val_ap
            click.echo(f"Saved new best model to {save_path}")

    click.echo("Training completed!")


if __name__ == "__main__":
    main()
