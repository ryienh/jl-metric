import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import click
import os


def create_violin_plot(
    correlations, output_dir="./figs", remove_entries=None, label_map=None
):
    perturbation_data = defaultdict(list)

    for key, corr in correlations.items():
        key_parts = key.split("_")

        for idx, part in enumerate(key_parts):
            if part.startswith("seed"):
                perturb_type = "_".join(key_parts[1:idx])
                metric_name = "_".join(key_parts[idx + 2 :])
                break

        # Remove LastFM from event task -- dataset contains no events
        if key_parts[0] == "LastFM" and key_parts[1] == "event":
            assert key_parts[2] == "permutation"
            continue

        perturbation_data[perturb_type].append((metric_name, corr))

    # Create a violin plot for each perturbation type
    for perturb_type, values in perturbation_data.items():

        print(f"Plotting for perturb type: {perturb_type}")
        df = pd.DataFrame(values, columns=["Metric", "Correlation"])

        if remove_entries:
            df = df[~df["Metric"].isin(remove_entries)]

        if perturb_type == "event_permutation":
            df.loc[df["Metric"] == "softmax_mmd", "Correlation"] = 0.0

        df["Suffix"] = df["Metric"].apply(lambda x: x.split("_")[-1])
        df["Base"] = df["Metric"].apply(lambda x: x.split("_")[0])

        # Sort with `jl_embd_cosine` last
        jl_embd_cosine_df = df[df["Metric"] == "jl_embd_cosine"]
        df = df[df["Metric"] != "jl_embd_cosine"]
        df = pd.concat([df.sort_values(by=["Base", "Suffix"]), jl_embd_cosine_df])

        unique_bases = df["Base"].unique()
        colors = sns.color_palette("husl", len(unique_bases))
        base_to_color = dict(zip(unique_bases, colors))

        plt.rcParams.update(
            {
                "font.size": 18,
                "axes.titlesize": 18,
                "axes.labelsize": 18,
                "xtick.labelsize": 15,
                "ytick.labelsize": 18,
                "legend.fontsize": 18,
            }
        )

        if perturb_type == "edge_rewiring":
            plt.figure(figsize=(15, 6))
        else:
            plt.figure(figsize=(15, 4))

        ax = plt.gca()
        ax.xaxis.set_ticks_position("top")

        grouped_stats = (
            df.groupby(["Metric", "Suffix"])["Correlation"]
            .agg(Median="median")
            .reset_index()
        )

        print(grouped_stats)

        if label_map:
            df["Metric"] = df["Metric"].map(label_map).fillna(df["Metric"])

        sns.violinplot(
            x="Metric",
            y="Correlation",
            data=df,
            hue="Base",
            scale="width",
            palette=base_to_color,
            dodge=False,
            cut=0,
            bw_method=0.3,
            legend=False,
        )

        if perturb_type == "edge_rewiring":
            plt.xlabel("Edge Rewiring")
        elif perturb_type == "event_permutation":
            plt.xlabel("Event Permutation")
        elif perturb_type == "temporal_perturbation":
            plt.xlabel("Time Perturbation")
        elif perturb_type == "mode_droping":
            plt.xlabel("Mode Dropping")
        elif perturb_type == "mode_collapse":
            plt.xlabel("Mode Collapse")
        elif perturb_type == "feature_perturbation":
            plt.xlabel("Event Perturbation")
        plt.xticks(rotation=90, ha="right")
        if perturb_type != "edge_rewiring":  # Set this condition depending on run
            plt.gca().set_xticklabels([])

        plt.ylim((-0.5, 1.0))
        plt.yticks((-0.5, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/violin_plot_{perturb_type}.pdf")
        plt.close()


@click.command()
@click.option(
    "--correlation-file", default="./correlations.pt", help="Path to saved correlations"
)
@click.option(
    "--output-dir", default="./violin_plots", help="Directory to save violin plots"
)
def main(correlation_file, output_dir):
    """Generate violin plots from correlation data."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Loading correlations from {correlation_file}")
    correlations = torch.load(correlation_file)

    keywords = [
        "Wikipedia",
        "Reddit",
        "LastFM",
        "MOOC",
        "Grid",
    ]
    filtered_correlations = {
        key: value
        for key, value in correlations.items()
        if any(keyword in key for keyword in keywords)
    }

    label_map = {
        "jl_embd_cosine": "JL Metric (Ours)",
        "AND_1.0_ks": "Node Degree (KS)",
        "AND_1.0_mmd": "Node Degree (MMD)",
        "LCC_1.0_ks": "LCC (KS)",
        "LCC_1.0_mmd": "LCC (MMD)",
        "NC_1.0_ks": "NC (KS)",
        "NC_1.0_mmd": "NC (MMD)",
        "PLE_1.0_ks": "PLE (KS)",
        "PLE_1.0_mmd": "PLE (MMD)",
        "activity_rate_ks": "Activity Rate (KS)",
        "activity_rate_mmd": "Activity Rate (MMD)",
        "softmax_js": "Feat. Distance (JS)",
        "softmax_kl": "Feat. Distance (KL-Div.)",
        "softmax_ks": "Feat. Distance (KS)",
        "softmax_mmd": "Feat. Distance (MMD)",
    }

    remove_entries = ["jl_embd_mmd", "jl_embd_ks"]

    create_violin_plot(
        filtered_correlations,
        output_dir=output_dir,
        remove_entries=remove_entries,
        label_map=label_map,
    )
    click.echo(f"Violin plots saved in {output_dir}")


if __name__ == "__main__":
    main()
