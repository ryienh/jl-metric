import torch
from scipy.stats import kstest
import pandas as pd
from torch_geometric.data import TemporalData

from mmd import mmd_linear
from perturb_fidelity import get_datasets
from function_descriptors import (
    calc_static_descriptors,
    calc_dynamic_descriptors,
)
from distances import calculate_cosine_similarity


N_MAX = 1000


def strip_ctdg(ctdg):
    new_ctdg = TemporalData(
        src=ctdg.src, dst=ctdg.dst, t=ctdg.t, msg=torch.ones((ctdg.src.shape[0], 2))
    )

    return new_ctdg


def calculate_average_idx(results):
    sum_count = {}

    for dataset, global_results in results.items():
        for pair_key, idx_value in global_results.items():
            if pair_key not in sum_count:
                sum_count[pair_key] = {"sum": 0, "count": 0}

            sum_count[pair_key]["sum"] += idx_value
            sum_count[pair_key]["count"] += 1

    data = {"Metric-Distance Pair": [], "Average idx": []}

    for pair_key, values in sum_count.items():
        average_idx = values["sum"] / values["count"]
        data["Metric-Distance Pair"].append(pair_key)
        data["Average idx"].append(average_idx)

    df = pd.DataFrame(data)
    print(df)


def calc_distances(S_r_1_desc, S_r_2_desc, S_g_desc):

    result_rr = {}
    result_rg = {}

    for metric in S_r_1_desc.keys():

        print(f"Calculating {metric}")

        if metric not in result_rr:
            result_rr[metric] = {}
        if metric not in result_rg:
            result_rg[metric] = {}

        if metric != "jl_embd":

            ks_stat_rr, _ = kstest(
                S_r_1_desc[metric].numpy(),
                S_r_2_desc[metric].numpy(),
            )
            result_rr[metric]["ks_distances"] = ks_stat_rr

            ks_stat_rg, _ = kstest(
                S_r_1_desc[metric].numpy(),
                S_g_desc[metric].numpy(),
            )
            result_rg[metric]["ks_distances"] = ks_stat_rg

            mmd_dist_rr = mmd_linear(
                S_r_1_desc[metric].to(torch.float32).reshape(-1, 1).numpy(),
                S_r_2_desc[metric].to(torch.float32).reshape(-1, 1).numpy(),
            )
            result_rr[metric]["mmd_distances"] = mmd_dist_rr.item()

            mmd_dist_rg = mmd_linear(
                S_r_1_desc[metric].to(torch.float32).reshape(-1, 1).numpy(),
                S_g_desc[metric].to(torch.float32).reshape(-1, 1).numpy(),
            )
            result_rg[metric]["mmd_distances"] = mmd_dist_rg.item()

        if metric == "jl_embd":

            cosine_sim_rr = calculate_cosine_similarity(
                S_r_1_desc[metric], S_r_2_desc[metric]
            )
            result_rr[metric]["cosine_distances"] = cosine_sim_rr

            cosine_sim_rg = calculate_cosine_similarity(
                S_r_1_desc[metric], S_g_desc[metric]
            )
            result_rg[metric]["cosine_distances"] = cosine_sim_rg

            ks_stat_rr, _ = kstest(
                torch.mean(S_r_1_desc[metric], dim=0).numpy(),
                torch.mean(S_r_2_desc[metric], dim=0).numpy(),
            )
            result_rr[metric]["ks_distances"] = ks_stat_rr

            ks_stat_rg, _ = kstest(
                torch.mean(S_r_1_desc[metric], dim=0).numpy(),
                torch.mean(S_g_desc[metric], dim=0).numpy(),
            )
            result_rg[metric]["ks_distances"] = ks_stat_rg

    return result_rr, result_rg


def calculate_sample_efficiency(dataset, gt, seed=442):

    global_results = {}

    for idx in range(3, N_MAX // 2):

        S_r_1 = dataset[:idx]
        S_r_2 = dataset[-idx:]
        S_g = gt[:idx]

        metadata = {"seed": seed}

        S_r_1_static = calc_static_descriptors(
            data=S_r_1, metadata=metadata, nyquist_interval=1
        )

        S_r_2_static = calc_static_descriptors(
            data=S_r_2, metadata=metadata, nyquist_interval=1
        )

        S_g_static = calc_static_descriptors(
            data=S_g, metadata=metadata, nyquist_interval=1
        )

        S_r_1_dynamic = calc_dynamic_descriptors(
            data=S_r_1,
            metadata=metadata,
            types=["jl_embd", "activity_rate"],
            node_dim=100,
            graph_dim=100,
        )

        S_r_2_dynamic = calc_dynamic_descriptors(
            data=S_r_2,
            metadata=metadata,
            types=["jl_embd", "activity_rate"],
            node_dim=100,
            graph_dim=100,
        )

        S_g_dynamic = calc_dynamic_descriptors(
            data=S_g,
            metadata=metadata,
            types=["jl_embd", "activity_rate"],
            node_dim=100,
            graph_dim=100,
        )

        S_r_1_all = {**S_r_1_static, **S_r_1_dynamic}
        S_r_2_all = {**S_r_2_static, **S_r_2_dynamic}
        S_g_all = {**S_g_static, **S_g_dynamic}

        rr, rg = calc_distances(S_r_1_all, S_r_2_all, S_g_all)

        for metric in rr.keys():
            for distance_type in rr[metric].keys():
                pair_key = f"{metric}_{distance_type}"

                if (
                    rr[metric][distance_type] < rg[metric][distance_type]
                    and pair_key not in global_results
                ):
                    global_results[pair_key] = idx

    return global_results


if __name__ == "__main__":

    # grid info must match that in tgn_train_and_save.py
    datasets = get_datasets(grid_nn=10000, grid_max_time=10000, grid_seed=101)[0]

    dataset_names = ["Reddit", "Wikipedia", "MOOC", "LastFM", "Grid"]

    trimmed_datasets = []
    for dataset in datasets:
        trimmed_datasets.append(dataset[:N_MAX])  # type: ignore

    stripped_datasets = []
    for dataset in trimmed_datasets:
        sdata = strip_ctdg(dataset)
        stripped_datasets.append(sdata)  # type: ignore

    gt = datasets[-1]

    results = {}
    for dataset, name in zip(stripped_datasets, dataset_names):
        if name != "Grid":
            print(f"Calculating efficiency for {name}")
            results[name] = calculate_sample_efficiency(dataset, gt)

    calculate_average_idx(results)
