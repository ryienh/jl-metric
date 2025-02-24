# Quality Measures for Continuous-Time Dynamic Graph Generative Models

This repository contains the implementation of experiments presented in *Quality Measures for Continuous-Time Dynamic Graph Generative Models*.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Installing PyTorch and PyTorch Geometric may require platform-specific procedures. Please refer to the official documentation for [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Reproducing Experiments

The experiments presented in Section 4 of our paper can be reproduced through the following steps:

1. Generate perturbed datasets
2. Calculate function descriptors
3. Compute distances and correlations
4. Generate visualizations and analysis

Each step is detailed below.

### Generate Perturbed Data

#### Fidelity Experiments
Generate perturbed datasets using our perturbation methods:
```bash
python perturb_fidelity.py [OPTIONS]
```

The script provides a command-line interface with several customizable parameters. Default values match our experimental settings, but can be modified according to needs:
```bash
Options:
  --output-dir TEXT        Directory to save perturbed datasets [default: ./perturbed_data]
  --datasets TEXT          Datasets to process. Use "all" or specify individual ones:
                          Reddit, Wikipedia, MOOC, LastFM, Grid [default: all]
  --num-seeds INTEGER      Number of random seeds for reproducibility [default: 10]
  --subset-events INTEGER  Number of events to use from each dataset. Use -1 for
                          all events [default: 1000]
  --jodie-path TEXT       Path to save JODIE datasets [default: ./data/]
  --grid-nn INTEGER       Number of nodes for synthetic grid dataset [default: 1024]
  --grid-max-time INTEGER Maximum time for synthetic grid dataset [default: 10000]
  --grid-seed INTEGER     Random seed for synthetic grid dataset [default: 101]
  --help                  Show this message and exit
```

The JODIE [1] datasets (Reddit, Wikipedia, MOOC, LastFM) will download automatically upon first execution. All datasets are used according to their respective licenses.

#### Diversity Experiments
Diversity experiments require two steps:

1. Train Temporal Graph Networks (TGN) [2]:
```bash
python train_tgn.py [OPTIONS]
```

Available options:
```bash
Options:
  --dataset TEXT        Dataset to train on (Reddit, Wikipedia, MOOC, LastFM,
                       Grid) [default: Grid]
  --model-dir TEXT     Directory to save the model [default: ./tgn]
  --data-path TEXT     Path to the data directory [default: ./data]
  --epochs INTEGER     Number of epochs to train [default: 50]
  --batch-size INTEGER Batch size for training [default: 200]
  --help              Show this message and exit
```

2. Generate perturbations using trained models:
```bash
python perturb_diversity.py [OPTIONS]
```

Available options:
```bash
Options:
  --output-dir TEXT      Directory to save perturbed datasets [default: ./DATA_DIR]
  --model-dir TEXT       Directory containing trained TGN models [default: ./tgn]
  --num-events INTEGER   Number of events to use from each dataset [default: 1000]
  --num-seeds INTEGER    Number of random seeds for perturbations [default: 5]
  --cache-dir TEXT       Directory to cache computed modes [default: modes_cache]
  --datasets TEXT        Datasets to process. Use "all" or specify individual ones
  --grid-nn INTEGER      Number of nodes for synthetic grid [default: 10000]
  --grid-max-time INTEGER Maximum time for synthetic grid [default: 10000]
  --grid-seed INTEGER    Random seed for synthetic grid [default: 101]
  --help                Show this message and exit
```

### Calculate Function Descriptors

Compute descriptors for perturbed datasets:
```bash
python calc_descriptors.py [OPTIONS]
```

Available options:
```bash
Options:
  --descriptor-root TEXT     Directory containing computed descriptors
                            [default: ./DESCRIPTOR_DIR]
  --node-dim INTEGER        Dimension for node embeddings in JL metric [default: 100]
  --graph-dim INTEGER       Dimension for graph embeddings in JL metric [default: 100]
  --event-subset INTEGER    Number of events to process. Use -1 for all [default: 1000]
  --num-processes INTEGER   Number of parallel processes [default: 64]
  --static-types TEXT       Static descriptor types to compute (AND, LCC, NC, PLE)
  --feature-types TEXT      Feature descriptor types to compute (softmax)
  --dynamic-types TEXT      Dynamic descriptor types to compute (jl_embd, activity_rate)
  --help                    Show this message and exit
```

### Calculate Distances and Correlations

Compute distances between original and perturbed descriptors:
```bash
python distances.py [OPTIONS]
```

Available options:
```bash
Options:
  --descriptor-root TEXT           Directory containing computed descriptors
                                  [default: ./DESCRIPTOR_DIR]
  --correlation-type [pearson|spearman]
                                  Type of correlation to compute [default: spearman]
  --progress-file TEXT            File to save/load intermediate progress
  --correlation-file TEXT         File to save final correlations
  --unnormalized-correlation-file TEXT
                                  File to save unnormalized correlations
  --batch-size INTEGER            Batch size for MMD calculations [default: 10000]
  --help                          Show this message and exit
```

### Generate Visualizations

Generate correlation distribution plots:
```bash
python correlation_distribution_plots.py [OPTIONS]
```

Available options:
```bash
Options:
  --correlation-file TEXT  Path to saved correlations [default: ./correlations.pt]
  --output-dir TEXT       Directory to save violin plots [default: ./VIZ_DIR]
  --help                  Show this message and exit
```

The script generates violin plots showing correlation distributions across perturbation types and metrics, and prints median correlation values corresponding to Table 1 in our paper.

Calculate sample and computational efficiency:
```bash
python sample_efficiency.py
python computational_efficiency.py
```

## Reproducibility

All experiments are seeded for reproducibility. Each experiment is run 10 times with random seeds from 0 to 9 (incrementing by 1 for each run).

## Citation

```bibtex
@inproceedings{jl-metric,
    title={Quality Measures for Dynamic Graph Generative Models},
    author={Ryien Hosseini and Filippo Simini and Venkatram Vishwanath and Rebecca Willett and Henry Hoffmann},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=8bjspmAMBk}
}
```

## References

[1] Kumar, S., Zhang, X., & Leskovec, J. (2019). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

[2] Rossi, E., et al. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637.