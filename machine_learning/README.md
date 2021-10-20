# Machine Learning

This is the repository containing the machine learning code for the paper "Towards a metagenomics machine learning interpretable model for understanding the transition from adenoma to colorectal cancer".

## Setup

### Software

Yo need the [conda](https://docs.conda.io/en/latest/index.html) package manager to create a virtual environment where the code can be executed. We recommend the [miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution. Once a version of `conda` is installed and the current repository is cloned, run the following command from the ML project root:
```
conda env create -p ./.venv -f environment.yml
```
To activate the environment run:
```
conda activate ./.venv
```

The user need to create an environment file (called `.env`) on the ML project root. An example environment file is provided in `example.env`, just copy it to `.env` and change the variables in accordance with the user paths.

Note that the code has only been tested in `GNU/Linux x64`.

### Data

First the data must be downloaded and prrocessed using the guidelness provided in the bioinformatics repository. Once the bioinformatics pipeline has ended the data should be synced to `data` using the `sync` method provided in `main.py`. The ML datasets should be built using the `build_data_sources` method of `main.py`.

### Experiments

All the CV splits, training and evaluation can be performed with the `train_interpreter` method of `main.py`.

The `main.sh` BASH file could be used as an example of how to run the experiments (this is how the paper experiments were done).

The summary graphics and tables can be produced with the `mlgut/analysis.py` file. Note that the analysis is automatically performed as part of the main pipeline.


## Results summary

![Rank stability analysis](figures/crc_signature_permutation_analysis.svg?raw=true "Rank stability analysis")

![Performance analysis](figures/crc_auroc_analysis.svg?raw=true "Performance analysis")

![Stability analysis](figures/crc_stability_analysis.svg?raw=true "Stability analysis")

![Rank stability analysis](figures/crc_rank_analysis.svg?raw=true "Rank stability analysis")

![Adenoma analysis](figures/crc_adenoma_radar.svg?raw=true "Adenoma analysis")

![Kegg cross-project analysis](figures/crc_kegg_kos_score_matrix.svg?raw=true "Kegg cross-project analysis")

![Taxonomic cross-project analysis](figures/crc_centrifuge_score_matrix.svg?raw=true "Taxonomic cross-project analysis")

![eggNog cross-project analysis](figures/crc_ogs_score_matrix.svg?raw=true "eggNog cross-project analysis")
