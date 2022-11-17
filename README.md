<a href="https://www.jpmorgan.com/technology/artificial-intelligence">
<img align="middle" src="./assets/jpmorgan-logo.svg" alt="JPMorgan Logo" height="40">
<img align="middle" src="./assets/xai_coe-logo.png" alt="Explainale AI Center of Excellence Logo" height="75">
</a>

[![License](https://img.shields.io/github/license/jpmorganchase/cf-shap-facct22)](https://github.com/jpmorganchase/cf-shap-facct22/blob/master/LICENSE)
[![Contact](https://img.shields.io/badge/contact-Emanuele_Albini-lightgrey)](https://www.emanuelealbini.com)


# Counterfactual Shapley Additive Explanations: Experiments

This repository contains the code to run the experiments presented in the following paper.

**Counterfactual Shapley Additive Explanations**  
Emanuele Albini, Jason Long, Danial Dervovic and Daniele Magazzeni  
J.P. Morgan AI Research  
[ACM](https://dl.acm.org/doi/abs/10.1145/3531146.3533168) | [ArXiv](https://arxiv.org/abs/2110.14270)

```
@inproceedings{Albini2022,
  title = {Counterfactual {{Shapley Additive Explanations}}},
  booktitle = {2022 {{ACM Conference}} on {{Fairness}}, {{Accountability}}, and {{Transparency}}},
  author = {Albini, Emanuele and Long, Jason and Dervovic, Danial and Magazzeni, Daniele},
  year = {2022},
  series = {{{FAccT}} '22},
  pages = {1054--1070},
  doi = {10.1145/3531146.3533168}
}
```

**Note that this repository only contains the code to reproduce the experiments presented in the paper.**
**If you are instead looking for the actual algorithms proposed in the paper see the [cf-shap](https://www.github.com/jpmorganchase/cf-shap) repository (that the scripts in this repo depends on).**

_NOTE: This repository is provided as is, and it is not actively maintained._

## 1. Experiments

## 1.1 Environment Setup

Note all the experiments have been run an **Ubuntu 16.04** machine in a conda environment with the following channels. 
Make sure that you have the following channels in your conda setup.

```bash
$ conda config --show channels
channels:
  - defaults
  - conda-forge
```

### 1.1.A Clone environemnt from YML (`cfshap22.yml`)
To create an exact copy of the environment used for the experiments, run the following command. Note that you must use an Ubuntu 16.04 machine to do this. This will not work on Windows. We strongly recommend using an Ubuntu machine for reproducibility.

```bash
conda env create -f cfshap22.yml
```

### 1.1.B Create environment from scratch (`requirements.txt`)
_Note that since this is not a snapshot (like the YML file) the environment might not be exactly the same as the one used for the experiments._
The package installed are reported in the `requirements.txt` file.  In order to setup the environment one can follow the following setup.

```bash

```bash
# Create a new Python 3.6 environment (with conda)
conda create --yes --name cfshap22 python=3.6

# Activate the environment
conda activate cfshap22

# Check that the right environment has been activated
# (the path must contain .../cfshap22/...)
which python
which pip

# Install requirements prioritizing conda packages (over pip)
while read requirement; do echo $requirement; conda install --yes $requirement || pip install $requirement; done < requirements.txt 
```
### > Reset Environment

In order to delete the environment and start from scratch use the following command.
```bash
# Deactivate environament: we cannot delete active environments
conda deactivate

# Delete environment
conda remove --yes --name cfshap22 --all
```

## 1.2 Install package with algorithms and download datasets

```bash
# Activate the environment
conda activate cfshap22

# Clone the package with the algorithms
git clone https://github.com/jpmorganchase/cf-shap.git
# git clone git@github.com:jpmorganchase/cf-shap.git

# Install the package in editable mode
pip install -e cf-shap

# Download lending club dataset
# You must setup your Kaggle API in order to dowload this dataset (see Kaggle website at https://www.kaggle.com/docs/api)
kaggle datasets download --unzip -d wordsforthewise/lending-club -p ./cf-shap/src/emutils/data/lendingclub
```

## 1.3 Run experiments

1. Before running any experiment, we must pre-process the data and train the models. The hyperparameters of the models are reported in the `json` files in the `models` directory. The pre-processing and training can be done with the following command:
```bash

# Activate environment
conda activate cfshap22

# Run the preprocessing
./preprocess.sh

# Train the models
./model.sh
```

2. Then we can generate the explanations and run the experiments with the following command:
```bash
# Compute the explanations
./explanations.sh

# Run the experiments on counterfactual-ability and plausibility with the induced counterfactuals
./experiments.sh

# Compute the explanations runtime
./explanations_runtime.sh
```

3. Export the results of the expertimennt in plots and tables.

(3.A) In order to export the results for the experiments on _counterfactual-ability_ and _plausibility_ use the `Export - Main Experiments.ipynb`. It will generate most of the plots and tables in the paper.

(3.B) In order to export the experiments on _explanations computation runtime_ use the `Export - Computation Time.ipynb` to generate the plots contained in the paper.

(3.C) In order to run the experiments on _explanations feature additivity_ use the `Export - Feature Additivity.ipynb` to generate the plots contained in the paper.


### > Cleanup experiments

```bash
# Remove pre-processed data
rm -R ./data

# Remove models (keeping the hyperparameters)
mv ./models/*.json .
rm -R ./models
mkdir ./models
mv *.json ./models

# Remove experiments results and exports
rm -R ./explanations
rm -R ./experiments
rm -R ./results
rm -R ./export
```

## 2. Contacts

For further information or queries on this work you can contact the _Explainable AI Center of Excellence at J.P. Morgan_ ([xai.coe@jpmchase.com](mailto:xai.coe@jpmchase.com)) or [Emanuele Albini](https://www.emanuelealbini.com), the main author of the paper.

## 3. Disclamer

This repository was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co. and its affiliates (``JP Morgan''), and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation and warranty whatsoever and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document and code is not intended as investment research or investment advice, or a recommendation,
offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful.
The code is provided for illustrative purposes only and is not intended to be used for trading or investment purposes. The code is provided "as is" and without warranty of any kind, express or implied, including without limitation, any warranty of merchantability or fitness for a particular purpose. In no event shall JP Morgan be liable for any direct, indirect, incidental, special or consequential damages, including, without limitation, lost revenues, lost profits, or loss of prospective economic advantage, resulting from the use of the code.