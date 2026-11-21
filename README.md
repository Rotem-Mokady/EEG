# EEG-PD-Bench

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**A benchmark for clinical prediction tasks from resting-state EEG (rsEEG) in Parkinson's disease.**

> 🚧 **Status: work in progress.** This is an undergraduate final project, currently in the data-exploration phase. The scope below follows the initial project description and will be expanded as the work progresses.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Motivation: Why a Benchmark?](#motivation-why-a-benchmark)
- [Goals](#goals)
- [Evaluation Framework](#evaluation-framework)
- [Prediction Tasks](#prediction-tasks)
- [Data](#data)
- [Design Requirements](#design-requirements)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage (planned API)](#usage-planned-api)
- [Roadmap](#roadmap)
- [Team & Acknowledgements](#team--acknowledgements)
- [License](#license)

---

## Overview

EEG-PD-Bench is an open-source Python package for training and evaluating machine-learning models on clinical prediction tasks in Parkinson's disease (PD), using features extracted from resting-state EEG.

The package takes three inputs:

| Input | Description |
|-------|-------------|
| **X** | EEG features from one or more datasets, structured as `subject × segment × feature` |
| **Y** | Target values to predict (diagnosis, clinical scores) |
| **M** | A set of **untrained** prediction models (classification or regression) |

It then trains, tunes and evaluates all models under a shared, reproducible protocol, so that results are directly comparable across models, tasks and datasets.

The package will be released publicly as part of this project, for future use by the lab and by the wider research community.

## Background

### Parkinson's disease

Parkinson's disease is a progressive neurodegenerative disorder. Dopamine-producing neurons in the **substantia nigra** gradually die, and the **basal ganglia**, which depend on that dopamine to start movements and hold back unwanted ones, stop working properly. Surviving neurons accumulate misfolded alpha-synuclein protein in clumps called *Lewy bodies*. Symptoms usually appear only after a large share of the dopamine neurons has already been lost. Growing evidence shows that the disease also involves the **cortex**, which is the layer EEG can measure.

| Motor symptoms | Non-motor symptoms |
|----------------|--------------------|
| Tremor, rigidity, slowness of movement (bradykinesia), falls | Memory and attention problems, sleep disorders, depression |

**Diagnosis is clinical.** No single blood test or scan confirms PD. A neurologist takes a history, performs a neurological examination and rules out other causes. This process depends on an expert and can be long and costly. Disease severity is measured with two standard scales:

- **MDS-UPDRS Part III**: a motor examination scored by a neurologist.
- **MoCA (Montreal Cognitive Assessment)**: a short test of cognitive impairment.

### Why EEG?

Electroencephalography (EEG) records the brain's electrical activity through electrodes placed on the scalp. Each electrode picks up the combined activity of millions of neurons, mainly in the cortex, sampled at about 250 times per second.

- **Available:** EEG is already standard equipment in neurology departments.
- **Non-invasive, cheap, fast and portable**, especially compared with imaging such as fMRI.
- **Objective:** it gives a quantitative measure that does not depend on a clinician's rating.
- **High temporal resolution:** it captures changes in brain activity at the millisecond scale.
- **Research evidence:** earlier studies have found differences in EEG features between PD patients and healthy controls, between milder and more severe disease, and between patients with and without cognitive impairment.

### What is already known

About three decades of research point to a few consistent findings:

- **Slowing of brain activity** is the most consistent one: more slow-wave power and a lower dominant frequency, especially in patients with cognitive impairment.
- The **ratio of fast to slow waves decreases** as the disease progresses.
- EEG slowing **predicts future cognitive decline**.

Several problems still stand in the way of clinical use:

- The link between EEG findings and **motor symptoms is inconsistent**.
- **Samples are small**, and every study uses its own processing and feature pipeline.
- Most studies come from a **single center**. Before clinical use, results need standardization and validation across several sites.

## Motivation: Why a Benchmark?

| Challenge | How EEG-PD-Bench addresses it |
|-----------|-------------------------------|
| **Single-center studies:** a model trained at one hospital often loses performance at another | Use data from **two sites** and test models on a site they have never seen (leave-one-site-out) |
| **No baseline:** it is unclear how much improvement new methods actually bring | Build an **open benchmark** with simple models, publishing the code, data splits and predictions so future work can be measured against them |
| **Inconsistent methods:** each study processes data and computes features differently | Process all data the **same way**, with **classical features chosen from the literature** rather than from our own data, to avoid overfitting |

**The long-term vision:** to lay the groundwork for an EEG-based tool with real clinical value, one that helps diagnose PD objectively, simply and cheaply, and helps match treatment to each patient.

## Goals

1. **Build a benchmark** for clinical prediction tasks from rsEEG in PD.
2. **Provide a generic software package** that evaluates any classification or regression model under a fixed, rigorous protocol.
3. **Run a first study** of how well simple, classical ML models perform on classical EEG features across three PD prediction tasks.

## Evaluation Framework

The package evaluates models from three complementary angles:

### 1. Subject Generalization
How well does a model generalize to **new subjects** from the same data distribution?

- **Repeated, Stratified, Nested K-Fold Cross-Validation.**
- The *inner* loop performs hyperparameter tuning; the *outer* loop estimates performance on held-out subjects.
- Splits are always made at the **subject level**, so segments from the same subject never appear in both training and test sets.
- Data from all sites is pooled, and folds are stratified so that **every site is represented in each fold**.

### 2. Dataset Generalization
How well does a model generalize to **a dataset recorded at a different site**?

- **Leave-One-Dataset-Out (LODO)** evaluation.
- Transfer between datasets is **not symmetric**, so each direction is reported separately (e.g. `LEMON → Iowa` and `Iowa → LEMON`).

### 3. Clinical Correlations
Are the model's predictions related to other subject characteristics?

- Tests the relationship between **out-of-fold predictions** and:
  - demographic variables: **age**, **sex**
  - clinical variables: **LEDD** (levodopa equivalent daily dose)

### Metrics
Performance is reported with a predefined set of metrics suited to each task type (classification vs. regression).

## Prediction Tasks

The initial study covers three tasks:

| Task | Type | Target |
|------|------|--------|
| **HC vs. PD** | Binary classification | Healthy control vs. Parkinson's disease |
| **Motor Severity Prediction** | Regression | MDS-UPDRS Part III (motor examination) score |
| **Cognitive Performance Prediction** | Regression | MoCA (Montreal Cognitive Assessment) score |

At minimum, each task is evaluated with:

- one **linear** model (e.g. Logistic Regression / a regularized linear regressor), and
- one **non-linear** model (e.g. XGBoost).

## Data

### Why features and not raw signals?

A 2-minute recording with 19 electrodes sampled at 250 Hz already contains about **570,000 values per subject**, while the datasets contain only a few hundred subjects. With far more values than subjects, a model memorizes the data instead of learning real patterns (the *curse of dimensionality*). Instead, each channel is summarized by a small set of meaningful features: about **10 features × 19 channels ≈ 190 values per subject**.

### Dataset description

- rsEEG features from recordings of **~300 subjects**, each recorded with **19 channels**.
- Every recording is split into short **10-second segments**, giving roughly **10–20 segments per recording**.
- A fixed set of features is computed **for every segment and every electrode**.
- Models should also use the **spatial** properties of the features (i.e. differences between electrodes).

### Unit of analysis

> ⚠️ **The sample unit for the models is a single EEG segment, not a subject.**
>
> Models are **trained at the segment level**, while **data splitting and evaluation are done at the subject level**. This prevents data leakage between segments of the same subject and keeps the reported performance honest.

### Data sources

| Site | Lab / Source | OpenNeuro ID | Notes |
|------|--------------|--------------|-------|
| Site 1 | **LEMON** lab, Tel Aviv Sourasky Medical Center (Ichilov) | `ds007526` | 144 subjects (116 PD, 28 HC), openly published by the lab |
| Site 2 | **University of Iowa** | `ds004584` | Open dataset |

> **Note:** Raw data and extracted features are **not** included in this repository. Both datasets are available on OpenNeuro under their own terms.

### Clinical variables

For every subject, the following clinical data is available alongside the EEG features: **sex**, **age**, **disease duration**, **MDS-UPDRS-III** and **MoCA**. **LEDD** is used in the clinical-correlation analysis.

### Features

The features are classical EEG measures, chosen from the literature. They are computed for every segment and every electrode. Examples include:

- **Band power** in each frequency band (delta, theta, alpha, beta)
- **Dominant frequency** (peak frequency)
- **Complexity measures**, which capture how irregular or "chaotic" the signal is

The final feature list will be documented here once it is finalized.

## Design Requirements

The package is designed around the following principles:

- **Simple code-level API** – an easy-to-use programmatic interface (no GUI) for running the full evaluation pipeline.
- **Model-agnostic** – works with any classification or regression model that follows a standard `fit` / `predict` interface.
- **Built-in hyperparameter tuning** – tuning happens inside training and evaluation, which is why nested CV is used rather than plain CV.
- **Identical splits for all models** – every model is evaluated on exactly the same data partitions, so comparisons are fair.
- **Extensible evaluation strategies** – new evaluation approaches can be added later (for example, testing HC-vs-PD classifiers on a healthy-controls-only dataset).
- **Configurable clinical covariates** – additional clinical variables can be added to analyze their relationship with model predictions (initially: age, sex and LEDD).

## Repository Structure

```text
EEG-PD-Bench/
├── data/          # Feature files and clinical tables (not tracked in git)
├── images/        # Figures and plots
├── notebooks/     # Exploration and analysis notebooks
├── src/           # Shared Python code (data preparation, evaluation, models)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Rotem-Mokady/EEG-PD-Bench.git
cd EEG-PD-Bench

# Create and activate an environment
conda create -n eeg-pd-bench python=3.11
conda activate eeg-pd-bench

# Install dependencies
pip install -r requirements.txt
```

Place the feature files in the `data/` directory before running the notebooks.

## Usage (planned API)

> The API below is a **design sketch** and may change as the implementation develops.

```python
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data import EEGDataset
from src.benchmark import Benchmark

# Load features (subject × segment × feature) and targets from both sites
data = EEGDataset.from_files(
    datasets=["LEMON", "Iowa"],
    target="diagnosis",                 # or "UPDRS_III", "MoCA"
    covariates=["age", "sex", "LEDD"],
)

# Untrained models + hyperparameter search spaces
models = {
    "logreg": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "xgboost": (XGBClassifier(), {"max_depth": [3, 5], "n_estimators": [100, 300]}),
}

bench = Benchmark(task="classification", models=models, random_state=42)

# Subject generalization: repeated stratified nested K-fold CV
subject_results = bench.evaluate_subject_generalization(data, n_splits=5, n_repeats=10)

# Dataset generalization: leave-one-dataset-out (each direction reported separately)
dataset_results = bench.evaluate_dataset_generalization(data)

# Clinical correlations of out-of-fold predictions
clinical_results = bench.clinical_correlations(subject_results)
```

## Roadmap

- [x] Background research and presentation
- [ ] Data exploration (both sites)
- [ ] Data-preparation module (`subject × segment × feature` structure)
- [ ] Subject-generalization pipeline (repeated stratified nested K-fold)
- [ ] Dataset-generalization pipeline (leave-one-dataset-out)
- [ ] Clinical-correlation analysis (age, sex, LEDD)
- [ ] Baseline results: linear vs. non-linear models on all three tasks
- [ ] Documentation and public release of the package

## Team & Acknowledgements

**Students:** Noa Ben Shimon, Rotem Mokady

**Lab supervisors:** Prof. Inbal Maidan, Daniel Vered (LEMON lab)

**Academic advisor:** Prof. Ruth Bar-Shir

This project is an undergraduate final project at Tel Aviv University, carried out in collaboration with the **LEMON lab** (*Laboratory for Early Markers Of Neurodegeneration*), headed by **Prof. Anat Mirelman**. The lab is part of the Center for the Study of Movement, Cognition and Mobility (CMCM) at the Neurological Institute, Tel Aviv Sourasky Medical Center (Ichilov). It works to identify early markers of aging and neurodegenerative disease and to find measures that can predict their course.

We thank the University of Iowa researchers for making their EEG data openly available on OpenNeuro.

## License

This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for details.
