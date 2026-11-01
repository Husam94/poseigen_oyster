# Oyster

[![PyPI version](https://img.shields.io/pypi/v/poseigen_oyster)](https://pypi.org/project/poseigen_oyster/)
[![Python versions](https://img.shields.io/pypi/pyversions/poseigen_oyster)](https://pypi.org/project/poseigen_oyster/)
[![License](https://img.shields.io/github/license/Husam94/poseigen_oyster)](LICENSE.txt)
[![bioRxiv preprint](https://img.shields.io/badge/preprint-bioRxiv-B31B1B)](https://www.biorxiv.org/content/10.1101/2025.11.01.686014)

Oyster is a glass-box neural network for genomic sequence-to-function modeling. It learns the effects of k-mer composition and position while intentionally excluding k-mer associations (interactions among k-mers). The position-specific k-mer contributions can be read directly from the learned model weights.

See the paper: [Oyster: A Neural Network for Modeling Genomic Sequences with Exact Position-Specific k-mer Contributions](https://www.biorxiv.org/content/10.1101/2025.11.01.686014). The paper compares Oyster with models that learn k-mer associations. Those models can achieve better predictive performance, while Oyster predictions remain in strong agreement and can serve as interpretable proxies.

## Installation

Requires Python 3.12 or newer. Install from PyPI with:

```bash
python -m pip install poseigen_oyster
```

## Package

- **`model`** provides `Oyster` and `Dual_Oyster`. `Dual_Oyster` combines two Oyster models, allowing each branch to use different inputs, k-mer sizes, position-pooling settings, and chunking.
- **`analysis`** provides tools for inspecting Oyster models, including sequence and k-mer contributions, PWM-derived contributions, clustering, and cluster refinement.
- `Oyster_BaseDict` and `Dual_Oyster_BaseDict` define hyperparameter search spaces for `Oyster` and `Dual_Oyster` to use with [poseigen_compass](https://github.com/Husam94/poseigen_compass).
- Oyster is built using [poseigen_trident](https://github.com/Husam94/poseigen_trident).

## Case studies

Oyster has been exemplified in the following case studies:

- **Protein-DNA interactions**
  - [CTCF case study](https://github.com/Husam94/CS-CTCF)
  - [YY1 case study](https://github.com/Husam94/CS-YY1)
- **Enhancer activity**
  - [Enhancer activity case study](https://github.com/Husam94/CS-EnhAct)
