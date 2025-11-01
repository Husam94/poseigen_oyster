# Oyster

Oyster is a Neural Network architecture that can be toggled to consider (non-Exact) or ignore (Exact) k-mer associations. 

This package implements Oyster. It has two modules: 

1. model
- Here you will find the Oyster model. It is built using the [Trident](https://github.com/Husam94/poseigen_trident).
- Dual Oyster combines two Oyster models to model different types of biological sequences. It is exemplified by modelling [YY1 interactions](https://github.com/Husam94/YY1) using genomic DNA sequences and histone signals. Dual Oyster also serves as a template for combining more than one Oyster into a single model. 
- Recommended search spaces for hyperparameter optimizations are provided in [poseigen_compass](https://github.com/Husam94/poseigen_compass) format
- Reset functions are provided to initialize or reinitialize models for model fitting.

2. analysis 
- Here you will find functions to analyze Exact Oyster architectures, to uncover learned concepts. These functions include determining the contribution of each sequence (for multi-sequence inputs), k-mer contributions, k-ppm contributions, and k-mer clustering and refining. 
- Follow the example in modelling [YY1 interactions](https://github.com/Husam94/YY1). 