# RNA-Protein Binding Prediction Using Deep Learning

## Introduction

This project aims to predict RNA-protein binding strength using a deep learning model trained on RNAcompete data. The model takes RNA sequences and predicts their binding affinities to specific proteins at various concentrations.

## Project Background
In biological systems, RNA molecules play a critical role in protein synthesis. The strength of binding between RNA and proteins is essential for various biological processes. This project focuses on predicting the binding affinity between RNA molecules and a specific protein based on experimental data. The binding strength is influenced by the concentration of the protein: the lower the concentration at which binding occurs, the stronger the binding affinity.

## Biological Experiment
The experiment involves introducing RNA molecules into a solution with a specific protein at varying concentrations (e.g., 5 nM, 20 nM, etc.). The experiment measures how many RNA molecules bind to the protein, and the sequences of the bound RNA molecules are recorded. The goal is to predict the binding strength of approximately 240,000 RNA molecules with a given protein.

## How to Run

To execute the program, Use the following command:

`python3 MLBProject.py RBP1_output.txt RNACMPT_seq.txt RBP1_input.seq RBP1_5nM.seq RBP1_20nM.seq RBP1_80nM.seq RBP1_320nM.seq RBP1_1300nM.seq`.


- **MLBProject.py**: The main Python script.
- **RBP1_output.txt**: The output file where the model's predicted binding affinities will be saved.
- **RNACMPT_seq.txt**: Contains RNA sequences for which the binding affinities need to be predicted.
- **RBP1_input.seq**: Contains all sequences from the experiment, including those that bound to the protein and those that did not.
- **RBP1_5nM.seq, RBP1_20nM.seq, RBP1_80nM.seq, etc**: Files from the experiment that contain sequences of RNA molecules that bound to the protein at different concentrations.

## Model Overview
The model has five layers:
- **Input Layer**: Accepts input sequences of RNA with a shape defined by (sequence_length, 4), where sequence_length is the length of the longest sequence, and 4 corresponds to the one-hot encoding of the RNA bases (A, C, G, U/T).
- **Conv1D Layer**: This convolutional layer has 64 filters with a kernel size of 3 and uses the ReLU activation function.
- **Global Max Pooling 1D Layer:** This layer reduces the dimensionality of the data by selecting the maximum value from each feature map, retaining the most significant information.
- **Dense Layer**: A dense layer with 128 units and ReLU activation, responsible for combining features learned in previous layers.
- **Output Layer**: The final layer with as many units as the number of concentration levels provided. Each unit represents the probability of RNA binding at a specific concentration.

## Preprocessing
- **Sequences**: One-hot encoded with a special case for unknown bases ('N').
- **Labels**: Indicate the probability of binding at different concentrations.
## Output
- **Predictions**: Stored in RBP1_output.txt, representing the binding strength of RNA sequences.
- **Pearson Correlation**: Average correlation of 0.1543 with actual experimental data, indicating moderate prediction accuracy.


