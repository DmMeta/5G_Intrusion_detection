# Intrusion Detection using Machine Learning

This project involves the implementation of various machine learning models for intrusion detection in a 5G network simulation. The task is to classify network traffic as benign or malicious based on the NIDD dataset.

## Models Used

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- CNN with Long Short-Term Memory (LSTM) layer
- Decision Tree
- Random Forest

## Feature Extraction Techniques

Pearson correlation is a statistical measure of the strength and direction of the linear relationship between two variables. In the context of intrusion detection, we used it to identify the correlation between different features in the dataset and determine their relevance to the classification task. Features with high correlation to the target variable (malicious or benign) are likely to be more informative for the models and so were among the ones finally selected.

### Analysis of Variance (ANOVA) F-test
ANOVA F-test is a statistical technique used to compare the means of two or more groups to determine if they are significantly different from each other. In the context of feature extraction for intrusion detection, ANOVA F-test were used to identify features that have significantly different means between the benign and malicious classes. Features with high F-statistics and low p-values are considered more important for classification.

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that aims to reduce the number of features in a dataset while preserving the most important information. It does this by transforming the original features into a new set of orthogonal components called principal components. As far as intrusion detection is concerned, PCA was used to reduce the dimensionality of the dataset while retaining as much variance as possible. This can help improve the performance of machine learning models by reducing their computational needs.

> **Note:** Apart from these techniques, several unimportnant to the task features were dropped as well as features with high concetration of NaN and constant values.

## Standard Scaling Preprocessing
Standard scaling (or Z-score normalization) is a preprocessing technique used to standardize the range of features in the dataset. It involves transforming the features such that they have a mean of 0 and a standard deviation of 1. This ensures that all features have the same scale and can help improve the convergence speed and performance of the models, especially when dealing with features that have different units or scales.

## Deployment to Kubernetes (k8s) Cluster

The intrusion detection service has been deployed to a Kubernetes cluster to ensure scalability, availability, and ease of management, using two replicas of a fastapi server with one endpoint for prediction of the network flow.


## Installation

```bash
# Clone the repo
git clone https://github.com/DmMeta/5G_Intrusion_detection.git

# Navigate into the directory
cd 5G_Intrusion_detection

# Install dependencies
pip3 install .