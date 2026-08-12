# AppliedResearchProject_code_file
Explainable Fake News Detection System using BERT, LIME and SHAP, developed for an MSc Applied Research Project.

# Explainable Fake News Detection System using BERT, LIME and SHAP

## Overview

This repository contains the implementation of an Explainable Fake News Detection System developed as part of an MSc Applied Research Project.

The project investigates automated fake news classification using traditional machine learning, deep learning and transformer-based Natural Language Processing (NLP) approaches. Six classification models are evaluated:

- Logistic Regression
- Multinomial Naïve Bayes
- Random Forest
- Support Vector Machine (SVM)
- Bidirectional Long Short-Term Memory (BiLSTM)
- Bidirectional Encoder Representations from Transformers (BERT)

The models are evaluated using Accuracy, Precision, Recall, F1-score and ROC-AUC. BERT achieved the strongest overall performance in the final experiment and was selected as the classifier for the final application.

To improve model transparency, LIME and SHAP are integrated to provide local explanations of individual BERT predictions. The final system is presented through an interactive Streamlit application.

## Project Objectives

The main objectives of the project are to:

1. Prepare and preprocess a labelled fake and real news dataset.
2. Develop and compare traditional machine-learning, deep-learning and transformer-based classifiers.
3. Evaluate the models using multiple classification metrics.
4. Select the strongest-performing model based on comparative evaluation.
5. Apply LIME and SHAP to explain individual BERT predictions.
6. Integrate the final BERT classifier and explainability functionality into an interactive Streamlit application.

## Dataset

The project uses the ISOT Fake News Dataset containing labelled fake and real news articles.

Dataset link:

[INSERT DATASET LINK HERE]

The dataset is used for academic research and is not redistributed through this repository unless permitted by its original licence.

## Methodology

The experimental workflow consists of:

Dataset Preparation  
↓  
Data Cleaning and Duplicate Removal  
↓  
Stratified Train-Test Split  
↓  
Model-Specific Text Representation  
↓  
Model Training  
↓  
Performance Evaluation  
↓  
Model Comparison  
↓  
BERT Selection  
↓  
LIME and SHAP Explainability  
↓  
Streamlit Application

Traditional machine-learning models use TF-IDF-based representations, while BiLSTM and BERT use representations appropriate to their respective neural architectures.

## Model Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The final experimental results were:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.5449% | 98.2926% | 99.0205% | 98.6552% | 99.8756% |
| Multinomial Naïve Bayes | 94.9910% | 95.3644% | 95.3416% | 95.3530% | 98.7339% |
| Random Forest | 99.5879% | 99.4759% | 99.7611% | 99.6183% | 99.9640% |
| Support Vector Machine | 99.2403% | 99.2130% | 99.3789% | 99.2959% | 99.9273% |
| BiLSTM | 99.8069% | 99.9282% | 99.7133% | 99.8206% | 99.9859% |
| BERT | 99.9227% | 100.0000% | 99.8567% | 99.9283% | 99.9992% |

BERT achieved the strongest overall combination of classification metrics and was therefore selected for the final prediction and explainability system.

## Explainable AI

Two Explainable Artificial Intelligence (XAI) techniques are incorporated:

### LIME

LIME provides a local explanation of an individual prediction by identifying textual features that influence the classifier's output.

### SHAP

SHAP provides feature-attribution information that helps identify how individual textual features contribute towards or away from the selected model output.

These explanations are intended to improve transparency of the classifier. They explain model behaviour and should not be interpreted as independent evidence that a news article is factually true or false.

## Streamlit Application

The final BERT classifier is integrated into an interactive Streamlit application.

The application allows users to:

- Enter or paste an English-language news article.
- Generate a Fake News or Real News prediction.
- View prediction confidence and class probabilities.
- Examine LIME explanations.
- Examine SHAP feature contributions.
- Visualise important features influencing individual predictions.

## Repository Structure

```text
project/
│
├── lastcode.ipynb
├── [STREAMLIT_APP_FILENAME].py
├── README.md
├── requirements.txt
│
├── data/
│   └── Dataset information
│
├── results/
│   └── Model evaluation outputs
│
└── models/
    └── Saved model files (where applicable)
