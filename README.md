Overview

This project aims to identify whether two given questions are semantically similar or duplicate, inspired by Quora’s real-world problem of reducing redundant content.
The system uses Natural Language Processing (NLP) and Machine Learning / Deep Learning techniques to predict duplicate question pairs.

🚀 Features

Detects duplicate question pairs with high accuracy

Text preprocessing (cleaning, tokenization, lemmatization)

Feature engineering (TF-IDF / Word Embeddings / Similarity metrics)

Machine Learning models (Logistic Regression, XGBoost, etc.)

Optional Deep Learning models (LSTM, Siamese Network)

User-friendly prediction interface (optional: Streamlit/Flask)



🧪 Dataset

Source: Quora Question Pairs Dataset (Kaggle)

Description:

question1: First question

question2: Second question

is_duplicate: Target label (1 = duplicate, 0 = not duplicate)

⚙️ Tech Stack

Programming Language: Python 🐍

Libraries:

NumPy, Pandas

Scikit-learn

NLTK / spaCy

XGBoost

TensorFlow / PyTorch (if DL used)

Visualization: Matplotlib, Seaborn

Deployment (Optional): Streamlit / Flask

🔍 Approach

Data Cleaning

Lowercasing

Removing punctuation & stopwords

Lemmatization

Feature Engineering

TF-IDF vectors

Word overlap

Cosine similarity

Length-based features

Model Training

Logistic Regression / Random Forest / XGBoost

Evaluation

Accuracy

Precision, Recall, F1-score

Prediction

Predicts whether two questions are duplicates






# Duplicate Question App

This folder contains the deployment code for the Quora Duplicate Question Pair Detection project.

## Contents
- app.py – Streamlit/Flask application
- model.pkl – Trained ML model
- preprocessing.py – Text cleaning logic
- requirements.txt – Dependencies

## How to Run
```bash
pip install -r requirements.txt
python app.py

