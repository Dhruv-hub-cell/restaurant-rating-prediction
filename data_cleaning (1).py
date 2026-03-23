# ============================================================
# DATA CLEANING AND PREPROCESSING
# Contributed by: Suprateek
# Project: Restaurant Rating Prediction
# Course: Artificial Intelligence - VUIP111
# ============================================================

import pandas as pd
import numpy as np

def load_data(filepath="zomato.csv"):
    """
    Load the Zomato dataset from a CSV file.
    """
    data = pd.read_csv(filepath)
    print("Dataset loaded successfully!")
    print("Shape of dataset (rows, columns):", data.shape)
    return data

def clean_rate_column(data):
    """
    Clean the 'rate' column.
    - Remove rows with 'NEW' or '-' ratings
    - Extract numeric value from format like '4.1/5'
    - Convert to float
    """
    # Remove invalid ratings
    data = data[data['rate'] != 'NEW']
    data = data[data['rate'] != '-']

    # Extract number before '/' and convert to float
    data['rate'] = data['rate'].astype(str).apply(lambda x: x.split('/')[0])
    data['rate'] = data['rate'].astype(float)

    print("Cleaned 'rate' column successfully.")
    print("Sample values:", data['rate'].head().tolist())
    return data

def clean_cost_column(data):
    """
    Clean the 'approx_cost(for two people)' column.
    - Remove commas (e.g. '1,200' becomes '1200')
    - Convert to numeric
    """
    data['approx_cost(for two people)'] = (
        data['approx_cost(for two people)']
        .astype(str)
        .str.replace(',', '')
        .str.strip()
    )
    data['approx_cost(for two people)'] = pd.to_numeric(
        data['approx_cost(for two people)'], errors='coerce'
    )

    print("Cleaned cost column successfully.")
    print("Sample values:", data['approx_cost(for two people)'].head().tolist())
    return data

def select_features(data):
    """
    Select the relevant features and target variable.
    Drop rows with missing values.
    """
    features = data[['online_order', 'book_table', 'votes',
                      'approx_cost(for two people)', 'listed_in(type)']]
    target = data['rate']

    # Combine and drop missing rows
    dataset = pd.concat([features, target], axis=1).dropna()

    print("Final dataset shape after removing missing values:", dataset.shape)
    print("Missing values per column:")
    print(dataset.isnull().sum())

    return dataset

def encode_features(dataset):
    """
    Encode categorical variables using One-Hot Encoding.
    Separate X (features) and y (target).
    """
    X = dataset.drop('rate', axis=1)
    y = dataset['rate']

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.astype(int)

    print("Encoded feature shape:", X_encoded.shape)
    return X_encoded, y

def preprocess_pipeline(filepath="zomato.csv"):
    """
    Run the complete preprocessing pipeline.
    Returns X_encoded and y ready for model training.
    """
    data = load_data(filepath)
    data = clean_rate_column(data)
    data = clean_cost_column(data)
    dataset = select_features(data)
    X_encoded, y = encode_features(dataset)
    print("\nPreprocessing complete! Data is ready for model training.")
    return X_encoded, y

# Run the pipeline
if __name__ == "__main__":
    X_encoded, y = preprocess_pipeline("zomato.csv")
    print("\nFinal X shape:", X_encoded.shape)
    print("Final y shape:", y.shape)
