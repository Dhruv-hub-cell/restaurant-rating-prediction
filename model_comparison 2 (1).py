# ============================================================
# MODEL TRAINING AND COMPARISON
# Contributed by: Akshay
# Project: Restaurant Rating Prediction
# Course: Artificial Intelligence - VUIP111
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_encoded_data(filepath="zomato.csv"):
    """
    Load and preprocess data for model training.
    Returns X_encoded and y.
    """
    data = pd.read_csv(filepath)

    # Clean rate column
    data = data[data['rate'] != 'NEW']
    data = data[data['rate'] != '-']
    data['rate'] = data['rate'].astype(str).apply(lambda x: x.split('/')[0])
    data['rate'] = data['rate'].astype(float)

    # Clean cost column
    data['approx_cost(for two people)'] = (
        data['approx_cost(for two people)']
        .astype(str).str.replace(',', '').str.strip()
    )
    data['approx_cost(for two people)'] = pd.to_numeric(
        data['approx_cost(for two people)'], errors='coerce'
    )

    # Select and clean features
    features = data[['online_order', 'book_table', 'votes',
                      'approx_cost(for two people)', 'listed_in(type)']]
    target = data['rate']
    dataset = pd.concat([features, target], axis=1).dropna()

    X = dataset.drop('rate', axis=1)
    y = dataset['rate']

    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)
    print("Data ready for model training. Shape:", X_encoded.shape)
    return X_encoded, y

def split_data(X_encoded, y):
    """
    Split data into 80% training and 20% testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def evaluate_model(name, y_test, y_pred):
    """
    Calculate MAE, RMSE and R2 score for a model.
    """
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name:20s} → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return mae, rmse, r2

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 4 regression models and collect results.
    """
    results = []

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    mae, rmse, r2 = evaluate_model("Linear Regression", y_test, lr.predict(X_test))
    results.append({"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R²": r2})

    # Model 2: Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    mae, rmse, r2 = evaluate_model("Decision Tree", y_test, dt.predict(X_test))
    results.append({"Model": "Decision Tree", "MAE": mae, "RMSE": rmse, "R²": r2})

    # Model 3: Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    mae, rmse, r2 = evaluate_model("Random Forest", y_test, rf.predict(X_test))
    results.append({"Model": "Random Forest", "MAE": mae, "RMSE": rmse, "R²": r2})

    # Model 4: Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    mae, rmse, r2 = evaluate_model("Gradient Boosting", y_test, gb.predict(X_test))
    results.append({"Model": "Gradient Boosting", "MAE": mae, "RMSE": rmse, "R²": r2})

    return pd.DataFrame(results), rf

def plot_model_comparison(comparison_df):
    """
    Plot a bar chart comparing R² scores of all models.
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Model', y='R²', data=comparison_df, palette='Blues_d')
    plt.title('Model Comparison — R² Score (Higher is Better)', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('plot_model_comparison.png')
    plt.show()
    print("Model comparison chart saved: plot_model_comparison.png")

def plot_feature_importance(rf_model, X_encoded):
    """
    Plot the top 10 most important features from Random Forest.
    """
    importance_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(9, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket')
    plt.title('Top 10 Most Important Features (Random Forest)', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plot_feature_importance.png')
    plt.show()
    print("Feature importance chart saved: plot_feature_importance.png")

def run_model_comparison(filepath="zomato.csv"):
    """
    Full pipeline: load data, train models, compare, and plot results.
    """
    print("Starting Model Training and Comparison...\n")
    X_encoded, y = load_encoded_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    print("\nModel Performance:")
    comparison_df, rf_model = train_all_models(X_train, X_test, y_train, y_test)

    print("\nFull Comparison Table:")
    print(comparison_df.to_string(index=False))

    plot_model_comparison(comparison_df)
    plot_feature_importance(rf_model, X_encoded)

    print("\nModel comparison complete!")
    return rf_model, X_encoded

# Run model comparison
if __name__ == "__main__":
    rf_model, X_encoded = run_model_comparison("zomato.csv")
