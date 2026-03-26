# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Contributed by: Nikhil
# Project: Restaurant Rating Prediction
# Course: Artificial Intelligence - VUIP111
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare(filepath="zomato.csv"):
    """
    Load and minimally prepare the dataset for EDA.
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

    # Select features and drop missing
    features = data[['online_order', 'book_table', 'votes',
                      'approx_cost(for two people)', 'listed_in(type)']]
    target = data['rate']
    dataset = pd.concat([features, target], axis=1).dropna()

    print("Data prepared for EDA. Shape:", dataset.shape)
    return dataset

def plot_rating_distribution(dataset):
    """
    Plot 1: Distribution of Restaurant Ratings
    Shows how ratings are spread across all restaurants.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(dataset['rate'], bins=20, kde=True, color='steelblue')
    plt.title('Distribution of Restaurant Ratings', fontsize=14)
    plt.xlabel('Rating')
    plt.ylabel('Number of Restaurants')
    plt.tight_layout()
    plt.savefig('plot_rating_distribution.png')
    plt.show()
    print("Plot 1 saved: plot_rating_distribution.png")

def plot_online_order_vs_rating(dataset):
    """
    Plot 2: Online Order vs Rating
    Shows whether accepting online orders affects the rating.
    """
    plt.figure(figsize=(7, 4))
    sns.boxplot(x='online_order', y='rate', data=dataset, palette='Set2')
    plt.title('Online Order vs Restaurant Rating', fontsize=14)
    plt.xlabel('Accepts Online Orders')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.savefig('plot_online_order_vs_rating.png')
    plt.show()
    print("Plot 2 saved: plot_online_order_vs_rating.png")

def plot_book_table_vs_rating(dataset):
    """
    Plot 3: Table Booking vs Rating
    Shows whether table booking availability affects the rating.
    """
    plt.figure(figsize=(7, 4))
    sns.boxplot(x='book_table', y='rate', data=dataset, palette='Set3')
    plt.title('Table Booking vs Restaurant Rating', fontsize=14)
    plt.xlabel('Allows Table Booking')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.savefig('plot_book_table_vs_rating.png')
    plt.show()
    print("Plot 3 saved: plot_book_table_vs_rating.png")

def plot_type_vs_rating(dataset):
    """
    Plot 4: Average Rating by Restaurant Type
    Shows which type of restaurant tends to have higher ratings.
    """
    plt.figure(figsize=(10, 5))
    type_rating = dataset.groupby('listed_in(type)')['rate'].mean().sort_values(ascending=False)
    sns.barplot(x=type_rating.index, y=type_rating.values, palette='viridis')
    plt.title('Average Rating by Restaurant Type', fontsize=14)
    plt.xlabel('Restaurant Type')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plot_type_vs_rating.png')
    plt.show()
    print("Plot 4 saved: plot_type_vs_rating.png")

def plot_votes_vs_rating(dataset):
    """
    Plot 5: Votes vs Rating (Scatter Plot)
    Shows whether more popular restaurants have higher ratings.
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(dataset['votes'], dataset['rate'], alpha=0.3, color='coral')
    plt.title('Votes vs Rating', fontsize=14)
    plt.xlabel('Number of Votes')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.savefig('plot_votes_vs_rating.png')
    plt.show()
    print("Plot 5 saved: plot_votes_vs_rating.png")

def run_eda(filepath="zomato.csv"):
    """
    Run all EDA plots.
    """
    print("Starting Exploratory Data Analysis...\n")
    dataset = load_and_prepare(filepath)

    plot_rating_distribution(dataset)
    plot_online_order_vs_rating(dataset)
    plot_book_table_vs_rating(dataset)
    plot_type_vs_rating(dataset)
    plot_votes_vs_rating(dataset)

    print("\nAll EDA plots generated and saved successfully!")

# Run EDA
if __name__ == "__main__":
    run_eda("zomato.csv")
