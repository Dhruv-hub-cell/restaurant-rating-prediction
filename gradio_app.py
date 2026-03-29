# ============================================================
# GRADIO FRONTEND INTERFACE AND CLASSIFICATION MODEL
# Contributed by: Nimit
# Project: Restaurant Rating Prediction
# Course: Artificial Intelligence - VUIP111
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ── PART 1: CLASSIFICATION MODEL ────────────────────────────────────────────

def load_encoded_data(filepath="zomato.csv"):
    """
    Load and preprocess data for classification.
    """
    data = pd.read_csv(filepath)

    data = data[data['rate'] != 'NEW']
    data = data[data['rate'] != '-']
    data['rate'] = data['rate'].astype(str).apply(lambda x: x.split('/')[0])
    data['rate'] = data['rate'].astype(float)

    data['approx_cost(for two people)'] = (
        data['approx_cost(for two people)']
        .astype(str).str.replace(',', '').str.strip()
    )
    data['approx_cost(for two people)'] = pd.to_numeric(
        data['approx_cost(for two people)'], errors='coerce'
    )

    features = data[['online_order', 'book_table', 'votes',
                      'approx_cost(for two people)', 'listed_in(type)']]
    target = data['rate']
    dataset = pd.concat([features, target], axis=1).dropna()

    X = dataset.drop('rate', axis=1)
    y = dataset['rate']
    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)

    return X_encoded, y

def build_classification_model(X_encoded, y):
    """
    Build and evaluate a Logistic Regression model
    to classify restaurants as Good (>=3.5) or Bad (<3.5).
    """
    # Convert ratings to binary labels
    y_class = (y >= 3.5).astype(int)  # 1 = Good, 0 = Bad

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_encoded, y_class, test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_c, y_train_c)
    y_pred_class = log_model.predict(X_test_c)

    # Print classification report
    print("Classification Report (Good vs Bad Restaurant):")
    print(classification_report(y_test_c, y_pred_class,
                                 target_names=['Bad (<3.5)', 'Good (>=3.5)']))

    # Confusion matrix
    cm = confusion_matrix(y_test_c, y_pred_class)
    cm_df = pd.DataFrame(cm,
                         index=['Actual Bad', 'Actual Good'],
                         columns=['Predicted Bad', 'Predicted Good'])
    print("Confusion Matrix:")
    print(cm_df)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix — Logistic Regression', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot_confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved: plot_confusion_matrix.png")

    return log_model

# ── PART 2: GRADIO INTERFACE ─────────────────────────────────────────────────

def launch_gradio_interface():
    """
    Load the saved model and launch the Gradio web interface.
    Make sure to run the main notebook first to generate the .pkl files.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return

    # Load saved model and feature columns
    model = joblib.load("restaurant_rating_model.pkl")
    feature_columns = joblib.load("model_features.pkl")

    def predict_rating(votes, cost, online_order, book_table, rest_type):
        """
        Takes restaurant details from user and returns predicted rating.
        """
        # Create blank input row with all features set to 0
        input_df = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        # Fill in the user's values
        if 'votes' in input_df.columns:
            input_df['votes'] = votes

        for col in input_df.columns:
            if 'cost' in col.lower():
                input_df[col] = cost

        if online_order == "Yes" and 'online_order_Yes' in input_df.columns:
            input_df['online_order_Yes'] = 1

        if book_table == "Yes" and 'book_table_Yes' in input_df.columns:
            input_df['book_table_Yes'] = 1

        type_col = f'listed_in(type)_{rest_type}'
        if type_col in input_df.columns:
            input_df[type_col] = 1

        # Make prediction
        prediction = model.predict(input_df)[0]
        rating = round(float(prediction), 2)

        # Give a label based on predicted rating
        if rating >= 4.0:
            label = "Excellent Restaurant!"
        elif rating >= 3.5:
            label = "Good Restaurant"
        elif rating >= 3.0:
            label = "Average Restaurant"
        else:
            label = "Below Average"

        return f"Predicted Rating: {rating} / 5.0\n{label}"

    # Build and launch the interface
    interface = gr.Interface(
        fn=predict_rating,
        inputs=[
            gr.Number(label="Number of Votes", value=100),
            gr.Number(label="Average Cost for Two (Rs)", value=500),
            gr.Radio(["Yes", "No"], label="Online Order Available?", value="Yes"),
            gr.Radio(["Yes", "No"], label="Table Booking Available?", value="No"),
            gr.Dropdown(
                ["Delivery", "Dine-out", "Desserts", "Cafes",
                 "Drinks & nightlife", "Buffet"],
                label="Restaurant Type",
                value="Delivery"
            )
        ],
        outputs=gr.Textbox(label="Prediction Result"),
        title="Restaurant Rating Predictor",
        description="Enter restaurant details to predict its Zomato rating using Machine Learning."
    )

    print("Launching Gradio interface...")
    interface.launch()

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PART 1: Classification Model ===\n")
    X_encoded, y = load_encoded_data("zomato.csv")
    log_model = build_classification_model(X_encoded, y)

    print("\n=== PART 2: Gradio Interface ===\n")
    launch_gradio_interface()
