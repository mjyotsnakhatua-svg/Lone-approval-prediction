"""
Educational loan approval prediction demo.
The dataset below is synthetic and created only to demonstrate a machine-learning workflow.
Do not use this model to make real lending decisions.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_demo_data():
    # Synthetic example records; not real applicants or a real lender's decisions.
    return pd.DataFrame([
        {"income": 65000, "loan_amount": 180000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 28000, "loan_amount": 240000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 72000, "loan_amount": 150000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 35000, "loan_amount": 210000, "credit_history": "Average", "employment": "Self-employed", "approved": 0},
        {"income": 58000, "loan_amount": 130000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 42000, "loan_amount": 260000, "credit_history": "Poor", "employment": "Salaried", "approved": 0},
        {"income": 80000, "loan_amount": 200000, "credit_history": "Good", "employment": "Self-employed", "approved": 1},
        {"income": 30000, "loan_amount": 160000, "credit_history": "Poor", "employment": "Salaried", "approved": 0},
        {"income": 54000, "loan_amount": 170000, "credit_history": "Average", "employment": "Salaried", "approved": 1},
        {"income": 39000, "loan_amount": 230000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 91000, "loan_amount": 220000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 32000, "loan_amount": 190000, "credit_history": "Average", "employment": "Self-employed", "approved": 0},
        {"income": 61000, "loan_amount": 140000, "credit_history": "Good", "employment": "Self-employed", "approved": 1},
        {"income": 46000, "loan_amount": 280000, "credit_history": "Poor", "employment": "Salaried", "approved": 0},
        {"income": 75000, "loan_amount": 180000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 27000, "loan_amount": 250000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 67000, "loan_amount": 160000, "credit_history": "Average", "employment": "Salaried", "approved": 1},
        {"income": 41000, "loan_amount": 245000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 88000, "loan_amount": 175000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 36000, "loan_amount": 215000, "credit_history": "Average", "employment": "Self-employed", "approved": 0},
        {"income": 59000, "loan_amount": 155000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 33000, "loan_amount": 270000, "credit_history": "Poor", "employment": "Salaried", "approved": 0},
        {"income": 78000, "loan_amount": 190000, "credit_history": "Good", "employment": "Self-employed", "approved": 1},
        {"income": 44000, "loan_amount": 225000, "credit_history": "Average", "employment": "Salaried", "approved": 0},
        {"income": 70000, "loan_amount": 145000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 29000, "loan_amount": 235000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 64000, "loan_amount": 185000, "credit_history": "Average", "employment": "Salaried", "approved": 1},
        {"income": 37000, "loan_amount": 255000, "credit_history": "Poor", "employment": "Self-employed", "approved": 0},
        {"income": 83000, "loan_amount": 165000, "credit_history": "Good", "employment": "Salaried", "approved": 1},
        {"income": 48000, "loan_amount": 205000, "credit_history": "Average", "employment": "Self-employed", "approved": 0},
    ])

def main():
    df = build_demo_data()
    features = ["income", "loan_amount", "credit_history", "employment"]
    X = df[features]
    y = df["approved"]

    numeric_features = ["income", "loan_amount"]
    categorical_features = ["credit_history", "employment"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("LOAN APPROVAL PREDICTION — EDUCATIONAL DEMO")
    print("Note: the sample data is synthetic and results are not meaningful for real lending.")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    sample_applicant = pd.DataFrame([{
        "income": 60000,
        "loan_amount": 170000,
        "credit_history": "Good",
        "employment": "Salaried"
    }])
    predicted = model.predict(sample_applicant)[0]
    print("\nExample applicant (synthetic inputs):")
    print("Predicted class:", "Approved" if predicted == 1 else "Not approved")

if __name__ == "__main__":
    main()
