# Loan Approval Prediction

A beginner machine-learning project demonstrating a classification workflow with Python, Pandas, and Scikit-learn.

## What it demonstrates
- Organizing applicant features and a target label
- Splitting data into training and test sets
- Handling numeric and categorical features
- Scaling numeric values and one-hot encoding categories
- Training a Logistic Regression classifier
- Evaluating predictions with accuracy, a confusion matrix, and a classification report

## Important data note
The included dataset is **synthetic demo data**, not real applicant data. The printed metrics are only a demonstration of code execution and do not show real-world model quality. This model must not be used to approve or reject actual loans.

## Requirements
- Python 3.10 or newer
- pip

## Run locally
1. Open a terminal in this project folder.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python train_model.py
   ```

## Project structure
- `train_model.py` — demo dataset, preprocessing pipeline, model training and evaluation
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

## Possible improvements
- Replace synthetic data with a properly licensed, documented public dataset
- Perform exploratory data analysis and visualize feature distributions
- Use cross-validation and compare multiple classifiers
- Check for data leakage, fairness, and model calibration before any real-world research use
