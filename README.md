# Credit Card Fraud Detection using Machine Learning and Streamlit

A beginner-friendly end-to-end ML project. It loads a CSV, explores the class balance,
trains a Logistic Regression pipeline, evaluates it, and lets you test a transaction.

## Project structure

```text
credit_card_fraud_detection_streamlit/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_creditcard.csv
```

## 1. Requirements

- Python 3.9 or newer
- VS Code (optional)
- Internet connection for the first package installation

## 2. Open the project

1. Download and extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder** and choose `credit_card_fraud_detection_streamlit`.
4. Open the VS Code terminal: **Terminal → New Terminal**.

## 3. Create a virtual environment (recommended)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt in VS Code, or run:
```powershell
venv\Scripts\Activate.ps1
```

## 4. Install libraries

Run this from the main project folder (the folder containing `requirements.txt`):
```bash
pip install -r requirements.txt
```

If `pip` is not recognized, try:
```bash
python -m pip install -r requirements.txt
```

## 5. Run the Streamlit app

```bash
python -m streamlit run app.py
```

The terminal will show a local address, usually `http://localhost:8501`.
Open that address in your browser. Keep the terminal running while using the app.
Stop the app with `Ctrl + C` in the terminal.

## 6. Use your own dataset

Use the sidebar to upload a CSV. It must contain:
- Numeric input feature columns
- A target column named `Class`
- `Class = 0` for genuine transactions
- `Class = 1` for fraudulent transactions

The well-known Kaggle credit-card dataset uses `Time`, `V1` through `V28`,
`Amount`, and `Class`, so it is compatible. If you use another dataset, the
app will use its numeric columns as model features.

## 7. Where does EDA happen?

EDA (Exploratory Data Analysis) is performed in `app.py` under:
- **Dataset overview**: number of rows, features, genuine and fraud cases
- **Preview dataset**: first 10 rows
- **Class distribution**: bar chart showing class imbalance

For a college report, you can add further EDA such as missing-value counts,
feature histograms, correlation heatmaps, and outlier analysis.

## 8. ML workflow

1. Load the transaction dataset.
2. Separate features (`X`) and target (`y`).
3. Split into training and testing sets using stratification.
4. Impute missing numeric values with the median.
5. Standardize features.
6. Train Logistic Regression with `class_weight="balanced"` to account for class imbalance.
7. Evaluate with precision, recall, F1-score, ROC-AUC and a confusion matrix.
8. Predict the fraud probability for a transaction.

## Important note about the demo data

`data/sample_creditcard.csv` is **synthetic sample data created for demonstrating the app**.
It is not actual customer or banking data and should not be used to claim real-world model
performance. For a more meaningful academic experiment, obtain a legitimate dataset such
as the public Kaggle Credit Card Fraud Detection dataset, follow its terms, and upload its CSV.

## Troubleshooting

- **No such file or directory for requirements.txt**: make sure the terminal is in the
  extracted main project folder. Run `dir` on Windows and check that `requirements.txt` appears.
- **Streamlit command not found**: use `python -m streamlit run app.py`.
- **Port 8501 is busy**: run `python -m streamlit run app.py --server.port 8502`.
- **Module not found**: activate the virtual environment and rerun `python -m pip install -r requirements.txt`.

## Suggested report title

**Credit Card Fraud Detection Using Machine Learning and Streamlit**

This project is for learning and demonstration only. It is not a production fraud
detection system or financial advice.
