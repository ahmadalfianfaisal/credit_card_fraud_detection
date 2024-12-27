# Credit Card Fraud Detection | Project Machine Learning

## Background
Credit card fraud is a significant challenge in the financial sector, and the ability to detect fraudulent transactions quickly and accurately is essential. This project focuses on identifying fraudulent transactions using exploratory data analysis (EDA) and machine learning techniques.

## Objectives
1. Understand the dataset through EDA.
2. Preprocess and generate features for improved model performance.
3. Develop and evaluate a machine learning model to classify fraudulent transactions.

## Dataset
The dataset used in this project contains information on credit card transactions, including:
- `isFraud`: Target variable indicating if a transaction is fraudulent.
- `type`: Type of transaction (e.g., `TRANSFER`, `CASH_OUT`).
- `amount`: Transaction amount.
- `oldbalanceOrg` and `newbalanceOrig`: Balances for the originating account.
- `oldbalanceDest` and `newbalanceDest`: Balances for the destination account.

### Data Source
The dataset is loaded from: `/content/drive/MyDrive/Project Credit Card Fraud Detection/Credit_Card_Fraud_Detection.csv`

## Steps
### 1. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Understanding distributions of features like `amount`, `type`, and `isFraud`.
- **Bivariate Analysis**: Exploring relationships between numerical and categorical variables.

### 2. Feature Preprocessing
Generated features to improve model performance, including:
- Encoding categorical variables (e.g., transaction types).
- Categorizing numerical variables into bins (e.g., `categorize_amount`).

### 3. Model Development
The primary model used is the **Decision Tree Classifier**, with the following steps:
- Splitting the dataset into training and validation sets.
- Training the model on the training set.
- Evaluating performance using metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Libraries Used
This project uses the following Python libraries:
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `matplotlib` and `seaborn` for data visualization.
- `scikit-learn` for machine learning and evaluation metrics.
- `joblib` for saving and loading model artifacts.

## Results
The Decision Tree Classifier achieved the following metrics on the testing set:
- **Precision (Fraud)**: 85.94%
- **Recall (Fraud)**: 92.10%
- **F1-Score (Fraud)**: 88.92%
- **False Positive Rate (FPR)**: 0.02%
- **Accuracy**: 99.96%
- **ROC-AUC Score**: 94.68%

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-fraud-detection.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Credit_Fraud_Detection_EDA_Modeling.ipynb
   ```

## Potential Impact
This project demonstrates the application of machine learning to detect fraud, which can help financial institutions reduce losses and improve security.

## Expected Outcomes
- A robust machine learning pipeline for fraud detection.
- Insights into key factors contributing to fraudulent transactions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to modify or extend this README to suit your project's needs.
