# Customer Churn Prediction 📉

This project is focused on analyzing and predicting customer churn using a real-world telecom dataset. It aims to help businesses understand why customers leave and how to prevent churn using machine learning techniques.

## 🚀 Project Structure

Churn Analysis Project/
│
├── Dataset/ # Contains the churn dataset (CSV)
├── Model/ # Trained ML model (.pkl file)
├── Notebook/ # Jupyter Notebooks with analysis & ML
├── requirements.txt # List of Python dependencies
├── .gitignore # Ignored files/folders for Git
└── README.md # Project documentation


## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
- Pickle

## 📊 Tasks Covered

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (Logistic Regression, Random Forest, etc.)
- Model Evaluation
- Saving and Loading the Model

## 📦 How to Run

```bash
# Clone the repo
git clone https://github.com/NiteshToma/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual env (optional)
python -m venv venv
venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook

📁 Dataset
The dataset used is: WA_Fn-UseC_-Telco-Customer-Churn.csv
It contains information about telecom customers and whether they churned.