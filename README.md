# Credit Card Default Prediction

This project aims to predict the risk of credit card defaults using various machine learning techniques. The project includes data preprocessing, exploratory data analysis, handling data imbalance, and training different machine learning models to classify the risk of credit card defaults.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Handling Data Imbalance](#handling-data-imbalance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-default-prediction.git
   cd credit-card-default-prediction
## Usage

1. Load the dataset:
   ```python
   ccd = pd.read_csv('/path/to/UCI_Credit_Card.csv', index_col='ID')
   ccd.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
2. Check for missing values:
   ```python
   print(ccd.isnull().sum())
3. Visualize initial data:
   ```python
   import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x="default.payment.next.month", data=ccd, palette="rocket")
    plt.xlabel("Default?")
    plt.ylabel("Number of Clients")
    plt.show()
4. Conduct exploratory data analysis:
   ```python
   plt.figure(figsize=(18,15))
   sns.heatmap(ccd.corr(), annot=True)
   plt.show()
5. Encode categorical variables and scale numerical features:
   ```python
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   onehotencoder = OneHotEncoder()
   cat_df = ccd[['SEX', 'EDUCATION', 'MARRIAGE']]
   cat = onehotencoder.fit_transform(cat_df).toarray()
   cat = pd.DataFrame(cat)
    
   scaler = StandardScaler()
   num_df = ccd.drop(['SEX', 'EDUCATION', 'MARRIAGE'], axis=1)
   num_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
6. Split the data:
   ```python
   from sklearn.model_selection import train_test_split
   X = ccd.drop('default.payment.next.month', axis=1)
   y = ccd['default.payment.next.month']
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
7. Train and evaluate a baseline model:
   ```python
   from sklearn.dummy import DummyClassifier
   from sklearn.metrics import accuracy_score
   dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
   dummy_prediction = dummy.predict(X_test)
   print(accuracy_score(y_test, dummy_prediction))
8. Handle data imbalance using SMOTE:
   ```python
   from imblearn.over_sampling import SMOTE
   sm = SMOTE(sampling_strategy='auto')
   X_res, y_res = sm.fit_resample(X, y)
9. Train and evaluate models:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   print(accuracy_score(y_test, predictions))

## Features

- Data preprocessing
- Exploratory data analysis
- Model training with various machine learning algorithms
- Handling data imbalance using SMOTE
- Hyperparameter tuning

## Data Preprocessing

1. Check for missing values.
2. Visualize initial data distributions.
3. Encode categorical variables using OneHotEncoder.
4. Scale numerical features using StandardScaler.

## Exploratory Data Analysis

1. Generate correlation heatmap.
2. Scatter plots to identify outliers and relationships.
3. Distribution plots for numerical features.

##Model Training and Evaluation

1. Split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split
   X = ccd.drop('default.payment.next.month', axis=1)
   y = ccd['default.payment.next.month']
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
2. Train and evaluate a baseline model:
   ```python
   from sklearn.dummy import DummyClassifier
   from sklearn.metrics import accuracy_score
   dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
   dummy_prediction = dummy.predict(X_test)
   print(accuracy_score(y_test, dummy_prediction))
   
## Handling Data Imbalance

1. Use SMOTE to handle class imbalance:
   ```python
   from imblearn.over_sampling import SMOTE
   sm = SMOTE()
   X_res, y_res = sm.fit_resample(X, y)
   
## Hyperparameter Tuning

1. Perform hyperparameter tuning using RandomizedSearchCV:
   ```python
   from sklearn.model_selection import RandomizedSearchCV
   params = {
        'n_estimators': [200, 400, 600],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
   }
   random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=10, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
   random_search.fit(X_res, y_res)

## Results

Final Accuracy Score: 86.58

Detailed classification report and confusion matrix are provided in the Jupyter Notebook.







