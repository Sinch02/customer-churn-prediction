# customer-churn-prediction
Predicting customer churn using machine learning models.



Customer Churn Prediction
This repository contains the implementation of a machine learning model to predict customer churn for a subscription-based service or business. The project demonstrates the end-to-end process of building and evaluating models to identify customers at risk of canceling their subscriptions.

Task Objectives
The primary goal of this project is:

To predict whether a customer is likely to churn or continue using the service.
To evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
Approach
Data Loading and Exploration:

Dataset: Churn_Modelling.csv.
Loaded and analyzed the data to understand features and identify potential missing or irrelevant data.
Data Preprocessing:

Handled categorical features using one-hot encoding.
Scaled numerical features for effective model training.
Split the data into training and testing sets.
Model Building and Evaluation:

Implemented and compared three algorithms:
Logistic Regression
Random Forest
Gradient Boosting
Evaluated models using metrics:
Accuracy
Precision
Recall
F1-Score
Chose the best-performing model based on evaluation results.
Challenges Faced
Handling class imbalance in the dataset (churners vs. non-churners).
Selecting the optimal algorithm for prediction.
Fine-tuning hyperparameters for improved performance.
Results Achieved
Logistic Regression:

Accuracy: 81.10%
F1-Score: 0.29
Random Forest:

Accuracy: 86.65%
F1-Score: 0.58
Gradient Boosting:

Accuracy: 86.75%
F1-Score: 0.59
Final Model: Gradient Boosting performed best in terms of overall metrics.

How to Run the Code
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/customer-churn-prediction.git
Navigate to the project directory:
bash
Copy code
cd customer-churn-prediction
Install dependencies:
Copy code
pip install -r requirements.txt
Run the script:
Copy code
python churn_prediction.py
View the evaluation results in the terminal.
Technologies and Libraries Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Future Enhancements
Use advanced algorithms like XGBoost or CatBoost for further improvement.
Implement techniques to address class imbalance (e.g., SMOTE).
Explore deep learning models for churn prediction.
