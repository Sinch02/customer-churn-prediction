# customer-churn-prediction
Predicting customer churn using machine learning models.

# Customer Churn Prediction
This project involves building a machine learning model to predict whether a customer is likely to churn (cancel their subscription) or remain with a subscription-based service. The analysis is performed using a Jupyter Notebook.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Project Overview
The primary objective of this project is to:

- Analyze customer data to identify trends and factors influencing churn.
- Build machine learning models (Logistic Regression, Random Forest, and Gradient Boosting) to predict churn.
- Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

## Features
The dataset includes customer demographics, subscription details, and usage behavior. Key features analyzed:

- **CreditScore:** Customer's credit score.
- **Geography:** Country of residence.
- **Gender:** Customer's gender.
- **Age:** Customer's age.
- **Balance:** Account balance.
- **NumOfProducts:** Number of products owned.
- **IsActiveMember:** Whether the customer is an active member.
- **EstimatedSalary:** Customer's estimated salary.

## Dataset
The dataset used for this project is located at:

```plaintext
D:\IMMERSIVIFY Intern\Customer Churn Prediction\archive\Churn_Modelling.csv
```

## Tools and Libraries Used
- **Python:** Programming language for data analysis and machine learning.
- **Jupyter Notebook:** Interactive environment for code and visualizations.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical data visualization.
- **Scikit-learn:** Machine learning algorithms and evaluation metrics.

## Installation
Follow these steps to set up the project environment:

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

## Project Workflow
### Data Loading and Exploration:
- Load the dataset using Pandas.
- Explore and clean the data to remove missing values or inconsistencies.

### Data Preprocessing:
- Encode categorical variables (e.g., Geography, Gender).
- Normalize numerical features for better model performance.
- Split the data into training and testing sets.

### Model Training:
- Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting models.

### Evaluation:
- Use metrics like accuracy, precision, recall, and F1-score to evaluate models.

## Results
- **Logistic Regression:** Accuracy = 81.10%
- **Random Forest:** Accuracy = 86.65%
- **Gradient Boosting:** Accuracy = 86.75%

## Results and Insights
Gradient Boosting achieved the highest F1-score and balanced recall, making it the most suitable model for this task.

The key factors influencing customer churn included:
- Customer age and balance.
- Activity level (IsActiveMember).
- Number of products owned (NumOfProducts).

## Challenges Faced
- Dealing with class imbalance in the dataset. (Fewer churned customers compared to non-churned ones.)
- Tuning hyperparameters for better performance.

## How to Run the Project
Ensure all required libraries are installed.

1. Open the `Customer_Churn_Prediction.ipynb` file in Jupyter Notebook.
2. Run the cells sequentially to execute the project.

## Future Improvements
- Experiment with additional machine learning models like XGBoost or Neural Networks.
- Explore feature engineering to identify new predictive features.
- Use oversampling or undersampling techniques to handle class imbalance better.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author
- Sinchana B R
- Contact: sinchanabr02@gmail.com 

Feel free to reach out for questions or suggestions regarding this project!.
Future Enhancements.
Use advanced algorithms like XGBoost or CatBoost for further improvement.
Implement techniques to address class imbalance (e.g., SMOTE).
Explore deep learning models for churn prediction.


