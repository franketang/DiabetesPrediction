Diabetes Prediction Project

Project Overview
This project leverages Python and machine learning to predict diabetes from clinical datasets. The approach involves data preprocessing, model training with logistic regression, and evaluating model effectiveness through various metrics and visual analyses.

Steps

Step 1: Setup Environment
Set up the Python development environment, ensuring all dependencies are correctly managed using a virtual environment.
- Commands:
  Bash Command:
  pip install virtualenv
  python -m virtualenv venv
  source venv/bin/activate
  pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab
 

![image](https://github.com/franketang/DiabetesPrediction/assets/29631514/11a10057-7aef-4276-ad5d-f89b21770219)

Picture 1: Demonstrates the command-line output while installing virtualenv and activating the environment.

Step 2: Data Loading
Load the diabetes dataset using pandas, a powerful Python data analysis toolkit. The dataset features several clinical predictors and a binary outcome variable indicating diabetes presence.
- Code Snippet:
 python
  import pandas as pd
  df = pd.read_csv('diabetes.csv')
  print(df.head())
 

![image](https://github.com/franketang/DiabetesPrediction/assets/29631514/0200bf9f-e045-4d96-9044-25524c9d4c65)

Picture 2: Displays the initial rows of the dataset as seen in Jupyter Notebook.

Step 3: Data Preprocessing
Data is cleaned and preprocessed to ensure quality and compatibility with the machine learning model. This includes filling missing values, encoding categorical variables, and normalizing data.
- Code Snippet:
 python
  df.fillna(method='ffill', inplace=True)
 

Step 4: Train-Test Split
The dataset is split into training and testing sets to ensure unbiased evaluation of the model.
- Code Snippet:
 python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome', axis=1), df['Outcome'], test_size=0.2, random_state=42)
 

Step 5: Model Training
A logistic regression model is trained using the scikit-learn library. This model is well-suited for binary classification tasks.
- Code Snippet:
 python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  model.fit(X_train, y_train)
 

Step 6: Model Evaluation
The model's performance is evaluated using accuracy and a confusion matrix, which provides insights into the true positives, false positives, true negatives, and false negatives.
- Code Snippet:
 python
  from sklearn.metrics import accuracy_score, confusion_matrix
  predictions = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, predictions))



 ![image](https://github.com/franketang/DiabetesPrediction/assets/29631514/aa94735b-1a75-4197-b388-bba70bd51b73)
Picture 3: The confusion matrix visualized using seaborn.

Step 7: ROC Curve Analysis
The Receiver Operating Characteristic (ROC) curve is plotted to evaluate the trade-offs between true positive rate and false positive rate at various threshold settings.
- Code Snippet:
 python
  from sklearn.metrics import roc_curve, auc
  fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))



 ![image](https://github.com/franketang/DiabetesPrediction/assets/29631514/6900007c-2d1e-43a8-a75d-ad55616ed8c2)
Picture 4: ROC curve, demonstrating model diagnostic ability.

Visualizations
Several plots are generated to visualize the steps and outcomes of the analysis:
- Setup and Initial Data Inspection: Initial command-line setups and data inspection in Jupyter Notebook.
- Intermediate Analysis and Outputs: Detailed Python command executions and outputs showcasing data manipulations and initial model results.
- Final Evaluation: Advanced visual outputs such as confusion matrix and ROC curve plots.


Conclusion
The project successfully implements a logistic regression model to predict diabetes with an accuracy of approximately 75%, as demonstrated by the confusion matrix and ROC curve. The ROC curve, with an area under the curve (AUC) of about 0.81, suggests that the model has good discriminatory ability between the positive and negative classes. These metrics provide crucial insights into the model's effectiveness and potential areas for improvement, such as parameter tuning or trying different algorithms for better performance.

How to Run
To replicate this project, follow the detailed steps provided in this README, ensuring each command and code snippet is executed in sequence in a Python-enabled environment like JupyterLab.
