# Obesity Dataset Exploration, Model Development, and Evaluation

Welcome to the project! Here's what we aim to accomplish:

Exploratory Data Analysis (EDA): <br>
We'll conduct a thorough analysis of an obesity dataset to uncover important factors that influence obesity levels. This involves exploring relationships between various features and the target variable, gaining insights into trends, distributions, and potential correlations.

Machine Learning Model Development: <br>
The aim is to build a robust machine learning model capable of accurately predicting obesity levels based on the identified predictors. This phase involves data preprocessing, feature selection, model training, evaluation, and fine-tuning to achieve optimal performance.

Machine Learning Model Testing: <br>
Finally we utilize the power of GitHub Actions to ensure continuous integration (CI), monitoring our model's performance rigorously, with a target F1 score surpassing 0.95.

---

Here's a breakdown of what you'll find in our project:

obesity.ipynb:
This Jupyter Notebook houses both the analysis of the dataset and the development of the machine learning model. It's where we explore the data, identify key predictors of obesity, and construct our predictive model.

tools.py:
In this file, you'll discover a collection of helper functions essential for our project. These functions are utilized in various aspects, aiding in tasks such as data loading, transformation, and preprocessing.

test_performance.py:
This dedicated test file serves a crucial role in our project. It's responsible for testing two critical aspects:

a. The accuracy of functions tasked with loading and transforming datasets. We ensure these functions operate correctly and produce the expected outputs.

b. The performance of our model. We set specific criteria, such as achieving an F1 score greater than 0.95, to evaluate the effectiveness of our model in predicting obesity levels accurately.

Through these components, we ensure the reliability of our data processing pipeline and the effectiveness of our predictive model.