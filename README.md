# SMS-Email-Spam-Detection
This project builds and evaluates a machine learning model to classify SMS messages as Spam or Ham (legitimate). It demonstrates data preprocessing, exploratory data analysis (EDA), feature extraction, and model building using natural language processing (NLP) techniques.

# 📜 Project Overview
Objective: Detect spam messages in SMS text data.

Approach:

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Text Vectorization using TF-IDF

Model Building using various Machine Learning algorithms

Model Evaluation and Comparison

# 🛠️ Technologies Used
Python 3

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

nltk (Natural Language Toolkit)

Algorithms:

Naive Bayes Classifier

Support Vector Machine (SVM)

Logistic Regression

Random Forest Classifier

# 📂 Dataset
Source: UCI Machine Learning Repository - SMS Spam Collection

Description: The dataset contains 5,572 SMS messages tagged as either 'ham' (legitimate) or 'spam'.

Attributes:

label: 'ham' or 'spam'

message: Text content of the SMS

# 🔥 Key Steps
1. Data Loading and Exploration
Load the dataset using pandas.

Check for missing values and perform basic statistics.

Visualize class distributions.

2. Data Preprocessing
Lowercasing text

Removing punctuation and special characters

Tokenization

Stopwords removal

Stemming using NLTK

3. Feature Engineering
Text data transformed into numerical features using TF-IDF Vectorization.

4. Model Building and Evaluation
Split data into training and testing sets (e.g., 80/20 split).

Train multiple models and compare performance using:

Accuracy

Precision

Recall

F1-Score

Confusion matrix and ROC-AUC score for deeper insights.

5. Conclusion
The best-performing model is identified based on evaluation metrics.

Possible future improvements like hyperparameter tuning and deep learning models are suggested.

# 📈 Results
Achieved high accuracy in distinguishing between spam and ham messages.

Models like Multinomial Naive Bayes and SVM showed strong performances due to their suitability for text classification.
