Spam Email Detection using Logistic Regression
📘 Project Overview

This project builds a Machine Learning model to classify emails as Spam or Ham (Not Spam) using Logistic Regression.
Text data is converted into numerical form using the Bag-of-Words model (CountVectorizer), and the classifier learns to detect spam patterns based on word frequencies.

🧠 Objective

To identify whether a given email message is spam or not spam based on its content using machine learning techniques.

📂 Dataset

Source: SMS Spam Collection Dataset (UCI / GitHub)

Dataset Format:
Separated by tabs (\t)

Columns:

label → message type (spam / ham)
text → actual message content

🧩 Steps Performed

Data Loading – Loaded the dataset from GitHub using Pandas.
Data Cleaning – Renamed columns and encoded target labels (ham = 1, spam = 0).
Feature Extraction – Used CountVectorizer to convert text into numeric features.
Train-Test Split – Split data into 80% training and 20% testing.
Model Training – Trained a LogisticRegression classifier.
Evaluation – Checked accuracy, confusion matrix, and classification report.
Visualization – Visualized confusion matrix using Seaborn heatmap.

Testing – Predicted outcomes for new custom messages.

🧾 Model Used

Algorithm: Logistic Regression
Feature Extraction: Bag-of-Words (CountVectorizer)
Accuracy: ~97–99%
Libraries: scikit-learn, pandas, seaborn, matplotlib

📊 Visualizations

Confusion Matrix (Heatmap for classification results)
Accuracy and precision-recall metrics
Example predictions for custom messages

🧠 Example Output

Custom Messages Tested:

Message	Prediction
Congratulations! You won a free ticket	🚨 Spam
Hey, are we meeting today?	✅ Ham
Claim your $1000 prize now	🚨 Spam
Don’t forget to submit the assignment	✅ Ham

⚙️ Technologies Used

Python 🐍
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

💡 Future Improvements

Apply TF-IDF Vectorizer for better text representation.
Try Naive Bayes, SVM, or Random Forest classifiers.
Implement deep learning with LSTM / BERT for advanced NLP.
Build a Flask / Streamlit web app for live message prediction.

👨‍💻 Author

Ankit Kashyap
Data Science & Machine Learning Enthusiast
