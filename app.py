import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Load dataset
data = pd.read_csv("dataset.csv")

#Remove empty rows
data = data.dropna()

#Input and output
X = data["description"]
y = data["category"]

#Convert text into numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

#Train model
model = LogisticRegression()
model.fit(X_vectorized,y)

#App title
st.title("Smart Spend-AI")
st.write("AI-Powered Expense Category Predictor")

#User input
expense = st.text_input("Enter your expense description:")

#Prediction

if st.button("Predict Category"):
    if expense:
        expense_vectorized = vectorizer.transform([expense])
        prediction = model.predict(expense_vectorized)

        st.success(f"Predicted Category:{prediction[0]}")
    else:
        st.warming("Please enter an expense description.")
