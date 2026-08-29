import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load dataset
data = pd.read_csv("dataset.csv")

#remove empty rows
data = data.dropna()

#Input and output
X = data["description"]
y = data["category"]

#convert text into numbers
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

#Split data into training and testing
X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size = 0.2, random_state = 42)

#Train AI model
model = LogisticRegression()

model.fit(X_train,y_train)

#Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

print("Model Accuracy:",accuracy)

#Take new expense from user
expense = input("Enter your expense descriptions:")

expense_vectorized = vectorizer.transform([expense])

prediction = model.predict(expense_vectorized)

print("Predicted Category:",prediction[0])

#Split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X_vectorized,y,test_size= 0.2,random_state = 42)

#Train the AI model
model = LogisticRegression()
model.fit(X_train,y_train)

#Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:",accuracy)

#Predict multiple expenses

while True:
    expense = input("Enter your expense description(or type'exit' to stop):")
    if expense.lower() == "exit":
      print("Thank you for using Smart Spend-AI!")
    break

expense_vectorized = vectorizer.transform([expense])
prediction = model.predict(expense_vectorized)

print("Predicted Category:",prediction[0])