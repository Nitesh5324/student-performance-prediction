
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# Load student dataset
data = pd.read_csv("data/students.csv")

# Select input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed"
    ]
]

# Select the score we want to predict
y = data["final_score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Save the trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
