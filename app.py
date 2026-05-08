import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page title
st.title("Breast Cancer Prediction using Machine Learning")

st.write("This application predicts whether the tumor is Benign or Malignant.")

# Load dataset
df = pd.read_csv("Breast_cancer_dataset.csv")

# Drop unnecessary columns if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Convert diagnosis column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Enter Patient Data")

input_data = []

for column in X.columns[:5]:
    value = st.number_input(f"{column}", value=0.0)
    input_data.append(value)

# Fill remaining columns with mean values
for column in X.columns[5:]:
    input_data.append(X[column].mean())

if st.button("Predict Cancer"):

    prediction = model.predict([input_data])

    if prediction[0] == 1:
        st.error("Malignant Cancer Detected")
    else:
        st.success("Benign Tumor Detected")