import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from feedback import update_model

# Load the trained model and vectorizer
model = joblib.load('model/best_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter the text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Transform the user input text
        input_vector = vectorizer.transform([user_input])
        
        # Predict the sentiment
        prediction = model.predict(input_vector)[0]
        
        # Display the prediction
        if prediction == 1:
            st.success("The sentiment is Positive.")
        elif prediction == 0:
            st.warning("The sentiment is Neutral.")
        else:
            st.error("The sentiment is Negative.")
        
        # Display feedback options
        st.write("Was the prediction correct?")
        if st.button("Yes"):
            update_model(user_input, prediction)
            st.success("Thank you for your feedback!")
        if st.button("No"):
            st.write("Please select the correct sentiment:")
            feedback_label = st.radio("", ["Positive", "Neutral", "Negative"])
            feedback_value = 1 if feedback_label == "Positive" else (0 if feedback_label == "Neutral" else -1)
            update_model(user_input, feedback_value)
            st.success("Thank you for your feedback!")
    else:
        st.error("Please enter some text for analysis.")

st.write("## Model Evaluation")

# Load test data
test_data = pd.read_csv('data/Emotions.csv')
corpus = test_data['text']
y_test = test_data['label'].astype(int)

# Preprocess the test data
X_test = vectorizer.transform(corpus)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write("Confusion Matrix:")
st.text(conf_matrix)
st.text("Classification Report:")
st.text(classification_rep)

# Plot Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
st.pyplot(fig)
