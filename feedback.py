import joblib
import pandas as pd

def update_model(feedback_text, feedback_label, model_path='model/best_model.pkl', vectorizer_path='model/vectorizer.pkl', data_path='data/feedback_data.csv'):
    # Load the existing model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Transform the feedback text using the vectorizer
    feedback_vector = vectorizer.transform([feedback_text])
    
    # Update the model (we'll use incremental learning here if applicable)
    # Otherwise, save the feedback to a file for later retraining
    feedback_data = pd.DataFrame({
        'text': [feedback_text],
        'label': [feedback_label]
    })
    
    feedback_data.to_csv(data_path, mode='a', header=False, index=False)
    
    print("Feedback saved. Retrain the model later with the new feedback data.")

# Example usage:
# update_model("This is a positive text", 1)
