import pandas as pd
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from modelTrain import train_model
from utils import preprocess_data
from tqdm import tqdm

def main():
    # Load the dataset
    df = pd.read_csv('data/Emotions.csv')
    corpus = df['text']  # Assuming the text column is named 'text'

    # Preprocess the data
    print("Preprocessing data...")
    X, y, cv = preprocess_data(df, corpus)

    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    total_size = len(y)
    train_size = len(y_train)
    test_size = len(y_test)

    print(f"Total dataset size: {total_size}")
    print(f"Training dataset size: {train_size}")
    print(f"Testing dataset size: {test_size}")

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)

    # Train and evaluate the model
    start_time = time.time()
    print("Training the model...")

    progress_bar = tqdm(total=100, position=0, leave=True)
    progress = 0
    while progress < 100:
        time.sleep(0.1)  # Simulating the training process
        progress += 1
        progress_bar.update(1)
        elapsed_time = time.time() - start_time
        print(f"\r[{'=' * progress}{' ' * (100 - progress)}] {progress}% [{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}]", end='')
    
    progress_bar.close()

    best_model, best_params = train_model(X_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nBest Parameters: {best_params}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    # Save the model and the vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(best_model, 'model/best_model.pkl')
    joblib.dump(cv, 'model/vectorizer.pkl')
    print("Model and vectorizer saved successfully!")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_rep)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
    disp.plot()
    plt.show()

    # Check if the model folder is empty
    model_dir_empty = len(os.listdir('model')) == 0
    if model_dir_empty:
        print("Model folder is empty. There may be an issue with saving the model.")
    else:
        print("Model trained successfully.")

if __name__ == "__main__":
    main()



