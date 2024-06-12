# Sentiment Analysis App

This Streamlit web application analyzes the sentiment of text inputs using a pre-trained machine learning model. It provides insights into the sentiment distribution of a dataset, model performance metrics, and allows users to input text for sentiment analysis in real-time.

## Features

- **Sentiment Analysis**: Users can enter text into the provided text area, and the app will predict the sentiment of the input text (positive, negative, or neutral).
- **Model Performance Metrics**: Display of model accuracy, confusion matrix, and classification report.
- **Dataset Distribution**: Visualization of sentiment distribution within the dataset.
- **Sentiment Means Over Time**: Line plot showing the trend of sentiment means over time.
- **Density Plot**: Bar plot showing the density of sentiment categories.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/CodeSage4D/SentiModel_Analysis.git
```
## Install the required dependencies:
```bash
pip install -r requirements.txt
```
pip install -r requirements.txt

## Run the Streamlit app:
```bash
streamlit run app.py
```
# Project Directory: 
    SentiAnalysis/
    │
    ├── data/
    │   └── Emotions.csv
    │
    ├── model/
    │   ├── best_model.pkl
    │   └── vectorizer.pkl
    │
    ├── final.py
    ├── modelTrain.py
    ├── utils.py
    ├── app.py
    ├── feedback.py
    └── venv/


# Use View:
1. Open the Streamlit app in your browser.
2. Explore the different sections of the app:
3. Model and Dataset Details: Sidebar section displaying details about the model and dataset.
4. Model Performance: Metrics and visualizations related to model performance.
5. Confusion Matrix and Dataset Distribution: Parallel display of confusion matrix and dataset distribution.
6. Sentiment Analysis: Input text area and sentiment analysis results.
7. Enter text into the text area and click on the "Analyze" button to see the sentiment analysis results.

# Built With
Streamlit - The web framework used for building the app.
Pandas - Data manipulation and analysis library.
scikit-learn - Machine learning library for model training and evaluation.
Matplotlib and Seaborn - Data visualization libraries.
