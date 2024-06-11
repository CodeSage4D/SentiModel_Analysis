from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(df, corpus):
    # Drop rows with NaN values in the 'label' column
    df.dropna(subset=['label'], inplace=True)

    # Create the CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    y = df['label']

    # Ensure that the target variable is of integer type
    y = y.astype(int)
    
    return X, y, cv
