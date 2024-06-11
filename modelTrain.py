from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

def train_model(X_train, y_train):
    parameters = {
        'max_features': ('auto', 'sqrt'),
        'n_estimators': [500, 1000, 1500],
        'max_depth': [5, 10, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Grid Search Elapsed Time: {elapsed_time:.2f} seconds")
    
    return grid_search.best_estimator_, grid_search.best_params_
