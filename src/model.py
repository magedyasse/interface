import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_models():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'SVC': SVC(probability=True),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
        
        
    }
    return models

def get_parameters():
    parameters = {
       'LogisticRegression': {
        'clf__C': np.logspace(-3, 3, 7),
        'clf__class_weight': ['balanced'],
        'clf__penalty': ['l2', 'l1'],
        'clf__solver': ['liblinear', 'lbfgs']
    },
    'SVC': {
        'clf__C': [0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],
        'clf__class_weight': [None, 'balanced']
    },
    'DecisionTreeClassifier': {
        'clf__max_depth': [None, 5, 10, 15],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__class_weight': [None, 'balanced'],
        'clf__criterion': ['gini', 'entropy']
    },
    'RandomForestClassifier': {
        'clf__n_estimators': [10, 20, 50, 100],
        'clf__max_depth': [None, 5, 10, 15],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
        }
    }
    return parameters

def train_models(x_train, y_train, models, parameters):
    results = []
    os.makedirs("saved_models", exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}........")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])

        param_grid = parameters.get(name, {})

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(x_train, y_train)

        model_path = f"saved_models/{name}_model.pkl"
        joblib.dump(grid.best_estimator_, model_path)

        results.append({
            'name': name,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_,
            'model': grid.best_estimator_,
            'model_path': model_path
        })
    return results




