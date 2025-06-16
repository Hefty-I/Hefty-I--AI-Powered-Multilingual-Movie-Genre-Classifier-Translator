import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def train_genre_model(X_train, X_test, y_train, y_test, model_type='Logistic Regression', hyperparams=None):
    """
    Train a machine learning model for genre prediction.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_type (str): Type of model to train
        hyperparams (dict): Hyperparameters for the model
        
    Returns:
        tuple: (model, evaluation_metrics, confusion_matrix)
    """
    if hyperparams is None:
        hyperparams = {}
    
    # Verify number of classes
    num_classes = y_train.shape[1]
    print(f"Training model with {num_classes} genre classes")
    
    # Check active classes per genre
    active_classes = [np.sum(y_train[:, col] > 0) > 0 for col in range(num_classes)]
    print(f"Found {sum(active_classes)} active genre classes with at least one positive sample")
    
    # Initialize classifiers for each genre
    classifiers = []
    for col in range(num_classes):
        if not active_classes[col]:
            # Use DummyClassifier for genres with no positive samples
            classifiers.append(DummyClassifier(strategy='most_frequent'))
        else:
            # Use specified model for genres with sufficient data
            if model_type == 'Logistic Regression':
                hyperparams.pop('multi_class', None)  # Remove deprecated parameter
                base_model = LogisticRegression(**hyperparams)
            elif model_type == 'Random Forest':
                base_model = RandomForestClassifier(**hyperparams)
            elif model_type == 'Support Vector Machine':
                base_model = SVC(**hyperparams)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            classifiers.append(base_model)
    
    # Create MultiOutputClassifier with mixed classifiers
    model = MultiOutputClassifier(estimator=None)
    model.estimators_ = classifiers
    
    # Train each classifier
    for idx, estimator in enumerate(model.estimators_):
        estimator.fit(X_train, y_train[:, idx])
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics with zero_division handling
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    # Calculate confusion matrix
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
    
    return model, metrics, conf_matrix

def predict_genre(text, model, vectorizer, mlb):
    """
    Predict genres for a given text.
    
    Args:
        text (str): Preprocessed text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        mlb: MultiLabelBinarizer
        
    Returns:
        list: Predicted genres
    """
    # Convert text to TF-IDF features
    text_features = vectorizer.transform([text])
    
    # Predict genres
    genre_predictions = model.predict(text_features)
    
    # Convert predictions to genre labels
    predicted_genres = mlb.inverse_transform(genre_predictions)
    
    # Flatten the list of lists
    predicted_genres = [genre for sublist in predicted_genres for genre in sublist]
    
    return predicted_genres

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics with zero_division handling
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_test, average='micro', zero_division=0),
        'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    return metrics

def get_top_features(vectorizer, model, genre_idx, n=10):
    """
    Get top features (words) for a specific genre.
    
    Args:
        vectorizer: TF-IDF vectorizer
        model: Trained model
        genre_idx (int): Index of the genre
        n (int): Number of top features to return
        
    Returns:
        list: List of tuples (feature, coefficient)
    """
    try:
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Check if we have a dummy classifier or another model that doesn't have coefficients
        if isinstance(model.estimators_[genre_idx], DummyClassifier):
            return [("No feature importance available for DummyClassifier", 0)]
        
        # Get coefficients or feature importances
        if hasattr(model.estimators_[genre_idx], 'feature_importances_'):
            coef = model.estimators_[genre_idx].feature_importances_
        elif hasattr(model.estimators_[genre_idx], 'coef_'):
            coef = model.estimators_[genre_idx].coef_[0]
        else:
            return [("No feature importance available for this model type", 0)]
        
        # Create tuples of (feature, coefficient)
        feature_coefs = list(zip(feature_names, coef))
        
        # Sort by absolute coefficient value
        feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_coefs[:n]
    except Exception as e:
        print(f"Error getting top features: {str(e)}")
        return [("Error getting features", 0)]