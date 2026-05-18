"""
Train a machine learning model on the heart disease dataset and save it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_save_model(dataset_path: str = "dataset.csv", model_path: str = "model.joblib", scaler_path: str = "scaler.joblib"):
    """
    Train a Random Forest classifier on the dataset and save the model.
    """
    print("📚 Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target value counts:\n{df['target'].value_counts()}\n")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Training accuracy: {train_score:.4f}")
    print(f"✅ Testing accuracy: {test_score:.4f}\n")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("📊 Feature Importance:")
    print(importance_df.to_string(index=False))
    print()
    
    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, "feature_names.joblib")
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Feature names saved to feature_names.joblib")


if __name__ == "__main__":
    train_and_save_model()
