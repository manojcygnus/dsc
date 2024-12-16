import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

def load_data(file_path):
    """
    Load and preprocess the IoT dataset
    
    Args:
        file_path (str): Path to the dataset CSV file
    
    Returns:
        tuple: Preprocessed features and labels
    """
    # Load Dataset
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df.drop(["Flow Bytes/s", "Timestamp", "Flow Packets/s"], axis=1, inplace=True)
    
    # Prepare features and target
    features = df.columns.tolist()
    features.remove("Label")
    features.remove("Flow ID")
    features.remove("Src IP")
    features.remove("Dst IP")
    
    # One-hot encode features
    X = pd.get_dummies(df[features], dtype=float)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Label"])
    
    return X_scaled, y, encoder

def split_data(X, y, test_size=0.2, val_size=0.25):
    """
    Split data into train, validation, and test sets
    
    Args:
        X (array): Features
        y (array): Labels
        test_size (float): Proportion of test set
        val_size (float): Proportion of validation set
    
    Returns:
        tuple: Train, validation, and test sets for features and labels
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
