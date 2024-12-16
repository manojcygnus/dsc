import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report

class F1ScoreCallback(keras.callbacks.Callback):
    """
    Custom callback to track F1 score during training
    """
    def __init__(self, validation_data, test_data):
        super(F1ScoreCallback, self).__init__()
        self.x_val, self.y_val = validation_data
        self.x_test, self.y_test = test_data
        self.f1_scores_val = []
        self.f1_scores_test = []

    def on_epoch_end(self, epoch, logs=None):
        val_preds = np.argmax(self.model.predict(self.x_val), axis=1)
        test_preds = np.argmax(self.model.predict(self.x_test), axis=1)

        val_true = np.argmax(self.y_val, axis=1)
        test_true = np.argmax(self.y_test, axis=1)

        f1_val = f1_score(val_true, val_preds, average='macro', zero_division=0)
        f1_test = f1_score(test_true, test_preds, average='macro', zero_division=0)

        self.f1_scores_val.append(f1_val)
        self.f1_scores_test.append(f1_test)

        print(f'\nEpoch {epoch + 1} - Val F1: {f1_val:.4f} - Test F1: {f1_test:.4f}')

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, encoder):
    """
    Train the model with early stopping and F1 score tracking
    
    Args:
        model (keras.Model): Compiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        X_test, y_test: Test data and labels
        encoder (LabelEncoder): Label encoder for class names
    
    Returns:
        dict: Training history and model
    """
    # Convert labels to one-hot encoding
    y_train_onehot = keras.utils.to_categorical(y_train)
    y_val_onehot = keras.utils.to_categorical(y_val)
    y_test_onehot = keras.utils.to_categorical(y_test)

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_f1_score_metric',
        mode='max',
        min_delta=0.001,
        patience=15,
        restore_best_weights=True
    )

    f1_callback = F1ScoreCallback(
        validation_data=(X_val, y_val_onehot),
        test_data=(X_test, y_test_onehot)
    )

    # Train the model
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        batch_size=256,
        epochs=100,
        callbacks=[early_stopping, f1_callback],
        verbose=1
    )

    # Evaluate the model
    test_preds = np.argmax(model.predict(X_test), axis=1)
    test_true = np.argmax(y_test_onehot, axis=1)

    # Print classification report
    classification_rep = classification_report(
        test_true, test_preds, target_names=encoder.classes_
    )
    print("\nClassification Report:")
    print(classification_rep)

    return {
        'history': history,
        'model': model,
        'f1_callback': f1_callback,
        'classification_report': classification_rep
    }
