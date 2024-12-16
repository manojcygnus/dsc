import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def f1_score_metric(y_true, y_pred):
    """
    Custom F1 score metric for Keras
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        F1 score as a tensor
    """
    # Convert probabilities to predicted class
    y_pred = K.cast(K.argmax(y_pred, axis=-1), tf.float32)
    y_true = K.cast(K.argmax(y_true, axis=-1), tf.float32)

    # Calculate precision and recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1

def create_mlp_model(input_shape, num_classes):
    """
    Create a Multi-Layer Perceptron model
    
    Args:
        input_shape (int): Number of input features
        num_classes (int): Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_shape,)),

        # First dense layer with batch normalization
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second dense layer with batch normalization
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Third dense layer with batch normalization
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output layer with softmax activation
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model with F1 score as metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[f1_score_metric]
    )

    return model
