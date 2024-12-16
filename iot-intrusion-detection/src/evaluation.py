import matplotlib.pyplot as plt

def plot_training_metrics(history, f1_callback):
    """
    Plot training and validation metrics
    
    Args:
        history (keras.callbacks.History): Model training history
        f1_callback (F1ScoreCallback): Custom F1 score callback
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['f1_score_metric'], label='Training F1 Score')
    plt.plot(history.history['val_f1_score_metric'], label='Validation F1 Score')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
