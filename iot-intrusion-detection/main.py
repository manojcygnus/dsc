from src.data_preprocessing import load_data, split_data
from src.model import create_mlp_model
from src.train import train_model
from src.evaluation import plot_training_metrics

def main():
    # File path for dataset
    file_path = 'content/drive/MyDrive/Dsc Files/ACI-IoT-2023.csv'

    # Load and preprocess data
    X, y, encoder = load_data(file_path)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create model
    num_classes = len(encoder.classes_)
    model = create_mlp_model(X_train.shape[1], num_classes)

    # Train model
    training_results = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        encoder
    )

    # Plot training metrics
    plot_training_metrics(
        training_results['history'], 
        training_results['f1_callback']
    )

if __name__ == "__main__":
    main()
