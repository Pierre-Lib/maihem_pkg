from ultralytics import YOLO


def train_model(data_yaml, model_arch='yolo11n', epochs=100, batch_size=16, img_size=640):
    """
    Train a YOLO11 model on a custom dataset.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        model_arch (str): Model architecture to use (e.g., 'yolo11n', 'yolo11s').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        img_size (int): Image size for training.
    """
    model = YOLO(f'{model_arch}.yaml')  # Initialize model
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size)
    print(f"Training completed. Model saved to {model_arch}.pt")
