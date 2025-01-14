from ultralytics import YOLO


def predict_image(model_path, image_path, conf_threshold=0.25):
    """
    Perform inference on an image using a trained YOLO11 model.

    Args:
        model_path (str): Path to the trained model file.
        image_path (str): Path to the input image.
        conf_threshold (float): Confidence threshold for predictions.
    """
    model = YOLO(model_path)  # Load trained model
    results = model(image_path, conf=conf_threshold)
    return results
