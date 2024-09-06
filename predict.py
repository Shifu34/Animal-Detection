import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import get_pretrained_model  # Import your custom/pretrained model
import os

# Define the device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation (must be consistent with what the model expects)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match model input
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Load the model
model = get_pretrained_model(num_classes=2).to(device)  # Adjust number of classes for cat/dog classifier
model.load_state_dict(torch.load("cat_dog_classifier.pt"))  # Load the saved model weights
model.eval()  # Set model to evaluation mode

def predict_image(image_path):
    """
    Predict the label (cat or dog) for an input image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        str: Predicted label (cat or dog).
    """
    # Open the image
    image = Image.open(image_path)

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Move the image tensor to the device (GPU/CPU)
    image = image.to(device)

    # Disable gradient calculations (no need during inference)
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

    # Map prediction to label (0 -> 'cat', 1 -> 'dog')
    label = 'dog' if predicted.item() == 1 else 'cat'
    return label

if __name__ == "__main__":
    # Path to the image you want to predict
    image_path = "D:\\MLOps\\Cat-Dog-Classifier\\dataset\\train\\cat\\Bengal_165_jpg.rf.31185846cf5892c6e7e3f02784e04fef.jpg"  # Replace with the path to the image you want to test

    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
    else:
        # Run the prediction
        prediction = predict_image(image_path)
        print(f"Prediction: {prediction}")
