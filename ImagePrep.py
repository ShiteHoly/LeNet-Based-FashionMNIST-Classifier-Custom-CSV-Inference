from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the input image for LeNet model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28 pixels
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize to the same as MNIST dataset
    ])

    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale if not already

    # Apply the transformations
    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)  # Add batch dimension

def preprocess_csv(csv_file):
    """
    Preprocess the input CSV file for LeNet model.

    Args:
        csv_file (str): Path to the input CSV file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file, header=None, skiprows=1).values

    # Convert to numpy array and normalize
    images = data.astype(np.float32) / 255.0

    # Normalize to match MNIST dataset mean and std
    images = (images - 0.1307) / 0.3081

    # Reshape to [num_samples, 1, 28, 28]
    images = images.reshape((-1, 1, 28, 28))

    # Convert to PyTorch tensor
    images_tensor = torch.tensor(images)
    return images_tensor