echo "🧠 LeNet-Based FashionMNIST Classifier with Custom CSV/Image Inference

This project implements a classic LeNet-5 convolutional neural network using PyTorch, trained on the FashionMNIST dataset. It supports flexible input via either image files or CSV-formatted datasets, making it adaptable for real-world handwritten digit classification tasks.

## 📌 Features

- 🧱 Classic LeNet-5 CNN architecture
- 🧪 Training on FashionMNIST dataset with >88% accuracy
- 🖼️ Inference from single image files (.png, .jpg, etc.)
- 📊 Inference from flattened grayscale images in .csv format
- 🔧 Modular codebase: training, preprocessing, inference all separated
- 💾 GPU/CPU support and model checkpoint saving

## 📁 Project Structure

\`\`\`
├── LeNetTrain.py         # Train the LeNet model on FashionMNIST
├── LeNetTest.py          # Test the trained model on image or CSV input
├── ImagePrep.py          # Preprocessing for image and CSV input
├── lenet.pth             # Saved model weights (generated after training)
├── requirements.txt      # Project dependencies
└── README.md             # This file
\`\`\`

## 🚀 Getting Started

### 1. Clone this repository

\`\`\`bash
git clone https://github.com/ShiteHoly/LeNet-Based-FashionMNIST-Classifier-Custom-CSV-Inference.git
cd LeNet-Based-FashionMNIST-Classifier-Custom-CSV-Inference
\`\`\`

### 2. Install dependencies

Ensure you are using Python 3.10+, then run:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 🏋️‍♂️ Training the Model

To train the model from scratch on the FashionMNIST dataset:

\`\`\`bash
python LeNetTrain.py
\`\`\`

This will:
- Download the FashionMNIST dataset
- Train for 10 epochs using GPU if available
- Save the model weights as \`lenet.pth\`

## 🔍 Running Inference

Update the input path in \`LeNetTest.py\` to your test file:

\`\`\`python
input_path = r\"path/to/your/test.csv\"   # or image.png / image.jpg
\`\`\`

Then run:

\`\`\`bash
python LeNetTest.py
\`\`\`

The script will automatically:
- Detect file type (.csv or image)
- Preprocess the input using \`ImagePrep.py\`
- Output predicted labels

## 📊 CSV Format Example

Each row in your .csv should represent a single 28x28 grayscale image, flattened into 784 pixel values (0–255 range). Example:

\`\`\`
0, 0, 0, ..., 255
12, 45, 34, ...,  78
\`\`\`

The script normalizes this data using FashionMNIST's mean (0.1307) and std (0.3081) before inference.

## 🧠 Model Architecture (LeNet-5)

- Conv2D(1, 6, kernel_size=5, padding=2) → Sigmoid
- AvgPool2D(kernel_size=2)
- Conv2D(6, 16, kernel_size=5) → Sigmoid
- AvgPool2D(kernel_size=2)
- Flatten
- Linear(400 → 120) → Sigmoid
- Linear(120 → 84) → Sigmoid
- Linear(84 → 10)

## 💾 Requirements

Listed in \`requirements.txt\`. Install via:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Includes:

- torch
- torchvision
- pandas
- numpy
- pillow

## 🧪 Example Output

\`\`\`
[CSV] Image 0: Predicted digit = 3
[CSV] Image 1: Predicted digit = 7
[Image] Predicted digit: 2
\`\`\`

## 🧼 Recommended .gitignore

\`\`\`
__pycache__/
*.pth
*.csv
*.png
*.jpg
*.jpeg
*.ipynb_checkpoints/
\`\`\`

## 📫 Author

Made by [@ShiteHoly](https://github.com/ShiteHoly)  

## 🔖 License
