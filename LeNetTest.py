import torch
from torch import nn
from torchvision import datasets, transforms
import ImagePrep as prep
import os

class Reshape(torch.nn.Module):
    def forward (self, x):
        return x.view(-1,1,28,28)#bactch size unknown here, using -1

model = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*5*5, 120),nn.Sigmoid(),
    nn.Linear(120,84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
# Same net in LeNetTrain.py, gonna use it to test the model withe saved weights

model.load_state_dict(torch.load('lenet.pth'))
model.eval()

# 🧩 INPUT: path to either image or CSV
input_path = r'C:\Users\17317\PycharmProjects\Classic LeNetCNN\data\test.csv' # Change this to your input file path

# 🧠 Decide based on file extension
ext = os.path.splitext(input_path)[-1].lower()

with torch.no_grad():
    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        input_tensor = prep.preprocess_image(input_path)
        output = model(input_tensor)
        predicted = output.argmax(1)
        print(f"[Image] Predicted digit: {predicted.item()}")

    elif ext == '.csv':
        input_tensor = prep.preprocess_csv(input_path)
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=1)
        for i, p in enumerate(predictions):
            print(f"[CSV] Image {i}: Predicted digit = {p.item()}")
    else:
        print(f"❌ Unsupported file type: {ext}")