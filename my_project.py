# This project will try to use detectnet to figure out if a given person is wearing all articles of clothing.
#!/usr/bin/python3
#DO NOT RUN THIS ON COLAB

import jetson.inference
import jetson.utils
import argparse
import sys

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
net = detectNet(
    model="ssd-mobilenet.onnx",
    labels="labels.txt",
    input_blob="input_0",
    output_cvg="scores",
    output_bbox="boxes",
    threshold=0.5
)
camera = videoSource("/dev/video0")
display = videoOutput("webrtc://@:8554/output")

while True:
    img = camera.Capture()

    if img is None: # capture timeout
        continue

    detections = net.Detect(img)
    #use detections for something
    display.Render(img)

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.onnx
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

batch_size = 8
num_epochs = 10
learning_rate = 0.001
model_path = 'fashion_mnist_model.pth'
onnx_model_path = 'fashion_mnist_model.onnx'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./fashion_mnist_data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
torch.save(model.state_dict(), model_path)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)  
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'])
net = detectNet(
    model=onnx_model_path,
    labels="labels.txt", 
    input_blob="input",
    output_cvg="output",
    threshold=0.5
)
camera = videoSource("/dev/video0")
display = videoOutput("webrtc://@:8554/output")
while True:
    img = camera.Capture()
    if img is None: 
        continue
    detections = net.Detect(img)
    display.Render(img)
'''