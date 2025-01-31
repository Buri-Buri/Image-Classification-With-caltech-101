# Image Classification with Caltech-101

## Overview
This project focuses on building, training, and evaluating a Convolutional Neural Network (CNN) using the Caltech-101 dataset for image classification. Additionally, it explores Explainable AI (XAI) techniques by utilizing Grad-CAM to interpret model decisions.

## Prerequisites
- Basic proficiency in Python.
- Understanding of machine learning and deep learning concepts.
- Familiarity with Convolutional Neural Networks (CNNs).
- Basic knowledge of Explainable AI (XAI).

## Part 1: Understanding the Caltech-101 Dataset
### Dataset Summary
- Includes 101 object categories and one background category.
- Each category contains between 40 to 800 images, totaling approximately 9,146 images.
- Images vary in size but can be resized for consistency.

### Task
- Classify images into one of the 101 categories using a CNN.
- Utilize Grad-CAM to interpret model predictions.

## Part 2: Dataset Preparation
### Downloading the Dataset
Obtain the Caltech-101 dataset from official sources.

### Preprocessing and Loading (PyTorch Example)
```python
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='path_to_caltech101', transform=transform)
```

### Splitting the Dataset
```python
from torch.utils.data import random_split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
```

### Creating Data Loaders
```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
```

## Part 3: Leveraging Pre-trained Models
### Using VGG19
```python
from torchvision.models import vgg19
model = vgg19(pretrained=True)
model.classifier[6] = nn.Linear(4096, 101)  # Adjust final layer
```

### Using ResNet50
```python
from torchvision.models import resnet50
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, 101)
```

### Using EfficientNet-B0
```python
from torchvision.models import efficientnet_b0
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 101)
```

## Part 4: Training the Model
### Defining Loss and Optimizer
```python
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training Loop
```python
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Validation
```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import ParameterGrid
param_grid = {'lr': [0.1, 0.01, 0.001], 'batch_size': [16, 32, 64]}
best_params = None
best_accuracy = 0

for params in ParameterGrid(param_grid):
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    # Perform training and validation to determine best parameters
```

## Part 5: Model Evaluation
### Testing the Model
```python
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_pred, y_true = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.numpy())
        y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

### Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_names))
```

### Top-k Accuracy
```python
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k_preds = torch.topk(output, k, dim=1).indices
        correct = max_k_preds.eq(target.view(-1, 1).expand_as(max_k_preds))
        return correct.any(dim=1).float().mean().item()
```

### t-SNE Visualization
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

features, labels_list = [], []
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        features.append(output)
        labels_list.append(labels)

features = torch.cat(features).numpy()
labels_list = torch.cat(labels_list).numpy()
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(features)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels_list, cmap='tab10')
plt.colorbar()
plt.show()
```

## Part 6: Explainability with Grad-CAM
### Installing Grad-CAM
```bash
pip install grad-cam
```

### Applying Grad-CAM
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.layer4[-1]  # Adjust based on model
cam = GradCAM(model=model, target_layer=target_layer)

for images, labels in test_loader:
    grayscale_cam = cam(input_tensor=images, target_category=labels[0])
    cam_image = show_cam_on_image(images[0].permute(1, 2, 0).numpy(), grayscale_cam)
    plt.imshow(cam_image)
    plt.show()
```

### Note
Some code snippets may require minor adjustments to function correctly.

