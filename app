import fpdf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define a deeper DnCNN model for potentially better denoising
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=25):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

# Adjust transform for better training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),  # Example of data augmentation
    transforms.ToTensor(),
])

# Load dataset and dataloader
image_paths = [os.path.join('/content/drive/MyDrive/project/', img) for img in os.listdir('/content/drive/MyDrive/project/') if img.endswith(('.jpg', '.jpeg', '.png'))]
dataset = ImageDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, criterion, and optimizer
model = DnCNN(channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
noise_factor = 25
for epoch in range(num_epochs):
    for data in dataloader:
        model.train()
        optimizer.zero_grad()
        data = data.to(device)
        noisy_data = add_noise(data, noise_factor).to(device)
        output = model(noisy_data)
        loss = criterion(output, data - noisy_data)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save trained model
torch.save(model.state_dict(), 'dncnn.pth')

# Function to denoise image using the trained model
def denoise_image(model, noisy_image):
    model.eval()
    with torch.no_grad():
        noisy_image = noisy_image.to(device)
        output = model(noisy_image)
        denoised_image = noisy_image - output
    return denoised_image.cpu()

# Function to calculate PSNR
def calculate_psnr(original, denoised):
    mse = torch.mean((original - denoised.to(original.device)) ** 2)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# Load the trained model
model.load_state_dict(torch.load('dncnn.pth'))

# Test the model on a noisy image
test_image_path = '/content/drive/MyDrive/project/Figure-21-original-image-and-noisy-images-a-Original-image-without-noise-b-Image.jpg'
test_image = Image.open(test_image_path).convert('RGB')
test_image = transform(test_image).unsqueeze(0).to(device)  # Add batch dimension
noisy_test_image = add_noise(test_image, noise_factor)

# Denoise the image
denoised_image = denoise_image(model, noisy_test_image)

# Calculate PSNR
psnr = calculate_psnr(test_image, denoised_image)
print(f'PSNR: {psnr:.2f} dB')

# Display the images
def show_images(noisy, denoised):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(noisy.squeeze().cpu().permute(1, 2, 0))
    ax[0].set_title('Noisy Image')
    ax[0].axis('off')
    ax[1].imshow(denoised.squeeze().permute(1, 2, 0))
    ax[1].set_title('Denoised Image')
    ax[1].axis('off')
    plt.show()

show_images(noisy_test_image, denoised_image)
