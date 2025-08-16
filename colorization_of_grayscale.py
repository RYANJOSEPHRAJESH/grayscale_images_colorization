import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

# -------------------
# CONFIGURATION
# -------------------
# Check for CUDA availability
print("Cuda available:", torch.cuda.is_available())
# Path to your dataset directory
DATASET_DIR = "MyDataset"
# Hyperparameters
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 200
# Device to use for training (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# DATASET & AUGMENTATION (CORRECTED)
# ----------------------------------------------------------------------
class GrayscaleColorizationDataset(Dataset):
    """
    A custom dataset to load images for a colorization task.
    It applies transforms to the color image first and then
    derives the grayscale input from the transformed image
    to ensure perfect alignment between input and target.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Open the original color image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        color_img = Image.open(img_path).convert("RGB")

        # 2. Apply all geometric and color transforms to the color image
        # This is the crucial step to ensure alignment.
        if self.transform:
            color_img = self.transform(color_img)
        
        # 3. Create the grayscale version from the TRANSFORMED color image
        gray_img = transforms.functional.rgb_to_grayscale(color_img, num_output_channels=1)

        # 4. Convert both images to tensors
        color_tensor = transforms.functional.to_tensor(color_img)
        gray_tensor = transforms.functional.to_tensor(gray_img)

        # Return the grayscale input and the color target
        return gray_tensor, color_tensor

# Define transforms to be applied to the color image BEFORE creating the grayscale version.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Create the dataset and DataLoader
full_dataset = GrayscaleColorizationDataset(
    DATASET_DIR, 
    transform=transform
)
# Use a subset of the data for faster training, if needed
dataset = Subset(full_dataset, range(min(1900, len(full_dataset))))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------------------------------------
# U-NET MODEL ARCHITECTURE
# ----------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder path (Downsampling)
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder path (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1) # Skip connection
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1) # Skip connection
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1) # Skip connection
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1) # Skip connection
        d1 = self.dec1(d1)

        # Final output layer with sigmoid activation to scale to [0, 1]
        return torch.sigmoid(self.final_conv(d1))

# ----------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------
model = UNet().to(DEVICE)
criterion = nn.L1Loss() # Using L1 loss is a good choice for this task
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for gray_batch, color_batch in train_loader:
        gray_batch, color_batch = gray_batch.to(DEVICE), color_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(gray_batch)
        loss = criterion(outputs, color_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

# ----------------------------------------------------------------------
# SAVE THE TRAINED MODEL
# ----------------------------------------------------------------------
torch.save(model.state_dict(), "colorization_unet.pth")
print("\n✅ Training complete!")
print("Model saved as colorization_unet.pth")
