import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

# -------------------
# CONFIGURATION
# -------------------
# Set the input and output directories as requested
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
MODEL_PATH = "colorization_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TEST_IMAGES = 5

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# ----------------------------------------------------------------------
# U-NET MODEL ARCHITECTURE (Must be identical to the training model)
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

        # Final output layer with sigmoid activation
        return torch.sigmoid(self.final_conv(d1))

# ----------------------------------------------------------------------
# DATA PREPARATION (for testing)
# ----------------------------------------------------------------------
# Transformations for test images (no random transforms)
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), # Scales to [0, 1]
])

def load_image(image_path):
    """Loads and preprocesses a single image for the model."""
    img = Image.open(image_path).convert("RGB")
    color_img_tensor = test_transforms(img).unsqueeze(0).to(DEVICE)
    gray_img_tensor = transforms.functional.rgb_to_grayscale(color_img_tensor).to(DEVICE)
    return gray_img_tensor, color_img_tensor

def save_comparison_image(gray_input, colored_output, original_color, output_path, title="Colorization Results"):
    """
    Saves a single image with the input, output, and original images for comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Convert tensors back to PIL Images for display
    gray_pil = transforms.ToPILImage()(gray_input.squeeze().cpu())
    output_pil = transforms.ToPILImage()(colored_output.squeeze().cpu())
    original_pil = transforms.ToPILImage()(original_color.squeeze().cpu())

    axes[0].imshow(gray_pil, cmap='gray')
    axes[0].set_title("Input (Grayscale)")
    axes[0].axis('off')

    axes[1].imshow(output_pil)
    axes[1].set_title("Model Output (Colorized)")
    axes[1].axis('off')

    axes[2].imshow(original_pil)
    axes[2].set_title("Original (Color)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load the trained model
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model weights loaded from: {MODEL_PATH}")

    # Get a list of all image files in the input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: The input directory '{INPUT_DIR}' does not exist.")
    else:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
        if len(image_files) < NUM_TEST_IMAGES:
            print(f"Warning: Not enough images in '{INPUT_DIR}'. Found {len(image_files)}, will process all available images.")
            NUM_TEST_IMAGES = len(image_files)
        
        if NUM_TEST_IMAGES > 0:
            # Randomly select a few images to test
            test_images = random.sample(image_files, NUM_TEST_IMAGES)
            
            print(f"Testing on {NUM_TEST_IMAGES} random images...")
            with torch.no_grad():
                for img_name in test_images:
                    image_path = os.path.join(INPUT_DIR, img_name)
                    gray_input, original_color = load_image(image_path)
                    
                    # Predict the colorized image
                    colored_output = model(gray_input)
                    
                    # Save the comparison image
                    output_path = os.path.join(OUTPUT_DIR, f"colorized_{img_name}")
                    save_comparison_image(gray_input, colored_output, original_color, output_path, title=f"Colorization Result for '{img_name}'")
                    print(f"🖼️ Saved result to: {output_path}")
        else:
            print(f"No images found in '{INPUT_DIR}' to process. Please add images and try again.")
