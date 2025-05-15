import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, ShiftScaleRotate
from PIL import Image
import os
import numpy as np

# Function to resize predictions to original size
def resize_to_original(pred, original_size):
    return nn.functional.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)

# Define augmentations using Albumentations
augmentations = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
], additional_targets={'mask': 'mask'}, p=1.0)

class CustomDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None, augmentations=None, num_transformations=50):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augmentations = augmentations
        self.num_transformations = num_transformations
        self.image_filenames = sorted(os.listdir(input_dir))
        self.label_filenames = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_filenames) * self.num_transformations

    def __getitem__(self, idx):
        image_idx = idx // self.num_transformations
        input_path = os.path.join(self.input_dir, self.image_filenames[image_idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[image_idx])

        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("L")
        input_image_np, label_image_np = np.array(input_image), np.array(label_image)
        original_size = input_image.size[::-1]  # Reverse to (H, W) format

        if self.augmentations:
            augmented = self.augmentations(image=input_image_np, mask=label_image_np)
            input_image = Image.fromarray(augmented['image'])
            label_image = Image.fromarray(augmented['mask'])

        if self.transform:
            input_image = self.transform(input_image)
            label_image = torch.from_numpy(np.array(label_image, dtype=np.float32)).unsqueeze(0)

        return input_image, label_image, original_size

# Define hyperparameters
num_epochs = 50
learning_rate = 1e-4
batch_size = 8
num_transformations = 10

# Define data transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Dataset paths.  Adjust these to the correct paths on your system.
input_dir_train = "train/images"  # Replace with your actual training image directory
label_dir_train = "train/masks"  # Replace with your actual training mask directory
input_dir_val = "val/images"    # Replace with your actual validation image directory
label_dir_val = "val/masks"      # Replace with your actual validation mask directory


# Load datasets
train_dataset = CustomDataset(
    input_dir=input_dir_train,
    label_dir=label_dir_train,
    transform=data_transforms["train"],
    augmentations=augmentations,
    num_transformations=num_transformations
)

val_dataset = CustomDataset(
    input_dir=input_dir_val,
    label_dir=label_dir_val,
    transform=data_transforms["val"],
    augmentations=None  # Disable augmentations for validation
)

# Padding function
from torch.nn.functional import pad

def pad_image(img, target_size):
    _, h, w = img.shape
    pad_h = target_size[0] - h
    pad_w = target_size[1] - w
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    return pad(img, padding)

# Collate function
def collate_fn(batch):
    images, masks, original_sizes = zip(*batch)
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    padded_images = [pad_image(img, (max_height, max_width)) for img in images]
    padded_masks = [pad_image(mask, (max_height, max_width)) for mask in masks]

    return torch.stack(padded_images), torch.stack(padded_masks), original_sizes

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Load and modify the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))  # For binary segmentation
model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    train_loss = 0.0

    for images, masks, original_sizes in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]

        # Initialize batch loss for this iteration
        batch_loss = 0.0

        for output, mask, original_size in zip(outputs, masks, original_sizes):
            # Resize the output to the original size of the mask
            resized_output = resize_to_original(output.unsqueeze(0), original_size).squeeze(0)
            resized_mask = resize_to_original(mask.unsqueeze(0), original_size).squeeze(0)

            # Compute loss for each pair
            loss = criterion(resized_output, resized_mask)  # Use BCEWithLogitsLoss
            batch_loss += loss  # Accumulate as tensor

        # Average the loss over the batch (keep it as tensor)
        train_loss += batch_loss.item() / len(images)

        # Backpropagation
        batch_loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks, original_sizes in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]

            # Initialize batch loss for this iteration
            batch_loss = 0.0

            for output, mask, original_size in zip(outputs, masks, original_sizes):
                # Resize the output to the original size of the mask
                resized_output = resize_to_original(output.unsqueeze(0), original_size).squeeze(0)
                resized_mask = resize_to_original(mask.unsqueeze(0), original_size).squeeze(0)

                # Compute loss for each pair
                loss = criterion(resized_output, resized_mask) # Use BCEWithLogitsLoss
                batch_loss += loss  # Accumulate as tensor

            # Average the loss over the batch (keep it as tensor)
            val_loss += batch_loss.item() / len(images)

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    scheduler.step(val_loss)


# Save the model
save_path = "component_wetness_segmentation.pth"  # Changed save path
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}!")
