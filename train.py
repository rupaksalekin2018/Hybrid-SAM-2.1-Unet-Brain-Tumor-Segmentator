import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from ultralytics import SAM

# --- CONFIGURATION ---
DATASET_PATH = "/kaggle/input/brain-tumor-dataset-segmentation/DATASET/Segmentation"
SAM_CHECKPOINT = "sam2.1_t.pt" # Using Large model for best features
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- 1. FIXED DATASET (Crucial Fix Here) ---
class TumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Load Mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        
        # --- FIX: Do NOT divide by 255.0 here ---
        # The mask is already 0 or 255. We threshold it to be exactly 0.0 or 1.0
        mask = (mask > 127).astype(np.float32) 
        # Output is now strictly 0.0 or 1.0
        
        mask = np.expand_dims(mask, axis=0) 

        # Normalize Image (0-1) is fine
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = np.transpose(image_norm, (2, 0, 1)) 

        return torch.tensor(image_norm), torch.tensor(mask)

# Find Files
all_images = []
all_masks = []
for f in glob(os.path.join(DATASET_PATH, "**", "*.png"), recursive=True):
    if "_mask.png" not in f:
        mask_f = f.replace(".png", "_mask.png")
        if os.path.exists(mask_f):
            all_images.append(f)
            all_masks.append(mask_f)

print(f"Found {len(all_images)} pairs.")

# Split & Loaders
train_X, val_X, train_y, val_y = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)
train_loader = DataLoader(TumorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(TumorDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 2. FIXED FEATURE EXTRACTOR (Handles Lists/Dicts) ---
class SAMFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_name):
        super().__init__()
        print(f"Loading SAM Backbone: {checkpoint_name}...")
        sam_wrapper = SAM(checkpoint_name)
        
        if hasattr(sam_wrapper.model, 'image_encoder'):
            self.vision_encoder = sam_wrapper.model.image_encoder
        elif hasattr(sam_wrapper.model, 'model') and hasattr(sam_wrapper.model.model, 'image_encoder'):
             self.vision_encoder = sam_wrapper.model.model.image_encoder
        else:
             self.vision_encoder = sam_wrapper.model.model.image_encoder

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        del sam_wrapper

    def forward(self, x):
        with torch.no_grad():
            x_large = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
            features = self.vision_encoder(x_large)
            # Recursively unwrap the output to find the tensor
            features = self._unwrap(features)
        return features
    
    def _unwrap(self, x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, dict):
            return self._unwrap(list(x.values())[-1])
        elif isinstance(x, (list, tuple)):
            return self._unwrap(x[-1])
        return x

# --- 3. HYBRID MODEL ---
class HybridSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_features = SAMFeatureExtractor(SAM_CHECKPOINT)
        
        # Adapter: Auto-detect channels later if needed, but standard is 256
        self.adapter = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.ReLU()
        )
        
        self.unet = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        sam_feats = self.sam_features(x)
        
        # Safety check for channel mismatch (just in case SAM returns 1152 etc)
        if sam_feats.shape[1] != self.adapter[0].in_channels:
             # This is a one-time fix during first forward pass
             device = sam_feats.device
             self.adapter = nn.Sequential(
                nn.Conv2d(sam_feats.shape[1], 3, kernel_size=1),
                nn.ReLU()
             ).to(device)

        adapted = self.adapter(sam_feats)
        adapted = F.interpolate(adapted, size=x.shape[2:], mode='bilinear', align_corners=False)
        fused_input = x + adapted
        return self.unet(fused_input)

# --- 4. TRAINING LOOP ---
print("Building Hybrid Model...")
model = HybridSegmentationModel().to(DEVICE)

criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("Starting Training Loop...")
history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

for epoch in range(EPOCHS):
    model.train()
    model.sam_features.eval() 
    
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, masks in pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    # Validation
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Dice Calc
            preds = (outputs.sigmoid() > 0.5).float()
            intersection = (preds * masks).sum()
            dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-7)
            dice_scores.append(dice.item())
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_dice)
    
    print(f"Ep {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_dice:.4f}")
    scheduler.step()

# --- 5. REPORT ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.plot(history['train_loss'], label='Train'); plt.title("Loss"); plt.legend()
plt.subplot(1, 3, 2); plt.plot(history['val_dice'], color='green'); plt.title(f"Max Dice: {max(history['val_dice']):.4f}")

# Inference
model.eval()
images, masks = next(iter(val_loader))
images = images.to(DEVICE)
with torch.no_grad():
    preds = model(images).sigmoid() > 0.5

plt.subplot(1, 3, 3)
img_show = images[0].cpu().permute(1, 2, 0).numpy()
pred_show = preds[0].cpu().squeeze().numpy()
plt.imshow(img_show); plt.imshow(pred_show, cmap='jet', alpha=0.5)
plt.title("Prediction")
plt.axis('off')
plt.show()

torch.save(model.state_dict(), "final_model.pth")


def visualize_test_gallery(model, loader, num_samples=8, device="cuda"):
    """
    Visualizes multiple random samples from the validation set with a 4-column layout.
    Columns: Original Image | Ground Truth | Prediction | Overlay
    """
    model.eval()
    
    # 1. Collect Random Samples
    all_images = []
    all_masks = []
    all_preds = []
    all_scores = []
    
    # Iterator to fetch data
    data_iter = iter(loader)
    
    print(f"Collecting {num_samples} samples for visualization...")
    
    with torch.no_grad():
        while len(all_images) < num_samples:
            try:
                images, masks = next(data_iter)
            except StopIteration:
                break # Stop if we run out of data
                
            images = images.to(device)
            outputs = model(images)
            preds = (outputs.sigmoid() > 0.5).float()
            
            # Move to CPU for plotting
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            for i in range(len(images)):
                if len(all_images) >= num_samples:
                    break
                    
                # Calculate Dice for this specific image
                intersection = (preds_np[i] * masks_np[i]).sum()
                union = preds_np[i].sum() + masks_np[i].sum()
                dice = (2. * intersection) / (union + 1e-7)
                
                # Only add if it actually has a tumor (optional: remove this if you want to see negatives too)
                # if masks_np[i].sum() > 0: 
                all_images.append(images_np[i])
                all_masks.append(masks_np[i])
                all_preds.append(preds_np[i])
                all_scores.append(dice)

    # 2. Plotting
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    for i in range(num_samples):
        img = all_images[i].transpose(1, 2, 0) # CHW -> HWC
        gt = all_masks[i].squeeze()
        pred = all_preds[i].squeeze()
        score = all_scores[i]
        
        # Row Labels (only for the first row)
        if i == 0:
            axes[i, 0].set_title("Original MRI", fontsize=14, fontweight='bold')
            axes[i, 1].set_title("Ground Truth", fontsize=14, fontweight='bold')
            axes[i, 2].set_title("Model Prediction", fontsize=14, fontweight='bold')
            axes[i, 3].set_title("Overlay (Pred vs MRI)", fontsize=14, fontweight='bold')

        # 1. Original
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        
        # 2. Ground Truth
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')
        
        # 3. Prediction
        # Color code the title based on performance
        title_color = 'green' if score > 0.8 else ('orange' if score > 0.5 else 'red')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f"Dice: {score:.4f}", color=title_color, fontsize=12, y=-0.1)
        axes[i, 2].axis('off')
        
        # 4. Overlay
        # Create a semi-transparent colored mask (Red for prediction)
        overlay = np.zeros_like(img)
        overlay[:, :, 0] = pred # Red channel
        
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(overlay, alpha=0.4) # 40% transparency
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

# --- Run the Visualization ---
# You can change num_samples to see more or fewer images
visualize_test_gallery(model, val_loader, num_samples=8)
