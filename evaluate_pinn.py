import os
import glob
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class AcousticAttenuationLayer(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.absorption_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, feature_map):
        absorption = torch.sigmoid(self.absorption_conv(feature_map)) * 0.1
        transmission_map = torch.cumprod(1.0 - absorption + 1e-6, dim=2)
        return feature_map * transmission_map

def load_pinn_model(checkpoint_path, device="cuda"):
    print(f"Rebuilding PINN Architecture and loading {checkpoint_path}...")
    
    sam = sam_model_registry["vit_b"](checkpoint=None)
    
    physics_block = AcousticAttenuationLayer(in_channels=32)
    original_upscaling = sam.mask_decoder.output_upscaling
    sam.mask_decoder.output_upscaling = nn.Sequential(
        *list(original_upscaling.children()), 
        physics_block
    )
    
    sam.to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    sam.load_state_dict(clean_state_dict, strict=True)
    sam.eval()
    
    return SamPredictor(sam)

def get_bounding_box(mask):
    """Generates a bounding box from the ground truth mask (SAM requires a prompt)"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        x_min = max(0, x_min - 10)
        x_max = min(W, x_max + 10)
        y_min = max(0, y_min - 10)
        y_max = min(H, y_max + 10)
        return np.array([x_min, y_min, x_max, y_max])
    else:
        return np.array([0, 0, mask.shape[1], mask.shape[0]])

def calculate_dice(pred_mask, gt_mask):
    """Calculates the Dice Similarity Coefficient"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return 2.0 * intersection / (pred.sum() + gt.sum())

def main():
    TEST_DIR = "medsam1/train" 
    WEIGHTS_PATH = "medsam_pinn_attenuation.pth"
    SAVE_DIR = "pinn_results"
    NUM_IMAGES_TO_TEST = 10
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = load_pinn_model(WEIGHTS_PATH, device)
    all_tifs = glob.glob(os.path.join(TEST_DIR, "*.tif"))
    image_paths = sorted([f for f in all_tifs if "_mask" not in f])[:NUM_IMAGES_TO_TEST]
    
    total_dice = 0.0
    
    print(f"Starting inference on {len(image_paths)} images...")
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = img_path.replace(".tif", "_mask.tif")
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 128).astype(np.uint8)
        else:
            gt_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
            
        bbox = get_bounding_box(gt_mask)
        
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            box=bbox,
            multimask_output=False
        )
        pred_mask = masks[0].astype(np.uint8)
        
        dice = calculate_dice(pred_mask, gt_mask)
        total_dice += dice
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_rgb)
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title(f"Original + BBox Prompt")
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title(f"PINN Prediction (Dice: {dice:.3f})")
        axes[2].axis('off')
        
        save_path = os.path.join(SAVE_DIR, f"result_{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Processed image {i+1}/{len(image_paths)} - Dice: {dice:.4f}")

    print("-" * 30)
    print(f"Average Dice Score: {total_dice / len(image_paths):.4f}")
    print(f"Visualizations saved to ./{SAVE_DIR}/")

if __name__ == "__main__":
    main()