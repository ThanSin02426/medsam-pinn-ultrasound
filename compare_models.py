import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

class AcousticAttenuationLayer(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.absorption_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, feature_map):
        absorption = torch.sigmoid(self.absorption_conv(feature_map)) * 0.1
        transmission_map = torch.cumprod(1.0 - absorption + 1e-6, dim=2)
        return feature_map * transmission_map

def load_base_medsam(checkpoint_path="medsam_vit_b.pth", device="cuda"):
    print(f"Loading Baseline MedSAM from {checkpoint_path}...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)

def load_pinn_medsam(checkpoint_path="medsam_pinn_attenuation.pth", device="cuda"):
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
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask.shape
        x_min, x_max = max(0, x_min - 10), min(W, x_max + 10)
        y_min, y_max = max(0, y_min - 10), min(H, y_max + 10)
        return np.array([x_min, y_min, x_max, y_max])
    else:
        return np.array([0, 0, mask.shape[1], mask.shape[0]])

def calc_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if pred.sum() + gt.sum() == 0:
        return 1.0, 1.0 
        
    dice = 2.0 * intersection / (pred.sum() + gt.sum())
    iou = intersection / union if union > 0 else 0.0
    return dice, iou

def main():
    DATA_ROOT = "medsam1/brachial_real_world/data"
    BASE_WEIGHTS = "medsam_vit_b.pth"
    PINN_WEIGHTS = "medsam_pinn_attenuation.pth"
    SAVE_DIR = "comparison_results"
    MAX_SAVES_PER_VIDEO = 2 
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_predictor = load_base_medsam(BASE_WEIGHTS, device)
    pinn_predictor = load_pinn_medsam(PINN_WEIGHTS, device)
    
    base_total_dice, base_total_iou = 0.0, 0.0
    pinn_total_dice, pinn_total_iou = 0.0, 0.0
    valid_frames = 0
    
    machine_folders = [f.path for f in os.scandir(DATA_ROOT) if f.is_dir()]
    
    for machine_dir in machine_folders:
        machine_name = os.path.basename(machine_dir)
        videos_dir = os.path.join(machine_dir, "videos")
        masks_dir = os.path.join(machine_dir, "ac_masks")
        
        if not os.path.exists(videos_dir) or not os.path.exists(masks_dir):
            continue
            
        print(f"\n--- Processing Machine: {machine_name} ---")
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
        
        for vid_file in tqdm(video_files, desc="Videos"):
            vid_name = vid_file.replace(".mp4", "")
            vid_path = os.path.join(videos_dir, vid_file)
            mask_folder = os.path.join(masks_dir, vid_name)
            
            if not os.path.exists(mask_folder):
                continue
                
            cap = cv2.VideoCapture(vid_path)
            frame_idx = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                mask_name = f"{vid_name}_{frame_idx:03d}.jpg"
                mask_path = os.path.join(mask_folder, mask_name)
                
                if os.path.exists(mask_path):
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = (gt_mask > 128).astype(np.uint8)
                    bbox = get_bounding_box(gt_mask)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    valid_frames += 1
                    
                    base_predictor.set_image(frame_rgb)
                    base_masks, _, _ = base_predictor.predict(box=bbox, multimask_output=False)
                    base_pred = base_masks[0].astype(np.uint8)
                    b_dice, b_iou = calc_metrics(base_pred, gt_mask)
                    base_total_dice += b_dice
                    base_total_iou += b_iou
                    
                    pinn_predictor.set_image(frame_rgb)
                    pinn_masks, _, _ = pinn_predictor.predict(box=bbox, multimask_output=False)
                    pinn_pred = pinn_masks[0].astype(np.uint8)
                    p_dice, p_iou = calc_metrics(pinn_pred, gt_mask)
                    pinn_total_dice += p_dice
                    pinn_total_iou += p_iou
                    
                    if saved_count < MAX_SAVES_PER_VIDEO:
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(frame_rgb)
                        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2)
                        axes[0].add_patch(rect)
                        axes[0].set_title(f"Video: {vid_name} | Frame: {frame_idx}")
                        axes[0].axis('off')
                        
                        axes[1].imshow(gt_mask, cmap='gray')
                        axes[1].set_title("Ground Truth Mask")
                        axes[1].axis('off')
                        
                        axes[2].imshow(base_pred, cmap='gray')
                        axes[2].set_title(f"Base MedSAM\nDice: {b_dice:.3f}")
                        axes[2].axis('off')
                        
                        axes[3].imshow(pinn_pred, cmap='gray')
                        axes[3].set_title(f"PINN MedSAM\nDice: {p_dice:.3f}")
                        axes[3].axis('off')
                        
                        save_path = os.path.join(SAVE_DIR, f"{machine_name}_{vid_name}_{frame_idx:03d}.png")
                        plt.tight_layout()
                        plt.savefig(save_path)
                        plt.close()
                        saved_count += 1
                        
                frame_idx += 1
            cap.release()

    if valid_frames > 0:
        print("\n" + "="*40)
        print("🏆 REAL-WORLD VIDEO RESULTS 🏆")
        print("="*40)
        print(f"Total Video Frames Evaluated: {valid_frames}")
        print(f"Baseline MedSAM -> Average Dice: {base_total_dice/valid_frames:.4f} | Average IoU: {base_total_iou/valid_frames:.4f}")
        print(f"PINN MedSAM     -> Average Dice: {pinn_total_dice/valid_frames:.4f} | Average IoU: {pinn_total_iou/valid_frames:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()