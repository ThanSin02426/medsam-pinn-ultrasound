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
    DATA_ROOT = "medsam1/brachial_real_world/data/Butterfly"
    BASE_WEIGHTS = "medsam_vit_b.pth"
    PINN_WEIGHTS = "medsam_pinn_attenuation.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_predictor = load_base_medsam(BASE_WEIGHTS, device)
    pinn_predictor = load_pinn_medsam(PINN_WEIGHTS, device)
    
    pos_frames, neg_frames = 0, 0
    
    base_pos_dice, base_pos_iou = 0.0, 0.0
    base_neg_dice, base_neg_iou = 0.0, 0.0
    
    pinn_pos_dice, pinn_pos_iou = 0.0, 0.0
    pinn_neg_dice, pinn_neg_iou = 0.0, 0.0
    
    videos_dir = os.path.join(DATA_ROOT, "videos")
    masks_dir = os.path.join(DATA_ROOT, "ac_masks")
    
    if not os.path.exists(videos_dir) or not os.path.exists(masks_dir):
        print(f" Could not find {videos_dir} or {masks_dir}")
        return
        
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
    
    for vid_file in tqdm(video_files, desc="Processing Butterfly Videos"):
        vid_name = vid_file.replace(".mp4", "")
        vid_path = os.path.join(videos_dir, vid_file)
        mask_folder = os.path.join(masks_dir, vid_name)
        
        if not os.path.exists(mask_folder):
            continue
            
        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        
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
                
                base_predictor.set_image(frame_rgb)
                base_masks, _, _ = base_predictor.predict(box=bbox, multimask_output=False)
                base_pred = base_masks[0].astype(np.uint8)
                b_dice, b_iou = calc_metrics(base_pred, gt_mask)
                
                pinn_predictor.set_image(frame_rgb)
                pinn_masks, _, _ = pinn_predictor.predict(box=bbox, multimask_output=False)
                pinn_pred = pinn_masks[0].astype(np.uint8)
                p_dice, p_iou = calc_metrics(pinn_pred, gt_mask)
                
                if gt_mask.sum() > 0:
                    pos_frames += 1
                    base_pos_dice += b_dice
                    base_pos_iou += b_iou
                    pinn_pos_dice += p_dice
                    pinn_pos_iou += p_iou
                else:
                    neg_frames += 1
                    base_neg_dice += b_dice
                    base_neg_iou += b_iou
                    pinn_neg_dice += p_dice
                    pinn_neg_iou += p_iou
                    
            frame_idx += 1
        cap.release()

    print("\n" + "="*50)
    print("🏆 ISOLATED PERFORMANCE DIAGNOSTIC (Butterfly) 🏆")
    print("="*50)
    print(f"Total Visible Anatomy Frames : {pos_frames}")
    print(f"Total Empty (Blank) Frames   : {neg_frames}")
    print("-" * 50)
    
    if pos_frames > 0:
        print(" VISIBLE ANATOMY PERFORMANCE (True Positives)")
        print(f"   Baseline MedSAM -> Dice: {base_pos_dice/pos_frames:.4f}")
        print(f"   PINN MedSAM     -> Dice: {pinn_pos_dice/pos_frames:.4f}")
    
    if neg_frames > 0:
        print("\n HALLUCINATION RESISTANCE (True Negatives)")
        print("   (Score of 1.0 means it correctly output a blank mask)")
        print(f"   Baseline MedSAM -> Dice: {base_neg_dice/neg_frames:.4f}")
        print(f"   PINN MedSAM     -> Dice: {pinn_neg_dice/neg_frames:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()