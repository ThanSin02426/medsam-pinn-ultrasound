import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from segment_anything import sam_model_registry

class AcousticAttenuationLayer(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.absorption_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, feature_map):
        absorption = torch.sigmoid(self.absorption_conv(feature_map)) * 0.1
        transmission_map = torch.cumprod(1.0 - absorption + 1e-6, dim=2)
        physics_aware_features = feature_map * transmission_map
        return physics_aware_features

class ParallelMedSAM(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model

    def forward(self, imgs, bboxes):
        image_embeddings = self.sam.image_encoder(imgs)
        
        batch_size = imgs.shape[0]
        low_res_masks_list = []
        
        for i in range(batch_size):
            curr_embedding = image_embeddings[i].unsqueeze(0) # [1, 256, 64, 64]
            curr_bbox = bboxes[i].unsqueeze(0).unsqueeze(1)   # [1, 1, 4]
            
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, 
                boxes=curr_bbox, 
                masks=None,
            )
            
            low_res_mask, _ = self.sam.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            low_res_masks_list.append(low_res_mask)
            
        return torch.cat(low_res_masks_list, dim=0)

class UltrasoundDataset(Dataset):
    def __init__(self, data_dir):
        all_tifs = glob.glob(os.path.join(data_dir, "*.tif"))
        self.image_paths = sorted([f for f in all_tifs if "_mask" not in f])
        self.mask_paths = [f.replace(".tif", "_mask.tif") for f in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024))
        
        mask_path = self.mask_paths[idx]
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            
        mask_1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask_1024 = (mask_1024 > 128).astype(np.float32)

        y_indices, x_indices = np.where(mask_1024 > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = 1024, 1024
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            bbox = np.array([0, 0, 1024, 1024])

        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_256 = (mask_256 > 128).astype(np.float32)

        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.tensor(mask_256).unsqueeze(0).float()
        bbox_tensor = torch.tensor(bbox).float()

        return img_tensor, mask_tensor, bbox_tensor

def calc_dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def main():
    DATA_DIR = "medsam1/train" 
    CHECKPOINT_PATH = "medsam_vit_b.pth"
    SAVE_PATH = "medsam_pinn_attenuation.pth"
    EPOCHS = 20
    
    num_gpus = torch.cuda.device_count()
    BATCH_SIZE = 2 * max(1, num_gpus) 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing PINN Architecture...")
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    
    physics_block = AcousticAttenuationLayer(in_channels=32)
    original_upscaling = sam.mask_decoder.output_upscaling
    sam.mask_decoder.output_upscaling = nn.Sequential(
        *list(original_upscaling.children()), 
        physics_block
    )

    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    medsam_model = ParallelMedSAM(sam)
    if num_gpus > 1:
        medsam_model = nn.DataParallel(medsam_model)
    medsam_model.to(DEVICE)

    trainable_params = [p for p in medsam_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
    dataset = UltrasoundDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    print(f"Starting Training on {DEVICE} with batch size {BATCH_SIZE}...")
    
    medsam_model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for imgs, masks, bboxes in pbar:
                imgs, masks, bboxes = imgs.to(DEVICE), masks.to(DEVICE), bboxes.to(DEVICE)
                
                pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
                pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)
                imgs_norm = (imgs * 255.0 - pixel_mean) / pixel_std
                
                low_res_masks = medsam_model(imgs_norm, bboxes)
                pred_masks = F.interpolate(low_res_masks, size=(256, 256), mode="bilinear", align_corners=False)

                loss_dice = calc_dice_loss(pred_masks, masks)
                loss_bce = bce_loss_fn(pred_masks, masks)
                loss = loss_dice + loss_bce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    print(f"Saving PINN model to {SAVE_PATH}...")
    if isinstance(medsam_model, nn.DataParallel):
        torch.save(medsam_model.module.sam.state_dict(), SAVE_PATH)
    else:
        torch.save(medsam_model.sam.state_dict(), SAVE_PATH)
        
    print("Training Complete!")

if __name__ == "__main__":
    main()