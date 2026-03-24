import torch
checkpoint = torch.load('/home/saintsolo-deepl/ADeeplearning/lqh2025/CLAMP/work_dirs/CLAMP_ViTB_ap10k_256x256_visionmamba3/best_AP_epoch_460.pth')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total params: {total_params/1e6:.2f} M")