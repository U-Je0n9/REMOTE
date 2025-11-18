import sys, os
sys.path.append('REMOTE/Depth-Anything-V2')

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm  

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# init
encoder = 'vitl'  
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'REMOTE/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

input_dir = "REMOTE/datasets/UMKE_IMG"
output_dir = "REMOTE/datasets/depth_data_umke"
os.makedirs(output_dir, exist_ok=True)


# 처리 루프
for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    base_name = os.path.splitext(filename)[0]
    npz_path = os.path.join(output_dir, f"{base_name}.npz")

    # if os.path.exists(npz_path):
    #     continue

    img_path = os.path.join(input_dir, filename)
    raw_img = cv2.imread(img_path)

    if raw_img is None:
        tqdm.write(f"Failed to read {img_path}")
        continue

    # 추론 수행
    depth_map = model.infer_image(raw_img)

    # 저장
    np.savez(npz_path, depth_map=depth_map)
    tqdm.write(f"Saved depth map: {npz_path}")