import numpy as np
import lpips
import torch
from PIL import Image
from torchvision import transforms

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

lpips_loss = lpips.LPIPS(net='vgg')

preprocess = transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def calculate_lpips(original, compressed):
    # original = preprocess(original)
    # compressed = preprocess(compressed)
    with torch.no_grad():
        lpips_score = lpips_loss(original, compressed).item()
    return lpips_score

    