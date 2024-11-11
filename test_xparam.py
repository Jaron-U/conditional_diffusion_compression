import argparse
import os
import torch
import torchvision
import numpy as np
import pathlib
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA
import cv2
from metrics_cal import calculate_psnr, calculate_lpips

# ptpath = "fall_trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0_2.pt"
# ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt"
ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0064-x-cosine-01-float32-aux0.0_2.pt"
# ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0_2.pt"
# ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0512-x-cosine-01-float32-aux0.9lpips_2.pt"
# ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, default=ptpath) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=65) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--img_dir", type=str, default='./imgs')
parser.add_argument("--out_dir", type=str, default='./compressed_imgs')
parser.add_argument("--lpips_weight", type=float, default=0.0) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.

config = parser.parse_args()

def calculate_bpp(image_path):
    image_size_bytes = os.path.getsize(image_path) 
    
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    total_pixels = width * height
    total_bits = image_size_bytes * 8 
    
    bpp = total_bits / total_pixels
    
    return bpp


def main(rank):
    
    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1,2,3,4,5,6],
        context_dim_mults=[1,2,3,4],
        embd_type="01",
    )

    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1,2,3,4],
        reverse_dim_mults=[4,3,2,1],
        hyper_dims_mults=[4,4,4],
        channels=3,
        out_channels=64,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193, # 8193
        loss_type="l2",
        lagrangian=0.0064,
        pred_mode="x",
        aux_loss_weight=config.lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(
        config.ckpt,
        map_location=lambda storage, loc: storage,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"], strict=False)
    
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()
    
    result = []
    for img in os.listdir(config.img_dir):
        img_info = {}
        if img.endswith(".png") or img.endswith(".jpg"):
            to_be_compressed = torchvision.io.read_image(os.path.join(config.img_dir, img)).unsqueeze(0).float().to(rank) / 255.0
            compressed, bpp = diffusion.compress(
                to_be_compressed * 2.0 - 1.0,
                sample_steps=config.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(to_be_compressed) * config.gamma
            )
            compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
            pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(compressed.cpu(), os.path.join(config.out_dir, img))
            # compressed_size, compressed_avg_pix = print_file_size(os.path.join("./compressed_imgs", img))
            psnr = calculate_psnr(to_be_compressed.cpu().numpy(), compressed.cpu().numpy())
            lpips = calculate_lpips(to_be_compressed.cpu(), compressed.cpu())
            img_info["bpp"] = round(bpp.item(), 4)
            img_info["psnr"] = round(psnr, 4)
            img_info["lpips"] = round(lpips, 4)
        result.append(img_info)
    save_result(ptpath, result)

def print_file_size(file_path):
    size = os.path.getsize(file_path)
    print(f"File Size of {file_path}: {size} bytes")
    print("average pixel:", np.average(torchvision.io.read_image(file_path).numpy()))
    return size, np.average(torchvision.io.read_image(file_path).numpy())

def save_result(weight_file_name, result):
    with open("result.txt", "a") as f:
        f.write("\n" + weight_file_name + "\n")
        f.write("gamma: " + str(config.gamma) + "\n")
        f.write("lpips_weight (rho): " + str(config.lpips_weight) + "\n")
        for item in result:
            f.write("%s\n" % item)

if __name__ == "__main__":
    main(config.device)
