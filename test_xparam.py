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

# ptpath = "/home/jianglongyu/Documents/models/params_cdc_ldm/image-l2-no_weight-vimeo-d64-t200-b1e-05-noise-cosine-01-float32-aux0/image-l2-no_weight-vimeo-d64-t200-b1e-05-noise-cosine-01-float32-aux0_0.pt"
# ptpath = "pre-trained/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt"
ptpath = "image-l2-use_weight5-vimeo-d64-t8193-b1e-05-x-cosine-01-float32-aux0_2.pt"
parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, default=ptpath) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=65) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--img_dir", type=str, default='./imgs')
parser.add_argument("--out_dir", type=str, default='./compressed_imgs')
parser.add_argument("--lpips_weight", type=float, default=0.0) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.

config = parser.parse_args()


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


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
        lagrangian=0.0032,
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
            ori_size, ori_avg_pix = print_file_size(os.path.join(config.img_dir, img))
            img_info["ori_size"] = ori_size
            img_info["ori_avg_pix"] = ori_avg_pix
            ori_bpp = calculate_bpp(os.path.join(config.img_dir, img))
            print("c_o_bpp:", ori_bpp)
            img_info["ori_bpp"] = ori_bpp

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
            compressed_size, compressed_avg_pix = print_file_size(os.path.join("./compressed_imgs", img))
            img_info["compressed_size"] = compressed_size
            img_info["compressed_avg_pix"] = compressed_avg_pix
            print("bpp:", bpp)
            img_info["bpp"] = bpp
            cal_bpp = calculate_bpp(os.path.join(config.out_dir, img))
            print("c_bpp:", cal_bpp)
            img_info["c_bpp"] = cal_bpp
            psnr = calculate_psnr(to_be_compressed.cpu().numpy(), compressed.cpu().numpy())
            print("PSNR:", psnr)
            img_info["PSNR"] = psnr
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
        for item in result:
            f.write("%s\n" % item)

if __name__ == "__main__":
    main(config.device)
