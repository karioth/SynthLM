from PIL import Image
import numpy as np
import json
import torch
import os
import requests
from tqdm import tqdm
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn
from .tokenizer_models import AutoencoderKL, sigma_vae

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def download_pretrained_vae(overwrite=False):
    download_path = "/mnt/unilm/yutao/vae.ckpt"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get("https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)

def safe_blob_write(fn, text):
    try:
        if os.path.exists(fn):
            os.remove(fn)
        with open(fn, "w") as f:
            f.write(text)
    except:
        print('Failed to write blob:', fn, text)

def safe_blob_dump(fn, result):
    try:
        if os.path.exists(fn):
            os.remove(fn)
        with open(fn, "w") as f:
            json.dump(result, f)
    except:
        print('Failed to write blob:', fn, result)

def load_vae(vae_model_path, image_size):
    data = torch.load(vae_model_path, map_location="cpu")

    if "config" not in data:
        input_size = image_size // 16
        latent_size = 16
        flatten_input = False
        vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_model_path)
    else:
        model_config = data["config"]
        input_size = image_size // model_config["patch_size"]
        latent_size = model_config["latent_size"]
        flatten_input = False
        vae = sigma_vae(**model_config)
        vae.load_state_dict(data["model"])
    
    return vae, input_size, latent_size, flatten_input


class EMAWeightAveraging(WeightAveraging):
    """
    Matches Lightning's stable-doc example (starts after 100 optimizer steps).
    """
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn(decay=0.9999))

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)
