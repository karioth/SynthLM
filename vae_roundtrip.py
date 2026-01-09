import argparse
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Ensure local imports work when running from other directories.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import center_crop_arr, load_vae  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_random_image(root: Path) -> Path:
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found under: {root}")
    random.shuffle(class_dirs)
    for cls_dir in class_dirs:
        images = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if images:
            return random.choice(images)
    # Fallback to recursive search if some class dirs are empty.
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return path
    raise FileNotFoundError(f"No images found under: {root}")


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    save_image(x.detach().cpu(), str(path), normalize=True, value_range=(-1, 1))


def parse_args() -> argparse.Namespace:
    repo_root = SCRIPT_DIR.parent
    default_data = repo_root / "imagenet_imagefolder" / "train"
    default_out = SCRIPT_DIR / "vae_sanity"
    parser = argparse.ArgumentParser(description="VAE roundtrip sanity check.")
    parser.add_argument("--data_root", type=str, default=str(default_data))
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_mode", action="store_true", help="Use posterior mode instead of sampling.")
    parser.add_argument("--out_dir", type=str, default=str(default_out))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    vae, _, _, _ = load_vae(args.vae, args.image_size)
    vae.to(device)
    vae.eval()

    image_path = pick_random_image(data_root)
    pil = Image.open(image_path).convert("RGB")
    pil_cropped = center_crop_arr(pil, args.image_size)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    x = to_tensor(pil_cropped).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.mode() if args.use_mode else posterior.sample()
        recon = vae.decode(z)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pil_cropped.save(out_dir / "input_cropped.png")
    save_tensor_image(recon, out_dir / "recon.png")

    print(f"Selected image: {image_path}")
    print(f"Saved input crop: {out_dir / 'input_cropped.png'}")
    print(f"Saved reconstruction: {out_dir / 'recon.png'}")


if __name__ == "__main__":
    main()
