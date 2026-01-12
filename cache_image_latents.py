import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler

from src.utils import center_crop_arr, load_vae


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


def init_distributed_mode(device):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() and device.startswith("cuda") else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        return True, rank, world_size
    return False, 0, 1


def get_args_parser():
    parser = argparse.ArgumentParser("Cache image VAE latents", add_help=True)
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * # gpus)")
    parser.add_argument("--image_size", default=256, type=int,
                        help="Input image size")
    parser.add_argument("--vae", default=None, type=str,
                        help="Path to pre-trained VAE model")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="ImageFolder split directory to cache (e.g., .../train)")
    parser.add_argument("--cached_path", default=None, type=str,
                        help="Output path for cached latents (default: data_dir + '_cached')")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for caching")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    if args.vae is None:
        raise ValueError("--vae is required")

    device = args.device
    distributed, rank, world_size = init_distributed_mode(device)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cudnn.benchmark = True

    data_dir = os.path.normpath(args.data_dir)
    cached_path = os.path.normpath(args.cached_path or f"{data_dir}_cached")

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_train = ImageFolderWithFilename(data_dir, transform=transform_train)
    if rank == 0:
        print(dataset_train)
        print(f"Cache output: {cached_path}")

    sampler_train = None
    if distributed:
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=False,
        )

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    vae, _, _, _ = load_vae(args.vae, args.image_size)
    vae.to(device)
    vae.eval()

    if rank == 0:
        print("Start caching VAE latents")
    start_time = time.time()
    use_tqdm = rank == 0
    data_iter = data_loader_train
    if use_tqdm:
        data_iter = tqdm(data_loader_train, total=len(data_loader_train), desc="Caching", unit="batch")

    for data_iter_step, (samples, _, paths) in enumerate(data_iter):
        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters.detach()
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters.detach()

        for i, path in enumerate(paths):
            save_path = os.path.join(cached_path, path + ".npz")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                moments=moments[i].cpu().numpy(),
                moments_flip=moments_flip[i].cpu().numpy(),
            )

        if device.startswith("cuda"):
            torch.cuda.synchronize()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print("Caching time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
