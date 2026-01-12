import argparse
import os

import torch
from safetensors.torch import load_file

from src.lightning import LitModule


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a train_hf checkpoint into a Lightning checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to train_hf checkpoint directory (contains other_state.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .ckpt path (default: <checkpoint>/lightning.ckpt).",
    )
    parser.add_argument("--model", type=str, default="Transformer-L")
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prediction-type", type=str, default="flow")
    parser.add_argument("--ddpm-num-steps", type=int, default=1000)
    parser.add_argument("--ddpm-beta-schedule", type=str, default="cosine")
    parser.add_argument("--batch-mul", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA.")
    return parser.parse_args()


def load_raw_model_state(checkpoint_dir):
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        return load_file(safetensors_path)
    mp_rank_path = os.path.join(checkpoint_dir, "pytorch_model", "mp_rank_00_model_states.pt")
    if os.path.exists(mp_rank_path):
        return torch.load(mp_rank_path, map_location="cpu")["module"]
    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")


def main():
    args = parse_args()
    output_path = args.output or os.path.join(args.checkpoint, "lightning.ckpt")

    other_state_path = os.path.join(args.checkpoint, "other_state.pth")
    if not os.path.exists(other_state_path):
        raise FileNotFoundError(f"Missing other_state.pth in {args.checkpoint}")
    other_state = torch.load(other_state_path, map_location="cpu")

    lit = LitModule(
        model_name=args.model,
        input_size=args.input_size,
        latent_size=args.latent_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
        prediction_type=args.prediction_type,
        ddpm_num_steps=args.ddpm_num_steps,
        ddpm_beta_schedule=args.ddpm_beta_schedule,
        batch_mul=args.batch_mul,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    if args.no_ema:
        raw_state = load_raw_model_state(args.checkpoint)
        lit.model.load_state_dict(raw_state)
    else:
        ema_state = other_state.get("ema", None)
        if ema_state is None or "shadow_params" not in ema_state:
            raise ValueError("EMA state not found in other_state.pth. Use --no-ema to load raw weights.")
        for model_param, ema_param in zip(lit.model.parameters(), ema_state["shadow_params"]):
            model_param.data = ema_param.data.to(model_param)

    lit.scaling_factor.copy_(torch.tensor(other_state["scaling_factor"], dtype=torch.float32))
    lit.bias_factor.copy_(torch.tensor(other_state["bias_factor"], dtype=torch.float32))
    lit.has_scaling.fill_(True)

    ckpt = {
        "state_dict": lit.state_dict(),
        "hyper_parameters": dict(lit.hparams),
        "pytorch-lightning_version": "2.6.0",
    }
    if "steps" in other_state:
        ckpt["global_step"] = other_state["steps"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved Lightning checkpoint to {output_path}")


if __name__ == "__main__":
    main()
