# [Multimodal Latent Language Modeling with Next-Token Diffusion]

Official PyTorch implementation and pretrained models of LatentLM. 

---


<!-- ## Pretrained models -->

<!-- coming soon -->

## Setup & Usage

Coming soon!

## Legacy HF Commands

Training (Flow, cached latents, 2 GPUs):

```bash
ACCELERATE_MIXED_PRECISION=bf16 accelerate launch --mixed_precision bf16 --num_processes 2 train_hf.py \
  --train_data_dir /share/users/student/f/friverossego/imagenet_kl16 \
  --vae pretrained_models/kl16.ckpt \
  --use_cached \
  --image_size 256 \
  --num_classes 1000 \
  --model Transformer-Medium \
  --batch_size 128 \
  --num_epochs 40 \
  --learning_rate 1e-4 \
  --adam_weight_decay 0.1 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --prediction_type flow \
  --use_ema \
  --mixed_precision bf16 \
  --output_dir logs/flow_cached_transformer_medium_40e \
  --batch_mul 2
```

Sampling (Flow, CFG=3.0, 20 steps):

```bash
mkdir -p visuals
python sample_hf.py \
  --checkpoint logs/flow_cached_transformer_medium_40e/checkpoint-120000 \
  --vae pretrained_models/kl16.ckpt \
  --image_size 256 \
  --num-classes 1000 \
  --model Transformer-Medium \
  --prediction_type flow \
  --cfg-scale 3.0 \
  --num_inference_steps 20 \
  --image_name transformer_medium_flow_cfg3_120k.png
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using BEiT models, please submit a GitHub issue.

For other communications, please contact [Li Dong](https://dong.li/) (`lidong1@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
