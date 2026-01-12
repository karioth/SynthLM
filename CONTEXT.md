# Repo Context (LatentLM)

This note captures the current state of the repo, key scripts, and known gotchas so a new chat can pick up quickly.

## Quick Map
- Core code now lives under `src/`.
- `train_hf.py`: main training entrypoint (Accelerate). Supports DiT and Transformer.
- `src/models/Transformer.py`: causal Transformer + diffusion head (next-token diffusion).
- `src/models/DiT.py`: non-causal diffusion transformer.
- `src/schedule/ddpm.py`: custom DDPM scheduler (supports epsilon and v_prediction).
- `src/schedule/dpm_solver.py`: DPM-Solver scheduler for sampling.
- `src/utils.py`: `load_vae`, `center_crop_arr`, and VAE download helper.
- `sample_hf.py`: sample a grid image from a checkpoint.
- `sample_many.py`: sample many images (random classes).
- `evaluate_fid.py`: multi-GPU FID/IS evaluation (uses torch.distributed).
- `cache_image_latents.py`: cache image latents to .npz for faster training.
- `commands.md`: copy-paste commands for training and sampling.

## Data + VAE Expectations
- ImageNet must be local in ImageFolder layout (class subdirs).
- Training uses `--train_data_dir` and does not auto-download ImageNet.
- VAE is passed via `--vae`:
  - If checkpoint lacks `config`, `load_vae` uses `AutoencoderKL` (KL-16 style).
  - If checkpoint has `config`, `load_vae` uses `sigma_vae`.
- Optional cached latents:
  - Generate with `cache_image_latents.py --data_dir <.../train>` (defaults to `<data_dir>_cached`; can override with `--cached_path`).
  - Train with `--use_cached` and point `--train_data_dir` at the cached folder (ImageFolder only; no HF dataset).
- `train_hf.py` computes `scaling_factor` and `bias_factor` from latents and saves them in `other_state.pth`.
- Sampling/eval must read that `other_state.pth` for correct decoding.

## Paper Alignment (ImageNet)
- Medium model in paper corresponds to `Transformer-Medium` / `DiT-Medium` here:
  - hidden size 1024, depth 24, heads 16, diffusion head 3.
- Tokenizer analysis setup (appendix):
  - LR 1e-4, cosine schedule, warmup 100, weight decay 0.1.
  - Batch size 256, ~200k steps (about 40 epochs).
  - v_prediction and cosine beta schedule.
  - DPM-Solver with 20 steps at inference.
- Scaling setup in paper uses different LR and larger batch (2048).
- `batch_mul` is repo-specific (Transformer only); not mentioned in paper.

## Class Conditioning / CFG
- Class label dropout for CFG is enabled by default: `class_dropout_prob=0.1`.
- Not exposed as a CLI arg; change in model constructors if needed.
- Sampling uses classifier-free guidance by duplicating labels with a null class.

## AR Model Data Flow (Transformer)
This is the causal Transformer + diffusion head (next-token diffusion).

### Training (teacher-forced)
- Input images -> VAE encode -> latents (C,H,W). Latents are normalized using `scaling_factor` and `bias_factor`.
- The Transformer conditioning path consumes **clean** latents:
  - `forward_parallel(x_start, y)` embeds tokens, prepends class embedding, shifts tokens, runs causal blocks.
  - Output is a per-token `condition` sequence (via `condition_layer`).
- The diffusion head consumes **noisy** latents:
  - Each token is noised independently via DDPM at sampled timesteps.
  - `forward_diffusion(x_noise, t, condition)` adds timestep embeddings to `condition`, runs MLP diffusion blocks, predicts noise or velocity.
- Loss: MSE vs noise (epsilon) or vs velocity (v_prediction), depending on `--prediction_type`.

### Flow Matching (Rectified Flow)
- Uses the same diffusion head and model API; `--prediction_type flow` switches training/sampling behavior.
- Training samples t ~ U(0,1) and targets velocity (noise - x0); the model sees `t * 1000` for embeddings.
- Sampling uses Euler steps over descending timesteps in [1, 0); `--num_inference_steps` controls the step count.

### Inference (autoregressive sampling)
- Generate tokens left-to-right.
- For each token index `i`:
  - Use `forward_recurrent` with KV cache to produce `condition` for the current step (class embedding at step 0, then embedded previous token).
  - Run diffusion sampling (DPM-Solver) for that token, using `forward_with_cfg` to apply CFG.
  - The sampled token is appended; repeat until full sequence is generated.
- After all tokens, reshape to latent grid and decode via VAE.

### CFG (Transformer)
- Training: label dropout in `LabelEmbedder` creates implicit unconditional examples.
- Sampling: batch is duplicated with null labels; `forward_with_cfg` mixes cond/uncond predictions:
  - `eps = uncond + cfg_scale * (cond - uncond)`.

## Sampling + Evaluation
- `sample_hf.py` imports `load_vae` from `src.utils` and moves VAE to GPU/dtype.
- `sample_hf.py` writes images to `visuals/` (create it if missing).
- For tokenizer-analysis comparison, sample with:
  - `--prediction_type v_prediction`
  - `--num_inference_steps 20`
  - `--cfg-scale 2.5` or `3.0`
- `evaluate_fid.py` requires `torchrun` and `dist.init_process_group`.

## Added Scripts / Fixes
- `vae_roundtrip.py`: loads a random ImageNet image, applies the same crop/normalize, encodes/decodes, saves output.
- `commands.md`: includes training + sampling commands and LD_LIBRARY_PATH hints.

## Environment Gotcha (HPC)
- PIL/torchvision may fail with `GLIBCXX_3.4.29 not found` due to old system libstdc++.
- Fix by ensuring conda libstdc++ is loaded first:
  - `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`
  - if needed: `export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`
- For local testing in this repo, use the `jamendo` conda environment.

## Branching
- There is a `flow` branch created to experiment with flow matching.

## Notes on Performance
- Transformer uses flash-attn2 if available; otherwise SDPA.
- DiT uses SDPA only.
- Training speed is sensitive to `batch_mul` and dataloader workers.

## Paths in Commands
- Many commands in `commands.md` use absolute paths under `/share/users/student/f/friverossego/LatentLM`.
- If repo location changes, update those paths.
