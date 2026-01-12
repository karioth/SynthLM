import torch
import torch.distributed as dist
import torch.nn.functional as F
import lightning as L

from diffusers.optimization import get_scheduler

from .models import All_models, DiT, Transformer
from .schedule import DDPMScheduler, DPMSolverMultistepScheduler, FlowMatchingScheduler
from .tokenizer_models.vae import DiagonalGaussianDistribution


class LitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "Transformer-L",
        input_size: int = 32,
        latent_size: int = 16,
        num_classes: int = 1000,
        dropout: float = 0.0,
        prediction_type: str = "epsilon",
        ddpm_num_steps: int = 1000,
        ddpm_beta_schedule: str = "cosine",
        batch_mul: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = All_models[model_name](
            input_size=input_size,
            in_channels=latent_size,
            num_classes=num_classes,
            flatten_input=False,
            drop=dropout,
        )

        if prediction_type == "flow":
            self.noise_scheduler = FlowMatchingScheduler(
                num_train_timesteps=ddpm_num_steps,
                prediction_type=prediction_type,
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=ddpm_num_steps,
                beta_schedule=ddpm_beta_schedule,
                prediction_type=prediction_type,
            )

        self.register_buffer("sample_weight", torch.ones(ddpm_num_steps, dtype=torch.float32))
        self.register_buffer("scaling_factor", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("bias_factor", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("has_scaling", torch.tensor(False, dtype=torch.bool))
        self._inference_scheduler = None

    def training_step(self, batch, batch_idx):
        moments, labels = batch
        posterior = DiagonalGaussianDistribution(moments)
        x0 = posterior.sample()

        if not self.has_scaling.item():
            self._init_scaling(x0)

        x0 = self._normalize(x0)

        if isinstance(self.model, Transformer):
            loss = self._loss_transformer(x0, labels)
        elif isinstance(self.model, DiT):
            loss = self._loss_dit(x0, labels)
        else:
            raise NotImplementedError("Unsupported model type.")

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def _init_scaling(self, x0):
        x0_float = x0.float()
        scaling = 1.0 / x0_float.flatten().std()
        bias = -x0_float.flatten().mean()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scaling, op=dist.ReduceOp.SUM)
            dist.all_reduce(bias, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            scaling /= world_size
            bias /= world_size

        mu = -bias
        std = 1.0 / scaling
        self.set_data_stats(mu, std)
        self.print(
            f"Scaling factor: {self.scaling_factor.item()}, Bias factor: {self.bias_factor.item()}"
        )

    def set_data_stats(self, mu: torch.Tensor, std: torch.Tensor):
        assert mu.shape == self.bias_factor.shape
        assert std.shape == self.scaling_factor.shape
        std = std.clamp_min(1e-8)
        self.bias_factor.copy_(-mu)
        self.scaling_factor.copy_(1.0 / std)
        self.has_scaling.fill_(True)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_factor.to(dtype=x.dtype)
        scale = self.scaling_factor.to(dtype=x.dtype)
        return (x + bias) * scale

    def unnormalize_latents(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_scaling.item():
            raise RuntimeError("Scaling/bias not set; cannot unnormalize latents.")
        bias = self.bias_factor.to(dtype=x.dtype)
        scale = self.scaling_factor.to(dtype=x.dtype)
        return x / scale - bias

    def _get_inference_scheduler(self):
        if self._inference_scheduler is None:
            if self.hparams.prediction_type == "flow":
                self._inference_scheduler = FlowMatchingScheduler(
                    num_train_timesteps=self.hparams.ddpm_num_steps,
                    prediction_type=self.hparams.prediction_type,
                )
            else:
                self._inference_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=self.hparams.ddpm_num_steps,
                    beta_schedule=self.hparams.ddpm_beta_schedule,
                    prediction_type=self.hparams.prediction_type,
                )
        return self._inference_scheduler

    @torch.no_grad()
    def sample_latents(
        self,
        class_labels,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 250,
        scheduler=None,
    ):
        self.eval()
        if scheduler is None:
            scheduler = self._get_inference_scheduler()

        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=self.device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=self.device, dtype=torch.long)

        y_null = torch.full_like(class_labels, self.hparams.num_classes, device=self.device)
        y = torch.cat([class_labels, y_null], 0)

        def p_sample(model, image):
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
            timesteps_model = timesteps * 1000.0 if self.hparams.prediction_type == "flow" else timesteps
            for t, t_model in zip(timesteps, timesteps_model):
                model_output = model(image, t_model.repeat(image.shape[0]).to(image))
                image = scheduler.step(model_output, t, image).prev_sample
            return image

        latents = self.model.sample_with_cfg(y, cfg_scale, p_sample)
        return self.unnormalize_latents(latents)

    def _loss_transformer(self, x0, labels):
        bsz, latent_size, h, w = x0.shape
        batch_mul = self.hparams.batch_mul
        is_flow = self.hparams.prediction_type == "flow"

        noise = torch.randn(
            (bsz * batch_mul * h * w, latent_size),
            device=x0.device,
            dtype=x0.dtype,
        )
        if is_flow:
            timesteps = torch.rand(
                bsz * batch_mul * h * w,
                device=x0.device,
                dtype=x0.dtype,
            )
        else:
            timesteps = torch.multinomial(
                self.sample_weight,
                bsz * batch_mul * h * w,
                replacement=True,
            )

        x0_rep = (
            x0.repeat_interleave(batch_mul, dim=0)
            .permute(0, 2, 3, 1)
            .reshape(-1, latent_size)
        )
        x_noisy = self.noise_scheduler.add_noise(x0_rep, noise, timesteps)
        velocity = self.noise_scheduler.get_velocity(x0_rep, noise, timesteps)

        x_noisy, noise, velocity = [
            x.reshape(bsz * batch_mul, h, w, latent_size).permute(0, 3, 1, 2)
            for x in (x_noisy, noise, velocity)
        ]
        timesteps = timesteps.reshape(bsz * batch_mul, h * w)
        timesteps_model = timesteps * 1000.0 if is_flow else timesteps
        timesteps_model = timesteps_model.to(dtype=x0.dtype)

        model_output = self.model(
            x_noisy,
            timesteps_model,
            x_start=x0,
            y=labels,
            batch_mul=batch_mul,
        )
        if self.hparams.prediction_type == "epsilon":
            return F.mse_loss(model_output.float(), noise.float())
        if self.hparams.prediction_type in ("v_prediction", "flow"):
            return F.mse_loss(model_output.float(), velocity.float())
        raise NotImplementedError(f"Unsupported prediction_type: {self.hparams.prediction_type}")

    def _loss_dit(self, x0, labels):
        bsz = x0.shape[0]
        is_flow = self.hparams.prediction_type == "flow"

        noise = torch.randn_like(x0)
        if is_flow:
            timesteps = torch.rand(bsz, device=x0.device, dtype=x0.dtype)
        else:
            timesteps = torch.multinomial(self.sample_weight, bsz, replacement=True)

        x_noisy = self.noise_scheduler.add_noise(x0, noise, timesteps)
        velocity = self.noise_scheduler.get_velocity(x0, noise, timesteps)
        timesteps_model = timesteps * 1000.0 if is_flow else timesteps
        timesteps_model = timesteps_model.to(dtype=x0.dtype)

        model_output = self.model(x_noisy, timesteps_model, y=labels)
        if self.hparams.prediction_type == "epsilon":
            return F.mse_loss(model_output.float(), noise.float())
        if self.hparams.prediction_type in ("v_prediction", "flow"):
            return F.mse_loss(model_output.float(), velocity.float())
        raise NotImplementedError(f"Unsupported prediction_type: {self.hparams.prediction_type}")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay.append(p)   # RMSNorm/LN weights, biases, scalars
            else:
                decay.append(p)      # linear/conv weights, embedding matrices, etc.

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
