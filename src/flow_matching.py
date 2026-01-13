# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor`):
            The sample at the previous timestep.
        pred_original_sample (`torch.Tensor`, optional):
            The predicted x0 based on the current sample and model output.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class FlowMatchingBase(nn.Module):
    """
    Rectified flow scheduler with a linear path:
        x_t = (1 - t) * x0 + t * x1, where x1 ~ N(0, I).

    Shared base: common training utilities + basic ODE sampler.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
    ):
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = None
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.t_m = t_m
        self.t_s = t_s

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Sets descending timesteps in [1, 0) for Euler integration.
        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        if timesteps is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `timesteps` is None.")
            self.num_inference_steps = num_inference_steps
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)[:-1]
        else:
            timesteps = torch.tensor(timesteps, dtype=torch.float32, device=device)
            self.num_inference_steps = len(timesteps)

        self.timesteps = timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear path interpolation for rectified flow.
        """
        t = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * original_samples + t * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rectified flow target velocity (x1 - x0).
        """
        return noise - sample

    def sample_timesteps(
        self,
        shape,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Sample timesteps with a logit-normal distribution.

        Reference: https://arxiv.org/pdf/2403.03206
        """
        u = torch.randn(shape, device=device, dtype=dtype) * self.t_s + self.t_m
        return torch.sigmoid(u)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchingSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Single Euler step: x_{t_prev} = x_t + (t_prev - t) * v_theta(x_t, t).
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=sample.device, dtype=sample.dtype)
        else:
            t = t.to(device=sample.device, dtype=sample.dtype)

        if not torch.is_tensor(prev_t):
            prev_t = torch.tensor(prev_t, device=sample.device, dtype=sample.dtype)
        else:
            prev_t = prev_t.to(device=sample.device, dtype=sample.dtype)

        dt = prev_t - t
        pred_prev_sample = sample + dt * model_output
        pred_original_sample = sample - t * model_output

        if not return_dict:
            return (pred_prev_sample,)

        return FlowMatchingSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def previous_timestep(self, timestep: Union[torch.Tensor, float, int]) -> torch.Tensor:
        timesteps = self.timesteps
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=timesteps.device, dtype=timesteps.dtype)
        index = (timesteps == timestep).nonzero(as_tuple=True)[0][0]
        if index == timesteps.shape[0] - 1:
            return torch.tensor(0.0, device=timesteps.device, dtype=timesteps.dtype)
        return timesteps[index + 1]

    def forward(self, model_fn, sample: torch.Tensor) -> torch.Tensor:
        """
        ODE sampler entry point. Expects `set_timesteps(...)` to be called first.
        """
        if self.timesteps is None:
            raise RuntimeError("set_timesteps(...) must be called before sampling.")

        timesteps = self.timesteps
        if self.prediction_type == "flow":
            timesteps_model = timesteps * 1000.0
        else:
            timesteps_model = timesteps

        for t, t_model in zip(timesteps, timesteps_model):
            t_in = t_model.repeat(sample.shape[0]).to(sample)
            model_output = model_fn(sample, t_in)
            sample = self.step(model_output, t, sample).prev_sample
        return sample


class FlowMatchingSchedulerDiT(FlowMatchingBase):
    def get_losses(self, model, x0_seq, labels) -> torch.Tensor:
        bsz = x0_seq.shape[0]
        noise = torch.randn_like(x0_seq)
        timesteps = self.sample_timesteps(bsz, device=x0_seq.device, dtype=x0_seq.dtype)

        x_noisy = self.add_noise(x0_seq, noise, timesteps)
        velocity = self.get_velocity(x0_seq, noise, timesteps)
        timesteps_model = timesteps * 1000.0 if self.prediction_type == "flow" else timesteps
        timesteps_model = timesteps_model.to(dtype=x0_seq.dtype)

        model_output = model(x_noisy, timesteps_model, y=labels)
        return F.mse_loss(model_output.float(), velocity.float())


class FlowMatchingSchedulerTransformer(FlowMatchingBase):
    def __init__(self, *args, batch_mul: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_mul = batch_mul

    def get_losses(self, model, x0_seq, labels) -> torch.Tensor:
        bsz, seq_len, latent_size = x0_seq.shape
        noise = torch.randn(
            (bsz * self.batch_mul * seq_len, latent_size),
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )
        timesteps = self.sample_timesteps(
            bsz * self.batch_mul * seq_len,
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )

        x0_rep = x0_seq.repeat_interleave(self.batch_mul, dim=0).reshape(-1, latent_size)
        x_noisy = self.add_noise(x0_rep, noise, timesteps)
        velocity = self.get_velocity(x0_rep, noise, timesteps)

        x_noisy, noise, velocity = [
            x.reshape(bsz * self.batch_mul, seq_len, latent_size)
            for x in (x_noisy, noise, velocity)
        ]
        timesteps = timesteps.reshape(bsz * self.batch_mul, seq_len)
        timesteps_model = timesteps * 1000.0 if self.prediction_type == "flow" else timesteps
        timesteps_model = timesteps_model.to(dtype=x0_seq.dtype)

        model_output = model(
            x_noisy,
            timesteps_model,
            x_start=x0_seq,
            y=labels,
            batch_mul=self.batch_mul,
        )
        return F.mse_loss(model_output.float(), velocity.float())
