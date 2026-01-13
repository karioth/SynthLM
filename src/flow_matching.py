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
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
    ):
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = None
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
        timesteps_model = timesteps * 1000.0

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
        timesteps_model = timesteps * 1000.0 
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
        timesteps_model = timesteps * 1000.0 
        timesteps_model = timesteps_model.to(dtype=x0_seq.dtype)

        model_output = model(
            x_noisy,
            timesteps_model,
            x_start=x0_seq,
            y=labels,
            batch_mul=self.batch_mul,
        )
        return F.mse_loss(model_output.float(), velocity.float())


class FlowMatchingSchedulerARDiff(FlowMatchingBase):
    """
    AR-Diffusion scheduler with AD + FIFO sampling.

    AD builds a per-frame timestep schedule and an update mask; FIFO limits the
    active window for efficiency without changing the schedule.
    """

    def __init__(
        self,
        *args,
        ardiff_step: int = 0,
        base_num_frames: Optional[int] = None,
        t_start: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ardiff_step = ardiff_step
        self.base_num_frames = base_num_frames
        self.t_start = float(t_start)

        self.step_template = None
        self.step_template_full = None
        self.timestep_matrix = None
        self.step_index = None
        self.update_masks = None
        self.valid_intervals = None
        self._schedule_frames = None

    def sample_monotone_anchor_times(
        self,
        L: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Continuous monotone timestep sampler (anchor + logit-normal).

        Returns:
          u: (L,) tensor with 0 <= u[0] <= ... <= u[L-1] <= 1
          k: anchor index (python int)
          a: anchor timestep value (scalar tensor)
        """
        device = device or torch.device("cpu")

        # 1) Anchor position
        k_t = torch.randint(0, L, size=(), device=device)
        k = int(k_t.item())

        # 2) Anchor value (logit-normal)
        a = torch.sigmoid(torch.randn((), device=device, dtype=dtype) * self.t_s + self.t_m)

        u = torch.empty((L,), device=device, dtype=dtype)
        u[k] = a

        # 3) Prefix (<= a), then sort to enforce monotonicity
        if k > 0:
            prefix = torch.rand((k,), device=device, dtype=dtype) * a  # in [0, a]
            prefix, _ = torch.sort(prefix)
            u[:k] = prefix

        # 4) Suffix (>= a), then sort to enforce monotonicity
        n_after = L - (k + 1)
        if n_after > 0:
            suffix = a + torch.rand((n_after,), device=device, dtype=dtype) * (1.0 - a)  # in [a, 1]
            suffix, _ = torch.sort(suffix)
            u[k + 1:] = suffix

        return u, k, a

    def get_losses(self, model, x0_seq, labels) -> torch.Tensor:
        bsz, seq_len = x0_seq.shape[:2]
        noise = torch.randn_like(x0_seq)
        t_vec = torch.empty((bsz, seq_len), device=x0_seq.device, dtype=x0_seq.dtype)
        for b in range(bsz):
            u, _, _ = self.sample_monotone_anchor_times(
                seq_len, device=x0_seq.device, dtype=x0_seq.dtype
            )
            t_vec[b] = u

        x_noisy = self.add_noise(x0_seq, noise, t_vec)
        velocity = self.get_velocity(x0_seq, noise, t_vec)
        t_model = t_vec * 1000.0 
        t_model = t_model.to(dtype=x0_seq.dtype)

        model_output = model(x_noisy, t_model, y=labels)
        return F.mse_loss(model_output.float(), velocity.float())

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Build the base step template and reset the cached AD/FIFO schedule.
        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        if timesteps is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `timesteps` is None.")
            self.num_inference_steps = num_inference_steps
            step_template = torch.linspace(
                1.0,
                0.0,
                num_inference_steps + 1,
                device=device,
                dtype=torch.float32,
            )[1:]
        else:
            step_template = torch.tensor(timesteps, dtype=torch.float32, device=device)
            self.num_inference_steps = len(step_template)

        self.step_template = step_template
        self.timesteps = step_template

        self.step_template_full = None
        self.timestep_matrix = None
        self.step_index = None
        self.update_masks = None
        self.valid_intervals = None
        self._schedule_frames = None

    def _build_ad_schedule(
        self,
        total_num_frames: int,
        base_num_frames: int,
        ardiff_step: int,
        step_template: torch.Tensor,
    ):
        """
        Port of AR-Diffusion's generate_timestep_matrix.

        Returns:
          - step_matrix: per-iteration per-frame timesteps
          - step_index: integer step indices per iteration
          - step_update_mask: frames whose step index changed this iteration
          - valid_interval: FIFO window (start, end) per iteration
          - step_template_full: [t_start] + step_template
        """
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template)

        t_start = torch.tensor([self.t_start], device=step_template.device, dtype=step_template.dtype)
        step_template_full = torch.cat([t_start, step_template], dim=0)
        pre_row = torch.zeros(total_num_frames, dtype=torch.long, device=step_template.device)

        while torch.all(pre_row == num_iterations) == False:
            new_row = torch.zeros(total_num_frames, dtype=torch.long, device=step_template.device)
            for i in range(total_num_frames):
                if i == 0 or pre_row[i - 1] == num_iterations:
                    # First frame or previous frame has finished denoising.
                    new_row[i] = pre_row[i] + 1
                else:
                    # Otherwise lag behind by ardiff_step.
                    new_row[i] = new_row[i - 1] - ardiff_step
            new_row = new_row.clamp(0, num_iterations)

            # Update only frames whose step index changed.
            update_mask.append(new_row != pre_row)
            step_index.append(new_row)
            step_matrix.append(step_template_full[new_row])
            pre_row = new_row

        terminal_flag = base_num_frames
        for i in range(0, len(update_mask)):
            if terminal_flag < total_num_frames and update_mask[i][terminal_flag].item():
                # When the next frame becomes active, shift the FIFO window.
                terminal_flag += 1
            valid_interval.append((terminal_flag - base_num_frames, terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)
        return step_matrix, step_index, step_update_mask, valid_interval, step_template_full

    def forward(self, model_fn, sample: torch.Tensor) -> torch.Tensor:
        """
        AD/FIFO sampling loop with per-frame timesteps and update masks.
        """
        if self.step_template is None:
            raise RuntimeError("set_timesteps(...) must be called before sampling.")
        if self.ardiff_step < 0:
            raise ValueError("ardiff_step must be >= 0.")

        total_num_frames = sample.shape[1]
        base_num_frames = self.base_num_frames or total_num_frames
        if base_num_frames > total_num_frames:
            raise ValueError("base_num_frames must be <= total_num_frames.")

        if self.timestep_matrix is None or self._schedule_frames != total_num_frames:
            (
                self.timestep_matrix,
                self.step_index,
                self.update_masks,
                self.valid_intervals,
                self.step_template_full,
            ) = self._build_ad_schedule(
                total_num_frames,
                base_num_frames,
                self.ardiff_step,
                self.step_template,
            )
            self._schedule_frames = total_num_frames

        for i in range(self.timestep_matrix.shape[0]):
            valid_s, valid_e = self.valid_intervals[i]
            x_slice = sample[:, valid_s:valid_e, ...]

            t_cur = self.timestep_matrix[i][valid_s:valid_e].to(sample.device)
            step_index = self.step_index[i][valid_s:valid_e].to(sample.device)
            update_mask = self.update_masks[i][valid_s:valid_e].to(sample.device)

            update_mask = update_mask.unsqueeze(0)
            while update_mask.dim() < x_slice.dim():
                update_mask = update_mask.unsqueeze(-1)
            update_mask = update_mask.to(dtype=x_slice.dtype)

            # Map step indices to actual timesteps; newly-activated frames step from t_start.
            prev_index = torch.clamp(step_index - 1, min=0)
            t_prev = self.step_template_full.to(sample.device)[prev_index]

            t_prev = t_prev.unsqueeze(0).expand(x_slice.shape[0], -1)
            t_cur = t_cur.unsqueeze(0).expand(x_slice.shape[0], -1)

            t_prev_model = t_prev * 1000.0
            model_output = model_fn(x_slice, t_prev_model)

            # Freeze frames whose timestep did not change.
            model_output = model_output * update_mask + x_slice * (1 - update_mask)

            dt = (t_cur - t_prev).to(dtype=x_slice.dtype)
            while dt.dim() < x_slice.dim():
                dt = dt.unsqueeze(-1)
            dt = dt * update_mask
            x_slice = x_slice + dt * model_output

            sample[:, valid_s:valid_e, ...] = x_slice

        return sample
