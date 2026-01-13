"""SAMPLING ONLY (AD + FIFO schedule retained, DDIM removed)."""

from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from tqdm import tqdm


class ADFIFO(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.schedule = schedule

    def generate_timestep_matrix(self, total_num_frames, base_num_frames, ar_step, step_template):
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template)
        step_template = np.concatenate([np.array([999]), step_template], axis=0)
        step_template = torch.tensor(step_template, dtype=torch.long)
        pre_row = torch.zeros(total_num_frames, dtype=torch.long)

        # 当前矩阵的最后一列还不全为最后一个时间步时，说明还没有生成完毕
        while torch.all(pre_row == num_iterations) == False: 
            # 生成下一列
            new_row = torch.zeros(total_num_frames, dtype=torch.long)
            for i in range(total_num_frames):
                if i == 0 or pre_row[i-1] == num_iterations: # 首帧 / 前一个视频帧已经完全去噪音
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i-1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            # print(new_row, new_row != pre_row)
            update_mask.append(new_row != pre_row) # 仅更新步数改变了的视频帧，False: 不用更新， True: 需要更新
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row
        
        # 基于update_mask，计算feed mask
        terminal_flag = base_num_frames
        for i in range(0, len(update_mask)):
            if terminal_flag < total_num_frames and \
                update_mask[i][terminal_flag] == True: # 下一个视频帧要更新，则指向下下个视频帧
                terminal_flag += 1
            valid_interval.append((terminal_flag - base_num_frames, terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)
        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def sample(
               self,
               S,
               batch_size,
               base_num_frames,
               shape, # (batch_size, total_num_frames, n_token_per_frame, n_channel_per_token) or (B, T, D)
               conditioning=None,
               verbose=True,
               x_T=None,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ardiff_step=None, # [0, S]
               step_template=None, # descending timesteps (length S)
               step_fn: Optional[Callable] = None,
               **kwargs):
        """
        AD/FIFO sampling skeleton (DDIM update removed).

        Required:
          - step_template: descending list/array of timesteps (length S)
          - step_fn: callable that applies your update rule
        """
        if step_template is None:
            raise ValueError("step_template is required (descending timesteps).")
        if step_fn is None:
            raise ValueError("step_fn is required to perform the update step.")
        if ardiff_step is None:
            raise ValueError("ardiff_step is required (AD timestep gap).")

        assert ardiff_step >= 0 and ardiff_step <= S, f"ardiff_step must be in [0, {S}]"
        total_num_frames = shape[1]
        device = x_T.device if x_T is not None else torch.device("cpu")
        img = x_T if x_T is not None else torch.randn(shape, device=device)

        timestep_matrix, step_index, update_masks, valid_intervals = self.generate_timestep_matrix(
            total_num_frames, base_num_frames, ardiff_step, step_template
        )

        if verbose:
            iterator = tqdm(range(timestep_matrix.shape[0]), desc='AD Sampler', total=timestep_matrix.shape[0])
        else:
            iterator = list(range(timestep_matrix.shape[0]))

        for i in iterator:
            update_mask = update_masks[i][None, :, None, None].int().to(device)
            ts = timestep_matrix[i].to(device)

            valid_interval_s, valid_interval_e = valid_intervals[i]
            valid_update_mask = update_mask[:, valid_interval_s : valid_interval_e, :, :]
            valid_img = img[:, valid_interval_s : valid_interval_e, :, :]
            valid_ts = ts[valid_interval_s : valid_interval_e]
            valid_index = step_index[i][valid_interval_s : valid_interval_e]

            valid_img = step_fn(
                valid_img,
                valid_ts,
                conditioning,
                unconditional_conditioning,
                unconditional_guidance_scale,
                valid_update_mask,
                valid_index,
            )
            img[:, valid_interval_s : valid_interval_e, :, :] = valid_img

        return img, None

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, update_mask, 
                      repeat_noise=False, use_original_steps=False, 
                      quantize_denoised=False, temperature=1., noise_dropout=0., 
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # NOTE: Do not update values if the noise step is the same
        # print(f"t: {t.shape} update_mask: {update_mask.shape}, x: {x.shape}, model_output: {model_output.shape}")
        model_output = model_output * update_mask + x * (1 - update_mask)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        elif self.model.parameterization == "x0":
            e_t = self.model._predict_eps_from_xstart(x, t, model_output)
        elif self.model.parameterization == "eps":
            e_t = model_output
        else:
            raise NotImplementedError(self.model.parameterization)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = repeat(torch.tensor(alphas[index], dtype=x.dtype), 
                     't -> b t l c', b=b, l=1, c=1).to(device)
        a_prev = repeat(torch.tensor(alphas_prev[index], dtype=x.dtype), 
                        't -> b t l c', b=b, l=1, c=1).to(device)
        sigma_t = repeat(torch.tensor(sigmas[index], dtype=x.dtype), 
                         't -> b t l c', b=b, l=1, c=1).to(device)
        sqrt_one_minus_at = repeat(torch.tensor(sqrt_one_minus_alphas[index], dtype=x.dtype), 
                                   't -> b t l c', b=b, l=1, c=1).to(device)

        # current prediction for x_0
        if self.model.parameterization == "eps":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        elif self.model.parameterization == "v":
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        elif self.model.parameterization == "x0":
            pred_x0 = model_output
        else:
            raise NotImplementedError(self.model.parameterization)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


def cfg_combine(model, x, t, cond, uncond, scale):
    """
    Optional helper for classifier-free guidance (original logic).
    """
    if uncond is None or scale == 1.:
        return model(x, t, cond)
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)
    if isinstance(cond, dict):
        c_in = {}
        for k in cond:
            if isinstance(cond[k], list):
                c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
            else:
                c_in[k] = torch.cat([uncond[k], cond[k]])
    elif isinstance(cond, list):
        c_in = [torch.cat([uncond[i], cond[i]]) for i in range(len(cond))]
    else:
        c_in = torch.cat([uncond, cond])
    model_uncond, model_t = model(x_in, t_in, c_in).chunk(2)
    return model_uncond + scale * (model_t - model_uncond)
