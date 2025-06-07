from typing import *

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(self,
               model,
               noise,
               cond: Optional[Any] = None,
               steps: int = 50,
               rescale_t: float = 1.0,
               verbose: bool = True,
               **kwargs):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """

    @torch.no_grad()
    def sample(self,
               model,
               noise,
               cond,
               neg_cond,
               steps: int = 50,
               rescale_t: float = 1.0,
               cfg_strength: float = 3.0,
               verbose: bool = True,
               **kwargs):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model,
                              noise,
                              cond,
                              steps,
                              rescale_t,
                              verbose,
                              neg_cond=neg_cond,
                              cfg_strength=cfg_strength,
                              **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """

    @torch.no_grad()
    def sample(self,
               model,
               noise,
               cond,
               neg_cond,
               steps: int = 50,
               rescale_t: float = 1.0,
               cfg_strength: float = 3.0,
               cfg_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True,
               **kwargs):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model,
                              noise,
                              cond,
                              steps,
                              rescale_t,
                              verbose,
                              neg_cond=neg_cond,
                              cfg_strength=cfg_strength,
                              cfg_interval=cfg_interval,
                              **kwargs)

class FlowEulerRepaintSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with RePaint inpainting strategy.
    This implementation follows the RePaint paper (https://arxiv.org/abs/2201.09865).

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min=sigma_min)

    def q_sample(
        self, x_start: torch.Tensor, t: float, noise: Optional[torch.Tensor] = None
    ):
        """
        Forward diffusion process (q sampling) at time t.

        Args:
            x_start: Original clean image [N x C x ...].
            t: Current timestep.
            noise: Optional pre-generated noise. If None, will generate new noise.
        """
        if noise is None:
            if isinstance(x_start, sp.SparseTensor):
                noise = sp.SparseTensor(
                    coords=x_start.coords, feats=torch.randn_like(x_start.feats)
                )
                print(
                    f"[{t}] sampled noise std: {noise.feats.std()} {noise.feats.min()} {noise.feats.max()}"
                )
            else:
                noise = torch.randn_like(x_start)

        # Scale noise according to timestep
        scaled_noise = (self.sigma_min + (1 - self.sigma_min) * t) * noise
        # return x_start + scaled_noise, noise
        # return (1 - (1 - self.sigma_min) * t) * x_start + scaled_noise, noise
        return (1 - t) * x_start + scaled_noise, noise

    def q_sample_from_to(
        self, x_t: torch.Tensor, t: float, t_next: float, resample_method: int = 1
    ):
        """
        Sample from x_t at time t to x_{t_next} at time t_next (t < t_next) using Rectified Flow scheme.
        Used for resampling between timesteps.
        Intrinsically it's a forward diffusion process
        """
        # For rectified flow, we scale both the signal and noise
        if isinstance(x_t, sp.SparseTensor):
            noise = sp.SparseTensor(
                coords=x_t.coords, feats=torch.randn_like(x_t.feats)
            )
        else:
            noise = torch.randn_like(x_t)

        if resample_method == 1:
            # or we can term x_t as x_0
            return self.q_sample(x_t, t_next - t)[0]
        elif resample_method == 3:
            # formula: x_0 * (1 - t) + (self.sigma_min + (1 - self.sigma_min) * t) * eps = x_t
            # this way, x_0 = (x_{t-1} - (self.sigma_min + (1 - self.sigma_min) * t_next) * eps) / (1 - t_next)
            # x_t = x_0 * (1 - t) + (self.sigma_min + (1 - self.sigma_min) * t) * eps
            #     = (1 - t) / (1 - t_next) * x_{t-1} - (1 - t) / (1 - t_next) * (self.sigma_min + (1 - self.sigma_min) * t_next) * eps) + (self.sigma_min + (1 - self.sigma_min) * t) * eps

            x_next = (
                (1 - t_next) / (1 - t) * x_t
                - (1 - t_next)
                / (1 - t)
                * (self.sigma_min + (1 - self.sigma_min) * t)
                * noise
                + (self.sigma_min + (1 - self.sigma_min) * t_next) * noise
            )
            return x_next

        # Scale signal according to time ratio
        # x_next = t_next / t * x_t
        # Add scaled noise
        # noise_scale = torch.sqrt(t_next**2 - (t_next/t)**2 * t**2)
        # x_next = x_next + noise_scale * noise

        # For transition t -> t_next:
        # 1. Scale signal by ratio of signal coefficients at t_next vs t
        signal_scale = (1 - t_next) / (1 - t)

        # 2. Scale noise considering both:
        #    - The transition ratio of noise coefficients
        #    - Maintaining sum of coefficients = 1
        noise_scale = (self.sigma_min + (1 - self.sigma_min) * t_next) * (
            1 - signal_scale
        )

        x_next = signal_scale * x_t + noise_scale * noise
        return x_next

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        mask: torch.Tensor,
        cond: Optional[Any] = None,
        **kwargs,
    ):
        """
        Sample x_{t-1} from the model using RePaint-modified Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            mask: Binary mask indicating regions to inpaint [N x 1 x ...].
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        # Get model predictions
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(
            model, x_t, t, cond, **kwargs
        )

        if isinstance(x_t, sp.SparseTensor):
            print(
                f"[t: {t}, t_prev: {t_prev}] predv feats: {pred_v.feats.std()} {pred_v.feats.min()} {pred_v.feats.max()}"
            )

        # Standard Euler step
        pred_x_prev = x_t - (t - t_prev) * pred_v

        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: torch.Tensor,
        mask: torch.Tensor,
        known_x0: Union[torch.Tensor, sp.SparseTensor],
        cond: Optional[Any] = None,
        resample_times: int = 1,
        resample_method: int = 1,
        steps: int = 50,
        rescale_t: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate inpainted samples using RePaint algorithm.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            mask: Binary mask indicating regions to inpaint [N x 1 x ...].
            known_x0: Known regions of the image to preserve.
            resample_times: Number of resampling steps for each timestep.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the final inpainted samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        device = noise.device
        batch_size = noise.shape[0]

        # Initialize with known content if provided
        sample = noise * mask + known_x0 * (1 - mask)

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        # Generate timestep sequence
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        # Sampling loop
        for t, t_next in tqdm(t_pairs, disable=not verbose):
            is_last_timestep = t_next == 0

            # Multiple resampling steps at each timestep
            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                # Resample masked regions with noise
                # Add noise to known regions according to timestep
                # Q-sample step, sample x_{t-1}^{known} for known region
                noised_known, _ = (
                    self.q_sample(known_x0, t_next)
                    if not is_last_timestep
                    else (known_x0, 0)
                )

                # if isinstance(sample, sp.SparseTensor):
                #     print(
                #         "After q_sample and remix",
                #         sample.feats.min(),
                #         sample.feats.max(),
                #     )
                # else:
                #     print("After q_sample and remix", sample.min(), sample.max())

                # Single sampling step
                # Sample x_{t-1}^{unknown} from x_t
                # in this formulation, the known region will also be changed.
                step_output = self.sample_once(
                    model, sample, t, t_next, mask=mask, cond=cond, **kwargs
                )

                sample = step_output.pred_x_prev

                # Remix to get x_{t-1}
                sample = sample * mask + noised_known * (1 - mask)
                if isinstance(sample, sp.SparseTensor):
                    print(
                        f"After sample_once [t: {t}, r: {r}]",
                        sample.feats[mask.bool()[:, 0]].min(),
                        sample.feats[mask.bool()[:, 0]].max(),
                        sample.feats[~mask.bool()[:, 0]].min(),
                        sample.feats[~mask.bool()[:, 0]].max(),
                    )
                else:
                    print(
                        f"After sample_once [t: {t}, r: {r}]",
                        sample.min(),
                        sample.max(),
                    )

                # Forward diffusion from x_{t-1} to x_t
                # If not the last resample step and not the last timestep,
                # sample noise from t_next to t for next resample iteration
                if not (is_last_resample_step or is_last_timestep):
                    sample = self.q_sample_from_to(
                        sample, t_next, t, resample_method=resample_method
                    )
                    if isinstance(sample, sp.SparseTensor):
                        print(
                            f"After sample_from_to [t: {t}, r: {r}]",
                            sample.feats[mask.bool()[:, 0]].min(),
                            sample.feats[mask.bool()[:, 0]].max(),
                            sample.feats[~mask.bool()[:, 0]].min(),
                            sample.feats[~mask.bool()[:, 0]].max(),
                        )
                    else:
                        print(
                            f"After sample_from_to [t: {t}, r: {r}]",
                            sample.min(),
                            sample.max(),
                        )

                if is_last_timestep:
                    break

            # Store intermediate results
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(step_output.pred_x_0)

        # Final inpainting mask
        sample = sample * mask + known_x0 * (1 - mask)
        ret.samples = sample
        return ret


class FlowEulerRepaintGuidanceIntervalSampler(
    GuidanceIntervalSamplerMixin, FlowEulerRepaintSampler
):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        mask,
        known_x0,
        cond,
        neg_cond,
        resample_times: int = 10,
        resample_method: int = 1,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(
            model,
            noise,
            mask,
            known_x0,
            cond,
            resample_times,
            resample_method,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )
