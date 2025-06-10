import os, os.path as osp
import sys

sys.path.append("..")
import glob
import copy
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import diffusers
import accelerate
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from omegaconf import OmegaConf
from trellis import models as trellis_models
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.models.structured_latent_flow import SLatFlowModel
from trellis.models.conditioner.conditioner_multiview import MultiviewConditioner
from cond_datasets import create_multiview_dataset


def load_config(*yaml_files, cli_args=None, extra_args=None):
    if cli_args is None:
        cli_args = {}
    if extra_args is None:
        extra_args = []
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]

    yaml_confs += [OmegaConf.from_cli(extra_args)]
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf


def get_sigmas(timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="./configs/default.yaml",
    help="config of the training",
)
parser.add_argument(
    "--ss_pretrained",
    type=str,
    default="JeffreyXiang/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16",
    help="Pretrained ss dit model",
)
parser.add_argument(
    "--slat_pretrained",
    type=str,
    default="JeffreyXiang/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16",
    help="Pretrained slat dit model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./outputs",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--logging_dir",
    type=str,
    default="./logs",
    help="The logging directory.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
)
parser.add_argument(
    "--report_to",
    type=str,
    default="tensorboard",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument(
    "--weighting_scheme",
    type=str,
    default="logit_normal",
    choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
)
parser.add_argument(
    "--logit_mean",
    type=float,
    default=1.0,
    help="mean to use when using the `'logit_normal'` weighting scheme.",
)
parser.add_argument(
    "--logit_std",
    type=float,
    default=1.0,
    help="std to use when using the `'logit_normal'` weighting scheme.",
)
parser.add_argument(
    "--mode_scale",
    type=float,
    default=1.29,
    help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
)
# TODO: re-initialize the cross attention layers
parser.add_argument("--reinit_crossattn", action="store_true")

args, extra_args = parser.parse_known_args()

config = load_config(args.config, cli_args=vars(args), extra_args=extra_args)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
os.makedirs(config.output_dir, exist_ok=True)

ss_flow_model: SparseStructureFlowModel = trellis_models.from_pretrained(
    config.ss_pretrained
).cuda()
slat_flow_model: SLatFlowModel = trellis_models.from_pretrained(
    config.slat_pretrained
).cuda()
multiview_feature_builder = MultiviewConditioner().cuda()

cross_attn_lr = 1e-3
feature_builder_lr = 1e-3
params = []
ss_flow_model.requires_grad_(False)
for block in ss_flow_model.blocks:
    block.cross_attn.requires_grad_(True)
    params.append({"params": block.parameters(), "lr": cross_attn_lr})

slat_flow_model.requires_grad_(False)
for block in slat_flow_model.blocks:
    block.cross_attn.requires_grad_(True)
    params.append({"params": block.parameters(), "lr": cross_attn_lr})

# TODO1: fetch learnable parameters from ss_flow and slat_flow

train_dataset = create_multiview_dataset(config.train_data)
eval_dataset = create_multiview_dataset(config.eval_data)

train_loader = DataLoader(train_dataset, batch_size=config.train_bs)
eval_loader = DataLoader(eval_dataset, batch_size=config.train_bs)

params.append(
    {"params": multiview_feature_builder.parameters(), "lr": feature_builder_lr}
)

optimizer = optim.Adam(params, lr=1e-3)

tbar = tqdm(enumerate(train_loader), total=len(train_loader))

# notice that
# slat also has its own normalization mean/std

noise_scheduler = FlowMatchEulerDiscreteScheduler()
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

for batch_id, batch in tbar:
    optimizer.zero_grad()
    cond_images = batch["images"]
    extrinsics, intrinsics = batch["extrinsics"], batch["intrinsics"]
    # should convert the batch ss_latent to voxel form
    model_input = batch["ss_latent"]
    flattened_volume = multiview_feature_builder(cond_images, extrinsics, intrinsics)

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    # generate noise and predict
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    # Add noise according to flow matching.
    # zt = (1 - texp) * x + texp * z1
    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    # (sigmas - 1.0) * model_input - sigmas * noise = sigmas * (model_input - noise) - model_input ~ noise - model_input

    # Predict the noise residual
    model_pred = ss_flow_model(noisy_model_input, timesteps, flattened_volume)

    # these weighting schemes use a uniform timestep sampling
    # and instead post-weight the loss
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config.weighting_scheme, sigmas=sigmas
    )

    target = noise - model_input

    # Compute regular loss.
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        ),
        1,
    )
    loss = loss.mean()

    loss.backward()
    optimizer.step()
    tbar.set_postfix(loss=loss.item())
