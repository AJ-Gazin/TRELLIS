import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    cast_training_params,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from torchvision import transforms
from trellis import models as trellis_models
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.utils.render_utils import render_frames
from trellis.evaluation import Evaluator
from trellis.modules.sparse import SparseTensor, sparse_cat
from trellis.pipelines import TrellisImageTo3DPipeline
from training.cond_datasets import create_image_dataset


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


def get_sigmas(timesteps, noise_scheduler, n_dim=4, dtype=torch.float32, device="cuda"):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def custom_collate_fn(batch):
    """Custom collate function to handle SparseTensor batching."""
    elem = batch[0]
    collated = {}

    for key in elem:
        if isinstance(elem[key], SparseTensor):
            # Handle SparseTensor by concatenating them
            collated[key] = sparse_cat([d[key] for d in batch])
        elif isinstance(elem[key], torch.Tensor):
            # Handle regular tensors using default_collate
            collated[key] = torch.stack([d[key] for d in batch])
        else:
            # For other types (strings, etc.), just create a list
            collated[key] = [d[key] for d in batch]

    return collated


class ImageConditioner(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vitl14_reg",
        device: str = "cuda",
        weight_dtype=torch.float32,
    ):
        super(ImageConditioner, self).__init__()
        self.dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", name, pretrained=True
        )
        self.device = device
        self.dinov2_model.to(device=self.device, dtype=weight_dtype)
        self.dinov2_model.eval()

        self.image_cond_model_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, image):
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.dinov2_model(image, is_training=True)["x_prenorm"]
        cond = F.layer_norm(features, features.shape[-1:])
        neg_cond = torch.zeros_like(cond)
        return cond


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, device="cuda", weight_dtype=torch.float16):
        super(LoRALinear, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.linear = linear_layer
        self.rank = rank

        # Create tensors with the right dtype and device first
        # Parameters defined in this way WON'T be captured, seems weight dtype INCORRECT
        # self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features)).to(
        #     device=device, dtype=weight_dtype
        # )
        # self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank)).to(
        #     device=device, dtype=weight_dtype
        # )
        self.lora_a = nn.Parameter(
            torch.zeros(rank, self.in_features).to(device=device, dtype=weight_dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, rank).to(device=device, dtype=weight_dtype)
        )

        self.scaling = 1.0
        self.reset_parameters()
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)
        self.weight_dtype = weight_dtype

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        ori_dtype = x.dtype
        base_output = self.linear(x)
        x = x.type(self.weight_dtype)
        lora_output = (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return base_output + lora_output.type(ori_dtype)


def inject_lora_layers(model, rank=4, weight_dtype=torch.float16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # remove the `.weight` / `.bias` at the end
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model if parent_name == "" else model.get_submodule(parent_name)
            setattr(
                parent,
                child_name,
                LoRALinear(module, rank=rank, weight_dtype=weight_dtype),
            )
    return model


def train_step(
    config,
    batch,
    ss_flow,
    image_conditioner,
    noise_scheduler,
    optimizer,
    params_to_opt,
    accelerator: Accelerator,
    device="cuda",
    weight_dtype=torch.float32,
):
    optimizer.zero_grad()

    # Get batch data
    images = batch["image"].to(device=device, dtype=weight_dtype)
    ss_latents = batch["ss_latent"].to(device=device, dtype=weight_dtype)

    model_input = ss_latents
    cond = image_conditioner(images)

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
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(
        device=model_input.device, dtype=weight_dtype
    )

    # Add noise according to flow matching.
    # zt = (1 - texp) * x + texp * z1
    sigmas = get_sigmas(
        timesteps, noise_scheduler, n_dim=model_input.ndim, dtype=model_input.dtype
    )
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    # (sigmas - 1.0) * model_input - sigmas * noise = sigmas * (model_input - noise) - model_input ~ noise - model_input

    # Predict the noise residual
    model_pred = ss_flow(noisy_model_input, timesteps, cond)

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

    accelerator.backward(loss)
    if accelerator.sync_gradients:
        # TMP VALUE
        accelerator.clip_grad_norm_(params_to_opt, 5.0)

    # # PRINT THE GRAD AND WEIGHT TO DEBUG
    # for name, param in ss_flow.blocks.named_parameters():
    #     if not param.requires_grad:
    #         if torch.isnan(param).any():
    #             import ipdb

    #             ipdb.set_trace()
    #         continue
    #     print(name, param.grad.max(), param.max())
    #     if torch.isnan(param.grad).any() or torch.isnan(param).any():
    #         import ipdb

    #         ipdb.set_trace()

    optimizer.step()

    # # PRINT THE GRAD AND WEIGHT TO DEBUG
    # for name, param in ss_flow.blocks.named_parameters():
    #     if not param.requires_grad:
    #         if torch.isnan(param).any():
    #             import ipdb

    #             ipdb.set_trace()
    #         continue
    #     print(name, param.grad.max(), param.max())
    #     if torch.isnan(param.grad).any() or torch.isnan(param).any():
    #         import ipdb

    #         ipdb.set_trace()

    return loss.detach()


def validation_step(ss_flow, evaluator, val_dataloader, accelerator, device="cuda"):
    ss_flow.eval()

    # Load the complete TRELLIS pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    pipeline.to(accelerator.device)

    # Replace the pipeline's models with our LoRA-updated versions
    pipeline.ss_flow_model = ss_flow

    all_rendered_images = []
    all_real_images = []

    with torch.no_grad():
        for batch in tqdm(
            val_dataloader, total=len(val_dataloader), desc="Validating..."
        ):
            # Notice that images are B x 3 x H x W
            images = batch["image"].to(device=device)
            # Extrinsics are B x 4 x 4
            extrinsics = batch["extrinsics"].to(device=device)
            intrinsics = batch["intrinsics"].to(device=device)

            # Run the complete pipeline with our updated models
            # It's also OK to input a B x 3 x H x W image as the input
            # Use fewer sampler steps for faster validation
            outputs = pipeline.run(
                images,
                seed=42,
                preprocess_image=False,
                num_samples=len(images),
                sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3,
                },
                verbose=False,  # Use first view as input
            )

            # Get the generated mesh
            # TODO: consider batch rendering
            rendered_views = []
            for gid, gaussian in enumerate(outputs["gaussian"]):
                # Render views using the pipeline's mesh
                # Outside dim: batch, inside dim: view
                rendered_views.extend(
                    render_frames(
                        gaussian,
                        extrinsics[gid : gid + 1],
                        intrinsics[gid : gid + 1],
                        get_tensor=True,
                        verbose=False,
                    )["color"]
                )
            rendered_images = torch.stack(rendered_views, dim=0)  # [B, C, H, W]

            # Collect images for evaluation
            all_rendered_images.extend([img.cpu() for img in rendered_images])
            all_real_images.extend([img.cpu() for img in images])

            # TODO: checkout multi-sample in inference
            torch.cuda.empty_cache()

    # Calculate metrics
    metrics = evaluator.evaluate(
        all_rendered_images, all_real_images, input_images=all_real_images
    )

    ss_flow.train()

    # Clean up pipeline to free memory
    del pipeline
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./training/lora_configs/default.yaml",
        help="config of the training",
    )
    parser.add_argument(
        "--ss_pretrained",
        type=str,
        default="JeffreyXiang/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16",
        help="Pretrained ss dit model",
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
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=2,
        help="Number of steps between validation runs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after warmup period).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA layers.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16", "fp32"],
        default="fp16",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers for memory efficient attention.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading.",
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
    parser.add_argument(
        "--feature_extractor_eval",
        type=str,
        default="inception_v3",
        choices=["inception_v3", "dinov2"],
        help="The feature extractor used to calculate FID/KD score during evaluation.",
    )
    parser.add_argument(
        "--precompute_feature_path",
        type=str,
        default="./datasets/Toys4k/eval_features/inception_v3/features.npy",
        help="The precompute feature path for FID/KD score calculation.",
    )

    args, extra_args = parser.parse_known_args()

    config = load_config(args.config, cli_args=vars(args), extra_args=extra_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="no",
        log_with=config.report_to,
        project_dir=config.logging_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)

    logging.info("Loading the ss_flow model...")
    # Load models
    ss_flow: SparseStructureFlowModel = trellis_models.from_pretrained(
        config.ss_pretrained
    ).cuda()
    ss_flow.requires_grad_(False)

    logging.info("Loading the image conditioner...")
    image_conditioner = ImageConditioner(device="cuda")

    logging.info("injecting lora layers...")
    # Inject LoRA layers
    # Trainable params are all FP16
    ss_flow.blocks = inject_lora_layers(
        ss_flow.blocks, rank=config.lora_rank, weight_dtype=torch.float32
    )
    # Using these seems to work, trainable params is FP32
    # ss_flow.input_layer = LoRALinear(ss_flow.input_layer, weight_dtype=torch.float32)
    # the convert_to_fp16 op seems will affect the `parameters()` results
    # ss_flow.convert_to_fp32()
    # ss_flow.dtype = torch.float32

    # Freeze base models
    # for param in ss_flow.parameters():
    # if not isinstance(param, (nn.Parameter)) or not param.requires_grad:
    # param.requires_grad = False

    # Load dataset
    print("Loading dataset...")
    train_dataset = create_image_dataset(config.train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )

    # Create validation dataset if needed
    if config.validation_interval > 0:
        val_dataset = create_image_dataset(config.eval_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn,
        )

    params_to_opt = [p for p in ss_flow.parameters() if p.requires_grad]
    # Setup optimizer
    optimizer = optim.Adam(
        [
            {
                "params": params_to_opt,
                "lr": config.learning_rate,
            },
        ]
    )
    num_total_params = sum([p.numel() for p in ss_flow.parameters()])
    num_trainable_params = sum([p.numel() for p in params_to_opt])
    logging.info(
        "{:.3f}M parameters in total, {:.3f}M params are trainable.".format(
            num_total_params / 1e6,
            num_trainable_params / 1e6,
        )
    )

    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
    )

    # Setup evaluator
    evaluator = Evaluator(
        feature_extractor=config.feature_extractor_eval,
        device=accelerator.device,
        real_feature_path=config.precompute_feature_path,
    )

    # Prepare for training
    ss_flow, image_conditioner, optimizer, train_dataloader = accelerator.prepare(
        ss_flow, image_conditioner, optimizer, train_dataloader
    )

    # Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        ss_flow.train()

        for batch in train_dataloader:
            loss = train_step(
                config,
                batch,
                ss_flow,
                image_conditioner,
                noise_scheduler,
                optimizer,
                params_to_opt,
                accelerator,
            )
            # loss = torch.tensor([0.0])
            global_step += 1

            if accelerator.is_main_process:
                if global_step % 100 == 0:
                    print(f"Step {global_step}: loss = {loss.item():.4f}")

                if global_step % (config.save_interval * len(train_dataloader)) == 0:
                    # Save checkpoint
                    checkpoint = {
                        "ss_flow_state_dict": accelerator.unwrap_model(
                            ss_flow
                        ).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(config.output_dir, f"checkpoint_{global_step}.pt"),
                    )

                if (
                    global_step % (config.validation_interval * len(train_dataloader))
                    == 0
                ):
                    # Run validation
                    metrics = validation_step(
                        ss_flow, evaluator, val_dataloader, accelerator
                    )
                    print(f"Validation metrics at step {global_step}:")
                    for k, v in metrics.items():
                        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
