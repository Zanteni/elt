import os
import torch
import wandb
import torchvision

from data import build_dataloader, extract_images
from model import build_model
from losses import build_loss

from utils import (
    load_config,
    set_seed,
    setup_environment,
    build_accelerator,
    build_logger,
    build_optimizer,
    EMA,
    save_checkpoint,
    maybe_resume,
    move_to_device,
    InfiniteDataLoader,
)

from eval import build_evaluators


# =====================================================
# Setup
# =====================================================

cfg = load_config(
    "configs/default.yaml",
    "configs/vae.yaml"
)

set_seed(
    cfg["seed"]
)

checkpoint_dir = setup_environment(
    cfg
)

accelerator = build_accelerator(
    cfg
)

device = accelerator.device


# =====================================================
# WandB
# =====================================================

if accelerator.is_main_process:

    build_logger(cfg,accelerator)

# =====================================================
# Dataset
# =====================================================

train_loader = build_dataloader(
    cfg["data"],
    split="train"
)

test_loader = build_dataloader(
    cfg["data"],
    split="test"
)


# =====================================================
# Model + Loss
# =====================================================

model = build_model(
    cfg
)

criterion = build_loss(
    cfg
)


# =====================================================
# Optimizer
# =====================================================

optimizer = build_optimizer(
    model,
    cfg
)


# =====================================================
# Accelerator
# =====================================================

model, optimizer, train_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader
)

raw_model = accelerator.unwrap_model(
    model
)


if cfg["train"].get(
    "use_compile",
    False
):

    model = torch.compile(
        model
    )


infinite_loader = InfiniteDataLoader(
    train_loader
)


# =====================================================
# EMA
# =====================================================

ema = EMA(
    raw_model,
    decay=float(
        cfg["train"]["ema_decay"]
    )
)

# =====================================================
# Evaluators
# =====================================================

evaluators = build_evaluators(
    cfg,
    model=raw_model,
    loaders={
        "test": test_loader
    },
    fid_metric=None,
    device=device
)

# =====================================================
# Resume
# =====================================================

start_step = maybe_resume(
    cfg,
    raw_model,
    optimizer,
    ema,
    device
)


total_steps = cfg["train"]["total_steps"]


if start_step >= total_steps:

    print(
        f"start_step ({start_step}) >= total_steps ({total_steps}) -- nothing to do."
    )

    if accelerator.is_main_process:
        wandb.finish()

    exit()


# =====================================================
# Training
# =====================================================

running_sums = {}
running_count = 0


for step in range(
    start_step,
    total_steps
):

    raw_model.train()


    batch = move_to_device(
        next(infinite_loader),
        device
    )


    images = extract_images(
        batch
    )


    # -------------------------------------------------
    # Forward + Backward
    # -------------------------------------------------

    with accelerator.accumulate(model):

        with accelerator.autocast():

            out = model(
                images
            )


            losses = criterion(
                out["recon"],
                images,
                out["mu"],
                out["logvar"]
            )


            loss = losses["loss"]


        optimizer.zero_grad(
            set_to_none=True
        )


        accelerator.backward(
            loss
        )


        accelerator.clip_grad_norm_(
            model.parameters(),
            max_norm=float(
                cfg["train"]["grad_clip_norm"]
            )
        )


        optimizer.step()



    # -------------------------------------------------
    # EMA
    # -------------------------------------------------

    ema.update(
        raw_model
    )


    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    for k,v in losses.items():

        if k not in running_sums:

            running_sums[k] = torch.zeros(
                (),
                device=device
            )


        running_sums[k] += (
            v.detach()
            *
            images.size(0)
        )


    running_count += images.size(0)



    # =================================================
    # Logging
    # =================================================

    if step % cfg["train"]["log_every"] == 0:


        metrics = {

            k:(
                v / running_count
            ).item()

            for k,v in running_sums.items()

        }


        if accelerator.is_main_process:


            wandb.log(

                {

                    **{

                        f"train/{k}":v

                        for k,v in metrics.items()

                    },

                    "step":step

                }

            )


            print(

                f"step {step} | "

                +

                " ".join(

                    f"{k}:{v:.5f}"

                    for k,v in metrics.items()

                )

            )


        running_sums = {}

        running_count = 0

    # =================================================
    # Evaluation
    # =================================================

    if (
        step % cfg["eval"]["recon_every"] == 0
        and step > 0
    ):

        if accelerator.is_main_process:

            backup = ema.apply_shadow(
                raw_model
            )

            raw_model.eval()

            result = evaluators["reconstruction"].evaluate()

            originals = result["images"]["originals"]

            reconstructions = result["images"]["reconstructions"]

            comparison = torch.cat(
                [
                    originals,
                    reconstructions
                ],
                dim=0
            )

            grid = torchvision.utils.make_grid(
                comparison,
                nrow=len(originals),
                normalize=True,
                value_range=(-1, 1)
            )

            wandb.log(
                {
                    "eval/reconstruction_grid":
                        wandb.Image(
                            grid,
                            caption="top: original | bottom: reconstruction"
                        ),

                    "eval/reconstruction_mse":
                        result["metrics"]["reconstruction_mse"],

                    "step": step
                }
            )

            ema.restore(
                raw_model,
                backup
            )

            raw_model.train()


    # =================================================
    # Checkpoint
    # =================================================

    if (
        step % cfg["train"]["ckpt_every"] == 0
        and step > 0
    ):

        if accelerator.is_main_process:

            save_checkpoint(
                os.path.join(
                    checkpoint_dir,
                    f"{cfg['model']['name']}_{step}.pt"
                ),
                raw_model,
                optimizer,
                ema=ema,
                epoch=step,
                cfg=cfg
            )


# =====================================================
# Final checkpoint
# =====================================================

if accelerator.is_main_process:

    save_checkpoint(
        os.path.join(
            checkpoint_dir,
            f"{cfg['model']['name']}_final.pt"
        ),
        raw_model,
        optimizer,
        ema=ema,
        epoch=total_steps,
        cfg=cfg
    )

    wandb.finish()