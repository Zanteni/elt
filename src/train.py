import os
import torch
import wandb

from accelerate import Accelerator

from data import (
    build_cifar10_datasets,
    build_dataloader,
)

from model import build_model

from losses import VAELoss

from utils import (
    load_config,
    set_seed,
    AverageMeter,
    EMA,
    save_checkpoint,
    move_to_device,
)

from eval import evaluate_reconstruction



# =====================================================
# Setup
# =====================================================

cfg = load_config(
    "configs/default.yaml",
    "configs/vae.yaml"
)


set_seed(cfg["seed"])


os.makedirs(
    "checkpoints",
    exist_ok=True
)


accelerator = Accelerator()

device = accelerator.device



# =====================================================
# Wandb
# =====================================================

if accelerator.is_main_process:

    wandb.init(
        project="elt-vae",
        name="vae-stage1",
        config=cfg
    )



# =====================================================
# Data
# =====================================================

train_dataset, test_dataset = build_cifar10_datasets(
    cfg["data"]["root"]
)


train_loader = build_dataloader(
    train_dataset,
    batch_size=cfg["data"]["batch_size"],
    num_workers=cfg["data"]["num_workers"],
)


test_loader = build_dataloader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=cfg["data"]["num_workers"],
)



# =====================================================
# Model
# =====================================================

model = build_model(cfg)



# =====================================================
# Loss
# =====================================================

criterion = VAELoss(
    beta=float(cfg["loss"]["beta"]),
    lpips_weight=float(cfg["loss"]["lpips_weight"]),
    reconstruction=cfg["loss"]["reconstruction"],
)



# =====================================================
# Optimizer
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(cfg["optimizer"]["lr"]),
    weight_decay=float(cfg["optimizer"]["weight_decay"]),
)



# =====================================================
# Accelerator prepare
# =====================================================

model, optimizer, train_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
)



# =====================================================
# EMA
# =====================================================

ema = EMA(
    accelerator.unwrap_model(model),
    decay=float(cfg["train"]["ema_decay"]),
)



# =====================================================
# Training
# =====================================================

step = 0

meter = AverageMeter()



while step < cfg["train"]["max_steps"]:

    model.train()


    for batch in train_loader:


        # ImageFolder returns:
        # image, label

        images, labels = batch


        images = move_to_device(
            images,
            device
        )


        # -------------------------
        # Forward
        # -------------------------

        out = model(images)



        losses = criterion(
            recon=out["recon"],
            target=images,
            mu=out["mu"],
            logvar=out["logvar"],
        )


        loss = losses["loss"]



        # -------------------------
        # Backward
        # -------------------------

        optimizer.zero_grad()

        accelerator.backward(loss)

        optimizer.step()



        # -------------------------
        # EMA
        # -------------------------

        ema.update(
            accelerator.unwrap_model(model)
        )



        meter.update(
            loss.item(),
            images.size(0)
        )



        # -------------------------
        # Logging
        # -------------------------

        if step % cfg["train"]["log_every"] == 0:


            if accelerator.is_main_process:


                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/reconstruction":
                            losses["reconstruction"].item(),
                        "train/kl":
                            losses["kl"].item(),
                        "step": step,
                    }
                )


                print(
                    f"step {step} | "
                    f"loss {meter.avg:.5f}"
                )



        # -------------------------
        # Evaluation
        # -------------------------

        if (
            step % cfg["train"]["eval_every"] == 0
            and step > 0
        ):


            if accelerator.is_main_process:


                metrics = evaluate_reconstruction(
                    model,
                    test_loader,
                    device=device
                )


                wandb.log(
                    {
                        "eval/reconstruction_mse":
                            metrics["reconstruction_mse"],
                        "step": step,
                    }
                )



        # -------------------------
        # Checkpoint
        # -------------------------

        if (
            step % cfg["train"]["ckpt_every"] == 0
            and step > 0
        ):


            if accelerator.is_main_process:


                save_checkpoint(
                    f"checkpoints/vae_{step}.pt",
                    accelerator.unwrap_model(model),
                    optimizer,
                    epoch=step,
                )


        step += 1


        if step >= cfg["train"]["max_steps"]:
            break



# =====================================================
# Finish
# =====================================================

if accelerator.is_main_process:

    wandb.finish()