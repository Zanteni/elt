import os
import torch
import wandb
import torchvision

from data import build_dataloader, extract_images
from model import build_model,build_vae
from diffusion import build_diffusion
from losses import build_loss

from utils import (
    load_config,
    set_seed,
    setup_environment,
    build_accelerator,
    build_optimizer,
    EMA,
    save_checkpoint,
    maybe_resume,
    move_to_device,
    InfiniteDataLoader,
)


# =====================================================
# Setup
# =====================================================

cfg = load_config(
    "configs/default.yaml",
    "configs/dit.yaml"
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

    wandb.init(
        project="elt-dit",
        name="dit-stage2",
        config=cfg
    )



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
# Load VAE (Frozen)
# =====================================================

# Load VAE architecture config
vae_cfg = load_config(
    "configs/default.yaml",
    "configs/vae.yaml"
)


# Build VAE
vae = build_vae(
    vae_cfg
)


# Load pretrained weights
vae_checkpoint = torch.load(
    cfg["vae"]["checkpoint"],
    map_location="cpu"
)


vae.load_state_dict(
    vae_checkpoint["model"]
)


# Freeze VAE
vae.eval()

for p in vae.parameters():
    p.requires_grad = False


vae.to(device)

# =====================================================
# Build Diffusion
# =====================================================

diffusion = build_diffusion(
    cfg,
    device=device
)



# =====================================================
# Build DiT
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



infinite_loader = InfiniteDataLoader(
    train_loader
)



# =====================================================
# Training
# =====================================================

running_loss = 0.0
count = 0



for step in range(
    start_step,
    total_steps
):

    model.train()


    batch = move_to_device(
        next(infinite_loader),
        device
    )


    images = extract_images(
        batch
    )


    # -------------------------------------------------
    # VAE Encoder
    # -------------------------------------------------

    with torch.no_grad():

        mu, logvar = vae.encoder(
            images
        )

        z0 = mu



        # B,C,H,W -> B,N,D

        B,C,H,W = z0.shape


        z0 = z0.permute(
            0,2,3,1
        )


        z0 = z0.reshape(
            B,
            H*W,
            C
        )



    # -------------------------------------------------
    # Diffusion forward
    # -------------------------------------------------

    zt,t,noise = diffusion(
        z0
    )



    # -------------------------------------------------
    # DiT prediction
    # -------------------------------------------------

    with accelerator.accumulate(model):


        with accelerator.autocast():


            eps_pred = model(
                zt,
                t
            )


            if isinstance(
                eps_pred,
                tuple
            ):

                eps_pred,_ = eps_pred



            losses = criterion(
                eps_pred,
                noise
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
            cfg["train"]["grad_clip_norm"]
        )


        optimizer.step()



    # -------------------------------------------------
    # EMA
    # -------------------------------------------------

    ema.update(
        raw_model
    )



    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    running_loss += loss.detach()

    count += 1



    if step % cfg["train"]["log_every"] == 0:


        avg_loss = (
            running_loss / count
        ).item()



        if accelerator.is_main_process:


            wandb.log(
                {
                    "train/loss":avg_loss,
                    "step":step
                }
            )


            print(
                f"step {step} | loss {avg_loss:.5f}"
            )


        running_loss = 0
        count = 0



    # -------------------------------------------------
    # Checkpoint
    # -------------------------------------------------

    if (

        step % cfg["train"]["ckpt_every"] == 0

        and step > 0

    ):

        if accelerator.is_main_process:


            save_checkpoint(

                os.path.join(
                    checkpoint_dir,
                    f"dit_{step}.pt"
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
            "dit_final.pt"
        ),

        raw_model,

        optimizer,

        ema=ema,

        epoch=total_steps,

        cfg=cfg
    )


    wandb.finish()