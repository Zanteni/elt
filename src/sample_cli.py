# ============================================================
# sample_cli.py
#
# Generate images from a trained DiT checkpoint
# ============================================================

import os
import torch
import torchvision

from utils import (
    load_config,
    build_vae_from_checkpoint,
    denormalize,
)

from model import build_model
from diffusion import build_diffusion
from sample import build_sampler



# ============================================================
# Load DiT checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    return model



# ============================================================
# Main
# ============================================================

def main():


    # ----------------------------
    # Config
    # ----------------------------

    cfg = load_config(
        "configs/default.yaml",
        "configs/dit.yaml",
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(f"Using device: {device}")



    # ----------------------------
    # Build DiT
    # ----------------------------

    print("Building DiT...")

    model = build_model(cfg)


    model = load_checkpoint(
        model,
        cfg["sampling"]["checkpoint"],
        device,
    )


    model.to(device)

    model.eval()



    # ----------------------------
    # Build VAE
    # ----------------------------

    print("Loading VAE...")


    vae = build_vae_from_checkpoint(
        cfg["vae"]["checkpoint"],
        device=device,
        freeze=True,
    )
    print("checkpoint latent_dim:", vae.cfg.latent_dim)
    print("latent_proj.in_features:", vae.decoder.latent_proj.in_features)
    

    vae.eval()



    # ----------------------------
    # Build diffusion
    # ----------------------------

    print("Building diffusion...")


    diffusion = build_diffusion(cfg)

    diffusion.to(device)

    diffusion.eval()



    # ----------------------------
    # Build sampler
    # ----------------------------

    print("Creating sampler...")


    sampler = build_sampler(
    cfg=cfg,
    model=model,
    device=device,
    vae=vae,
    diffusion=diffusion,
)


    # ----------------------------
    # Generate
    # ----------------------------

    print("Generating images...")


    with torch.no_grad():

        outputs = sampler.generate()



    images = outputs["diffusion"]["images"]


    images = denormalize(images)



    # ----------------------------
    # Save
    # ----------------------------

    save_dir = cfg["sampling"].get(
        "save_dir",
        "samples"
    )


    os.makedirs(
        save_dir,
        exist_ok=True,
    )


    for i, img in enumerate(images):

        path = os.path.join(
            save_dir,
            f"{i}.png"
        )

        torchvision.utils.save_image(
            img,
            path,
        )


    print(
        f"Saved {len(images)} images to {save_dir}"
    )



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()