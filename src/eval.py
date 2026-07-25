import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from data import extract_images 


@torch.no_grad()
def evaluate_reconstruction(model, dataloader, device=None, num_images=8):
    """
    Evaluate VAE reconstruction quality over the full dataloader.

    Returns:
        reconstruction mse (averaged over every batch in dataloader)
        originals, reconstructions (first num_images, for visualization only)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    was_training = model.training
    model.eval()

    total_loss = 0
    count = 0
    originals = []
    reconstructions = []

    try:
        for batch in dataloader:
            images = extract_images(batch).to(device, non_blocking=True)

            out = model(images)
            recon = out["recon"]

            loss = torch.mean((images - recon) ** 2)
            total_loss += loss.item()
            count += 1

            if len(originals) < num_images:
                remaining = num_images - len(originals)
                originals.append(images[:remaining].cpu())
                reconstructions.append(recon[:remaining].cpu())
    finally:
        model.train(was_training)

    originals = torch.cat(originals, dim=0)[:num_images]
    reconstructions = torch.cat(reconstructions, dim=0)[:num_images]

    return {
        "reconstruction_mse": total_loss / count,
        "originals": originals,
        "reconstructions": reconstructions,
    }


def visualize_reconstruction(originals, reconstructions, save_path=None):
    comparison = torch.cat([originals, reconstructions], dim=0)

    grid = vutils.make_grid(
        comparison,
        nrow=len(originals),
        normalize=True,
        value_range=(-1, 1),
    )

    if save_path is not None:
        vutils.save_image(grid, save_path)
    else:
        plt.figure(figsize=(12, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

@torch.no_grad()
def compute_fid(fid_metric, real_loader, model, vae, diffusion, cfg, device):
    fid_metric.reset()

    num_samples = cfg["eval"]["fid_num_samples"]  # e.g. 1000

    for batch in real_loader:
        images = extract_images(batch).to(device)
        images = (images + 1) / 2  # [-1,1] -> [0,1] for torchmetrics
        fid_metric.update(images, real=True)

    generated = 0
    while generated < num_samples:
        batch_size = min(cfg["eval"]["sample_batch_size"], num_samples - generated)
        y = ...  # per your conditioning setup
        z0 = diffusion.ddim_sample(model, shape=(batch_size, ...), y=y,
                                    steps=cfg["eval"]["ddim_steps"], device=device)
        images = vae.decode(z0)
        images = (images + 1) / 2
        fid_metric.update(images, real=False)
        generated += batch_size

    return fid_metric.compute().item()