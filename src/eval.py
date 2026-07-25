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