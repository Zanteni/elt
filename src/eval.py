import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_reconstruction(
    model,
    dataloader,
    device="cuda",
    num_images=8,
):
    """
    Evaluate VAE reconstruction quality.

    Returns:
        reconstruction mse
        original images
        reconstructed images
    """

    model.eval()
    model.to(device)


    total_loss = 0
    count = 0

    originals = []
    reconstructions = []


    for images in dataloader:

        images = images.to(device)


        out = model(images)

        recon = out["recon"]


        loss = torch.mean(
            (images - recon) ** 2
        )


        total_loss += loss.item()
        count += 1


        if len(originals) < num_images:

            originals.append(
                images.cpu()
            )

            reconstructions.append(
                recon.cpu()
            )


        if len(originals) >= num_images:
            break



    originals = torch.cat(
        originals,
        dim=0
    )[:num_images]


    reconstructions = torch.cat(
        reconstructions,
        dim=0
    )[:num_images]


    return {
        "reconstruction_mse": total_loss/count,
        "originals": originals,
        "reconstructions": reconstructions
    }



def visualize_reconstruction(
    originals,
    reconstructions,
):

    comparison = torch.cat(
        [
            originals,
            reconstructions
        ],
        dim=0
    )


    grid = vutils.make_grid(
        comparison,
        nrow=len(originals),
        normalize=True,
        value_range=(-1,1)
    )


    plt.figure(
        figsize=(12,4)
    )

    plt.imshow(
        grid.permute(1,2,0)
    )

    plt.axis("off")
    plt.show()