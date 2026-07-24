"""
Dataset utilities for ELT / ViT-VAE.

Responsibilities
----------------
1. Dataset builders
2. Dataset wrappers
3. DataLoader builders
4. Latent preprocessing utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ===========================================================================
# Normalization
# ===========================================================================

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """
    [0,1] -> [-1,1]
    """
    return x * 2.0 - 1.0


# ===========================================================================
# Dataset Builders
# ===========================================================================

def build_cifar10_datasets(root: str):
    """
    Returns
    -------
    train_dataset
    test_dataset

    Images:
        shape : (3,32,32)
        range : [-1,1]
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_to_neg_one_to_one,
    ])

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )

    return train_dataset, test_dataset


# ===========================================================================
# Dataset Wrappers
# ===========================================================================

class ImageOnlyDataset(Dataset):
    """
    Converts

        (image, label)

    into

        image
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        return sample[0]


class LatentDataset(Dataset):
    """
    Dataset of precomputed latents.

    Unconditional:
        z

    Conditional:
        (z,label)
    """

    def __init__(
        self,
        latents: torch.Tensor,
        labels=None,
    ):

        assert isinstance(latents, torch.Tensor)

        if labels is not None:
            assert len(labels) == len(latents)

        self.latents = latents
        self.labels = labels

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):

        z = self.latents[idx]

        if self.labels is None:
            return z

        return z, self.labels[idx]


# ===========================================================================
# DataLoader Builders
# ===========================================================================

def build_dataloader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
):
    """
    Generic DataLoader builder.

    Works for

        ImageDataset

    and

        LatentDataset
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )


# ===========================================================================
# Utilities
# ===========================================================================

def extract_images(batch):
    """
    Supports

        image

    or

        (image,label)

    or

        (image,label,...)

    Returns

        image
    """

    if isinstance(batch, (list, tuple)):
        return batch[0]

    return batch


@torch.no_grad()
def build_latent_cache(
    encoder,
    dataloader,
    device="cuda",
    save_path=None,
):
    """
    Encodes an entire dataset once.

    Pipeline

        images
            ↓
        encoder
            ↓
           mu
            ↓
        latent tensor

    Returns
    -------
    latents :
        (N, tokens, latent_dim)
    """

    encoder.eval()
    encoder.to(device)

    latent_list = []

    for batch in dataloader:

        images = extract_images(batch)

        images = images.to(
            device,
            non_blocking=True,
        )

        mu, _ = encoder(images)

        latent_list.append(
            mu.cpu()
        )

    latents = torch.cat(
        latent_list,
        dim=0,
    )

    if save_path is not None:

        torch.save(
            latents,
            save_path,
        )

    return latents