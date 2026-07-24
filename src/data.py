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

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_to_neg_one_to_one,
    ])


    train_dataset = datasets.ImageFolder(
        root=f"{root}/train",
        transform=transform,
    )


    test_dataset = datasets.ImageFolder(
        root=f"{root}/test",
        transform=transform,
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

    def __init__(self, latents: torch.Tensor, labels=None):
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
    drop_last=True,
):
    """
    Generic DataLoader builder.

    Works for ImageDataset and LatentDataset.

    NOTE on drop_last: defaults to True (standard for training), but should
    be set False for eval/validation loaders -- FID and other eval metrics
    need the full dataset, not a batch-size-dependent, silently-truncated
    subset of it. Pass drop_last=False explicitly when building eval loaders.

    NOTE on shuffle: defaults to True (standard for training), but eval
    loaders should typically pass shuffle=False for reproducible, comparable
    runs across checkpoints.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


# ===========================================================================
# Utilities
# ===========================================================================

def extract_images(batch):
    """
    Supports image, or (image,label), or (image,label,...). Returns image.
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


@torch.no_grad()
def build_latent_cache(
    encoder,
    dataloader,
    device=None,
    save_path=None,
):
    """
    Encodes an entire dataset once.

    Pipeline
        images -> encoder -> mu -> latent tensor

    Returns
    -------
    latents : (N, tokens, latent_dim)

    NOTE: device defaults to None -> auto-selects "cuda" if available, else
    "cpu" (matches build_dataloader's pin_memory logic, avoids crashing on
    CPU-only setups that don't pass an explicit device).

    NOTE: encoder's original .training mode is saved and restored afterward,
    rather than being left permanently flipped to .eval().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    was_training = encoder.training
    encoder.eval()
    encoder.to(device)

    latent_list = []

    try:
        for batch in dataloader:
            images = extract_images(batch)
            images = images.to(device, non_blocking=True)
            mu, _ = encoder(images)
            latent_list.append(mu.cpu())
    finally:
        encoder.train(was_training)

    latents = torch.cat(latent_list, dim=0)

    if save_path is not None:
        torch.save(latents, save_path)

    return latents