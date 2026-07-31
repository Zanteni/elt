"""
Dataset utilities for ELT / ViT-VAE.

Rule: every build_*dataloader* function takes only `cfg` (the data sub-config).
No dataset objects are ever passed in from the caller -- construction happens
entirely inside these functions, driven by config values.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ===========================================================================
# Normalization
# ===========================================================================

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]"""
    return x * 2.0 - 1.0


# ===========================================================================
# Dataset Wrappers
# ===========================================================================

class ImageOnlyDataset(Dataset):
    """Converts (image, label) into image."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


class LatentDataset(Dataset):
    """
    Dataset of precomputed latents.

    Unconditional: z
    Conditional:   (z, label)
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
# Internal: shared DataLoader wrapping, not exposed directly
# ===========================================================================

def _make_loader(dataset, cfg, is_train):
    """
    Shared by every public build_*dataloader* function below.

    is_train controls shuffle/drop_last:
        train -> shuffle=True,  drop_last=True  (standard training)
        eval  -> shuffle=False, drop_last=False (need every sample, reproducible order)
    """
    num_workers = cfg["num_workers"]
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def extract_images(batch):
    """Supports image, or (image,label), or (image,label,...). Returns image."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def extract_labels(batch):
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        return batch[1]
    return None


# ===========================================================================
# Public: image DataLoader (config only)
# ===========================================================================
def build_dataloader(cfg, split="train"):

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        normalize_to_neg_one_to_one,
    ])

    dataset_name = cfg["dataset"].lower()

    if dataset_name == "cifar10":

        dataset = datasets.CIFAR10(
            root=cfg["root"],
            train=(split == "train"),
            transform=transform,
            download=cfg.get("download", True),
        )

    elif dataset_name == "imagefolder":

        dataset = datasets.ImageFolder(
            root=f"{cfg['root']}/{split}",
            transform=transform,
        )

    else:

        raise ValueError(
            f"Unknown dataset '{dataset_name}'"
        )

    return _make_loader(
        dataset,
        cfg,
        is_train=(split == "train"),
    )
# ===========================================================================
# Public: latent DataLoader (config only)
# ===========================================================================

def build_latent_dataloader(cfg, split="train"):
    """
    Loads a precomputed latent cache from disk and wraps it in a DataLoader.

    Expects cfg to contain:
        latent_cache_path_<split>  (e.g. latent_cache_path_train)
        label_cache_path_<split>   (optional; None/absent if unconditional)

    Does not take a dataset or tensor argument -- the cache must already
    exist on disk (see build_latent_cache below), consistent with every
    other loader builder here taking only cfg.
    """
    latents = torch.load(cfg[f"latent_cache_path_{split}"])

    label_key = f"label_cache_path_{split}"
    labels = torch.load(cfg[label_key]) if cfg.get(label_key) else None

    dataset = LatentDataset(latents, labels)
    return _make_loader(dataset, cfg, is_train=(split == "train"))


# ===========================================================================
# Latent cache builder (produces what build_latent_dataloader reads)
# ===========================================================================

@torch.no_grad()
def build_latent_cache(encoder, cfg, split="train", device=None):
    """
    Encodes an entire split once and saves it to cfg["latent_cache_path_<split>"].

    Still takes `encoder` explicitly (a live, already-trained VAE encoder) --
    that can't come from config, it's a runtime object. Everything else
    (which images to encode, where to save) comes from cfg, via build_dataloader
    internally, rather than the caller building and passing an image loader.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_loader = build_dataloader(cfg, split=split)

    was_training = encoder.training
    encoder.eval()
    encoder.to(device)

    latent_list = []
    try:
        for batch in image_loader:
            images = extract_images(batch).to(device, non_blocking=True)
            mu, _ = encoder(images)
            latent_list.append(mu.cpu())
    finally:
        encoder.train(was_training)

    latents = torch.cat(latent_list, dim=0)

    save_path = cfg[f"latent_cache_path_{split}"]
    torch.save(latents, save_path)

    return latents