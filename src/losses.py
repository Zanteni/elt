import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


# ============================================================================
# Reconstruction Loss Registry
# ============================================================================

SUPPORTED_RECON_LOSSES = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
}


# ============================================================================
# Reconstruction Loss
# ============================================================================

class ReconstructionLoss(nn.Module):
    """
    Computes pixel reconstruction loss.

    Supported:
        - mse
        - l1
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()

        assert loss_type in SUPPORTED_RECON_LOSSES, (
            f"Unknown reconstruction loss '{loss_type}'. "
            f"Supported: {list(SUPPORTED_RECON_LOSSES.keys())}"
        )

        self.loss_fn = SUPPORTED_RECON_LOSSES[loss_type]

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss_fn(recon, target)


# ============================================================================
# KL Divergence
# ============================================================================

class KLDivergenceLoss(nn.Module):
    """
    KL(q(z|x) || N(0,I))

    Inputs:
        mu      : (B, N, latent_dim)
        logvar  : (B, N, latent_dim)

    Returns:
        scalar KL divergence
    """

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        kl = -0.5 * (
            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        # Sum over latent dimension
        kl = kl.sum(dim=-1)

        # Average over tokens and batch
        return kl.mean()


# ============================================================================
# LPIPS Loss
# ============================================================================

class LPIPSLoss(nn.Module):
    """
    Learned perceptual similarity.
    """

    def __init__(self):
        super().__init__()

        self.model = lpips.LPIPS(net="vgg")

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False


    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        # move LPIPS to same device as input
        if next(self.model.parameters()).device != x.device:
            self.model = self.model.to(x.device)

        return self.model(x, y).mean()
    
# ============================================================================
# Complete VAE Loss
# ============================================================================

class VAELoss(nn.Module):
    """
    Total VAE objective

        L = L_rec + beta * L_kl + lambda * L_lpips
    """

    def __init__(
        self,
        beta: float = 1e-6,
        lpips_weight: float = 0.0,
        reconstruction: str = "mse",
    ):
        super().__init__()

        self.reconstruction_loss = ReconstructionLoss(reconstruction)
        self.kl_loss = KLDivergenceLoss()

        self.beta = beta
        self.lpips_weight = lpips_weight

        if lpips_weight > 0:
            self.lpips_loss = LPIPSLoss()
        else:
            self.lpips_loss = None

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):

        reconstruction = self.reconstruction_loss(
            recon,
            target,
        )

        kl = self.kl_loss(
            mu,
            logvar,
        )

        total = reconstruction + self.beta * kl

        if self.lpips_loss is not None:

            perceptual = self.lpips_loss(
                recon,
                target,
            )

            total = total + self.lpips_weight * perceptual

        else:

            perceptual = recon.new_tensor(0.0)

        return {
            "loss": total,
            "reconstruction": reconstruction,
            "kl": kl,
            "lpips": perceptual,
        }

def build_loss(cfg):

    if cfg["model"]["name"] == "vae":

        return VAELoss(
            beta=cfg["loss"]["beta"],
            lpips_weight=cfg["loss"]["lpips_weight"],
            reconstruction=cfg["loss"]["reconstruction"],
        )

    raise ValueError(
        f"Unknown loss for {cfg['model']['name']}"
    )