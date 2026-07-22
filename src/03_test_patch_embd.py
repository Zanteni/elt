import torch
from model import PatchEmbed


def test_output_shape():
    B, N, patch_dim, d_model = 10, 196, 768, 512
    x = torch.randn(B, N, patch_dim)
    embed = PatchEmbed(patch_dim=patch_dim, d_model=d_model)
    y = embed(x)
    assert y.shape == (B, N, d_model)
    print("[OK] Output shape test passed")


def test_parameter_registration():
    embed = PatchEmbed(patch_dim=768, d_model=512)
    params = list(embed.parameters())
    assert len(params) > 0
    print("[OK] Parameter registration test passed")


def test_numerical_stability():
    x = torch.randn(4, 196, 768)
    embed = PatchEmbed(patch_dim=768, d_model=512)
    y = embed(x)
    assert torch.isfinite(y).all()
    assert not torch.isnan(y).any()
    print("[OK] Numerical stability test passed")


def test_gradient_flow():
    x = torch.randn(2, 196, 768, requires_grad=True)
    embed = PatchEmbed(patch_dim=768, d_model=512)
    y = embed(x)
    loss = y.mean()
    loss.backward()
    assert embed.embed.weight.grad is not None
    assert torch.isfinite(embed.embed.weight.grad).all()
    assert not torch.isnan(embed.embed.weight.grad).any()
    print("[OK] Gradient flow test passed")


if __name__ == "__main__":
    print("===== PATCHEMBED TESTS =====")
    test_output_shape()
    test_parameter_registration()
    test_numerical_stability()
    test_gradient_flow()
    print("\nALL TESTS PASSED")