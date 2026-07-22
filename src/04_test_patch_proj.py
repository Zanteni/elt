import torch
from model import PatchProj


def test_output_shape():
    B, N, d_model, patch_dim = 10, 196, 512, 768
    x = torch.randn(B, N, d_model)
    proj = PatchProj(d_model=d_model, patch_dim=patch_dim)
    y = proj(x)
    assert y.shape == (B, N, patch_dim)
    print("[OK] Output shape test passed")


def test_parameter_registration():
    proj = PatchProj(d_model=512, patch_dim=768)
    params = list(proj.parameters())
    assert len(params) > 0
    print("[OK] Parameter registration test passed")


def test_numerical_stability():
    x = torch.randn(4, 196, 512)
    proj = PatchProj(d_model=512, patch_dim=768)
    y = proj(x)
    assert torch.isfinite(y).all()
    assert not torch.isnan(y).any()
    print("[OK] Numerical stability test passed")


def test_gradient_flow():
    x = torch.randn(2, 196, 512, requires_grad=True)
    proj = PatchProj(d_model=512, patch_dim=768)
    y = proj(x)
    loss = y.mean()
    loss.backward()
    assert proj.proj.weight.grad is not None
    assert torch.isfinite(proj.proj.weight.grad).all()
    assert not torch.isnan(proj.proj.weight.grad).any()
    print("[OK] Gradient flow test passed")


if __name__ == "__main__":
    print("===== PATCHPROJ TESTS =====")
    test_output_shape()
    test_parameter_registration()
    test_numerical_stability()
    test_gradient_flow()
    print("\nALL TESTS PASSED")