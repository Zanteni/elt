import torch
from model import patchify


def test_output_shape():
    B, C, H, W, P = 10, 3, 224, 224, 16
    x = torch.randn(B, C, H, W)
    y = patchify(x, patch_size=P)
    N = (H // P) * (W // P)
    d_patch = C * P * P
    assert y.shape == (B, N, d_patch)
    print("[OK] Output shape test passed")


def test_3d_input_now_works():
    """Regression test: x=x.unsqueeze(0) fix -- 3D (C,H,W) input should
    now auto-add a batch dim instead of crashing."""
    C, H, W, P = 3, 32, 32, 8
    x = torch.randn(C, H, W)
    y = patchify(x, patch_size=P)
    N = (H // P) * (W // P)
    d_patch = C * P * P
    assert y.shape == (1, N, d_patch)
    print("[OK] 3D input (batch-optional) test passed")


def test_numerical_stability():
    x = torch.randn(4, 3, 224, 224)
    y = patchify(x, patch_size=16)
    assert torch.isfinite(y).all()
    assert not torch.isnan(y).any()
    print("[OK] Numerical stability test passed")


def test_gradient_flow():
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    y = patchify(x, patch_size=16)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    print("[OK] Gradient flow test passed")


if __name__ == "__main__":
    print("===== PATCHIFY TESTS =====")
    test_output_shape()
    test_3d_input_now_works()
    test_numerical_stability()
    test_gradient_flow()
    print("\nALL TESTS PASSED")