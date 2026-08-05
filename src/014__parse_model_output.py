import torch


def parse_model_output(output):

    if isinstance(output, tuple):

        # LoopedDiT
        if len(output) == 3:
            pred, _, history = output

            return {
                "pred": pred,
                "history": history
            }

        # DiT learn_sigma
        elif len(output) == 2:
            pred, _ = output

            return {
                "pred": pred,
                "history": None
            }

    # DiT no learn_sigma
    return {
        "pred": output,
        "history": None
    }



def main():

    B = 2
    N = 64
    latent_dim = 4


    # ---------------------------
    # Test DiT without sigma
    # ---------------------------

    output = torch.randn(B, N, latent_dim)

    result = parse_model_output(output)

    assert result["pred"].shape == (B, N, latent_dim)
    assert result["history"] is None

    print("DiT no sigma")


    # ---------------------------
    # Test DiT with sigma
    # ---------------------------

    eps = torch.randn(B, N, latent_dim)
    variance = torch.randn(B, N, latent_dim)

    output = (
        eps,
        variance
    )

    result = parse_model_output(output)

    assert torch.equal(
        result["pred"],
        eps
    )

    assert result["history"] is None

    print(" DiT learn sigma")


    # ---------------------------
    # Test LoopedDiT
    # ---------------------------

    history = {
        1: torch.randn(B,N,latent_dim),
        2: torch.randn(B,N,latent_dim),
        3: torch.randn(B,N,latent_dim),
    }


    output = (
        eps,
        variance,
        history
    )


    result = parse_model_output(output)


    assert torch.equal(
        result["pred"],
        eps
    )

    assert result["history"] == history

    print(" LoopedDiT")


    print("\nAll tests passed ")


if __name__ == "__main__":
    main()