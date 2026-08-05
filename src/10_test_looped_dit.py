import torch

from model import (
    DiT,
    LoopedDiT,
    LoopedDiTConfig,
    DiTConfig,
    AttentionConfig,
)


def check_close(a, b, name, tol=1e-6):
    diff = (a - b).abs().max().item()

    print(
        f"{name}: max diff = {diff:.8f}",
        "ok" if diff < tol else "no"
    )

    return diff < tol



def unpack_output(out, learn_sigma):

    if learn_sigma:
        eps, var = out[:2]
        history = out[2] if len(out) == 3 else None
        return eps, var, history

    else:
        x = out[0] if isinstance(out, tuple) else out
        history = out[1] if isinstance(out, tuple) else None
        return x, None, history



def build_configs():

    dit_cfg = DiTConfig(
        latent_dim=4,
        hidden_size=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        grid_h=8,
        grid_w=8,
        num_classes=10,
        cfg_dropout=0.1,
        dropout=0.0,
    )


    attn_cfg = AttentionConfig(
        d_model=192,
        n_heads=3,
        attention_type="rope",
        grid_h=8,
        grid_w=8,
    )


    return dit_cfg, attn_cfg



# ---------------------------------------------------
# Test vanilla DiT forward
# ---------------------------------------------------

def test_dit_forward():

    print("\n==============================")
    print("Test 1: Vanilla DiT forward")
    print("==============================")


    cfg, attn_cfg = build_configs()


    model = DiT(
        cfg,
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=False,
    )


    z = torch.randn(2,64,4)
    t = torch.randint(0,1000,(2,))
    y = torch.randint(0,10,(2,))


    out = model(z,t,y)


    print("Output:",out.shape)



# ---------------------------------------------------
# Test L=1 equivalence
# ---------------------------------------------------

def test_looped_equals_dit(learn_sigma):

    print("\n==============================")
    print(
        f"Test 2: DiT == LoopedDiT L=1 sigma={learn_sigma}"
    )
    print("==============================")


    torch.manual_seed(0)


    cfg, attn_cfg = build_configs()


    dit = DiT(
        cfg,
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=learn_sigma,
    )


    loop_cfg = LoopedDiTConfig(
        dit_config=cfg,
        loop_steps=1
    )


    looped = LoopedDiT(
        loop_cfg,
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=learn_sigma,
    )


    looped.load_state_dict(
        dit.state_dict()
    )


    z = torch.randn(2,64,4)
    t = torch.randint(0,1000,(2,))
    y = torch.randint(0,10,(2,))


    out_dit = dit(z,t,y)

    out_loop = looped(
        z,
        t,
        y,
        record="all"
    )


    a1,a2,h1 = unpack_output(
        out_dit,
        learn_sigma
    )

    b1,b2,h2 = unpack_output(
        out_loop,
        learn_sigma
    )


    check_close(
        a1,
        b1,
        "main output"
    )


    if learn_sigma:

        check_close(
            a2,
            b2,
            "variance output"
        )


    print(
        "History:",
        h2.keys()
    )



# ---------------------------------------------------
# Test multiple loops
# ---------------------------------------------------

def test_multiple_loops():

    print("\n==============================")
    print("Test 3: Multiple loops")
    print("==============================")


    cfg, attn_cfg = build_configs()


    model = LoopedDiT(
        LoopedDiTConfig(
            dit_config=cfg,
            loop_steps=4
        ),
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=False
    )


    z = torch.randn(2,64,4)
    t = torch.randint(0,1000,(2,))
    y = torch.randint(0,10,(2,))


    out = model(
        z,
        t,
        y,
        record="all"
    )


    x, history = out


    print(
        "Final:",
        x.shape
    )


    print(
        "Recorded loops:"
    )


    for k,v in history.items():

        print(
            k,
            v.shape
        )



# ---------------------------------------------------
# Test history modes
# ---------------------------------------------------

def test_history_modes():

    print("\n==============================")
    print("Test 4: History modes")
    print("==============================")


    cfg, attn_cfg = build_configs()


    model = LoopedDiT(
        LoopedDiTConfig(
            dit_config=cfg,
            loop_steps=5
        ),
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=False
    )


    z = torch.randn(1,64,4)
    t = torch.randint(0,1000,(1,))
    y = torch.randint(0,10,(1,))


    for record in [
        None,
        3,
        [1,3,5],
        "all"
    ]:

        print("\nrecord =",record)


        out = model(
            z,
            t,
            y,
            record=record
        )


        x,history = out


        print(
            "output:",
            x.shape
        )

        print(
            "history:",
            history.keys()
        )



# ---------------------------------------------------
# Test gradients
# ---------------------------------------------------

def test_gradients():

    print("\n==============================")
    print("Test 5: Gradient flow")
    print("==============================")


    cfg, attn_cfg = build_configs()


    model = LoopedDiT(
        LoopedDiTConfig(
            dit_config=cfg,
            loop_steps=4
        ),
        attn_cfg,
        num_timesteps=1000,
        learn_sigma=False
    )


    z = torch.randn(2,64,4)
    t = torch.randint(0,1000,(2,))
    y = torch.randint(0,10,(2,))


    out,_ = model(z,t,y)


    loss = out.mean()

    loss.backward()


    grads = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]


    print(
        "Number of gradients:",
        len(grads)
    )


    print(
        "Gradient flow:",
        "OK" if len(grads)>0 else ""
    )




def main():

    test_dit_forward()

    test_looped_equals_dit(False)

    test_looped_equals_dit(True)

    test_multiple_loops()

    test_history_modes()

    test_gradients()


    print(
        "\nALL TESTS FINISHED "
    )



if __name__ == "__main__":
    main()