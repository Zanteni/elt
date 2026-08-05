"""
Comprehensive isolated test for DiTTrainer.train_step:

Checks:
- ELT forward path
- intermediate loop recording
- diffusion loss
- ELT distillation loss
- ELT schedule
- backward
- optimizer update
- baseline path without ELT
"""

import torch
import torch.nn.functional as F
import tempfile

from accelerate import Accelerator

from model import (
    DiTConfig,
    AttentionConfig,
    LoopedDiTConfig,
    LoopedDiT
)

from diffusion import (
    DiffusionConfig,
    GaussianDiffusion
)

from losses import DiffusionLoss
from utils import ELTSchedule



class DummyVAE(torch.nn.Module):

    def __init__(self, num_tokens, latent_dim):

        super().__init__()

        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

        self.param = torch.nn.Parameter(
            torch.zeros(1)
        )


    def encoder(self, images):

        B = images.shape[0]

        mu = torch.randn(
            B,
            self.num_tokens,
            self.latent_dim
        )

        logvar = torch.zeros_like(mu)

        return mu, logvar




def build_cfg(
    elt=True,
    total_steps=20
):

    return {

        "elt":{
            "enabled":elt,
            "strategy":"random",
            "l_min":1,
            "distillation":{
                "schedule":{
                    "name":"linear_warmup",
                    "warmup_ratio":0.5
                },
                "lambda":1.0,
            },
        },


        "model":{
            "name":"looped_dit"
        },


        "conditioning":{
            "enabled":False
        },


        "train":{
            "grad_clip_norm":1.0,
            "total_steps":total_steps,
            "use_compile":False,
            "log_every":5,
            "ckpt_every":1000
        },


        "eval":{
            "every":1000
        },


        "sampling":{
            "enabled":False,
            "every":1000
        },


        "repa":{
            "lambda":0.0
        }

    }





def build_model():

    dit_cfg = DiTConfig(
        latent_dim=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        grid_h=4,
        grid_w=4,
        num_classes=None
    )


    attn_cfg = AttentionConfig(
        d_model=32,
        n_heads=4,
        attention_type="rope",
        grid_h=4,
        grid_w=4
    )


    loop_cfg = LoopedDiTConfig(
        dit_config=dit_cfg,
        loop_steps=4
    )


    return LoopedDiT(
        loop_cfg,
        attn_cfg,
        num_timesteps=50,
        learn_sigma=True
    )





def build_trainer(elt=True):

    from train import DiTTrainer


    cfg = build_cfg(elt)


    model = build_model()


    diffusion = GaussianDiffusion(
        DiffusionConfig(
            timestep=50,
            schedule="cosine",
            learn_sigma=True
        )
    )


    criterion = DiffusionLoss(
        loss_type="mse"
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )


    accelerator = Accelerator()


    trainer = DiTTrainer(
        cfg=cfg,
        model=model,
        vae=DummyVAE(16,4),
        diffusion=diffusion,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=None,
        accelerator=accelerator,
        device="cpu",
        checkpoint_dir=tempfile.mkdtemp(),
        distill=F.mse_loss
    )


    trainer.distill_scheduler = ELTSchedule(
        cfg["elt"]["distillation"],
        cfg["train"]["total_steps"]
    )


    trainer.raw_model=model

    trainer.model, trainer.optimizer = accelerator.prepare(
        model,
        optimizer
    )


    return trainer, model





def test_elt_train_step():


    print("\n========== ELT TRAIN STEP TEST ==========")


    trainer, model = build_trainer(True)


    images = torch.randn(
        2,
        3,
        32,
        32
    )


    batch=(images,None)


    before = {
        n:p.clone()
        for n,p in model.named_parameters()
    }



    output = trainer.train_step(
        batch,
        step=10
    )



    losses = output["losses"]



    print(
        "losses:",
        {
            k:v.item()
            for k,v in losses.items()
        }
    )


    assert "loss" in losses
    assert "elt" in losses


    assert torch.isfinite(
        losses["loss"]
    )


    assert torch.isfinite(
        losses["elt"]
    )


    history = output["history"]


    assert history is not None


    print(
        "history loops:",
        history.keys()
    )


    changed = any(
        not torch.equal(
            before[n],
            p
        )
        for n,p in model.named_parameters()
    )


    assert changed


    print("optimizer update OK")



def test_baseline_train_step():


    print("\n========== BASELINE TRAIN STEP TEST ==========")


    trainer,_ = build_trainer(False)


    batch=(
        torch.randn(
            2,
            3,
            32,
            32
        ),
        None
    )


    output = trainer.train_step(
        batch,
        step=0
    )


    assert "loss" in output["losses"]

    assert "elt" not in output["losses"]


    print(
        "baseline loss:",
        output["losses"]["loss"].item()
    )


    print("baseline OK")

def test_loop_difference_after_training():

    print("\n========== LOOP DIFFERENCE AFTER TRAINING ==========")


    trainer, model = build_trainer(False)

    model.train()


    batch = (
        torch.randn(
            2,
            3,
            32,
            32
        ),
        None
    )


    # -----------------------------
    # normal training warmup
    # -----------------------------

    for step in range(5):

        output = trainer.train_step(
            batch,
            step
        )

        print(
            f"step {step} | loss {output['losses']['loss'].item():.6f}"
        )


    # -----------------------------
    # now ELT forward
    # -----------------------------

    model.eval()


    with torch.no_grad():

        images,_ = batch


        mu,logvar = trainer.vae.encoder(images)

        latents = mu


        x_t,t,noise = trainer.diffusion(latents)


        output = model(
            x_t,
            t,
            None,
            record=[2,4]
        )


        pred,c,history = output



        print(
            "history:",
            history.keys()
        )


        predictions = {
            k:model.final_layer(v,c)
            for k,v in history.items()
        }


        for k,v in predictions.items():

            print(
                f"loop {k} mean:",
                v.abs().mean().item()
            )


        diff = (
            predictions[2]
            -
            predictions[4]
        ).abs().max()


        print(
            "prediction diff:",
            diff.item()
        )


        assert torch.isfinite(diff)


    print("Loop difference test OK")


if __name__=="__main__":

    test_elt_train_step()

    test_baseline_train_step()
    test_loop_difference_after_training()


    print(
        "\nALL DiT TRAINER TESTS PASSED"
    )