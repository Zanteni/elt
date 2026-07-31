import copy
import random
import yaml
import os
import numpy as np
import torch
import wandb
import math
from  model  import build_model
# ============================================================================
# Configuration
# ============================================================================

def load_yaml(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)



def merge_configs(default_cfg, stage_cfg):

    merged = copy.deepcopy(default_cfg)

    for key, value in stage_cfg.items():

        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):

            merged[key] = merge_configs(
                merged[key],
                value
            )

        else:

            merged[key] = value

    return merged



def load_config(default_path, stage_path):

    default_cfg = load_yaml(
        default_path
    )

    stage_cfg = load_yaml(
        stage_path
    )

    return merge_configs(
        default_cfg,
        stage_cfg
    )



# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)



# ============================================================================
# Environment setup
# ============================================================================

def setup_environment(cfg):

    train_cfg = cfg["train"]


    torch.backends.cudnn.benchmark = train_cfg.get(
        "benchmark",
        True
    )


    torch.backends.cudnn.deterministic = train_cfg.get(
        "deterministic",
        False
    )


    allow_tf32 = train_cfg.get(
        "allow_tf32",
        True
    )


    if torch.cuda.is_available():

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        torch.backends.cudnn.allow_tf32 = allow_tf32



    checkpoint_dir = train_cfg.get(
        "checkpoints_dir",
        "checkpoints"
    )


    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )


    return checkpoint_dir



# ============================================================================
# Accelerator
# ============================================================================

def build_accelerator(cfg):

    from accelerate import Accelerator


    train_cfg = cfg["train"]


    accelerator = Accelerator(

        mixed_precision=train_cfg.get(
            "mixed_precision",
            "bf16"
        ),

        gradient_accumulation_steps=train_cfg.get(
            "gradient_accumulation_steps",
            1
        )
    )


    return accelerator
# ============================================================================
#  BUILD
# ============================================================================


def build_logger(cfg, accelerator):
    """
    Initialize experiment logger (Weights & Biases).

    Only the main process creates a logger.
    """

    if not accelerator.is_main_process:
        return

    project = f"elt-{cfg['model']['name']}"

    run_name = (
        f"{cfg['model']['name']}:"
        f"{cfg['model']['variant']}"
    )

    wandb.init(
        project=project,
        name=run_name,
        config=cfg,
    )

# ============================================================================
# Optimizer
# ============================================================================
# ============================================================
# Optimizer Builders
# ============================================================


def build_adamw(params, optim_cfg):

    return torch.optim.AdamW(

        params,

        lr=float(
            optim_cfg["lr"]
        ),

        betas=tuple(
            optim_cfg.get(
                "betas",
                (0.9, 0.999)
            )
        ),

        eps=float(
            optim_cfg.get(
                "eps",
                1e-8
            )
        ),

        fused=optim_cfg.get(
            "fused",
            torch.cuda.is_available()
        ),

    )

OPTIMIZER_BUILDERS = {

    "adamw": build_adamw,

}

def build_optimizer(model, cfg):

    optim_cfg = cfg["optimizer"]


    name = optim_cfg["name"]


    if name not in OPTIMIZER_BUILDERS:

        raise ValueError(
            f"Unknown optimizer '{name}', "
            f"expected one of {list(OPTIMIZER_BUILDERS)}"
        )


    params = build_param_groups(
        model,
        optim_cfg["weight_decay"]
    )


    optimizer = OPTIMIZER_BUILDERS[name](
        params,
        optim_cfg
    )


    return optimizer
# ============================================================================
# Scheduler
# ============================================================================

def build_constant_schedule(optimizer, cfg):
    return None  # VAETrainer already guards scheduler=None correctly -- no fake no-op object needed

def build_linear_warmup_cosine_schedule(optimizer, cfg):
    warmup_steps = cfg["scheduler"].get("warmup_steps", 0)
    total_steps = cfg["train"]["total_steps"]  
    min_lr_ratio = cfg["scheduler"].get("min_lr_ratio", 0.0)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


SCHEDULER_BUILDERS = {
    "constant": build_constant_schedule,
    "linear_warmup_cosine": build_linear_warmup_cosine_schedule,
}

def build_scheduler(optimizer, cfg):
    name = cfg["scheduler"]["name"]
    if name not in SCHEDULER_BUILDERS:
        raise ValueError(f"Unknown scheduler '{name}', expected one of {list(SCHEDULER_BUILDERS)}")
    return SCHEDULER_BUILDERS[name](optimizer, cfg)

# ============================================================================
# Resume training
# ============================================================================

def maybe_resume(
    cfg,
    model,
    optimizer=None,
    ema=None,
    device="cpu"
):

    resume = cfg["train"].get(
        "resume",
        None
    )


    if resume is None:

        return 0



    if resume == "latest":

        checkpoint_dir = cfg["train"].get(
            "checkpoints_dir",
            "checkpoints"
        )


        resume = get_latest_checkpoint(
            checkpoint_dir
        )


        if resume is None:

            print(
                "No checkpoint found, starting from step 0"
            )

            return 0



    step = load_checkpoint(

        resume,

        model,

        optimizer,

        ema=ema,

        device=device

    )


    print(
        f"Resumed from {resume} (step {step})"
    )


    return step



# ============================================================================
# Latest checkpoint
# ============================================================================
def get_latest_checkpoint(directory):

    if not os.path.exists(directory):
        return None


    checkpoints = []


    for f in os.listdir(directory):

        if not f.endswith(".pt"):
            continue

        try:

            step = int(
                f.split("_")[-1]
                .replace(".pt","")
            )

            checkpoints.append(
                (step,f)
            )

        except ValueError:

            continue



    if not checkpoints:

        return None


    checkpoints.sort(
        key=lambda x:x[0]
    )


    return os.path.join(
        directory,
        checkpoints[-1][1]
    )
# ============================================================================
# Model utilities
# ============================================================================

def count_parameters(
    model,
    trainable_only=False
):

    if trainable_only:

        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )


    return sum(
        p.numel()
        for p in model.parameters()
    )



def freeze_model(model):

    for p in model.parameters():

        p.requires_grad = False



def unfreeze_model(model):

    for p in model.parameters():

        p.requires_grad = True



# ============================================================================
# Param groups
# Exclude norm/bias from weight decay
# ============================================================================

def build_param_groups(
    model,
    weight_decay
):

    decay = []

    no_decay = []


    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue


        if (
            param.ndim <= 1
            or "norm" in name.lower()
        ):

            no_decay.append(param)

        else:

            decay.append(param)



    return [

        {
            "params": decay,
            "weight_decay": weight_decay
        },

        {
            "params": no_decay,
            "weight_decay": 0.0
        }

    ]



# ============================================================================
# Device utilities
# ============================================================================

def move_to_device(
    batch,
    device
):

    if torch.is_tensor(batch):

        return batch.to(
            device,
            non_blocking=True
        )


    if isinstance(
        batch,
        (list, tuple)
    ):

        return type(batch)(
            move_to_device(
                x,
                device
            )
            for x in batch
        )


    if isinstance(batch, dict):

        return {

            k: move_to_device(
                v,
                device
            )

            for k,v in batch.items()

        }


    return batch



# ============================================================================
# Average Meter
# ============================================================================

class AverageMeter:


    def __init__(self):

        self.reset()



    def reset(self):

        self.sum = 0.0

        self.count = 0

        self.avg = 0.0



    def update(
        self,
        value,
        n=1
    ):

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count



# ============================================================================
# EMA
# ============================================================================

class EMA:


    def __init__(
        self,
        model,
        decay=0.9999
    ):

        self.decay = decay

        self.shadow = {}


        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = param.data.clone()



    @torch.no_grad()
    def update(self, model):

        for name, param in model.named_parameters():

            if param.requires_grad:

                shadow = self.shadow[name]

                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    self.shadow[name] = shadow


                shadow.mul_(
                    self.decay
                )

                shadow.add_(
                    param.data,
                    alpha=1 - self.decay
                )



    def apply_shadow(self, model):

        backup = {}


        for name, param in model.named_parameters():

            if param.requires_grad:

                backup[name] = param.data.clone()

                param.data.copy_(
                    self.shadow[name]
                )


        return backup



    def restore(
        self,
        model,
        backup
    ):

        for name, param in model.named_parameters():

            if param.requires_grad:

                param.data.copy_(
                    backup[name]
                )



# ============================================================================
# Checkpoints
# ============================================================================
def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch=0,
    cfg=None
):

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict()
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if ema is not None:
        checkpoint["ema"] = ema.shadow

    if cfg is not None:
        checkpoint["config"] = cfg

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device
    )



    model.load_state_dict(
        checkpoint["model"]
    )



    if (
        optimizer is not None
        and "optimizer" in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )



    if (
        scheduler is not None
        and "scheduler" in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )



    if (
        scaler is not None
        and "scaler" in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint["scaler"]
        )



    if (
        ema is not None
        and "ema" in checkpoint
    ):

        ema.shadow = checkpoint["ema"]



    return checkpoint["epoch"]



# ============================================================================
# Infinite DataLoader
# ============================================================================

class InfiniteDataLoader:


    def __init__(
        self,
        dataloader
    ):

        self.loader = dataloader

        self.iterator = iter(
            dataloader
        )



    def __iter__(self):

        return self



    def __next__(self):

        try:

            return next(
                self.iterator
            )


        except StopIteration:

            self.iterator = iter(
                self.loader
            )

            return next(
                self.iterator
            )

def build_vae_from_checkpoint(
    path,
    device="cpu",
    freeze=True
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    cfg = checkpoint["config"]

    vae = build_model(cfg)

    load_checkpoint(
        path,
        vae,
        device=device
    )
    vae.to(device) 
    vae.eval()

    if freeze:
        freeze_model(vae)

    return vae

def denormalize(x):

    x = (x + 1) / 2

    return x.clamp(0,1)

def sample_labels(num_classes, num_images, device, mode="random"):
    if num_classes is None:
        return None
    if mode == "cycle":
        return (torch.arange(num_images, device=device) % num_classes)
    return torch.randint(0, num_classes, (num_images,), device=device)