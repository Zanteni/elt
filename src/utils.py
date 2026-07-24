import copy
import random
import yaml

import numpy as np
import torch



# ============================================================================
# Configuration
# ============================================================================


def load_yaml(path):

    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)



def merge_configs(default_cfg, stage_cfg):

    merged = copy.deepcopy(default_cfg)


    for key,value in stage_cfg.items():

        if (
            key in merged
            and isinstance(merged[key],dict)
            and isinstance(value,dict)
        ):

            merged[key] = merge_configs(
                merged[key],
                value
            )

        else:

            merged[key]=value


    return merged



def load_config(default_path, stage_path):

    default_cfg = load_yaml(default_path)

    stage_cfg = load_yaml(stage_path)

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


    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False



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
        p.requires_grad=False



def unfreeze_model(model):

    for p in model.parameters():
        p.requires_grad=True



# ============================================================================
# Device
# ============================================================================


def move_to_device(batch,device):

    if torch.is_tensor(batch):

        return batch.to(
            device,
            non_blocking=True
        )


    if isinstance(batch,(list,tuple)):

        return type(batch)(
            move_to_device(x,device)
            for x in batch
        )


    if isinstance(batch,dict):

        return {
            k:move_to_device(v,device)
            for k,v in batch.items()
        }


    return batch



# ============================================================================
# Average meter
# ============================================================================


class AverageMeter:


    def __init__(self):

        self.reset()



    def reset(self):

        self.sum=0.
        self.count=0
        self.avg=0.



    def update(self,value,n=1):

        self.sum += value*n

        self.count += n

        self.avg = self.sum/self.count



# ============================================================================
# EMA
# ============================================================================


class EMA:


    def __init__(
        self,
        model,
        decay=0.9999
    ):

        self.decay=decay

        self.shadow={}


        for name,param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name]=param.data.clone()



    @torch.no_grad()
    def update(self,model):

        for name,param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name].mul_(self.decay)

                self.shadow[name].add_(
                    param.data,
                    alpha=1-self.decay
                )



    def apply_shadow(self,model):

        backup={}


        for name,param in model.named_parameters():

            if param.requires_grad:

                backup[name]=param.data.clone()

                param.data.copy_(
                    self.shadow[name]
                )


        return backup



    def restore(self,model,backup):

        for name,param in model.named_parameters():

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
):


    checkpoint={

        "epoch":epoch,

        "model":model.state_dict()

    }


    if optimizer:

        checkpoint["optimizer"]=optimizer.state_dict()



    if scheduler:

        checkpoint["scheduler"]=scheduler.state_dict()



    if scaler:

        checkpoint["scaler"]=scaler.state_dict()



    if ema:

        checkpoint["ema"]=ema.shadow



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


    checkpoint=torch.load(
        path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    if optimizer and "optimizer" in checkpoint:

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


    if scheduler and "scheduler" in checkpoint:

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )


    if scaler and "scaler" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler"]
        )


    if ema and "ema" in checkpoint:

        ema.shadow = checkpoint["ema"]


    return checkpoint["epoch"]



# ============================================================================
# Infinite loader
# ============================================================================


class InfiniteDataLoader:


    def __init__(self,dataloader):

        self.loader=dataloader

        self.iterator=iter(dataloader)



    def __iter__(self):

        return self



    def __next__(self):

        try:

            return next(self.iterator)


        except StopIteration:

            self.iterator=iter(self.loader)

            return next(self.iterator)