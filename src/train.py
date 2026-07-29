from utils import (maybe_resume,
                   InfiniteDataLoader,
                   move_to_device,
                   save_checkpoint,set_seed,
                   setup_environment,build_accelerator,
                   build_logger,
                   build_optimizer,
                   build_scheduler,
                   load_config,
                   freeze_model,
                   build_vae_from_checkpoint,
                   EMA
                   )
from eval import extract_images,visualize_reconstruction,build_evaluators
from model import build_model
from  diffusion  import  build_diffusion
from losses import build_loss
from  data import build_dataloader
import  torch
import wandb,os

# ============================================================
# Base Trainer
# ============================================================
class BaseTrainer:

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterion,
        train_loader,
        accelerator,
        device,
        checkpoint_dir,
        scheduler=None,
        ema=None,
        logger=None,
        evaluators=None,
    ):

        self.cfg = cfg

        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader

        self.accelerator = accelerator
        self.device = device

        self.scheduler = scheduler
        self.ema = ema

        self.logger = logger

        self.evaluators = (
            evaluators
            if evaluators is not None
            else {}
        )

        self.checkpoint_dir = checkpoint_dir


    # =====================================================
    # Interface
    # =====================================================

    def setup(self):
        """
        Optional initialization before training.
        """
        pass


    def train(self):
        raise NotImplementedError


    def train_step(self, batch):
        raise NotImplementedError


    def validate(self):
        raise NotImplementedError


    def log(self):
        raise NotImplementedError


    def save_checkpoint(self):
        raise NotImplementedError


    def save_final(self):
        raise NotImplementedError

class VAETrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler=None, ema=None, logger=None, evaluators=None):
        super().__init__(cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler, ema, logger, evaluators)
    def setup(self):
                
        self.model, self.optimizer,self.scheduler, self.train_loader = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_loader,
                )
            )
        
        self.raw_model = self.accelerator.unwrap_model(self.model)

        if self.cfg["train"]["use_compile"]:
            self.model = torch.compile(self.model)
                    
        self.start_step  = maybe_resume(cfg=self.cfg,
                                 model=self.raw_model,
                                 optimizer=self.optimizer,
                                 ema=self.ema,
                                 device=self.device)

        self.total_steps =  self.cfg["train"]["total_steps"]
        self.train_iter = InfiniteDataLoader(self.train_loader)
        self.running_sums = {}
        self.running_count = 0
        
    def train_step(self, batch):
        images = extract_images(batch)

        with self.accelerator.accumulate(self.model):

            with self.accelerator.autocast():

                out = self.model(images)

                losses = self.criterion(
                    out["recon"],
                    images,
                    out["mu"],
                    out["logvar"],
                )

                loss = losses["loss"]

            self.optimizer.zero_grad(set_to_none=True)

            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.cfg["train"]["grad_clip_norm"],
            )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        if self.ema is not None:
            self.ema.update(self.raw_model)

        return {
            "losses": losses,
            "batch_size": images.size(0)
            }

    def log(self, step, train_output):

        losses = train_output["losses"]
        batch_size = train_output["batch_size"]

        # accumulate
        for name, value in losses.items():

            if name not in self.running_sums:
                self.running_sums[name] = 0.0

            self.running_sums[name] += value.detach().item() * batch_size

        self.running_count += batch_size

        # only log every N steps
        if step % self.cfg["train"]["log_every"] != 0:
            return

        metrics = {
            name: total / self.running_count
            for name, total in self.running_sums.items()
        }

        if self.accelerator.is_main_process:

            wandb.log(
                {
                    **{f"train/{k}": v for k, v in metrics.items()},
                    "step": step,
                }
            )

            print(
                f"step {step} | "
                + " ".join(
                    f"{k}: {v:.5f}"
                    for k, v in metrics.items()
                )
            )

        self.running_sums = {}
        self.running_count = 0

    def validate(self, step):

        if step == 0:
            return

        if step % self.cfg["eval"]["every"] != 0:
            return

        if not self.accelerator.is_main_process:
            return

        if self.ema is not None:
            backup = self.ema.apply_shadow(self.raw_model)
        try:

            self.raw_model.eval()

            for name, evaluator in self.evaluators.items():

                result = evaluator.evaluate()

                self._log_evaluation(
                    step,
                    name,
                    result,
                )

        finally:
            if self.ema is not None:
                self.ema.restore(
                    self.raw_model,
                    backup,
                )

            self.raw_model.train()




    def _log_evaluation(self,step,name,result):

        log_dict = {"step": step}

        for metric_name, value in result.get("metrics", {}).items():

            log_dict[f"eval/{metric_name}"] = value

        images = result.get("images", {})

        if (
            "originals" in images
            and
            "reconstructions" in images
        ):

            grid = visualize_reconstruction(
                images["originals"],
                images["reconstructions"],
            )

            log_dict["eval/reconstruction"] = wandb.Image(grid)

        wandb.log(log_dict)

    def save_checkpoint(self, step):

        if step % self.cfg["train"]["ckpt_every"] != 0:
            return

        if not self.accelerator.is_main_process:
            return

        path = os.path.join(

            self.checkpoint_dir,

            f"{self.cfg['model']['name']}_{step}.pt"

        )

        save_checkpoint(

            path=path,

            model=self.raw_model,

            optimizer=self.optimizer,

            ema=self.ema,

            epoch=step,

            cfg=self.cfg,

        )
    def save_final_checkpoint(self):

        if not self.accelerator.is_main_process:
            return

        path = os.path.join(

            self.checkpoint_dir,

            f"{self.cfg['model']['name']}_final.pt"

        )

        save_checkpoint(

            path=path,

            model=self.raw_model,

            optimizer=self.optimizer,

            ema=self.ema,

            epoch=self.total_steps,
            cfg=self.cfg
            )   
    def train(self):

        self.setup()

        for step in range(self.start_step, self.total_steps):

            batch = next(self.train_iter)

            train_output = self.train_step(batch)

            self.log(step, train_output)

            self.validate(step)

            self.save_checkpoint(step)

        self.save_final_checkpoint()



def build_vae_trainer(cfg):

    set_seed(cfg["seed"])

    checkpoint_dir = setup_environment(cfg)

    accelerator = build_accelerator(cfg)

    device = accelerator.device
    logger = build_logger(cfg,accelerator)
    train_loader = build_dataloader(cfg["data"], split="train")
    test_loader = build_dataloader(cfg["data"],split="test")

    loaders = {
        "train": train_loader,
        "test": test_loader,
    }

    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)  
    ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))

    evaluators = build_evaluators(cfg, model, loaders,device)  # <- needs loaders/fid_metric, see below

    return VAETrainer(
        cfg, model, optimizer, criterion, train_loader, accelerator, device,
        checkpoint_dir, scheduler=scheduler, ema=ema,logger=logger, evaluators=evaluators,
    )
#======================================================================================
# DIT TRAINER
#======================================================================================


class DiTTrainer(BaseTrainer):
    def __init__(self, cfg, model,vae,diffusion, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler=None, ema=None, logger=None, evaluators=None):
        super().__init__(cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler, ema, logger, evaluators)
        self.vae = vae
        self.diffusion = diffusion

    def setup(self):
        self.model, self.optimizer,self.scheduler, self.train_loader = (
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.train_loader,
                        )
                    )
        
        freeze_model(self.vae)
        self.vae.to(self.device)
        self.vae.eval()

        self.diffusion.to(self.device)
        self.diffusion.eval()

        self.raw_model = self.accelerator.unwrap_model(self.model)
        if self.cfg["train"]["use_compile"]:
            self.model = torch.compile(self.model)
        
        self.start_step  = maybe_resume(cfg=self.cfg,
                                         model=self.raw_model,
                                         optimizer=self.optimizer,
                                         ema=self.ema,
                                         device=self.device)
        
        self.total_steps =  self.cfg["train"]["total_steps"]
        self.train_iter = InfiniteDataLoader(self.train_loader)
        self.running_sums = {}
        self.running_count = 0

    def train_step(self,batch):

        images = extract_images(batch)


        with torch.no_grad():

            mu, logvar = self.vae.encoder(images)

            latents = mu



        with self.accelerator.accumulate(self.model):

            with self.accelerator.autocast():


                x_t,t,noise = self.diffusion(
                    latents
                )


                output = self.model(
                    x_t,
                    t
                )


                eps_pred,_ = self.diffusion._split_output(
                    output
                )


                losses = self.criterion(
                    eps_pred,
                    noise
                )


                loss = losses["loss"]



            self.optimizer.zero_grad(
                set_to_none=True
            )


            self.accelerator.backward(loss)


            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.cfg["train"]["grad_clip_norm"]
            )


            self.optimizer.step()


            if self.scheduler is not None:
                self.scheduler.step()


        if self.ema is not None:
            self.ema.update(
                self.raw_model
            )


        return {
            "losses": losses,
            "batch_size": images.size(0)
        }


    def validate(self, step):

        if step % self.cfg["eval"]["every"] != 0:
            return

        if not self.accelerator.is_main_process:
            return


        if self.ema is not None:
            backup = self.ema.apply_shadow(
                self.raw_model
            )

        try:

            self.raw_model.eval()

            for name, evaluator in self.evaluators.items():

                result = evaluator.evaluate()

                self._log_evaluation(
                    step,
                    name,
                    result,
                )

        finally:

            if self.ema is not None:
                self.ema.restore(
                    self.raw_model,
                    backup,
                )

            self.raw_model.train()

    def log(self,step,train_output):

        losses = train_output["losses"]
        batch_size = train_output["batch_size"]


        for name,value in losses.items():

            if name not in self.running_sums:
                self.running_sums[name]=0.0


            self.running_sums[name]+=(
                value.detach().item()*batch_size
            )


        self.running_count += batch_size



        if step % self.cfg["train"]["log_every"] !=0:
            return



        metrics = {
            name: total/self.running_count
            for name,total in self.running_sums.items()
        }


        if self.accelerator.is_main_process:

            wandb.log(
                {
                    **{
                        f"train/{k}":v
                        for k,v in metrics.items()
                    },
                    "step":step
                }
            )


            print(
                f"step {step} | "
                +
                " ".join(
                    f"{k}: {v:.5f}"
                    for k,v in metrics.items()
                )
            )


        self.running_sums={}
        self.running_count=0


    def save_checkpoint(self,step):

        if step % self.cfg["train"]["ckpt_every"] !=0:
            return


        if not self.accelerator.is_main_process:
            return


        path = os.path.join(
            self.checkpoint_dir,
            f"{self.cfg['model']['name']}_{step}.pt"
        )


        save_checkpoint(
            path=path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            epoch=step,
            cfg=self.cfg
        )

    def save_final_checkpoint(self):

        if not self.accelerator.is_main_process:
            return


        path = os.path.join(
            self.checkpoint_dir,
            f"{self.cfg['model']['name']}_final.pt"
        )


        save_checkpoint(
            path=path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            epoch=self.total_steps,
            cfg=self.cfg
        )
        
    def train(self):

        self.setup()


        for step in range(
            self.start_step,
            self.total_steps
        ):

            batch = next(self.train_iter)


            train_output = self.train_step(batch)


            self.log(
                step,
                train_output
            )


            self.validate(
                step
            )


            self.save_checkpoint(
                step
            )


        self.save_final_checkpoint()

#=============
# DIT FACTORY
#==============

def build_dit_trainer(cfg):

    set_seed(cfg["seed"])

    checkpoint_dir = setup_environment(cfg)

    accelerator = build_accelerator(cfg)

    device = accelerator.device

    logger = build_logger(
        cfg,
        accelerator
    )


    # --------------------------------
    # Data
    # --------------------------------

    train_loader = build_dataloader(
        cfg["data"],
        split="train"
    )

    test_loader = build_dataloader(
        cfg["data"],
        split="test"
    )

    loaders = {
        "train": train_loader,
        "test": test_loader,
    }


    # --------------------------------
    # DiT model
    # --------------------------------

    model = build_model(cfg)


    # --------------------------------
    # VAE
    # --------------------------------

    vae = build_vae_from_checkpoint(
        cfg["vae"]["checkpoint"],
        device=device,
        freeze=True,
    )


    # --------------------------------
    # Diffusion
    # --------------------------------

    diffusion = build_diffusion(
        cfg,
        device=device
    )


    # --------------------------------
    # Loss
    # --------------------------------

    criterion = build_loss(cfg)


    # --------------------------------
    # Optimizer
    # --------------------------------

    optimizer = build_optimizer(
        model,
        cfg
    )

    scheduler = build_scheduler(
        optimizer,
        cfg
    )


    ema = EMA(
        model,
        decay=float(cfg["train"]["ema_decay"])
    )


    # --------------------------------
    # Evaluation
    # --------------------------------

    evaluators = build_evaluators(
        cfg,
        model,
        vae,
        diffusion,
        loaders,
        device
    )


    return DiTTrainer(
        cfg,
        model,
        vae,
        diffusion,
        optimizer,
        criterion,
        train_loader,
        accelerator,
        device,
        checkpoint_dir,
        scheduler=scheduler,
        ema=ema,
        logger=logger,
        evaluators=evaluators,
    )

TRAINER_BUILDERS = {
    "vae": build_vae_trainer,
    "dit": build_dit_trainer,
    # "elt": build_elt_trainer,
}


def build_trainer(cfg):
    name = cfg["model"]["name"]
    if name not in TRAINER_BUILDERS:
        raise ValueError(f"Unknown trainer type '{name}', expected one of {list(TRAINER_BUILDERS)}")
    return TRAINER_BUILDERS[name](cfg)

cfg = load_config("configs/default.yaml", "configs/dit.yaml")
trainer = build_trainer(cfg)
trainer.train()